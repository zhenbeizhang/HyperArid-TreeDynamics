import json
import numpy as np
from .io import read_csv, write_json

def run(root,output):
    rows=read_csv(root/'data/validation/archived_objects.csv')
    bins=[3,10,30,100,200];summary=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        r=[x for x in rows if x['role']=='reference' and lo<=x['area_m2']<hi]
        p=[x for x in rows if x['role']=='prediction' and lo<=x['area_m2']<hi]
        summary.append(dict(lower_area_m2=lo,upper_exclusive_m2=hi,reference_count=len(r),predicted_count=len(p),signed_count_difference=len(p)-len(r),signed_count_difference_percent=100*(len(p)-len(r))/len(r),reference_area_m2=sum(x['area_m2'] for x in r),predicted_area_m2=sum(x['area_m2'] for x in p)))
    assert [r['reference_count'] for r in summary]==[3081,2256,792,92]
    assert [r['predicted_count'] for r in summary]==[3523,1936,752,83]
    write_json(output/'archived_size_statistics.json',summary)
    scenarios=json.loads((root/'data/source_data/Fig_S1_six_scenarios.json').read_text(encoding='utf8'))
    for r in scenarios:
        assert r['late_count']-r['early_count']==r['net_count']
        np.testing.assert_allclose(100*r['net_count']/r['early_count'],r['change_pct'])
        np.testing.assert_allclose(r['cells']*.01,r['area_km2'])
    report=dict(reference_count=sum(r['reference_count'] for r in summary),predicted_count=sum(r['predicted_count'] for r in summary),reference_area_m2=sum(r['reference_area_m2'] for r in summary),predicted_area_m2=sum(r['predicted_area_m2'] for r in summary),six_exclusion_scenarios_arithmetic_verified=True)
    np.testing.assert_allclose([report['reference_area_m2'],report['predicted_area_m2']],[107589.25,100791.25])
    write_json(output/'summary_check.json',report)
    return report
