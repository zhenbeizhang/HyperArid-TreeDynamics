"""Aggregate reserved-test statistics from per-tile counts and fixed eligibility."""
from pathlib import Path
import numpy as np
from .io import read_csv, write_csv
from .object_metrics import aggregate, div

def run(root, output):
    rows=read_csv(root/'data/validation/tile_metrics.csv')
    audit=read_csv(root/'data/validation/tile_eligibility.csv')
    sizes=read_csv(root/'data/validation/tile_size_classes.csv')
    tests=[{'id':r['id'],'product':r['product'],'period':r['period']} for r in audit]
    def table(name, values):
        names={'总体及来源时期分层精度.csv':'validation_summary.csv','按来源产品聚类的条件区间.csv':'validation_intervals.csv','两流程配对差值条件区间.csv':'validation_contrasts.csv','面积分级精度汇总.csv':'validation_size_summary.csv'}
        write_csv(output/names[name],values)
    full={r['id'] for r in audit if r['complete_core']};clean={r['id'] for r in audit if r['complete_core'] and not any(r[k] for k in ['binary_edge_core','polygon_edge_core','A_edge_core','B_edge_core'])};allids={r['id'] for r in tests}
    summary=[];ci=[];contrasts=[];size_summary=[]
    SETS=[('all97_diagnostic',allids),('complete95',full),('shared_no_edge',clean)]
    keys=['pixel_tp','pixel_fp','pixel_fn','reference_count','predicted_count','tp','fp','fn','small_reference_count','small_fn','small_predicted_count','small_fp','count_error','absolute_count_error','no_foreground_count','small_no_foreground_count']
    def totals(rr):
     a=aggregate(rr);a['no_foreground_count']=sum(r['no_foreground_count'] for r in rr);a['small_no_foreground_count']=sum(r['small_no_foreground_count'] for r in rr);return a
    def rates(a):
     d={k:a[...,i] for i,k in enumerate(keys+['n'])}
     def ratio(x,y):return np.divide(x,y,out=np.full_like(np.asarray(x,dtype=float),np.nan),where=y!=0)
     return {'pixel_iou':ratio(d['pixel_tp'],d['pixel_tp']+d['pixel_fp']+d['pixel_fn']),'object_precision':ratio(d['tp'],d['tp']+d['fp']),'object_recall':ratio(d['tp'],d['tp']+d['fn']),'object_f1':ratio(2*d['tp'],2*d['tp']+d['fp']+d['fn']),'small_unmatched_rate':ratio(d['small_fn'],d['small_reference_count']),'small_no_foreground_rate':ratio(d['small_no_foreground_count'],d['small_reference_count']),'count_bias_mean':ratio(d['count_error'],d['n']),'count_MAE':ratio(d['absolute_count_error'],d['n'])}
    for set_no,(setname,ids) in enumerate(SETS):
     for period in ['all','2017-2020','2021-2024','unknown']:
      active={r['id'] for r in tests if r['id'] in ids and (period=='all' or r['period']==period)};products=sorted({r['product'] for r in tests if r['id'] in active});k=len(products)
      rng=np.random.default_rng(27+set_no);draws=rng.integers(0,k,size=(2000,k)) if k else np.empty((0,0),int);weights=np.array([np.bincount(d,minlength=k) for d in draws])
      for ref in ['binary','polygon']:
       for th in [.5,.25]:
        boot={};point={}
        for method in ['A','B']:
         rr=[r for r in rows if r['id'] in active and r['method']==method and r['reference']==ref and r['iou_threshold']==th];a=totals(rr)
         summary.append({'sample_set':setname,'period':period,'products':k,'method':method,'reference':ref,'iou_threshold':th,**a})
         arr=np.array([[sum(r[z] for r in rr if r['product']==p) for z in keys]+[sum(r['product']==p for r in rr)] for p in products],float)
         if k and setname!='all97_diagnostic' and k>=5:
          boot[method]=rates(weights@arr);point[method]=rates(arr.sum(0))
          for metric,v in boot[method].items():
           valid=v[np.isfinite(v)];low,high=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
           ci.append({'sample_set':setname,'period':period,'product_clusters':k,'method':method,'reference':ref,'iou_threshold':th,'metric':metric,'estimate':float(point[method][metric]),'conditional_CI_low':float(low),'conditional_CI_high':float(high),'valid_replicates':len(valid)})
         for size in ['small_3_10','medium_10_50','large_50_200']:
          sr=[r for r in sizes if r['id'] in active and r['method']==method and r['reference']==ref and r['iou_threshold']==th and r['size_class']==size];tot={z:sum(r[z] for r in sr) for z in ['reference_count','reference_matched','reference_unmatched','reference_no_foreground','predicted_count','predicted_unmatched']}
          size_summary.append({'sample_set':setname,'period':period,'method':method,'reference':ref,'iou_threshold':th,'size_class':size,**tot,'reference_recall':div(tot['reference_matched'],tot['reference_count']),'reference_no_foreground_rate':div(tot['reference_no_foreground'],tot['reference_count'])})
        if 'A' in boot and 'B' in boot:
         for metric in boot['A']:
          v=boot['B'][metric]-boot['A'][metric];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
          contrasts.append({'sample_set':setname,'period':period,'reference':ref,'iou_threshold':th,'contrast':'B-A','metric':metric,'difference':float(point['B'][metric]-point['A'][metric]),'conditional_CI_low':float(lo),'conditional_CI_high':float(hi),'product_clusters':k,'valid_replicates':len(v)})
    table('总体及来源时期分层精度.csv',summary);table('按来源产品聚类的条件区间.csv',ci);table('两流程配对差值条件区间.csv',contrasts);table('面积分级精度汇总.csv',size_summary)
