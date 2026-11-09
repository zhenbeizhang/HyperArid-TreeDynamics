"""Weighted spatial validation and fixed block resampling from prepared unit data."""
import json
import numpy as np
from .io import write_json, read_jsonl

def fit(x,y,w):
    mu=np.average(x,axis=0,weights=w)
    sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w))
    if np.any(sd<=1e-12):raise ValueError('Constant predictor')
    D=np.c_[np.ones(len(x)),(x-mu)/sd]
    b,_,rank,_=np.linalg.lstsq(D*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)
    if rank!=x.shape[1]+1:raise ValueError('Rank-deficient design')
    raw=np.r_[b[0]-np.dot(b[1:]/sd,mu),b[1:]/sd]
    return b,raw,mu,sd,D

def score(y,p,base,w):
    sse=np.dot(w,(y-p)**2)
    sst=np.dot(w,(y-np.average(y,weights=w))**2)
    sseb=np.dot(w,(y-base)**2)
    return dict(r2=1-sse/sst,skill_vs_train_mean=1-sse/sseb,rmse=np.sqrt(sse/w.sum()),mae=np.dot(w,abs(y-p))/w.sum(),train_mean_rmse=np.sqrt(sseb/w.sum()),SSE=sse)

def run(root,output):
    cases=json.loads((root/'config/climate_cases.json').read_text(encoding='utf8'))
    coefficient_rows=[];cv_rows=[];case_predictions={}
    for i,c in enumerate(cases):
        with np.load(root/'data/climate'/c['file'],allow_pickle=False) as z:
            x,y,w=z['X'],z['y'],z['fit_weights'];area=z['area_weights']
            b,raw,mu,sd,D=fit(x,y,w)
            ub,bi=np.unique(z['unit_block_ids'],return_inverse=True)
            np.testing.assert_array_equal(ub,z['bootstrap_block_ids'])
            mult=z['bootstrap_multiplicities'];pdim=x.shape[1]+1
            assert mult.shape==(1999,len(ub))
            H=np.zeros((len(ub),pdim,pdim));g=np.zeros((len(ub),pdim))
            for k in range(len(ub)):
                t=bi==k;H[k]=(D[t]*w[t,None]).T@D[t];g[k]=D[t].T@(w[t]*y[t])
            Hb=np.einsum('bk,kij->bij',mult,H);gb=mult@g
            condition=np.linalg.cond(Hb);good=np.isfinite(condition)&(condition<1e12)
            boot=np.full((1999,pdim),np.nan)
            boot[good]=np.linalg.solve(Hb[good],gb[good,...,None])[...,0]
            br=boot.copy();br[:,1:]/=sd;br[:,0]-=br[:,1:]@mu
            for j,v in enumerate(c['variables']):
                lo,hi=np.nanquantile(boot[:,j+1],[.025,.975]);lr,hr=np.nanquantile(br[:,j+1],[.025,.975])
                coefficient_rows.append(dict(model=c['model'],case_id=c['case_id'],predictor=v,n=len(y),blocks=len(ub),raw_coefficient=raw[j+1],raw_percentile_low=lr,raw_percentile_high=hr,effect_per_one_predictor_SD=b[j+1],effect_percentile_low=lo,effect_percentile_high=hi,predictor_SD=sd[j],bootstrap_draws=1999,valid_draws=int(good.sum())))
            for scheme in c['schemes']:
                pred=np.full(len(y),np.nan);base=np.full(len(y),np.nan);times=np.zeros(len(y),int)
                for status in z['folds__'+scheme]:
                    tr=status==0;te=status==1
                    if not te.any():continue
                    bb,rr,mm,ss,DD=fit(x[tr],y[tr],w[tr])
                    pred[te]=np.c_[np.ones(te.sum()),(x[te]-mm)/ss]@bb
                    base[te]=np.average(y[tr],weights=w[tr]);times[te]+=1
                assert np.all(times==1) and np.isfinite(pred).all()
                for weighting,ww in [('area',area),('equal',np.ones(len(y)))]:
                    cv_rows.append(dict(model=c['model'],case_id=c['case_id'],scheme=scheme,score_weights=weighting,n=len(y),**score(y,pred,base,ww)))
        if (i+1)%16==0:print('Climate fits:',i+1,'/',len(cases),flush=True)
    write_json(output/'climate_coefficients.json',coefficient_rows)
    write_json(output/'climate_validation.json',cv_rows)
    expected_c=read_jsonl(root/'data/source_data/Climate_all_coefficients_and_intervals.jsonl')
    expected_v=read_jsonl(root/'data/source_data/Climate_all_spatial_validation.jsonl')
    checked=0
    for actual,expected,key in [(coefficient_rows,expected_c,['model','case_id','predictor']),(cv_rows,expected_v,['model','case_id','scheme','score_weights'])]:
        index={tuple(r[k] for k in key):r for r in expected}
        for r in actual:
            e=index[tuple(r[k] for k in key)]
            for field,v in r.items():
                if field in key:continue
                if field not in e:continue
                np.testing.assert_allclose(v,e[field],rtol=1e-7,atol=1e-8,err_msg=str((r['case_id'],field)))
                checked+=1
    report=dict(model_case_combinations=len(cases),coefficient_rows=len(coefficient_rows),validation_rows=len(cv_rows),numerical_values_checked=checked,matched_saved_results=True,relative_tolerance=1e-7,absolute_tolerance=1e-8,scope='Fits, block intervals and held-out prediction metrics from prepared units; not raw climate-grid processing or detection uncertainty')
    write_json(output/'climate_check.json',report)
    return report
