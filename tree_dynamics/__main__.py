import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from .io import read_csv, write_json

def main():
    p=argparse.ArgumentParser(description='Tree-canopy model prediction and mask evaluation, with optional manuscript analyses.')
    p.add_argument('task',choices=['verify-files','predict','predict-tif','evaluate-masks','validation','climate','summaries','all'])
    p.add_argument('--output',type=Path,default=Path('outputs'))
    p.add_argument('--input',type=Path,help='Ordered four-band NumPy tile or four-band GeoTIFF')
    p.add_argument('--bands',type=int,nargs=4,metavar=('R','G','B','NIR'),help='Required for predict-tif: 1-based stored indices in model-input order; never inferred')
    p.add_argument('--weights',type=Path,help='Released checkpoint at a custom location')
    p.add_argument('--reference',type=Path,help='Reference binary mask for evaluate-masks')
    p.add_argument('--prediction',type=Path,help='Predicted binary mask for evaluate-masks')
    p.add_argument('--iou-threshold',type=float,default=0.5)
    p.add_argument('--code-only',action='store_true',help='Explicit code-only verification (also the default)')
    p.add_argument('--with-model',action='store_true',help='Also verify the GitHub Releases checkpoint')
    p.add_argument('--with-data',action='store_true',help='Also verify the optional manuscript-derived data archive')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if (a.code_only or a.with_model or a.with_data) and a.task!='verify-files':p.error('Verification flags are supported only for verify-files')
    if a.code_only and (a.with_model or a.with_data):p.error('--code-only cannot be combined with --with-model or --with-data')
    from .integrity import require_analysis_data, verify_weights
    if a.task in ['validation','climate','summaries','all'] or a.with_data:
        try:require_analysis_data(root)
        except (ValueError,FileNotFoundError) as exc:p.error(str(exc))
    if a.task=='verify-files':
        manifests=['file_manifest.json']+(['data_manifest.json'] if a.with_data else [])
        total=0
        for name in manifests:
            inventory=json.loads((root/name).read_text(encoding='utf8'))
            for r in inventory:
                path=(root/r['file']).resolve()
                if not path.is_relative_to(root.resolve()):raise ValueError('Invalid inventory path')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if path.stat().st_size!=r['bytes'] or digest!=r['sha256']:raise ValueError('File integrity mismatch: '+r['file'])
            total+=len(inventory)
        if a.with_model:
            try:verify_weights(a.weights)
            except (ValueError,FileNotFoundError) as exc:p.error(str(exc))
            print('Verified released model checkpoint')
        print('Verified',total,'files across',len(manifests),'manifest(s)');return
    if a.task=='predict-tif':
        if a.input is None or a.bands is None:p.error('predict-tif requires --input and --bands R G B NIR; spectral order is not inferred')
        from .geotiff import predict_geotiff
        print(json.dumps(predict_geotiff(root,a.input,a.output/'canopy_mask.tif',a.bands,a.weights),indent=2));return
    if a.task=='evaluate-masks':
        if a.reference is None or a.prediction is None:p.error('evaluate-masks requires --reference and --prediction')
        from .geotiff import evaluate_masks
        print(json.dumps(evaluate_masks(a.reference,a.prediction,a.output/'mask_metrics.json',a.iou_threshold),indent=2));return
    if a.task=='predict':
        if a.input is None:p.error('predict requires --input')
        from .inference import run
        run(root,a.input,a.output/'canopy_mask.npy',a.weights);return
    a.output.mkdir(parents=True,exist_ok=True);reports={}
    if a.task in ['validation','all']:
        from .validation import run
        run(root,a.output);checks=0
        for name in ['summary','intervals','contrasts','size_summary']:
            actual=read_csv(a.output/('validation_'+name+'.csv'))
            expected=read_csv(root/'data/validation'/('expected_'+name+'.csv'))
            assert len(actual)==len(expected)
            for r,e in zip(actual,expected):
                assert set(r)==set(e)
                for k,v in r.items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool):np.testing.assert_allclose(v,e[k],rtol=1e-9,atol=1e-10,equal_nan=True)
                    else:assert v==e[k],(name,k,v,e[k])
                    checks+=1
        reports['validation']={'matched_saved_results':True,'values_checked':checks}
    if a.task in ['summaries','all']:
        from .summaries import run
        reports['summaries']=run(root,a.output)
    if a.task in ['climate','all']:
        from .climate import run
        reports['climate']=run(root,a.output)
    write_json(a.output/'reproduction_check.json',reports)
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
