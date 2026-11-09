"""Exercise training, deterministic restart, evaluation and prediction on synthetic tiles."""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.common import write_json, load_checkpoint, seed_all
from datasets.tree_dataset import read_image, preprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('Use an empty output directory.')
    data=out/'data';data.mkdir(parents=True)
    rows=[]
    yy,xx=np.mgrid[:256,:256]
    for i in range(4):
        rng=np.random.default_rng(100+i)
        mask=(((xx-(50+35*i))**2+(yy-128)**2)<20**2).astype('uint8')*255
        image=rng.integers(100,2500,(4,256,256),dtype='uint16')
        profile=dict(driver='GTiff',width=256,height=256,crs='EPSG:32637',transform=from_origin(200000+i*128,3100000,.5,.5))
        with rasterio.open(data/f'i{i}.tif','w',count=4,dtype='uint16',**profile) as dst:dst.write(image)
        with rasterio.open(data/f'm{i}.tif','w',count=1,dtype='uint8',**profile) as dst:dst.write(mask,1)
        rows.append(dict(sample_id=f'S{i}',image_file=f'i{i}.tif',mask_file=f'm{i}.tif',source_id=f'G{i}',parent_id=f'P{i}'))
    with (data/'metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    runs=[]
    def run(label,args):
        result=subprocess.run([sys.executable,'-B','-X','utf8',*map(str,args)],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
                              capture_output=True,text=True,encoding='utf-8')
        (out/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(f'{label} failed: {result.stderr}')
        runs.append(label)
        print(label+' passed',flush=True)
    run('split',['scripts/prepare_splits.py','--metadata',data/'metadata.csv','--output',out/'splits'])
    base=['train.py','--data-root',data,'--splits',out/'splits','--bands',1,2,3,4,'--epochs',2,'--device','cpu']
    run('train_full',[*base,'--output',out/'full'])
    run('train_repeat',[*base,'--output',out/'repeat'])
    run('train_pause',[*base,'--output',out/'resumed','--stop-after-epoch',1])
    run('train_resume',[*base,'--output',out/'resumed','--resume',out/'resumed/last.pth'])
    full=torch.load(out/'full/last.pth',weights_only=True,map_location='cpu')
    for folder in ['repeat','resumed']:
        other=torch.load(out/folder/'last.pth',weights_only=True,map_location='cpu')
        assert full['history']==other['history'],folder
        assert all(torch.equal(v,other['model'][k]) for k,v in full['model'].items()),folder
        del other
    # Confirm optimization changed learnable parameters from the seeded initialization.
    from models import unet
    seed_all(2026,4);initial=unet(n_channels=5).state_dict()
    changed=any(not torch.equal(v,full['model'][k]) for k,v in initial.items() if k.endswith('weight'))
    assert changed
    del initial,full
    run('evaluate',['evaluate.py','--data-root',data,'--manifest',out/'splits/test.csv','--checkpoint',out/'full/best.pth','--output',out/'test_metrics.json'])
    run('predict',['predict.py','--input',data/'i0.tif','--checkpoint',out/'full/best.pth','--output',out/'prediction.tif'])
    model,saved=load_checkpoint(out/'full/best.pth',torch.device('cpu'))
    seed_all(2026,4);raw,profile=read_image(data/'i0.tif',[1,2,3,4])
    with torch.inference_mode():expected=(model(preprocess(raw)[None])[0,0].numpy()>.5).astype('uint8')*255
    with rasterio.open(out/'prediction.tif') as src:
        assert src.crs==profile['crs'] and src.transform==profile['transform']
        assert np.array_equal(src.read(1),expected)
    write_json(out/'smoke_report.json',dict(passed=True,steps=runs,parameter_update_verified=changed,
        repeated_run_tensor_equal=True,resumed_run_tensor_equal=True,prediction_equals_direct_model=True,
        scope='Synthetic CPU pipeline test; not ecological accuracy or reproduction of historical training results.'))
    print('All synthetic training workflow checks passed.',flush=True)


if __name__=='__main__':main()
