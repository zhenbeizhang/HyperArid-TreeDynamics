"""Generate synthetic four-band arrays and exercise the released CPU model.

This is an installation and interface demo, not an ecological accuracy experiment.
No commercial imagery or real tree annotations are used.
"""
from pathlib import Path
import argparse,json,platform,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tree_dynamics.inference import load_model,predict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/demo'))
    p.add_argument('--weights',type=Path,help='Released weight at a custom location')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);start=time.monotonic()
    root=Path(__file__).resolve().parents[1]
    model=load_model(a.weights or root/'models/canopy_state_dict.pt');results=[]
    for size in [256,512]:
        raw=np.random.default_rng(310+size).uniform(10,1000,(4,size,size)).astype('float32')
        source=a.output/f'synthetic_four_band_{size}.npy';np.save(source,raw)
        mask=predict(raw,model)
        assert mask.shape==(size,size) and mask.dtype==np.uint8
        assert set(np.unique(mask)).issubset({0,255})
        target=a.output/f'synthetic_mask_{size}.npy';np.save(target,mask)
        results.append({'input':source.name,'output':target.name,'shape':list(mask.shape),'dtype':str(mask.dtype),'allowed_values':[0,255]})
    report={'passed':True,'purpose':'Synthetic installation/interface test; not tree-detection accuracy','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':4,'elapsed_seconds':round(time.monotonic()-start,2),'outputs':results}
    (a.output/'demo_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
