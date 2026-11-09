"""CPU tile inference with the checkpoint bound to the revision evaluations."""
from pathlib import Path
import numpy as np
import torch
from .model import unet
from .preprocessing import Normalize
from .integrity import verify_weights

def load_model(checkpoint):
    checkpoint=verify_weights(checkpoint)
    model=unet(n_channels=5)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.eval()

def predict(raw,model):
    if raw.ndim!=3 or raw.shape[0]!=4 or raw.shape[1:] not in [(256,256),(512,512)]:
        raise ValueError('Expected four ordered bands with shape (4,256,256) or (4,512,512)')
    b=raw.astype(np.float32)
    if not np.isfinite(b).all():raise ValueError('Non-finite input')
    nd=(b[3]-b[0])/(b[3]+b[0]+1e-6)
    inp=Normalize()(torch.from_numpy(np.concatenate([b,nd[None]])))
    if not torch.isfinite(inp).all():raise ValueError('Non-finite normalized input')
    with torch.inference_mode():p=model(inp.unsqueeze(0))[0,0].numpy()
    return (p>.5).astype('uint8')*255

def run(root,input_path,output_path,weights=None):
    torch.set_num_threads(4)
    model=load_model(weights or root/'models/canopy_state_dict.pt')
    raw=np.load(input_path,allow_pickle=False)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.save(output_path,predict(raw,model))
