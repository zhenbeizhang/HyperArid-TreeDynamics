import unittest
import tempfile
import csv
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from datasets.tree_dataset import preprocess, read_pair, validate_splits, check_bands
from datasets.augment import Augment
from losses.bce_tversky import BCETverskyLoss
from training.metrics import counts, metrics


class PipelineTests(unittest.TestCase):
    def test_probability_loss_and_gradient(self):
        p=torch.tensor([[[[0.8,0.2]]]],requires_grad=True)
        y=torch.tensor([[[[1.,0.]]]])
        loss=BCETverskyLoss(alpha=.7,beta=.3)(p,y)
        expected=.5*(-np.log(.8))+.5*(1-(.8+1e-6)/(.8+.7*.2+.3*.2+1e-6))
        self.assertAlmostEqual(loss.item(),expected,places=6)
        loss.backward()
        self.assertLess(p.grad[0,0,0,0].item(),0)
        self.assertGreater(p.grad[0,0,0,1].item(),0)
        self.assertEqual(BCETverskyLoss()(torch.zeros_like(y),torch.zeros_like(y)).item(),0)

    def test_metrics(self):
        p=torch.tensor([.9,.7,.5,.1]);y=torch.tensor([1.,0.,1.,0.])
        self.assertEqual(counts(p,y,.5).tolist(),[1,1,1,1])
        self.assertEqual(metrics([1,1,1,1])['iou'],1/3)
        self.assertIsNone(metrics([0,0,0,10])['recall'])

    def test_preprocess_and_paired_augmentation(self):
        from tree_dynamics.preprocessing import Normalize
        raw=np.random.default_rng(23).uniform(100,2000,(4,256,256)).astype('float32')
        expected=Normalize()(torch.from_numpy(np.concatenate([raw,((raw[3]-raw[0])/(raw[3]+raw[0]+1e-6))[None]])))
        self.assertTrue(torch.equal(preprocess(raw),expected))
        marker=torch.zeros(1,256,256);marker[:,20:40,80:100]=1
        c=dict(hflip_probability=1,vflip_probability=1,rotate90=True,color_jitter_probability=0,
               brightness=.2,contrast=.2,saturation=.2,hue=.1)
        x,y=Augment(c)(marker.repeat(5,1,1),marker.clone())
        self.assertTrue(torch.equal(x[0],y[0]))
        self.assertEqual(set(y.unique().tolist()),{0.,1.})
        c['color_jitter_probability']=1
        x,y=Augment(c)(marker.repeat(5,1,1),marker.clone())
        self.assertTrue(torch.equal(x[3],y[0]))
        for invalid in [None,[1,1,3,4],[0,1,2,3]]:
            with self.assertRaises(ValueError):check_bands(invalid)

    def test_grid_and_group_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            profile=dict(driver='GTiff',width=256,height=256,crs='EPSG:32637',transform=from_origin(200000,3100000,.5,.5))
            rows=[]
            for i in range(2):
                image=np.random.default_rng(i).integers(100,4000,(4,256,256),dtype='uint16')
                with rasterio.open(root/f'i{i}.tif','w',count=4,dtype='uint16',**profile) as dst:dst.write(image)
                with rasterio.open(root/f'm{i}.tif','w',count=1,dtype='uint8',**profile) as dst:dst.write(np.zeros((1,256,256),dtype='uint8'))
                rows.append(dict(sample_id=str(i),image_file=f'i{i}.tif',mask_file=f'm{i}.tif',group_id='same'))
                with (root/f'{i}.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[i]));writer.writeheader();writer.writerow(rows[i])
            with self.assertRaisesRegex(ValueError,'group'):
                validate_splits(root,{'train':root/'0.csv','val':root/'1.csv'},[1,2,3,4])
            profile['transform']=from_origin(200000.5,3100000,.5,.5)
            with rasterio.open(root/'m0.tif','w',count=1,dtype='uint8',**profile) as dst:dst.write(np.zeros((1,256,256),dtype='uint8'))
            with self.assertRaisesRegex(ValueError,'grid mismatch'):read_pair(root,rows[0],[1,2,3,4])


if __name__=='__main__':unittest.main()
