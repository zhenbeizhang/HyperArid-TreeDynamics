"""Executable checks of known band permutations, grids and mask confusion counts."""
from pathlib import Path
import argparse,json,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from tree_dynamics.inference import load_model,predict
from tree_dynamics.geotiff import read_ordered_tile,predict_geotiff,evaluate_masks
from tree_dynamics.integrity import verify_weights


def rejects(call):
    try:call()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid input was not rejected')


def write_tif(path,array,transform=None,nodata=None):
    if array.ndim==2:array=array[None]
    with rasterio.open(path,'w',driver='GTiff',height=array.shape[1],width=array.shape[2],count=array.shape[0],dtype=array.dtype,
                       crs='EPSG:32637',transform=transform or from_origin(260000,3140000,.5,.5),nodata=nodata) as dst:dst.write(array)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--weights',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];weights=args.weights or root/'models/canopy_state_dict.pt'
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='canopy_model_io_') as name:
        tmp=Path(name);raw=np.random.default_rng(918).integers(1,2000,(4,256,256),dtype=np.uint16)
        # Store a deliberate B,G,R,NIR permutation; recover the known model order.
        image=tmp/'input.tif';write_tif(image,raw[[2,1,0,3]])
        restored,profile=read_ordered_tile(image,[3,2,1,4]);assert np.array_equal(restored,raw)
        rejects(lambda:read_ordered_tile(image,None));rejects(lambda:read_ordered_tile(image,[1,1,3,4]));rejects(lambda:read_ordered_tile(image,[1,2,3,5]))
        bad=tmp/'invalid.tif';nodata_raw=raw.copy();nodata_raw[:,0,0]=0;write_tif(bad,nodata_raw,nodata=0)
        rejects(lambda:read_ordered_tile(bad,[1,2,3,4]))
        out=tmp/'mask.tif';predict_geotiff(root,image,out,[3,2,1,4],weights)
        expected=predict(raw,load_model(weights))
        with rasterio.open(out) as ds:
            assert np.array_equal(ds.read(1),expected) and ds.transform==profile['transform'] and ds.crs==profile['crs']
            assert ds.count==1 and ds.dtypes==('uint8',) and ds.nodata is None
        rejects(lambda:predict_geotiff(root,image,image,[3,2,1,4],weights))
        ref=np.zeros((256,256),np.uint8);pred=ref.copy()
        ref[40:43,40:44]=255;ref[80:84,80:84]=255
        pred[40:43,40:44]=255;pred[120:123,120:124]=255
        rp=tmp/'reference.tif';pp=tmp/'prediction.tif';write_tif(rp,ref);write_tif(pp,pred)
        report=evaluate_masks(rp,pp,tmp/'metrics.json')
        assert (report['tp'],report['fp'],report['fn'])==(1,1,1)
        assert (report['pixel_tp'],report['pixel_fp'],report['pixel_fn'])==(12,12,16)
        assert report['object_precision']==.5 and report['object_recall']==.5 and report['pixel_iou']==.3
        shifted=tmp/'shifted.tif';write_tif(shifted,pred,from_origin(260000.5,3140000,.5,.5))
        rejects(lambda:evaluate_masks(rp,shifted,tmp/'wrong.json'))
        rejects(lambda:evaluate_masks(rp,pp,tmp/'wrong.json',threshold=0))
        invalid=tmp/'labels.tif';write_tif(invalid,np.full((256,256),2,np.uint8))
        rejects(lambda:evaluate_masks(rp,invalid,tmp/'wrong.json'))
        empty=tmp/'empty.tif';write_tif(empty,np.zeros((256,256),np.uint8))
        e=evaluate_masks(empty,empty,tmp/'empty.json');assert e['object_precision'] is None and e['pixel_iou'] is None
        fake=tmp/'wrong.pt';fake.write_bytes(b'wrong checkpoint');rejects(lambda:verify_weights(fake))
    print(json.dumps({'passed':True,'known_band_permutation_verified':True,'geotiff_matches_numpy_prediction':True,
                      'crs_and_grid_preserved':True,'known_pixel_and_object_counts_verified':True,'invalid_inputs_rejected':True,
                      'scope':'Synthetic software checks, not new model accuracy results.'},indent=2))

if __name__=='__main__':main()
