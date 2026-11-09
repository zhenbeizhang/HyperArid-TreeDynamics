"""GeoTIFF tile prediction and binary-mask evaluation with explicit channel order."""
from pathlib import Path
import numpy as np
from .io import write_json


def read_ordered_tile(path, bands):
    import rasterio
    if bands is None or len(bands) != 4 or len(set(bands)) != 4:
        raise ValueError('Specify four distinct 1-based bands in model-input order: red green blue NIR. No order is inferred from the sample TIFFs.')
    with rasterio.open(path) as src:
        if src.count != 4 or min(bands) < 1 or max(bands) > src.count:
            raise ValueError('Expected a four-band TIFF and four distinct band indices from 1 to 4.')
        if (src.height, src.width) not in [(256, 256), (512, 512)]:
            raise ValueError('Only 256 x 256 and 512 x 512 tiles are supported; regional mosaics require a separate tiling workflow.')
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units != 'metre' or not np.allclose(src.res, (0.5, 0.5), rtol=0, atol=1e-7):
            raise ValueError('Expected a projected metre-based CRS with 0.5 m pixel spacing.')
        if not np.all(src.read_masks(indexes=bands) == 255):
            raise ValueError('The tile contains invalid/NoData pixels; provide a fully observed tile rather than silently filling missing data.')
        return src.read(indexes=bands), src.profile.copy()


def predict_geotiff(root, input_path, output_path, bands, weights=None):
    import rasterio
    import torch
    from .inference import load_model, predict
    raw, profile = read_ordered_tile(input_path, bands)
    output_path = Path(output_path)
    if output_path.resolve() == Path(input_path).resolve():
        raise ValueError('Output must not overwrite the input image.')
    torch.set_num_threads(4)
    model = load_model(weights or Path(root) / 'models/canopy_state_dict.pt')
    mask = predict(raw, model)
    # A new single-band profile avoids carrying spectral or photometric TIFF options.
    out_profile = dict(driver='GTiff', height=mask.shape[0], width=mask.shape[1], count=1,
                       dtype='uint8', crs=profile['crs'], transform=profile['transform'],
                       nodata=None, compress='deflate')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(mask, 1)
    report = {'input':Path(input_path).name, 'output':output_path.name,
              'stored_band_indices_in_model_order':list(bands), 'model_input_order':['red','green','blue','near_infrared'],
              'shape':list(mask.shape), 'mask_values':[0,255], 'threshold':0.5,
              'comparison':'strictly greater', 'crs':str(profile['crs']),
              'scope':'Tile prediction with caller-specified spectral mapping.'}
    write_json(output_path.with_suffix('.json'), report)
    return report


def read_binary_mask(path):
    path = Path(path)
    geo = None
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {'.tif', '.tiff'}:
        import rasterio
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError('Mask TIFFs must have exactly one band.')
            if src.crs is None or not src.crs.is_projected or src.crs.linear_units != 'metre' or not np.allclose(src.res, (0.5,0.5), rtol=0, atol=1e-7):
                raise ValueError('Mask TIFFs require a projected metre-based CRS and 0.5 m pixels.')
            if not np.all(src.read_masks(1)==255):
                raise ValueError('Mask contains invalid pixels; missing labels cannot be scored as background.')
            array=src.read(1);geo=(src.crs,src.transform)
    else:
        raise ValueError('Masks must be .npy, .tif or .tiff files.')
    if array.shape != (256,256) or not np.isfinite(array).all():
        raise ValueError('Mask evaluation requires a finite 256 x 256 array at 0.5 m spacing.')
    if not set(np.unique(array)).issubset({0,1,255}):
        raise ValueError('Expected binary masks with values 0/1 or 0/255; instance labels and probability maps need explicit conversion.')
    return array>0,geo


def evaluate_masks(reference, prediction, output, threshold=0.5):
    from .object_metrics import CORE, components, selected, score, div
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be greater than zero and at most one.')
    ref,rg=read_binary_mask(reference);pred,pg=read_binary_mask(prediction)
    if rg is not None and pg is not None and (rg[0]!=pg[0] or not rg[1].almost_equals(pg[1],precision=1e-7)):
        raise ValueError('Reference and prediction TIFF grids do not align.')
    if (rg is None) != (pg is None):
        raise ValueError('Use two GeoTIFFs or two NumPy arrays; mixed inputs cannot verify spatial alignment.')
    refs,re=selected(components(ref));preds,pe=selected(components(pred))
    counts,pairs=score(refs,preds,threshold)
    tp=int(np.count_nonzero(ref&pred&CORE));fp=int(np.count_nonzero(~ref&pred&CORE));fn=int(np.count_nonzero(ref&~pred&CORE))
    report={**counts,'pixel_tp':tp,'pixel_fp':fp,'pixel_fn':fn,
            'pixel_precision':div(tp,tp+fp),'pixel_recall':div(tp,tp+fn),'pixel_iou':div(tp,tp+fp+fn),
            'object_precision':div(counts['tp'],counts['tp']+counts['fp']),
            'object_recall':div(counts['tp'],counts['tp']+counts['fn']),
            'object_f1':div(2*counts['tp'],2*counts['tp']+counts['fp']+counts['fn']),
            'reference_edge_core':re,'prediction_edge_core':pe,'iou_threshold':threshold,
            'pixel_core':'rows/columns 28:228 (200 x 200 pixels)',
            'object_rule':'4-connected components; 3 <= area < 200 m2; centroid in core',
            'reference':Path(reference).name,'prediction':Path(prediction).name,
            'matched_object_pairs':pairs,'undefined_ratio':None,
            'scope':'Evaluation of the supplied mask pair; not independent validation of the released model.'}
    write_json(Path(output),report)
    return report
