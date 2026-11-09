# Model inputs and released weights

## Weight file

Download `canopy_state_dict.pt` from GitHub Releases to `models/`. It is a tensor-only state dictionary with 346 entries, loaded on CPU with `weights_only=True` and strict parameter matching. Its SHA-256 is checked before model loading. Use `--weights PATH` if the released file is stored elsewhere; the same identity check applies. This option relocates the released checkpoint rather than silently substituting a different model.

## Array contract

NumPy input: numeric array `(4,256,256)` or `(4,512,512)` at 0.5 m sampling, ordered red/green/blue/NIR. All values must be finite. Values are converted to float32; the fifth input is `(NIR-red)/(NIR+red+1e-6)`. The released `Normalize` class applies per-channel normalization, retaining the checkpoint evaluation convention. Predictions are thresholded strictly above 0.5 and exported as uint8 0/255.

GeoTIFF input: four stored bands, explicit `--bands R G B NIR` indices, projected metre-based CRS, 0.5 m pixels, supported tile dimensions and valid data throughout. Output preserves the CRS and affine transform. Missing-data tiles are rejected rather than silently scored as observed background.

The public sample TIFFs retain their original stored order without authenticated spectral names. A channel mapping must come from the source product or another reliable record. A false-colour display choice does not establish the mapping required by this checkpoint. Synthetic examples provide a known ordered input for testing installation without making a claim about the samples' spectral metadata.

## Mask evaluation

Use two single-band GeoTIFFs on the same grid or two already aligned NumPy masks. The supported size is 256 × 256, with labels 0/1 or 0/255. GeoTIFFs must use 0.5 m pixels and matching CRS/transforms. Instance indices and floating probability maps require explicit conversion before evaluation.

The pixel core is rows/columns 28:228. Connected components are extracted in the full tile with four-neighbour connectivity. Eligible components have 3 ≤ area < 200 m² and a centroid within the core. Edge-intersecting component counts are recorded. Undefined metric ratios are JSON null. These are the packaged evaluation rules; scoring an arbitrary sample pair does not make it an independent test of the model.
