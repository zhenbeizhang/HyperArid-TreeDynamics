# HyperArid-TreeDynamics

Model architecture, checkpoint preprocessing, tile prediction and crown-mask evaluation for sparse trees in hyper-arid landscapes. The accompanying sample dataset contains 956 four-band image–mask pairs from northern Saudi Arabia.

## Get started

Use Python 3.10+ and run commands from the repository root.

```console
python -m pip install -r requirements-model.txt
python -B -m tree_dynamics verify-files
```

Download `canopy_state_dict.pt` from this repository's [Releases](https://github.com/zhenbeizhang/HyperArid-TreeDynamics/releases) assets and place it in `models/`. The expected hash and size are in `config/model_weights.json`.

```console
python -B -m tree_dynamics verify-files --with-model
python scripts/demo_inference.py --output outputs/demo
python -B -m tree_dynamics predict --input outputs/demo/synthetic_four_band_256.npy --output outputs/predict
```

The demo creates synthetic 256- and 512-pixel inputs locally and checks prediction interfaces. Prediction needs code and model weights; it does not require climate inputs or the manuscript analysis archive.

## Use GeoTIFF tiles

Install the GeoTIFF dependency:

```console
python -m pip install -r requirements-geotiff.txt
```

The model input order is red, green, blue, near-infrared; NDVI is calculated internally. Supply the corresponding **1-based stored TIFF band indices** explicitly. For example, **only if** an input file is documented as blue/green/red/NIR:

```console
python -B -m tree_dynamics predict-tif --input image.tif --bands 3 2 1 4 --output outputs/tile
```

Output: `canopy_mask.tif` (0/255, original CRS and grid) and its prediction-settings JSON. Supported tiles are 256 × 256 or 512 × 512, at 0.5 m spacing with fully valid pixels. This entry point does not tile or mosaic full satellite scenes.

**For the public samples:** their four stored bands are preserved, but their physical band names have not been authenticated. The command does not infer that mapping. The author's RGB=(4,3,2) display convention is separate from model preprocessing; verify the spectral mapping before interpreting predictions on those tiles. See [input guidance](docs/INPUTS_AND_WEIGHTS.md).

## Evaluate a mask pair

```console
python -B -m tree_dynamics evaluate-masks --reference reference.tif --prediction prediction.tif --iou-threshold 0.5 --output outputs/evaluation
```

This evaluates 256-pixel binary masks on a 200-pixel central core. Object matching uses four-connected components, 3 ≤ crown area < 200 m², a centroid-in-core rule and maximum-cardinality matching at the specified IoU threshold. Output includes pixel and object metrics plus matched-component indices. Two `.npy` masks are also supported; they must already be aligned at 0.5 m spacing.

## Code organization

| Location | Contents |
|---|---|
| `tree_dynamics/model.py` | Checkpoint-compatible model architecture |
| `tree_dynamics/preprocessing.py` | Bound per-channel normalization |
| `tree_dynamics/inference.py`, `geotiff.py` | NumPy and GeoTIFF prediction; mask evaluation |
| `tree_dynamics/object_metrics.py`, `photometric.py` | Object matching and photometric transformations |
| `config/prediction.json`, `model_weights.json` | Inference settings and weight identity |
| `scripts/` | Model demonstration, input checks and sample-index checks |
| `reference/historical_model_v2/` | Previously published training, augmentation and prediction source |
| `analysis/` | Optional manuscript-analysis guidance and figure/inventory scripts |
| `docs/` | Installation, model-input and historical-source notes |

The historical training source is provided for provenance and reuse with its [compatibility notes](docs/HISTORICAL_CODE_NOTES.md). It is not a verified retraining recipe for this checkpoint. The current prediction architecture, normalization and weights are kept together.

## Sample dataset

The expanded `Hyperarid_tree_crown_samples_956.zip` collection is prepared for a separate Zenodo release; its version-specific DOI will be added here after publication. It contains images, masks, source grouping, sample coordinates, the original sample-distribution figure, and CC BY 4.0 terms. The sample archive contains no weights or executable code. Once the archive is available and extracted, check it with:

```console
python scripts/check_samples.py --dataset path/to/Hyperarid_tree_crown_samples
```

Use source or spatial groups when defining evaluation splits. The dataset does not assign the historical training/validation/test membership.

## Optional manuscript analyses

Statistical reproduction and plotting are documented separately in [analysis/README.txt](analysis/README.txt). They use `Supplementary_source_data.zip`, a derived-input archive distinct from the Zenodo samples. They are not needed to run the model.

## Licence and citation

Code and released weights are licensed under CC BY 4.0, as authorized by the author. Cite the associated paper and the specific repository release/commit. Earlier source-record identifiers and licences are retained in [reference/public_archive_provenance.json](reference/public_archive_provenance.json). Existing notices for third-party contributions remain applicable. See [LICENSE](LICENSE.txt).
