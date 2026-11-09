# HyperArid-TreeDynamics

Train, evaluate and apply a five-channel tree-crown segmentation model to very-high-resolution imagery of hyper-arid landscapes. The repository includes the model architecture, paired-image loading, online augmentation, loss function, training loop and GeoTIFF prediction.

**You can train this model from scratch without downloading pretrained weights.** The training recipe below is an executable implementation for new experiments with the preserved model architecture. Its relationship to the previously published source is documented in [training notes](docs/TRAINING.md); it does not reproduce the unavailable historical training membership or certify the original paper's accuracy.

## Repository structure

```text
models/unet.py             Model architecture and attention modules
datasets/tree_dataset.py   GeoTIFF image/mask reader and shared preprocessing
datasets/augment.py        Training-only flips, rotations and RGB ColorJitter
losses/bce_tversky.py       BCE + Tversky for sigmoid probability outputs
training/                  Checkpoint helpers, deterministic seeds and pixel metrics
train.py                   Training and epoch-boundary resume
evaluate.py                Held-out test-set pixel evaluation
predict.py                 GeoTIFF prediction using your trained checkpoint
config/train.json          Training, channel, augmentation and loss parameters
scripts/prepare_splits.py  Reproducible source-group split
scripts/smoke_training.py  Synthetic end-to-end training example
tests/                     Numerical and data-contract checks
reference/                 Unchanged previously published training source
analysis/                  Optional manuscript-analysis routines
tree_dynamics/             Earlier prediction and crown-object evaluation commands
```

## 1. Install

Use Python 3.10+ and run commands from the repository root:

```console
python -m pip install -r requirements-training.txt
```

The exercised environment is Windows, Python 3.12.7, PyTorch 2.7.1, TorchVision 0.22.1 and Rasterio 1.4.3. Exact package versions are in `requirements-training-tested.txt`. The training test uses CPU; a fresh installation and GPU execution have not been tested.

## 2. Prepare images and labels

Provide four-band GeoTIFF image tiles and matching single-band binary masks. Tiles must be 256 × 256 or 512 × 512 at 0.5 m spacing in a projected metre-based CRS, without NoData; use a common size within a run. Masks contain 0/1 or 0/255 and share the image grid.

```text
data_local/
  images/S0001.tif
  masks/S0001.tif
  metadata/samples.csv
```

The metadata CSV needs `sample_id,image_file,mask_file,source_id`; paths are relative to `data_local`. Include `parent_id` when multiple tiles come from the same parent image. Source identifiers must refer to source images or meaningful geographic groups, rather than one identifier per small tile.

```console
python scripts/prepare_splits.py --metadata data_local/metadata/samples.csv --output splits --group-column source_id --seed 2026
```

This creates `train.csv`, `val.csv` and `test.csv`. Groups remain separate; default fractions are approximately 70/15/15% of groups and need not give those sample percentages. The split is created before online augmentation. Training checks group, parent and exact-image duplication across sets, image/mask alignment and file hashes. Group separation alone does not establish complete geographic or ecological independence.

The expanded 956-pair sample archive is prepared for a separate Zenodo release. Its version-specific DOI will be linked after publication. These sample TIFFs preserve their stored order but do not authenticate the physical band names.

## 3. Train

Set the four stored band indices in **red, green, blue, NIR** order using `--bands` or the `bands` field in `config/train.json`. The default is deliberately unset. The model computes NDVI internally, giving five input channels.

The following example assumes the input is documented as **red/green/blue/NIR**:

```console
python train.py --data-root data_local --splits splits --bands 1 2 3 4 --config config/train.json --output runs/experiment1 --device cpu
```

For a documented blue/green/red/NIR file, use `--bands 3 2 1 4`. Do not infer spectral names from a false-colour display. Use `--device cuda` for a suitable locally configured CUDA installation.

Training saves `best.pth` (selected by validation pixel IoU), `last.pth`, `history.json`, the resolved configuration, environment information, input file hashes, code hashes and all three membership CSVs. Test labels are checked for file validity but are never scored during training or used to select checkpoints. A checkpoint is saved after the first epoch even if validation IoU is zero.

Resume from the last completed epoch in the same run folder, with unchanged settings and inputs:

```console
python train.py --data-root data_local --splits splits --bands 1 2 3 4 --config config/train.json --output runs/experiment1 --resume runs/experiment1/last.pth --device cpu
```

## 4. Evaluate the reserved test set

```console
python evaluate.py --data-root data_local --manifest splits/test.csv --checkpoint runs/experiment1/best.pth --output runs/experiment1/test_metrics.json
```

This reports pooled whole-tile pixel confusion counts, precision, recall, F1, IoU and accuracy, at the threshold stored with the checkpoint. It requires the same held-out test files recorded by the run. For crown-object metrics on aligned 256-pixel mask pairs, the separate `python -m tree_dynamics evaluate-masks` command remains available; its central-core and area rules are in [input guidance](docs/INPUTS_AND_WEIGHTS.md).

## 5. Predict

```console
python predict.py --input data_local/images/S0001.tif --checkpoint runs/experiment1/best.pth --output runs/experiment1/prediction.tif
```

Prediction loads preprocessing, channel order and threshold from the checkpoint and preserves the input CRS/grid. If another input uses a different stored order, provide its explicit `--bands R G B NIR` mapping. Output values are 0 (background) and 255 (tree canopy). This is a tile workflow; full-scene tiling and mosaicking are not implemented by this entry point.

## Try the complete workflow without external data

```console
python -m unittest discover -s tests -v
python scripts/smoke_training.py --output runs/smoke
```

The example creates four synthetic GeoTIFF pairs, performs two short training epochs, repeats the run, verifies epoch-boundary resume, evaluates the held-out tile and writes a prediction. It checks numerical agreement and execution, not tree-detection accuracy. No pretrained weights or public sample download are needed.

## Earlier model and analyses

The earlier tensor-only checkpoint is separate from the checkpoints produced by `train.py`. Its planned distribution is through GitHub Releases; no download is required for the training workflow above. The older `python -m tree_dynamics predict` command requires that particular checkpoint and verifies its identity.

The earlier public source is retained in `reference/historical_model_v2/`. Its compatibility issues and the changes in the runnable recipe are documented in [training notes](docs/TRAINING.md). Optional statistics and plotting remain in [analysis/README.txt](analysis/README.txt).

## Licence and citation

Author-contributed code is provided under [CC BY 4.0](LICENSE.txt). Cite the associated study, “Neglected dynamics of individual tree growth and mortality in hyper-arid areas”, and the repository release/commit used. Historical public records and third-party notices remain documented in `reference/`. Sample data are distributed separately through Zenodo.
