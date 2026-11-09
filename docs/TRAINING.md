# Executable training recipe

## Model and numerical contract

`models/unet.py` preserves the model source used by the prior public inference package, byte for byte. The `tree_dynamics.model` module now re-exports that implementation for compatibility. Parameter names, attention modules, sigmoid output and five-channel architecture are unchanged.

Images are explicitly reordered to red/green/blue/NIR, converted to float32, extended with `(NIR-red)/(NIR+red+1e-6)`, and normalized using the existing `Normalize` implementation. This normalization is retained for compatibility: NDVI normalization bounds are derived from values in [0,1), without clipping all resulting NDVI values to [0,1]. Training and prediction call the same preprocessing function.

The loss combines probability-input binary cross-entropy and per-image Tversky loss. With the default coefficients, it is 0.5 BCE + 0.5 Tversky; Tversky penalizes false positives by 0.7 and false negatives by 0.3. No second sigmoid is applied to a sigmoid output.

## Relationship to the earlier source

| Component | Executable recipe | Earlier source |
|---|---|---|
| Architecture | Same five-channel model definition | RGB training example and five-channel prediction examples coexist |
| Loss interface | BCE on probabilities; Tversky directly on probabilities | Probability output passed to a logits-based loss and another sigmoid |
| Validation | Named metric dictionary, pooled pixel counts | Standalone validation entry unpacks six values from a seven-value return |
| Labels and pairing | Strict binary labels, aligned grids, explicit CSV membership | Implicit directory pairing and some interpolation/padding defaults |
| Augmentation | After split, training only; flips, 90-degree rotations and normalized RGB ColorJitter | Includes arbitrary rotations, scale and blur; historical execution is not reconstructed |
| Input channels | Explicit four-band order plus NDVI throughout | Example channel settings differ between scripts |
| Optimizer | AdamW, constant learning rate, uniform weight decay; float32 | Parameter decay groups, warm-up schedule and mixed precision examples |
| Randomness/checkpoints | Explicit seed per epoch, source/data hashes, best and last checkpoints | Earlier example configuration and checkpoint layout |

The recipe makes these choices explicit so that a new experiment can be run and repeated. The previous source remains unchanged under `reference/historical_model_v2/`; this recipe is not evidence that the historical checkpoint was trained with the corrected loss or the new augmentation sequence. Default epochs and hyperparameters are editable recipe settings, not recovered historical run records.

## Data and split provenance

The public sample metadata can be used to create a new source-group split. The helper preserves groups and optional parent identifiers, and writes actual sample memberships. It does not reconstruct the historical split or certify independence of samples used in the paper. Input validation checks exact decoded image duplication; it does not search for every spatial overlap or transformed near-duplicate in arbitrary new datasets.

Only training rows receive online augmentation. Geometry is applied jointly to image and mask; right-angle rotations and flips preserve binary labels without interpolation. ColorJitter runs after normalization on RGB only; NIR and the already computed NDVI channel remain unchanged. This is a feature perturbation, not a physical radiometric calibration experiment.

## Repetition and resume

The recipe uses seed + epoch for Python, NumPy and PyTorch, a seeded shuffle generator, no worker subprocesses and deterministic PyTorch algorithms. Epoch-boundary checkpoints include optimizer state and all resolved settings. Resume verifies input and source hashes and uses the same device type and run folder. Interrupted partial epochs are rerun from the last completed epoch. Exact agreement is checked on the tested CPU environment; software or hardware changes can alter numerical results.

For a planned two-epoch run interrupted after one epoch:

```console
python train.py --data-root data_local --splits splits --bands 1 2 3 4 --output runs/example --epochs 2 --stop-after-epoch 1
python train.py --data-root data_local --splits splits --bands 1 2 3 4 --output runs/example --epochs 2 --resume runs/example/last.pth
```

`best.pth` is selected only by validation pixel IoU; the first epoch is retained even if its IoU is zero. Empty-positive unions have an undefined IoU (JSON null) and a selection score of zero. Evaluation of `test.csv` is a separate command. Test results must not guide subsequent model selection if they are to serve as held-out performance estimates.

## Checkpoint formats

`train.py` writes tensor-safe dictionaries loadable with `weights_only=True`: model, optimizer, config, memberships/hashes, history and epoch. `predict.py` and `evaluate.py` accept this format. The optional earlier `canopy_state_dict.pt` contains only parameters and belongs to the separately documented checkpoint-bound commands. The two formats are deliberately distinguished to keep preprocessing and training provenance explicit.
