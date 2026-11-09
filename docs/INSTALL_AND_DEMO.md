# Installation and examples

The main README contains the supported training, held-out evaluation and prediction commands. Install requirements-training.txt to use them. requirements-training-tested.txt lists the exact versions used for the CPU checks on Python 3.12.7 / Windows; no fresh-environment install is claimed.

Run from the repository root:

```console
python -m unittest discover -s tests -v
python scripts/smoke_training.py --output runs/smoke
```

No external imagery or pretrained checkpoint is required. The smoke example creates synthetic four-band tiles with a known stored red/green/blue/NIR order and binary masks. It checks training, repeated seeds, resume, saved-checkpoint loading, held-out evaluation and GeoTIFF prediction. These checks are software checks and do not provide an ecological accuracy estimate.

Earlier command-line tasks under python -m tree_dynamics remain available. Those requiring the earlier released checkpoint or optional analysis archive retain their documented requirements. New training starts from random initialization and does not require either artifact.
