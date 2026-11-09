# Code guide

For model training, begin with train.py and config/train.json. The full architecture is in models/unet.py; datasets/ contains image-mask loading and augmentation; losses/ defines the probability-input BCE/Tversky loss; training/ provides checkpoint and metric helpers. evaluate.py scores the reserved test manifest, and predict.py applies the new checkpoints to GeoTIFF tiles.

The training source and original architecture under reference/historical_model_v2 remain unchanged. docs/TRAINING.md distinguishes the executable recipe from the historical source. The tree_dynamics/model.py compatibility module imports the shared models/unet.py definition, retaining older checkpoint-bound inference commands.

Optional manuscript statistics and plots are described in analysis/README.txt. They are not prerequisites for training or prediction. The public sample collection has a separate Zenodo release path; weights and code are not placed inside that sample archive.
