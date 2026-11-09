Model architecture

unet.py contains the full model definition, including the U-Net backbone, morphological attention, directional centrality and attention gates. Import it with: from models import unet

train.py creates this architecture with five input channels and trains it from scratch. No pretrained weight is needed. The model returns sigmoid probabilities. New training checkpoints are saved to the chosen runs directory.

The optional earlier canopy_state_dict.pt file is a separate artifact for the older checkpoint-specific commands. Its expected hash is in config/model_weights.json; it is not included in this source repository.
