"""Online training-only augmentation; validation/test inputs stay deterministic."""
import torch
from torchvision.transforms import ColorJitter


class Augment:
    """Apply shared geometry to image and mask, then perturb RGB features only."""
    def __init__(self, config):
        self.config = config
        self.jitter = ColorJitter(brightness=config['brightness'], contrast=config['contrast'],
                                 saturation=config['saturation'], hue=config['hue'])

    def __call__(self, image, mask):
        c = self.config
        if torch.rand(()).item() < c['hflip_probability']:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(()).item() < c['vflip_probability']:
            image, mask = image.flip(-2), mask.flip(-2)
        # Right-angle rotations preserve the pixel grid and binary mask labels.
        if c['rotate90']:
            k = int(torch.randint(4, ()).item())
            image, mask = image.rot90(k, [-2, -1]), mask.rot90(k, [-2, -1])
        if torch.rand(()).item() < c['color_jitter_probability']:
            # Normalized RGB only; NIR/NDVI retain their original normalized values.
            image = torch.cat([self.jitter(image[:3]), image[3:]])
        return image.contiguous(), mask.contiguous()
