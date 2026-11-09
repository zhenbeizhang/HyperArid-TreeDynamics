import random
from typing import List, Union
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


class Compose:
    """
    Compose multiple transforms together.
    Supports both training (image, target) and inference (image only).
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):

        for t in self.transforms:

            if target is None:
                image = t(image)

            else:
                image, target = t(image, target)

        return (image, target) if target is not None else image


class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.
    """

    def __call__(self, image, target=None):

        image = F.to_tensor(image)

        if target is not None:
            target = F.to_tensor(target)
            return image, target

        return image


class RandomHorizontalFlip:
    """
    Random horizontal flip for data augmentation.
    """

    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):

        if random.random() < self.flip_prob:

            image = F.hflip(image)
            target = F.hflip(target)

        return image, target


class Normalize:
    """
    Normalize each spectral band to the range [0, 1].

    The NDVI band (index 4) excludes abnormal values during normalization.
    """

    def __call__(self, image, target=None):

        for i in range(image.size(0)):

            band = image[i]

            if i == 4:

                mask = (band >= 0) & (band < 1)

                valid_band = band[mask]

                if valid_band.numel() == 0:
                    min_val, max_val = 0.0, 0.0
                else:
                    min_val = valid_band.min()
                    max_val = valid_band.max()

            else:

                min_val = band.min()
                max_val = band.max()

            if max_val != min_val:
                image[i] = (band - min_val) / (max_val - min_val)
            else:
                image[i] = band - min_val

        return image if target is None else (image, target)


class Resize:
    """
    Resize image and optionally mask.
    """

    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):

        self.size = size
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):

        image = F.resize(image, self.size)

        if target is not None and self.resize_mask:

            target = F.resize(target, self.size)

        return image if target is None else (image, target)


class RandomCrop:
    """
    Random crop with optional padding if the image is smaller than the crop size.
    """

    def __init__(self, size: int):

        self.size = size

    def pad_if_smaller(self, img, fill=0):

        min_size = min(img.shape[-2:])

        if min_size < self.size:

            ow, oh = img.size

            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0

            img = F.pad(img, [0, 0, padw, padh], fill=fill)

        return img

    def __call__(self, image, target):

        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)

        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))

        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)

        return image, target


class RandomRotation:
    """
    Random rotation augmentation.
    """

    def __init__(self, degrees, prob=0.5):

        self.degrees = (-degrees, degrees)
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:

            angle = random.uniform(self.degrees[0], self.degrees[1])

            image = F.rotate(image, angle)
            target = F.rotate(target, angle)

        return image, target


class ColorJitter:
    """
    Apply color jitter only to RGB bands.
    Other bands such as NIR or NDVI remain unchanged.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.5):

        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:

            rgb_bands = image[:3, ...]

            other_bands = image[3:, ...] if image.size(0) > 3 else None

            jittered_rgb = self.jitter(rgb_bands)

            if other_bands is not None:

                image = torch.cat([jittered_rgb, other_bands], dim=0)

            else:

                image = jittered_rgb

        return image, target


class RandomScale:
    """
    Randomly scale image and mask.
    """

    def __init__(self, scale_range=(0.8, 1.2), prob=0.5):

        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:

            scale = random.uniform(self.scale_range[0], self.scale_range[1])

            new_size = [int(dim * scale) for dim in image.shape[1:]]

            image = F.resize(image, new_size)

            target = F.resize(
                target,
                new_size,
                interpolation=F.InterpolationMode.NEAREST
            )

        return image, target


class GaussianBlur:
    """
    Apply Gaussian blur augmentation.
    """

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), prob=0.3):

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:

            sigma = random.uniform(self.sigma[0], self.sigma[1])

            image = F.gaussian_blur(image, self.kernel_size, [sigma, sigma])

        return image, target