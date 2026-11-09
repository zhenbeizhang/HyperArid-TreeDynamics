import os
import torch.utils.data as data
from osgeo import gdal
import numpy as np
import torch


class TreeDataset(data.Dataset):
    """
    PyTorch dataset for loading satellite imagery and tree crown masks.

    The dataset supports multi-band satellite imagery and optional NDVI
    computation. Images and masks are expected to be stored in TIFF format.
    """

    def __init__(
        self,
        root: str,
        mode: str = "train",
        transforms=None,
        band_combination: list = [0, 1, 2],
        use_ndvi: bool = False
    ):
        super().__init__()

        assert os.path.exists(root), f"path '{root}' does not exist."

        self.band_combination = band_combination
        self.use_ndvi = use_ndvi
        self.transforms = transforms

        if mode == "train":
            self.image_root = os.path.join(root, "Train", "Train-Image")
            self.mask_root = os.path.join(root, "Train", "Train-Mask")

        elif mode == "test":
            self.image_root = os.path.join(root, "Test", "Test-Image")
            self.mask_root = os.path.join(root, "Test", "Test-Mask")

        elif mode == "validation":
            self.image_root = os.path.join(root, "Validation", "val-Image")
            self.mask_root = os.path.join(root, "Validation", "val-Mask")

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Allowed modes are 'train', 'test', 'validation'."
            )

        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".tif")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".tif")]

        assert len(image_names) > 0, f"no images found in {self.image_root}."

        # Ensure masks correspond to images
        valid_mask_names = []
        for img_name in image_names:

            base_name = os.path.splitext(img_name)[0]
            mask_name = f"{base_name}.tif"

            if mask_name in mask_names:
                valid_mask_names.append(mask_name)

        mask_names = valid_mask_names

        self.images_path = [
            os.path.join(self.image_root, n)
            for n in image_names
            if os.path.splitext(n)[0] + ".tif" in mask_names
        ]

        self.masks_path = [
            os.path.join(self.mask_root, n)
            for n in mask_names
        ]

    def __getitem__(self, idx):

        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        img_ds = gdal.Open(image_path)

        if img_ds is None:
            raise RuntimeError(f"Unable to open image: {image_path}")

        # Read all bands
        bands = []

        for i in range(img_ds.RasterCount):

            band = img_ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
            bands.append(band)

        # Optional NDVI computation
        if self.use_ndvi and len(bands) >= 4:

            red = bands[0]
            nir = bands[3]

            ndvi = (nir - red) / (nir + red + 1e-6)

            bands.append(ndvi)

        # Select specific bands
        selected = [bands[i] for i in self.band_combination]
        image = np.stack(selected, axis=2)

        img_ds = None

        # Read mask using GDAL
        mask_ds = gdal.Open(mask_path)

        if mask_ds is None:
            raise RuntimeError(f"Unable to open mask: {mask_path}")

        target = mask_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

        mask_ds = None

        target = np.clip(target, 0, 255)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):

        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):

        images, targets = list(zip(*batch))

        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

    batch_shape = (len(images),) + max_size

    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)

    for img, pad_img in zip(images, batched_imgs):

        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    return batched_imgs