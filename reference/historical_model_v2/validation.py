import os
from typing import Union, List

import torch
from torch.utils import data

import src
from src import UNET

from train_utils import evaluate
from my_dataset import TreeDataset
import transforms as T


class TreePresetEval:
    """
    Preprocessing pipeline used during validation.
    """

    def __init__(self, base_size: Union[int, List[int]], num_bands: int = 3):

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize()
        ])

    def __call__(self, img, target):

        return self.transforms(img, target)


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Properties: {torch.cuda.get_device_properties(0)}")

    else:

        print("CUDA is not available. Using CPU.")

    band_combination = [0, 1, 2, 3, 4]

    use_ndvi = True

    num_bands = len(band_combination)

    val_dataset = TreeDataset(
        args.data_path,
        mode="validation",
        transforms=TreePresetEval([512, 512], num_bands=num_bands),
        band_combination=band_combination,
        use_ndvi=use_ndvi
    )

    num_workers = 4

    val_data_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    model = src.UNET.unet(n_channels=num_bands)

    weights = torch.load(args.weights, map_location="cpu")

    if "model" in weights:

        model.load_state_dict(weights["model"])

    else:

        model.load_state_dict(weights)

    model.to(device)

    mae_metric, f1_metric, precision_metric, recall_metric, iou_metric, oa_metric = evaluate(
        model,
        val_data_loader,
        device=device
    )

    print("Validation results:")
    print("MAE:", mae_metric)
    print("F1:", f1_metric)
    print("Precision:", precision_metric)
    print("Recall:", recall_metric)
    print("IoU:", iou_metric)
    print("Overall Accuracy:", oa_metric)


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description="Validation script for U-Net based tree detection"
    )

    parser.add_argument(
        "--data-path",
        default="./data",
        help="Dataset root directory"
    )

    parser.add_argument(
        "--weights",
        default="",
        help="Path to trained model weights"
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for validation"
    )

    parser.add_argument(
        "--print-freq",
        default=1,
        type=int,
        help="Logging frequency"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)