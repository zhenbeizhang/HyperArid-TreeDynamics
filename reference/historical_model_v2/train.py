import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import datetime
from typing import Union, List

import torch
from torch.utils import data

import src
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler, train_and_eval
from my_dataset import TreeDataset
import transforms as T
from src import UNET


class TreePresetTrain:
    """
    Data augmentation pipeline used for training.
    Includes geometric and radiometric transformations
    suitable for very high-resolution satellite imagery.
    """

    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 num_bands: int = 3, hflip_prob=0.5):

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            T.RandomScale(scale_range=(0.8, 1.2), prob=0.5),
            T.RandomRotation(45, prob=0.5),
            T.RandomHorizontalFlip(hflip_prob),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                prob=0.5
            ),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0), prob=0.3),
            T.Normalize()
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class TreePresetEval:
    """
    Preprocessing pipeline for validation/testing.
    Only deterministic transformations are applied.
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

    # Spectral band configuration
    band_comb = [0, 1, 2]  # example: RGB
    use_ndvi = False
    num_bands = len(band_comb)

    # Initialize model
    model = src.UNET.unet(n_channels=num_bands)
    model.to(device)

    model_name = model.__class__.__name__

    # Create experiment folder
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"{model_name}_{now}")
    os.makedirs(save_dir, exist_ok=True)

    loss_config = train_and_eval.LOSS_CONFIG

    # Save experiment configuration
    config_file = os.path.join(save_dir, f"{model_name}.txt")

    with open(config_file, "w", encoding="utf-8") as f:

        f.write("Hyperparameters\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

        f.write("\nData configuration\n")
        f.write(f"band_combination: {band_comb}\n")
        f.write(f"use_ndvi: {use_ndvi}\n")

        f.write("\nLoss configuration\n")

        def get_loss_config_string(loss_config):
            config_str = f"loss_function: {loss_config['name']}\n"
            for param_name, param_value in loss_config['params'].items():
                config_str += f"{param_name}: {param_value}\n"
            return config_str

        f.write(get_loss_config_string(loss_config))

        f.write("\nModel configuration\n")
        f.write(f"model = src.{model_name}(n_channels={num_bands})\n")
        f.write(f"input_channels: {num_bands}\n")
        f.write("output_classes: 1\n")
        f.write("bilinear_upsampling: True\n")

        f.write("\nTraining configuration\n")
        f.write(f"device: {device}\n")
        f.write("optimizer: AdamW\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"learning_rate: {args.lr}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"eval_interval: {args.eval_interval}\n")
        f.write(f"mixed_precision: {args.amp}\n")

        f.write("\nEvaluation metrics\n")
        f.write("Precision, Recall, F1 Score, Overall Accuracy, IoU, Dice, Kappa\n")

        f.write("\nBest model selection\n")
        f.write("weighted_score = IoU\n")

    results_file = os.path.join(save_dir, f"results{now}.txt")

    # Dataset initialization
    train_dataset = TreeDataset(
        args.data_path,
        mode="train",
        transforms=TreePresetTrain([256, 256], 256, num_bands=num_bands),
        band_combination=band_comb,
        use_ndvi=use_ndvi
    )

    test_dataset = TreeDataset(
        args.data_path,
        mode="test",
        transforms=TreePresetEval([256, 256], num_bands=num_bands),
        band_combination=band_comb,
        use_ndvi=use_ndvi
    )

    # DataLoader
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    test_data_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )

    params_group = get_params_groups(model, weight_decay=args.weight_decay)

    optimizer = torch.optim.AdamW(
        params_group,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    lr_scheduler = create_lr_scheduler(
        optimizer,
        len(train_data_loader),
        args.epochs,
        warmup=True
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_weighted_score = 0.0
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        mean_loss, lr = train_one_epoch(
            model,
            optimizer,
            train_data_loader,
            device,
            epoch,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            scaler=scaler
        )

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args
        }

        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # Evaluation
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:

            precision, recall, f1, oa, iou, dice, kappa = evaluate(
                model,
                test_data_loader,
                device=device
            )

            print(
                f"[epoch: {epoch}] "
                f"Precision: {precision:.3f} "
                f"Recall: {recall:.3f} "
                f"F1: {f1:.3f} "
                f"OA: {oa:.3f} "
                f"IoU: {iou:.3f} "
                f"Dice: {dice:.3f} "
                f"Kappa: {kappa:.3f}"
            )

            with open(results_file, "a") as f:

                write_info = (
                    f"[epoch: {epoch}] "
                    f"train_loss: {mean_loss:.4f} "
                    f"lr: {lr:.6f} "
                    f"Precision: {precision:.3f} "
                    f"Recall: {recall:.3f} "
                    f"F1: {f1:.3f} "
                    f"OA: {oa:.3f} "
                    f"IoU: {iou:.3f} "
                    f"Dice: {dice:.3f} "
                    f"Kappa: {kappa:.3f}\n"
                )

                f.write(write_info)

            weighted_score = iou

            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                torch.save(save_file, os.path.join(save_dir, "model_best.pth"))

        # keep only last two checkpoints
        old_ckpt = os.path.join(save_dir, f"model_{epoch - 2}.pth")
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

        torch.save(save_file, os.path.join(save_dir, f"model_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("training time {}".format(total_time_str))


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch training script for U-Net based tree detection"
    )

    parser.add_argument("--data-path", default='./data', help="Dataset root directory")
    parser.add_argument("--device", default="cuda", help="Training device")

    parser.add_argument("-b", "--batch-size", default=4, type=int)

    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay',
        dest='weight_decay'
    )

    parser.add_argument(
        "--epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of total training epochs"
    )

    parser.add_argument(
        "--eval-interval",
        default=1,
        type=int,
        help="validation interval"
    )

    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help='initial learning rate'
    )

    parser.add_argument(
        '--print-freq',
        default=10,
        type=int,
        help='logging frequency'
    )

    parser.add_argument(
        '--resume',
        default='',
        help='resume training from checkpoint'
    )

    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='start epoch'
    )

    parser.add_argument(
        "--amp",
        action='store_true',
        default=True,
        help="Use mixed precision training"
    )

    parser.add_argument(
        "--save-dir",
        default="save_weights",
        help="Directory for saving models"
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)