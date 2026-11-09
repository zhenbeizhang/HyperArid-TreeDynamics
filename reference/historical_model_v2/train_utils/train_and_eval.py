import math
import torch
from torch import nn
from torch.nn import functional as F

import train_utils.distributed_utils as utils
from train_utils.loss import BCE_TverskyLoss


"""
Loss configuration used for training.
"""
LOSS_CONFIG = {
    "name": "BCE_TverskyLoss",
    "params": {
        "tversky_weight": 0.5,
        "bce_weight": 0.5,
        "alpha": 0.7,
        "beta": 0.3
    }
}


def evaluate(model, data_loader, device):
    """
    Evaluate model performance using confusion-matrix-based metrics.
    """

    model.eval()

    metrics = utils.ConfusionMatrixMetrics(num_class=2)

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = "Test:"

    with torch.no_grad():

        for images, targets in metric_logger.log_every(data_loader, 100, header):

            images = images.to(device)
            targets = targets.to(device)

            output = model(images)

            metrics.update(output, targets)

        metrics.reduce_from_all_processes()

        precision, recall, f1, oa, iou, dice, kappa = metrics.compute()

    return precision, recall, f1, oa, iou, dice, kappa


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    lr_scheduler,
    print_freq=10,
    scaler=None
):
    """
    Train model for one epoch.
    """

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")

    metric_logger.add_meter(
        "lr",
        utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    header = f"Epoch: [{epoch}]"

    for image, target in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)
        target = target.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):

            output = model(image)

            output_dict = {"seg": output}

            loss = BCE_TverskyLoss.criterion(
                output_dict,
                target,
                tversky_weight=LOSS_CONFIG["params"]["tversky_weight"],
                bce_weight=LOSS_CONFIG["params"]["bce_weight"],
                alpha=LOSS_CONFIG["params"]["alpha"],
                beta=LOSS_CONFIG["params"]["beta"]
            )

        optimizer.zero_grad()

        if scaler is not None:

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=3,
    warmup_factor=0.1,
    end_factor=1e-4
):
    """
    Learning rate scheduler with warmup and cosine decay.
    """

    assert num_step > 0 and epochs > 0

    if warmup is False:
        warmup_epochs = 0

    def lr_lambda(x):

        if warmup and x <= (warmup_epochs * num_step):

            alpha = float(x) / (warmup_epochs * num_step)

            return warmup_factor * (1 - alpha) + alpha

        else:

            current_step = x - warmup_epochs * num_step

            cosine_steps = (epochs - warmup_epochs) * num_step

            return (
                (1 + math.cos(current_step * math.pi / cosine_steps)) / 2
            ) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    """
    Separate parameters into groups with and without weight decay.
    Bias and 1D parameters are excluded from weight decay.
    """

    params_group = [
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": weight_decay}
    ]

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias"):

            params_group[0]["params"].append(param)

        else:

            params_group[1]["params"].append(param)

    return params_group