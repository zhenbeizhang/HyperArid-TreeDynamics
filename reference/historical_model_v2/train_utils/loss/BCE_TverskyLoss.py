import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky loss for binary segmentation.

    Parameters
    ----------
    alpha : float
        Weight controlling false positives.
    beta : float
        Weight controlling false negatives.
    smooth : float
        Small constant to avoid division by zero.
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        tp = (probs * targets).sum(dim=(1, 2, 3))

        fp = ((1 - targets) * probs).sum(dim=(1, 2, 3))

        fn = (targets * (1 - probs)).sum(dim=(1, 2, 3))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return torch.mean(1 - tversky)


def criterion(
    inputs,
    target,
    tversky_weight=0.5,
    bce_weight=0.5,
    alpha=0.5,
    beta=0.5
):
    """
    Combined BCE + Tversky loss.

    Parameters
    ----------
    inputs : dict
        Model output dictionary. Expected key: "seg".
    target : torch.Tensor
        Ground truth segmentation mask.
    tversky_weight : float
        Weight for Tversky loss.
    bce_weight : float
        Weight for BCE loss.
    alpha : float
        Tversky alpha parameter.
    beta : float
        Tversky beta parameter.

    Returns
    -------
    torch.Tensor
        Combined loss value.
    """

    if not isinstance(inputs, dict):

        raise TypeError(
            f"inputs must be a dictionary, got {type(inputs)}"
        )

    tversky_loss = TverskyLoss(alpha=alpha, beta=beta)

    seg_loss = tversky_loss(inputs["seg"], target)

    bce_loss = F.binary_cross_entropy_with_logits(
        inputs["seg"],
        target
    )

    combined_loss = (
        tversky_weight * seg_loss +
        bce_weight * bce_loss
    )

    return combined_loss