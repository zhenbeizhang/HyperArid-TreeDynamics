"""BCE + Tversky for the unchanged model's sigmoid probability output."""
import torch
from torch import nn
from torch.nn import functional as F


class BCETverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, bce_weight=0.5, tversky_weight=0.5, smooth=1e-6):
        super().__init__()
        if min(alpha, beta, bce_weight, tversky_weight) < 0 or smooth <= 0 or bce_weight + tversky_weight <= 0:
            raise ValueError('Invalid loss coefficients.')
        self.alpha, self.beta, self.bce_weight, self.tversky_weight, self.smooth = alpha, beta, bce_weight, tversky_weight, smooth

    def forward(self, probabilities, target):
        if probabilities.shape != target.shape:
            raise ValueError('Prediction and target shapes must match.')
        if not torch.isfinite(probabilities).all() or not torch.isfinite(target).all():
            raise ValueError('Loss inputs must be finite.')
        if torch.any((probabilities < 0) | (probabilities > 1)) or torch.any((target != 0) & (target != 1)):
            raise ValueError('Expected probabilities in [0,1] and binary targets.')
        bce = F.binary_cross_entropy(probabilities, target)
        dims = (1, 2, 3)
        tp = (probabilities * target).sum(dims)
        fp = (probabilities * (1-target)).sum(dims)
        fn = ((1-probabilities) * target).sum(dims)
        tv = 1 - (tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
        return self.bce_weight*bce + self.tversky_weight*tv.mean()
