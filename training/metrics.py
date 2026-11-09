"""Whole-tile pixel counts aggregated before computing dataset metrics."""
import torch


def counts(probabilities, target, threshold):
    pred, ref = probabilities > threshold, target > 0.5
    return torch.tensor([torch.sum(pred & ref), torch.sum(pred & ~ref),
                         torch.sum(~pred & ref), torch.sum(~pred & ~ref)], dtype=torch.int64)


def metrics(c):
    tp, fp, fn, tn = [int(v) for v in c]
    div = lambda a, b: a/b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=div(tp,tp+fp), recall=div(tp,tp+fn),
                f1=div(2*tp,2*tp+fp+fn), iou=div(tp,tp+fp+fn), accuracy=div(tp+tn,tp+fp+fn+tn))
