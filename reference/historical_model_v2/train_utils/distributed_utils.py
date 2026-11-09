import datetime
import errno
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


class ConfusionMatrixMetrics:
    """
    Evaluation metrics based on the confusion matrix.
    Designed for binary segmentation tasks and supports distributed training.
    """

    def __init__(self, num_class, threshold=0.5):

        self.num_class = num_class
        self.threshold = threshold
        self.confusion_matrix = None
        self.total_samples = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor):

        batch_size, c, h, w = gt.shape
        assert batch_size == 1, "Validation batch size must be 1."

        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)

        pred_binary = (resize_pred > self.threshold).int().squeeze(1)
        gt_binary = gt.int().squeeze(1)

        pred_np = pred_binary.cpu().numpy().flatten()
        gt_np = gt_binary.cpu().numpy().flatten()

        mask = (gt_np >= 0) & (gt_np < self.num_class)
        label = self.num_class * gt_np[mask].astype("int") + pred_np[mask]

        count = np.bincount(label, minlength=self.num_class ** 2)

        batch_confusion_matrix = count.reshape(self.num_class, self.num_class)

        if self.confusion_matrix is None:

            self.confusion_matrix = torch.zeros(
                (self.num_class, self.num_class),
                dtype=torch.float64,
                device=gt.device
            )

        self.confusion_matrix += torch.from_numpy(batch_confusion_matrix).to(gt.device)

        self.total_samples += batch_size

    def compute(self):

        if self.confusion_matrix is None:
            return (0.0,) * 7

        cm = self.confusion_matrix.cpu().numpy()

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)

        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)

        oa = np.diag(cm).sum() / (cm.sum() + eps)

        iou = tp / (tp + fn + fp + eps)

        dice = 2 * tp / ((tp + fp) + (tp + fn) + eps)

        total = cm.sum()

        po = np.diag(cm).sum() / total

        row_sums = np.sum(cm, axis=1)
        col_sums = np.sum(cm, axis=0)

        pe = np.sum(row_sums * col_sums) / (total ** 2)

        kappa = (po - pe) / (1 - pe + eps)

        return (
            precision[1],
            recall[1],
            f1[1],
            oa,
            iou[1],
            dice[1],
            kappa
        )

    def reduce_from_all_processes(self):

        if not dist.is_available():
            return

        if not dist.is_initialized():
            return

        dist.barrier()

        dist.all_reduce(self.confusion_matrix)

        dist.all_reduce(
            torch.tensor(
                [self.total_samples],
                dtype=torch.float64,
                device=self.confusion_matrix.device
            )
        )


class SmoothedValue:
    """
    Track a series of values and provide smoothed statistics.
    """

    def __init__(self, window_size=20, fmt=None):

        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)

        self.total = 0.0
        self.count = 0

        self.fmt = fmt

    def update(self, value, n=1):

        self.deque.append(value)

        self.count += n
        self.total += value * n

    @property
    def median(self):

        d = torch.tensor(list(self.deque))

        return d.median().item()

    @property
    def avg(self):

        d = torch.tensor(list(self.deque), dtype=torch.float32)

        return d.mean().item()

    @property
    def global_avg(self):

        return self.total / self.count

    @property
    def max(self):

        return max(self.deque)

    @property
    def value(self):

        return self.deque[-1]

    def __str__(self):

        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger:
    """
    Logger for tracking metrics during training or evaluation.
    """

    def __init__(self, delimiter="\t"):

        self.meters = defaultdict(SmoothedValue)

        self.delimiter = delimiter

    def update(self, **kwargs):

        for k, v in kwargs.items():

            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int))

            self.meters[k].update(v)

    def add_meter(self, name, meter):

        self.meters[name] = meter

    def __str__(self):

        return self.delimiter.join(
            f"{name}: {str(meter)}"
            for name, meter in self.meters.items()
        )

    def log_every(self, iterable, print_freq, header=""):

        i = 0

        start_time = time.time()

        end = time.time()

        iter_time = SmoothedValue(fmt="{avg:.4f}")

        data_time = SmoothedValue(fmt="{avg:.4f}")

        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        if torch.cuda.is_available():

            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}"
            ])

        else:

            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}"
            ])

        MB = 1024.0 * 1024.0

        for obj in iterable:

            data_time.update(time.time() - end)

            yield obj

            iter_time.update(time.time() - end)

            if i % print_freq == 0:

                eta_seconds = iter_time.global_avg * (len(iterable) - i)

                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():

                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )

                else:

                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)
                        )
                    )

            i += 1

            end = time.time()

        total_time = time.time() - start_time

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print(f"{header} Total time: {total_time_str}")


def mkdir(path):

    try:

        os.makedirs(path)

    except OSError as e:

        if e.errno != errno.EEXIST:

            raise


def is_dist_avail_and_initialized():

    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_world_size():

    if not is_dist_avail_and_initialized():
        return 1

    return dist.get_world_size()


def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process():

    return get_rank() == 0


def save_on_master(*args, **kwargs):

    if is_main_process():

        torch.save(*args, **kwargs)