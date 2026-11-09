"""Evaluate a new training checkpoint on its held-out test manifest."""
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.tree_dataset import TreeDataset, validate_splits
from training.common import load_checkpoint, device_from, seed_all, write_json
from losses.bce_tversky import BCETverskyLoss
from train import evaluate_loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    a = p.parse_args()
    device = device_from(a.device)
    model, saved = load_checkpoint(a.checkpoint, device)
    cfg = saved['config']
    seed_all(cfg['seed'], cfg['threads'])
    audit = validate_splits(a.data_root, {'test':a.manifest}, cfg['bands'])
    # Match the reserved membership and file hashes; reject substituted test inputs.
    expected = [row for row in saved['input_audit'] if row['split']=='test']
    if audit != expected:
        raise ValueError('This command requires the unchanged test manifest and files reserved by the training run.')
    dataset = TreeDataset(a.data_root, a.manifest, cfg['bands'])
    result = evaluate_loader(model, DataLoader(dataset,batch_size=cfg['batch_size'],num_workers=0), device, BCETverskyLoss(**cfg['loss']), cfg['threshold'])
    result.update({'samples':len(dataset), 'scope':'Held-out whole-tile pixel metrics; not crown-object or regional change accuracy.',
                   'threshold':cfg['threshold'], 'checkpoint_epoch':saved['epoch']+1})
    write_json(a.output, result)
    print(result)


if __name__ == '__main__': main()
