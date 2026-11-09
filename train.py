"""Train the preserved architecture using an explicit, reproducible training recipe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import platform
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models import unet
from datasets.tree_dataset import TreeDataset, validate_splits, digest
from datasets.augment import Augment
from losses.bce_tversky import BCETverskyLoss
from training.common import load_config, seed_all, device_from, write_json, load_checkpoint
from training.metrics import counts, metrics


def evaluate_loader(model, loader, device, criterion, threshold):
    """Evaluate without gradients and pool pixel counts before computing ratios."""
    model.eval()
    confusion = torch.zeros(4, dtype=torch.int64)
    total_loss = 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probabilities = model(x)
            total_loss += criterion(probabilities, y).item() * len(x)
            confusion += counts(probabilities, y, threshold)
    return {'loss': total_loss/len(loader.dataset), **metrics(confusion)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True, help='Directory containing train.csv, val.csv and test.csv')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config/train.json')
    parser.add_argument('--bands', nargs=4, type=int, metavar=('R','G','B','NIR'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, help='Override the configured total epochs')
    parser.add_argument('--resume', type=Path, help='Resume from last.pth with the same configuration and input hashes')
    parser.add_argument('--stop-after-epoch', type=int, help='Stop after this many total epochs; retain the configured training horizon')
    args = parser.parse_args()
    cfg = load_config(args.config, args.bands)
    if args.epochs is not None:
        if args.epochs <= 0: parser.error('--epochs must be positive')
        cfg['epochs'] = args.epochs
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= cfg['epochs']:
        parser.error('--stop-after-epoch must be within the configured training horizon')
    device = device_from(args.device)
    seed_all(cfg['seed'], cfg['threads'])
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory or resume.')
    manifests = {s: args.splits/(s+'.csv') for s in ['train','val','test']}
    # Test files are inspected for integrity/separation only. No test score enters model selection.
    audit = validate_splits(args.data_root, manifests, cfg['bands'])
    root = Path(__file__).parent
    source_files = [Path('train.py'), Path('models/unet.py'), Path('tree_dynamics/preprocessing.py')]
    source_files += [p.relative_to(root) for folder in ['datasets','losses','training'] for p in (root/folder).glob('*.py')]
    # Exact source hashes make resume checks sensitive even to comment-only edits.
    source_hashes = {p.as_posix(): digest(root/p) for p in sorted(source_files)}
    train_set = TreeDataset(args.data_root, manifests['train'], cfg['bands'], Augment(cfg['augmentation']))
    val_set = TreeDataset(args.data_root, manifests['val'], cfg['bands'])
    model = unet(n_channels=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = BCETverskyLoss(**cfg['loss'])
    # A negative initial best score ensures the first checkpoint is retained at IoU=0.
    start, best, history = 0, -1.0, []
    if args.resume:
        model, saved = load_checkpoint(args.resume, device)
        if saved['config'] != cfg or saved['input_audit'] != audit or saved['source_hashes'] != source_hashes:
            raise ValueError('Resume configuration, source code or data differ from the saved run.')
        if saved['device'] != str(device):
            raise ValueError('Use the same device type for deterministic resume.')
        if args.resume.resolve() != (args.output/'last.pth').resolve() or not (args.output/'best.pth').is_file():
            raise ValueError('Resume from last.pth in its existing output folder, including best.pth.')
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['epoch']+1, saved['best_iou'], saved['history']
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'config.json', cfg)
    write_json(args.output/'input_audit.json', audit)
    write_json(args.output/'source_hashes.json', source_hashes)
    write_json(args.output/'environment.json', {'python':platform.python_version(), 'torch':str(torch.__version__),
                                              'device':str(device), 'threads':cfg['threads'], 'precision':'float32',
                                              'seed_rule':'seed + zero-based epoch; workers=0', 'optimizer':'AdamW, constant learning rate'})
    for split, path in manifests.items():
        shutil.copyfile(path, args.output/(split+'.csv'))
    stop = args.stop_after_epoch or cfg['epochs']
    if start >= stop:
        raise ValueError('No remaining epochs at the requested stopping point.')
    # Epoch-specific seeds let an interrupted epoch restart from its saved boundary.
    for epoch in range(start, stop):
        seed_all(cfg['seed']+epoch, cfg['threads'])
        generator = torch.Generator().manual_seed(cfg['seed']+epoch)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, generator=generator)
        val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], num_workers=0)
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x)
        # Select on validation IoU only; held-out test evaluation is a separate command.
        validation = evaluate_loader(model, val_loader, device, criterion, cfg['threshold'])
        score = validation['iou'] if validation['iou'] is not None else 0.0
        improved = score > best
        best = max(best, score)
        row = {'epoch':epoch+1, 'train_loss':total/len(train_set), 'validation':validation}
        history.append(row)
        saved = {'format':'hyperarid-training-v1', 'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                 'epoch':epoch, 'best_iou':best, 'config':cfg, 'input_audit':audit, 'source_hashes':source_hashes,
                 'history':history, 'device':str(device)}
        # Replace the last checkpoint only after its new file is fully written.
        temporary = args.output/'last.pth.tmp'
        torch.save(saved, temporary)
        temporary.replace(args.output/'last.pth')
        if improved:
            shutil.copyfile(args.output/'last.pth', args.output/'best.pth')
        write_json(args.output/'history.json', history)
        print(json.dumps(row), flush=True)
    print('Training complete. Evaluate best.pth on test.csv separately.', flush=True)


if __name__ == '__main__':
    main()
