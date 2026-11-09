"""Paired GeoTIFF loading and explicit, auditable sample membership."""
from pathlib import Path
import csv
import hashlib
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tree_dynamics.preprocessing import Normalize


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'sample_id', 'image_file', 'mask_file', 'group_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires columns {sorted(required)}')
        rows = list(reader)
    if not rows or any(not r[k].strip() for r in rows for k in required):
        raise ValueError('Manifest is empty or contains missing identifiers/paths.')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated sample_id in manifest.')
    return rows


def resolve_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Expected an existing file inside dataset root: {relative}')
    return path


def check_bands(bands):
    if not isinstance(bands, list) or len(bands) != 4 or any(type(v) is not int for v in bands) or set(bands) != {1, 2, 3, 4}:
        raise ValueError('Set bands to four distinct 1-based stored indices in red, green, blue, NIR order; no mapping is inferred.')


def read_image(path, bands):
    check_bands(bands)
    with rasterio.open(path) as src:
        if src.count != 4 or (src.height, src.width) not in [(256, 256), (512, 512)]:
            raise ValueError('Expected a four-band 256 or 512 square tile.')
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units != 'metre' or not np.allclose(src.res, [0.5, 0.5], rtol=0, atol=1e-7):
            raise ValueError('Expected a projected metre CRS and 0.5 m pixels.')
        if not np.all(src.read_masks() == 255):
            raise ValueError('Input image contains NoData; supply fully observed tiles.')
        image = src.read(bands).astype(np.float32)
        profile = src.profile.copy()
    if not np.isfinite(image).all():
        raise ValueError('Image contains non-finite values.')
    return image, profile


def read_pair(root, row, bands):
    image, profile = read_image(resolve_file(root, row['image_file']), bands)
    with rasterio.open(resolve_file(root, row['mask_file'])) as src:
        if src.count != 1 or src.shape != image.shape[1:] or src.crs != profile['crs'] or not src.transform.almost_equals(profile['transform'], precision=1e-7):
            raise ValueError(f"Image/mask grid mismatch: {row['sample_id']}")
        if not np.all(src.read_masks(1) == 255):
            raise ValueError('Mask contains NoData.')
        mask = src.read(1)
    if not np.isfinite(mask).all() or not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError('Masks must contain only 0/1 or 0/255 labels.')
    return image, (mask > 0).astype(np.float32)


def preprocess(raw):
    """Same channel normalization and NDVI convention as the supplied model."""
    b = raw.astype(np.float32, copy=True)
    ndvi = (b[3] - b[0]) / (b[3] + b[0] + 1e-6)
    image = Normalize()(torch.from_numpy(np.concatenate([b, ndvi[None]])))
    if not torch.isfinite(image).all():
        raise ValueError('Non-finite normalized input.')
    return image


def validate_splits(root, manifests, bands):
    """Audit grids and group/parent/identity separation before training."""
    owners = {key: {} for key in ['sample', 'group', 'parent', 'file', 'pixels']}
    records = []
    shapes = set()
    for split, path in manifests.items():
        for row in read_manifest(path):
            image, _ = read_pair(root, row, bands)
            shapes.add(image.shape[1:])
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            keys = {'sample': row['sample_id'], 'group': row['group_id'],
                    'file': str(resolve_file(root, row['image_file'])), 'pixels': pixel_hash}
            if row.get('parent_id'):
                keys['parent'] = row['parent_id']
            for kind, key in keys.items():
                previous = owners[kind].get(key)
                if previous is not None and (previous != split or kind in ['sample', 'file', 'pixels']):
                    raise ValueError(f'{kind} duplication or cross-split overlap in {split}: {row["sample_id"]}')
                owners[kind][key] = split
            records.append({'split': split, 'sample_id': row['sample_id'], 'group_id': row['group_id'],
                            'image_sha256': digest(resolve_file(root, row['image_file'])),
                            'mask_sha256': digest(resolve_file(root, row['mask_file']))})
    if len(shapes) != 1:
        raise ValueError('Use a common tile size for all samples; no implicit resizing is performed.')
    return records


class TreeDataset(Dataset):
    def __init__(self, root, manifest, bands, augmentation=None):
        check_bands(bands)
        self.root, self.rows, self.bands = Path(root), read_manifest(manifest), bands
        self.augmentation = augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, mask = read_pair(self.root, self.rows[index], self.bands)
        x, y = preprocess(image), torch.from_numpy(mask[None])
        if self.augmentation is not None:
            x, y = self.augmentation(x, y)
        return x, y
