"""Checksums for the released model and optional derived-data archive."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(path=None):
    path = Path(path) if path is not None else ROOT / 'models/canopy_state_dict.pt'
    spec = json.loads((ROOT / 'config/model_weights.json').read_text(encoding='utf-8'))
    if not path.is_file():
        raise FileNotFoundError('Download canopy_state_dict.pt from GitHub Releases and place it in models/, or supply --weights PATH.')
    if path.stat().st_size != spec['bytes'] or file_sha256(path) != spec['sha256']:
        raise ValueError('Model checksum mismatch: use the weight file associated with this code release.')
    return path


def require_analysis_data(root=ROOT):
    root = Path(root)
    manifest = root / 'data_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('This analysis command needs Supplementary_source_data.zip (derived manuscript inputs). The 956-pair Zenodo sample archive is a separate dataset. Extract the derived archive contents into the code root; prediction does not require this archive.')
    binding = json.loads((root / 'config/artifact_binding.json').read_text(encoding='utf-8'))
    if file_sha256(manifest) != binding['data_manifest_sha256']:
        raise ValueError('The derived-data manifest does not match this code release.')
    return manifest
