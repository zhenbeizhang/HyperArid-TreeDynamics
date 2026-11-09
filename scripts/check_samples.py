"""Validate the public sample index and optionally every file checksum."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json


def inside(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Path outside dataset root')
    if not path.is_file():raise FileNotFoundError(relative)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--verify-hashes',action='store_true')
    args=p.parse_args();root=args.dataset.resolve()
    desc=json.loads((root/'dataset_description.json').read_text(encoding='utf-8'))
    with (root/'metadata/samples.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==desc['sample_pairs'] and len({r['sample_id'] for r in rows})==len(rows)
    for r in rows:
        inside(root,r['image_file']);inside(root,r['mask_file'])
        assert int(r['foreground_pixels'])>=0 and int(r['foreground_pixels'])<=int(r['width_px'])*int(r['height_px'])
        assert int(r['mask_foreground_present'])==int(int(r['foreground_pixels'])>0)
    counts=Counter(r['parent_id'] for r in rows)
    assert len(counts)==desc['parent_tiles'] and set(counts.values())=={4}
    assert len({r['source_id'] for r in rows})==desc['source_products']
    checked=0
    if args.verify_hashes:
        with (root/'file_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
            for entry in csv.DictReader(f):
                path=inside(root,entry['file']);digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                assert path.stat().st_size==int(entry['bytes']) and digest.hexdigest()==entry['sha256'],entry['file']
                checked+=1
    print(json.dumps({'passed':True,'sample_pairs':len(rows),'parents':len(counts),'source_products':len({r['source_id'] for r in rows}),
                      'foreground_masks':sum(int(r['mask_foreground_present']) for r in rows),'file_hashes_verified':checked,
                      'scope':'Index consistency and optional file integrity; not visual label accuracy or spectral identification.'},indent=2))

if __name__=='__main__':main()
