"""Create a new reproducible group split, without reconstructing historical membership."""
import argparse
import csv
import random
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--group-column',default='source_id')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--val-fraction',type=float,default=0.15)
    p.add_argument('--test-fraction',type=float,default=0.15)
    a = p.parse_args()
    if not 0 < a.val_fraction < 1 or not 0 < a.test_fraction < 1 or a.val_fraction+a.test_fraction >= 1:
        p.error('Validation/test fractions must be positive and sum to less than one.')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be empty to avoid replacing an existing split.')
    with a.metadata.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or [];rows=list(reader)
    if not {'sample_id','image_file','mask_file',a.group_column}.issubset(fields) or not rows:
        p.error('Metadata requires sample_id,image_file,mask_file and the selected group column.')
    if any(not r.get(a.group_column) for r in rows) or len({r['sample_id'] for r in rows})!=len(rows):
        p.error('Missing groups or duplicate sample identifiers.')
    groups=sorted({r[a.group_column] for r in rows})
    if len(groups)<3:p.error('At least three groups are required.')
    random.Random(a.seed).shuffle(groups)
    nv=max(1,round(len(groups)*a.val_fraction));nt=max(1,round(len(groups)*a.test_fraction))
    if nv+nt>=len(groups):p.error('Fractions leave no training groups.')
    assignment={g:('val' if i<nv else 'test' if i<nv+nt else 'train') for i,g in enumerate(groups)}
    if 'parent_id' in fields:
        owner={}
        for row in rows:
            parent=row['parent_id'];split=assignment[row[a.group_column]]
            if parent and parent in owner and owner[parent]!=split:p.error('A parent spans multiple split groups.')
            if parent:owner[parent]=split
    a.output.mkdir(parents=True,exist_ok=True)
    outfields=fields if 'group_id' in fields else fields+['group_id']
    summary={}
    for split in ['train','val','test']:
        subset=[dict(r,group_id=r[a.group_column]) for r in rows if assignment[r[a.group_column]]==split]
        with (a.output/(split+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=outfields);writer.writeheader();writer.writerows(subset)
        summary[split]={'samples':len(subset),'groups':sorted({r['group_id'] for r in subset})}
    (a.output/'split_summary.json').write_text(json.dumps({'seed':a.seed,'group_column':a.group_column,
        'fraction_unit':'groups, not samples','splits':summary,'scope':'New split for this training recipe; not historical training membership.'},indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
