"""Validate retained file provenance against supplied tables, without original imagery."""
from pathlib import Path
from collections import Counter
import csv,json,re,hashlib,io

def validate(root):
 def j(n):return json.loads((root/n).read_text(encoding='utf-8-sig'))
 def csvrows(n):
  with (root/n).open(encoding='utf-8-sig',newline=None) as f:return list(csv.DictReader(f))
 def norm(s):return re.sub(r'\s+',' ',s).strip()
 m=j('Training_file_manifest.json');rows=m['files'];products=j('Training_source_inventory_17_products.json');pm={x['product_id']:x for x in products}
 assert len(rows)==6157 and len(pm)==17
 assert len({(r['collection'],r['directory_split'],r['image_filename']) for r in rows})==len(rows)
 roles={'F_train':4627,'F_test':574,'I_train':764,'I_test':95,'I_predict':97}
 count=Counter((r['product_id'],r['collection']+'_'+r['directory_split']) for r in rows)
 for p in products:
  for role in roles:
   assert count[p['product_id'],role]==p['current_patch_counts'][role]
   parents={r['parent_identifier_from_filename'] for r in rows if r['product_id']==p['product_id'] and r['collection']+'_'+r['directory_split']==role}
   assert len(parents)==p['parent_counts_by_directory'][role]
 for row in rows:
  p=pm[row['product_id']]
  assert row['scene_identifier']==p['scene_candidate']
  assert row['parent_identifier_from_filename'].startswith(p['scene_candidate']+'_')
  assert row['image_filename'].startswith(row['parent_identifier_from_filename'])
  assert row['inventory_date']==p['approved_date'] and row['filename_date']==p['filename_date_unverified']
  assert row['listed_date']==p['listed_date'] and row['listed_date_basis']==p['listed_date_basis']
  assert row['historical_final_training_membership']=='not_established' and row['acquisition_date_independently_verified'] is False
 for row in csvrows('Table_S9.csv'):
  for role,total in roles.items():
   value=int(row[role.replace('_',' ')].replace(',',''))
   assert value==(sum(n for (p,k),n in count.items() if k==role) if row['Product']=='Total' else count[row['Product'],role])
 for row in csvrows('Table_S8.csv'):
  row={norm(k):norm(v) for k,v in row.items()};pid=re.search(r'L2A\d+',row['Platform and product'])[0];p=pm[pid]
  assert row['Filename date']==p['filename_date_unverified']
  assert row['Inventory date']==(p['approved_date'] or 'Not listed')
  assert row['Listed date basis']==('Inventory' if p['approved_date'] else 'Filename only')
  assert row['Scene identifier'] in p['scene_candidate']
 assert Counter(p['listed_date_basis'] for p in products)=={'designated_inventory':10,'filename_only':7}
 for rec in csvrows('Source_data_index.csv'):
  b=(root/rec['file']).read_bytes();assert len(b)==int(rec['bytes']) and hashlib.sha256(b).hexdigest()==rec['sha256']
 return {'file_records':len(rows),'products':17,'product_directory_counts_checked':85,'parent_counts_checked':85,'directory_totals':roles,'date_basis_products':{'designated_inventory':10,'filename_only':7},'historical_final_membership_certified':False,'image_bytes_checked':False}
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'data/source_data')
 args=parser.parse_args()
 print(json.dumps(validate(args.data_dir),indent=2))
