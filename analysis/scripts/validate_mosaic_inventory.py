"""Check retained inventory arithmetic and linkage, not historical pixel provenance."""
from pathlib import Path
from collections import Counter
import json,csv,hashlib,datetime

def validate(root):
 d=json.loads((root/'Mosaic_inventory_provenance.json').read_text(encoding='utf8'));rr=d['records']
 assert len(rr)==len({r['product_id'] for r in rr})==565
 assert len({(r['period'],r['source_workbook'],r['excel_row']) for r in rr})==565
 assert Counter(r['period'] for r in rr)=={'2017-2020':278,'2021-2024':287}
 assert Counter(int(r['inventory_date'][:4]) for r in rr)=={2017:158,2018:65,2019:48,2020:7,2021:16,2022:15,2023:93,2024:163}
 for r in rr:
  datetime.date.fromisoformat(r['inventory_date'])
  assert r['historical_source_order'] is None and r['selected_mosaic_pixels'] is None
  assert r['footprint_record_present']==(r['footprint_record'] is not None)
  if r['footprint_record']:
   assert int(r['product_id'].replace('L2A',''))==int(r['footprint_record']['product_id_attribute'])
 assert Counter(r['period'] for r in rr if r['footprint_record_present'])=={'2017-2020':271,'2021-2024':287}
 missing=[r for r in rr if not r['footprint_record_present']]
 assert len(missing)==7 and {r['excel_row'] for r in missing}==set(range(273,280))
 assert all(r['period']=='2017-2020' and r['inventory_date'].startswith('2019') for r in missing)
 assert d['reported_manuscript_scene_count']-len(rr)==d['arithmetic_gap_without_identifiers']==62
 assert d['gap_identifiers'] is None and d['gap_native_resolution'] is None and d['gap_inferred_to_be_1m'] is False
 with (root/'Source_data_index.csv').open(encoding='utf-8-sig',newline='') as f:
  for x in csv.DictReader(f):
   b=(root/x['file']).read_bytes();assert len(b)==int(x['bytes']) and hashlib.sha256(b).hexdigest()==x['sha256']
 return {'records':565,'unique_products':565,'footprint_links':558,'missing_footprints_already_in_inventory':7,'unidentified_count_gap':62,'preferred_year_record_percentages':[223/278*100,256/287*100],'historical_pixel_source_order_recovered':False,'gap_resolution_identified':False}
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'data/source_data')
 args=parser.parse_args()
 print(json.dumps(validate(args.data_dir),indent=2))
