"""Replot the saved 32 coefficients; no model fitting or data selection."""
from pathlib import Path
import argparse,json,hashlib,csv,platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def draw(source,out):
 out.mkdir(parents=True,exist_ok=True);rows=json.loads(source.read_text(encoding='utf-8-sig'))
 assert len(rows)==32 and len({(r['response'],r['predictor']) for r in rows})==32
 assert all(r['bootstrap_draws']==1999 and r['blocks']==22 for r in rows)
 predictors=['PPT','Tmax','PET','VPD'];blue='#226A8A';orange='#B95D26';neutral='#344C5C'
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','legend.frameon':False})
 fig=plt.figure(figsize=(183/25.4,150/25.4))
 fig.text(.055,.965,'Changes between observation periods',fontsize=7,fontweight='bold')
 fig.text(.055,.555,'Spatial relationships within each observation period',fontsize=7,fontweight='bold')
 specs=[
 ('a','Crown-density change',['change_density'],[.11,.69,.345,.205],(-1,1.1),[-1,0,1],'Objects/ha per climate-trend SD'),
 ('b','Canopy-cover change',['change_cover'],[.61,.69,.345,.205],(-.065,.065),[-.05,0,.05],'Percentage points per climate-trend SD'),
 ('c','Tree density',['early_density','late_density'],[.075,.165,.235,.305],(-6,8.5),[-4,0,4,8],'Objects/ha per climate SD'),
 ('d','Canopy cover',['early_cover','late_cover'],[.405,.165,.235,.305],(-.16,.26),[-.1,0,.1,.2],'Percentage points per climate SD'),
 ('e','Mean crown area',['early_mean','late_mean'],[.735,.165,.235,.305],(-15,12),[-10,0,10],'m²/object per climate SD')]
 plotted=[]
 for letter,title,responses,rect,limits,ticks,xlabel in specs:
  ax=fig.add_axes(rect);ax.axvline(0,color='#8B8B8B',lw=.65,zorder=0)
  for j,response in enumerate(responses):
   data=[next(r for r in rows if r['response']==response and r['predictor']==v) for v in predictors]
   val=np.array([r['effect_per_one_predictor_SD'] for r in data]);lo=np.array([r['effect_percentile_low'] for r in data]);hi=np.array([r['effect_percentile_high'] for r in data]);offset=0 if len(responses)==1 else [-.14,.14][j];y=np.arange(4)+offset
   assert np.all(lo<=val) and np.all(val<=hi) and lo.min()>limits[0] and hi.max()<limits[1]
   color=neutral if len(responses)==1 else [blue,orange][j];marker='o' if j==0 else 's'
   ax.errorbar(val,y,xerr=[val-lo,hi-val],fmt=marker,color=color,ecolor=color,markersize=3.4,capsize=2,elinewidth=.8,markeredgewidth=.65,zorder=3)
   for row in data:plotted.append({'figure_panel':letter,'response':response,'predictor':row['predictor'],'estimate':row['effect_per_one_predictor_SD'],'low':row['effect_percentile_low'],'high':row['effect_percentile_high'],'n':row['n'],'predictor_SD':row['predictor_SD']})
  ax.set_yticks(range(4),predictors);ax.set_ylim(3.55,-.55);ax.set_xlim(limits);ax.set_xticks(ticks);ax.set_xlabel(xlabel,labelpad=5,fontsize=6.6 if letter in 'cde' else 7)
  ax.tick_params(axis='both',length=2.5,width=.65,pad=3);ax.spines[['top','right','left']].set_visible(False)
  ax.text(-.14 if letter in 'ab' else -.23,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=8,va='bottom')
  ax.text(0,1.08,title,transform=ax.transAxes,fontweight='bold',fontsize=7,va='bottom')
 handles=[Line2D([],[],color=blue,marker='o',lw=.8,markersize=3.4,label='Early period 2017–2020'),Line2D([],[],color=orange,marker='s',lw=.8,markersize=3.4,label='Late period 2021–2024')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.50),ncol=2,fontsize=6.8,handlelength=1.7,columnspacing=2.5)
 fig.text(.055,.045,'Coefficients per predictor SD; bars show pointwise 95% block-bootstrap intervals.',fontsize=6.5)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  bb=t.get_window_extent(renderer);assert bb.x0>=0 and bb.y0>=0 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(t.get_text(),bb.bounds)
  bounds.append({'text':t.get_text(),'bounds':list(bb.bounds)})
 for ext in ['svg','pdf','tiff','png']:fig.savefig(out/f'Figure_4.{ext}',dpi=600)
 fig.savefig(out/'Figure_4_preview.png',dpi=200)
 # Grayscale is only a QA preview; shape as well as colour distinguishes the periods.
 from PIL import Image
 Image.open(out/'Figure_4_preview.png').convert('L').save(out/'Figure_4_grayscale.png')
 plt.close(fig)
 with (out/'Figure_4_panel_values.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(plotted[0]));w.writeheader();w.writerows(plotted)
 assert len(plotted)==32
 (out/'Figure_4_QA.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'coefficients':32,'all_values_in_axis_limits':True,'all_text_in_canvas':True,'no_refitting':True,'size_mm':[183,150],'text_bounds':bounds,'python':platform.python_version(),'matplotlib':matplotlib.__version__},indent=2),encoding='utf8')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();draw(args.source,args.out)
