"""Plot saved SVD sweeps against layer/head INT8; does not run models."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--svd-run',type=Path,default=HERE/'results/sssm-all-models-kl-parallel-20260923T072951Z')
ap.add_argument('--head-run',type=Path,default=Path((HERE/'latest_per_head.txt').read_text().strip()))
args=ap.parse_args()
old=json.loads((args.svd_run/'summary.json').read_text())
new=json.loads((args.head_run/'summary.json').read_text())
rows=old+new
models=['DeltaNet-1.3B','GatedDeltaNet-1.3B','Mamba2-1.3B','Nemotron-3-Nano-4B','Qwen3.5-4B']
for model in models:
 for pos in (256,-1):
  rs=[r for r in rows if r['model']==model and r['position']==pos]
  assert len(rs)==8 and all(r['n']==6400 for r in rs)
rows=[r for r in rows if r['method']!='int8_layer_affine']
out=args.head_run/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for output_key,title,metrics in [('kl_p99','p99 KL divergence',['kl_p99']),('kl_mean','Mean KL divergence',['kl_mean']),('kl_mean_p99','Mean and p99 KL divergence',['kl_mean','kl_p99'])]:
 fig=plt.figure(figsize=(15,8.5))
 grid=fig.add_gridspec(2,6,left=.075,right=.98,bottom=.09,top=.83,wspace=1.15,hspace=.5)
 slots=[grid[0,0:2],grid[0,2:4],grid[0,4:6],grid[1,1:3],grid[1,3:5]]
 axes=[fig.add_subplot(slot) for slot in slots]
 values=[r[metric] for r in rows for metric in metrics]
 for ax,model in zip(axes,models):
  for pos,color in [(256,'#2563eb'),(-1,'#ea580c')]:
   rs=[r for r in rows if r['model']==model and r['position']==pos]
   svd=sorted([r for r in rs if r['method'].startswith('svd')],key=lambda r:int(r['method'][3:]))
   for metric in metrics:
    tail=metric=='kl_p99'
    ax.plot(range(6),[r[metric] for r in svd],linestyle=':' if tail else '-',marker='o',color=color,lw=2,ms=4,markerfacecolor='white' if tail else color)
    value=next(r[metric] for r in rs if r['method']=='int8_head_affine')
    ax.axhline(value,color=color,linestyle=':' if tail else '-',lw=1.4,alpha=.65,zorder=1)
  ax.set_yscale('log');ax.set_ylim(min(values)*.65,max(values)*1.6)
  ax.set_xticks(range(6),labels=['1','2','4','8','16','32'])
  ax.tick_params(axis='x',labelsize=9);ax.set_xlim(-.15,5.15)
  ax.set_title(model);ax.set_xlabel('SVD rank')
  ax.grid(True,axis='both',which='major',alpha=.2)
 for ax in (axes[0],axes[3]):ax.set_ylabel('KL divergence (log scale)' if len(metrics)>1 else title+' (log scale)')
 fig.suptitle(title+' vs SVD rank',fontsize=18,y=.97)
 handles=[Line2D([0],[0],color='#2563eb',lw=2,label='256-token prefix'),Line2D([0],[0],color='#ea580c',lw=2,label='Full prompt'),Line2D([0],[0],color='#444444',marker='o',lw=1.5,label='SVD'),Line2D([0],[0],color='#444444',lw=1.4,alpha=.65,label='INT8 per head (horizontal)')]
 if len(metrics)>1:
  handles.extend([Line2D([0],[0],color='#444444',ls='-',marker='o',label='Mean (filled)'),Line2D([0],[0],color='#444444',ls=':',marker='o',markerfacecolor='white',label='p99 (hollow)')])
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3 if len(metrics)>1 else 4,frameon=False)
 assert len(axes)==5
 assert abs((axes[0].get_position().x0+axes[2].get_position().x1)-(axes[3].get_position().x0+axes[4].get_position().x1))<1e-10
 for ext in ('png','svg','pdf'):fig.savefig(out/f'rank_vs_{output_key}.{ext}',dpi=200)
 plt.close(fig)
with (out/'plot_data.csv').open('w') as f:
 fields=['model','position','method','n','kl_mean','kl_p99']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
(out/'README.md').write_text(f'# Rank vs KL plots\n\nSVD source: {args.svd_run}\n\nPer-head INT8 source: {args.head_run}\n\nFive model panels: three above two centered below. Blue: 256-token prefix; orange: full prompt. Y axis is logarithmic; x axis shows SVD ranks. Per-head INT8 is shown as horizontal reference lines; per-layer INT8 is omitted. No experiments were rerun. The combined figure uses solid lines / filled markers for mean and dotted lines / hollow markers for p99. The mean and p99 figures use the corresponding saved aggregate statistics.\n')
print(out)
