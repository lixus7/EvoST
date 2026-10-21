"""Rebuild the checkpoint-age figure from the adjacent exported result values.

Example: python plot_checkpoint_age.py --output ../../fig/evost_checkpoint_age.pdf
Requires matplotlib. No training data or checkpoints are needed.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).with_name('checkpoint_age.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('evost_checkpoint_age.pdf'))
    args=parser.parse_args()
    data=json.loads(args.data.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.35,2.25),layout='constrained')
    for offset,key,color,label in [(-.18,'old','#b9c5bd','Old checkpoint'),(.18,'refresh','#377961','Identity refresh')]:
        vals=[p[key] for p in data['panels']]
        xs=[i+offset for i in range(len(vals))]
        ax.bar(xs,[v['gm']-1 for v in vals],bottom=1,width=.34,color=color,label=label,zorder=3)
        ax.errorbar(xs,[v['gm'] for v in vals],yerr=[[v['gm']-v['lo'] for v in vals],[v['hi']-v['gm'] for v in vals]],fmt='none',ecolor='#27352e',capsize=2,lw=.8,zorder=4)
    ax.axhline(1,color='#444444',lw=.8)
    ax.set_ylim(.985,1.285)
    ax.set_yticks([1,1.1,1.2])
    ax.set_xticks(range(len(data['panels'])),['%s years\n%s settings'%(p['age_years'],p['settings']) for p in data['panels']])
    ax.set_ylabel('MAE / target-year MAE')
    ax.grid(axis='y',alpha=.18,zorder=0)
    ax.legend(frameon=False,loc='upper center',ncol=2,fontsize=7)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)

if __name__=='__main__':
    main()
