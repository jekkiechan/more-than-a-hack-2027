"""Export aggregate research figures from the saved study readout."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results'


def main():
    readout=OUT/'seed_gap_context_readout.json'
    if readout.exists():
        data=json.loads(readout.read_text())
        baseline=next(r for r in data['diagnostic_summaries'] if r['recipe']=='seed_2026' and r['view']=='feed_like')
    else:
        sys.path.insert(0,str(ROOT))
        from scripts.summarize_seed_gap_context import diagnostic_summary
        data=json.loads((OUT/'seed_gap_context.json').read_text())
        baseline=diagnostic_summary([r for r in data['rows'] if r['recipe']=='seed_2026' and r['view']=='feed_like'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    groups=[['gap:no_gap','gap:3-10s','gap:10-20s','gap:20-40s','gap:40-infs'],
            ['contact:0-1s','contact:1-3s','contact:3-5s','contact:5-10s','contact:10-20s','contact:20-infs']]
    labels=[['Observed','3–<10','10–<20','20–<40','≥40'],['<1','1–<3','3–<5','5–<10','10–<20','≥20']]
    for ax,keys,ticks,title in zip(axes,groups,labels,['Missing-ball interval duration (s)','Distance to nearest reliable contact proxy (s)']):
        for metric,label,color in [('wrong_team_rate','Wrong team','#bf4d28'),('false_dead_rate','False dead ball','#3879a8')]:
            values=[100*baseline['frames'][k][metric]['value'] for k in keys]
            ax.plot(range(len(keys)),values,'o-',label=label,color=color,lw=2)
        ticks=[f'{label}\n(p={baseline["frames"][k]["team_accuracy"]["n_providers"]})' for label,k in zip(ticks,keys)]
        ax.set_xticks(range(len(keys)),ticks);ax.set_xlabel(title)
        ax.set_ylim(0,40);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Errors per 100 reference-live frames')
    axes[0].legend(frameon=False)
    fig.suptitle('Adopted GRU seed 2026: feed-like development views\nReplicates → matches → equal providers; p = providers with support',fontsize=12)
    for extension in ('png','svg'):
        fig.savefig(OUT/f'seed_gap_context_errors.{extension}',dpi=180)
    plt.close(fig)
    base=json.loads((OUT/'seed_gap_context.json').read_text())
    long_path=OUT/'seed_gap_context_long.json'
    long=json.loads(long_path.read_text()) if long_path.exists() else {'summaries':[]}
    summaries=base['summaries']+long['summaries']
    recipes=[('seed_2026','5s seed 2026'),('seed_2027','5s seed 2027'),
             ('equal_probability_ensemble','5s probability ensemble')]
    if long['summaries']: recipes += [('context15_2026','15s seed 2026'),('context15_2027','15s seed 2027')]
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    positions=np.arange(len(recipes));width=.36
    for offset,view,label,color in [(-.5,'clean','Clean','#3879a8'),(.5,'feed_like','Feed-like','#bf4d28')]:
        values=[next(r['spell_f1'] for r in summaries if r['recipe']==key and r['view']==view) for key,_ in recipes]
        bars=ax.bar(positions+offset*width,values,width,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(positions,[v for _,v in recipes],rotation=12,ha='right')
    ax.set_ylabel('Official spell F1, mean over IoU .3–.7');ax.set_ylim(0,.78)
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    ax.set_title('Matched full-provider-holdout development comparison\nTwo individual runs and an actual probability ensemble')
    for extension in ('png','svg'):
        fig.savefig(OUT/f'seed_gap_context_scores.{extension}',dpi=180)
    plt.close(fig)
    for path in OUT.glob('seed_gap_context*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__': main()
