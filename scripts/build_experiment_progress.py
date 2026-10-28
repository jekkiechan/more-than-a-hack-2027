"""Build the Page's source-backed aggregate progress data, without scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'docs/research/experiment_progress.json'


def read(relative):return json.loads((ROOT/relative).read_text())


def build():
    report=read('experiments/results/seed_gap_context.json')
    ablation=read('experiments/results/stress_component_ablation.json')
    long=read('experiments/results/seed_gap_context_long.json')
    old=read('experiments/baselines/gru_hgb_ballgap25_v1.json')
    single_path=ROOT/'experiments/baselines/joint_gru_slotdrop_v1.json'
    single=json.loads(single_path.read_text()) if single_path.exists() else read('experiments/baseline_manifest.json')
    current=read('experiments/baseline_manifest.json')
    summaries=report['summaries']+long['summaries']
    def scores(name):
        return {view:{'spell':next(r for r in summaries if r['recipe']==name and r['view']==view)['spell_f1'],
                      'macro':next(r for r in summaries if r['recipe']==name and r['view']==view)['macro_f1']}
                for view in ('clean','feed_like')}
    adopted=[dict(id='ballgap25',label='Ball gap mix',date=old['promoted_utc'][:10],
                  **{v:{'spell':ablation['summary']['baseline'][v]['spell_f1'],
                         'macro':ablation['summary']['baseline'][v]['macro_f1']} for v in ('clean','feed_like')},
                  status='Former baseline; scores re-evaluated under current protocol'),
             dict(id='single',label='Single GRU',date=single['promoted_utc'][:10],**scores('seed_2026'),status='Former single-seed baseline'),
             dict(id='ensemble',label='Two seed GRU',date='2026-10-28',**scores('equal_probability_ensemble'),status='User-adopted baseline')]
    compared=[dict(id=name,label=label,date='2026-10-28',**scores(name),status=status) for name,label,status in (
        ('seed_2026','5s seed 2026','Original single seed'),('seed_2027','5s seed 2027','Original single seed'),
        ('equal_probability_ensemble','Two seed GRU','User-adopted baseline'),
        ('context15_2026','15s seed 2026','Not adopted; D guardrail fails'),
        ('context15_2027','15s seed 2027','Not adopted; clean spell declines'))]
    checks=[]
    def add(path,key,label,stamp,status,identifier):
        d=read(path);value=d
        for k in key:value=value[k]
        checks.append(dict(id=identifier,label=label,date=stamp,spell=value['spell_f1_mean'],macro=value['macro_f1'],status=status,source=path))
    p='experiments/results/learned_temporal_public_check.json';d=read(p)
    add(p,('scores','bigru'),'Temporal GRU',d['frozen']['created_utc'],'Historical diagnostic','temporal')
    p='experiments/results/current_baseline_public_check.json';d=read(p)
    add(p,('scores','gru_hgb'),'GRU HGB',d['frozen']['started_utc'],'Former baseline','gru_hgb')
    p='experiments/results/light_corruption_public_check.json';d=read(p)
    add(p,('score',),'Ball gap mix',d['frozen']['created_utc'],'Former baseline','ballgap')
    p='experiments/results/joint_gru_public_check.json';d=read(p)
    add(p,('score',),'Joint player D',d['frozen']['created_utc'],'Diagnostic only; earlier stress guard failed','joint')
    p='experiments/results/temporal_competition_check.json';d=read(p)
    for key,label in [('full127_lgbm','Full127 tree'),('clean_leader_blend','Clean blend'),('robust_blend','Robust blend'),('dropout_gru','Single GRU')]:
        add(p,('scores',key),label,d['completed_utc'],'Former baseline' if key=='dropout_gru' else 'Diagnostic only',key)
    checks.append(dict(id='ensemble',label='Two seed GRU',date='2026-10-28',spell=None,macro=None,status='User-adopted; organizer score unavailable',source='experiments/results/seed_gap_context.json'))
    sources=sorted({r['source'] for r in checks}|{'experiments/results/stress_component_ablation.json',
        'experiments/results/seed_gap_context_long.json','experiments/baselines/gru_hgb_ballgap25_v1.json'})
    result=dict(version=1,updated_utc=datetime.now(timezone.utc).isoformat(),
        progress_page='https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9',
        current_selection='joint_gru_seed_ensemble_v1',
        development=dict(version=1,adoptions=adopted,comparisons=compared,
            protocol='Full H/A+D provider holdout, five previously exposed matches; replicate → match → equal provider mean',
            date_basis='Adoption dates; historical recipes retrospectively compared under the same current protocol'),
        organizer=dict(version=1,checkpoints=checks,protocol='Same public 7500 frames and 17 reference spells; saved diagnostic scores only'),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SAVED',RESULT,'without fitting or scoring')
    return result


if __name__=='__main__':build()
