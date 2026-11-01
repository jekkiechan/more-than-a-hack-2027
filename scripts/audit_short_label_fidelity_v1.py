"""Aggregate TRAIN-only short-label provenance and observable support audit."""
from pathlib import Path
from collections import Counter, defaultdict
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments import short_quality_diversity_v1 as study

p=study.p


def main():
    records=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
    bins=defaultdict(Counter);games=defaultdict(Counter);native=Counter()
    for r in records:
        for s in study.quality.spans(r):
            bucket='<=1s' if s['seconds']<=1 else '(1,2]s' if s['seconds']<=2 else '(2,3]s'
            for counts in (bins[bucket],games[str(r['match'])]):
                counts['spans']+=1
                for field in ('quality','anchor_conflict','controlled_proof','no_observed_ball','held'):
                    counts[field]+=int(s[field])
                counts['sampled_ha_rows']+=len(s['sampled_rows'])
                counts['corroborating_sampled_rows']+=s['supported_rows']
            a,b=s['start'],s['stop'];runs=r['touch_runs']
            proof=runs[(runs[:,0]<b)&(runs[:,1]>=a)&(runs[:,2]==s['team'])&(runs[:,4]==1)&(runs[:,5]==0)]
            if not np.any((proof[:,1]-proof[:,0]+1)/float(r['native_fps'])>=.2):
                native['no_control_run_at_least_point2s']+=1
            native['touch_provenance_frames']+=int((r['full_prov'][a:b]==1).sum())
            native['native_ha_frames']+=b-a
            native['native_observed_ball_frames']+=int(np.asarray(r['full_ball_observed'])[a:b].sum())
    p.write(p.RESULTS/f'{study.STUDY}_label_fidelity_audit.json',dict(train_only=True,
        records=len(records),games=len(games),duration_bins={k:dict(v) for k,v in bins.items()},
        per_game={k:dict(v) for k,v in sorted(games.items())},native_summary=dict(native),
        limitations='Touch-control, sparse anchors and geometry remain imperfect proxies. These are consistency and observability checks; no gold fidelity estimate, changed labels, new games, evaluation or organizer access.',
        source_sha256=p.sha(Path(__file__)),quality_source_sha256=p.sha(Path(study.quality.__file__))))


if __name__=='__main__':main()
