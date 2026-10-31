"""Unlabeled fusion-quality sensitivity, separate from possession evaluation.

All proxy gates are reported; ball agreement is not coordinate ground truth.
This audit neither changes the fusion output nor selects/promotes a model.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import broadcast_fusion as bf, current_baseline, tracking_research as tr
from experiments.eval_set.protocol import sha256


def audit(tactical, broadcast, alignment):
    mapped = np.flatnonzero(alignment >= 0)
    target = alignment[mapped]
    unique = np.bincount(target, minlength=len(tactical['ball']))[target] == 1
    mapped, target = mapped[unique], target[unique]
    errors = np.array([bf.geometry(tactical['players'][t], tactical['teams'],
                                  broadcast['players'][b], broadcast['teams'])[0]
                       for b,t in zip(mapped,target)])
    source = broadcast['ball']
    seen = np.isfinite(source).all(axis=1)
    seen &= ((source >= -5) & (source <= tactical['pitch']+5)).all(axis=1)
    speed = np.linalg.norm(np.diff(source, axis=0), axis=1)*float(broadcast['fps'])
    reliable = seen.copy()
    reliable[1:] &= np.isfinite(speed) & (speed <= 65)
    reliable[:-1] &= np.isfinite(speed) & (speed <= 65)
    sequence = np.r_[0, np.cumsum(np.diff(alignment) != 1)]
    persistent = np.zeros(len(source), dtype=bool)
    for a,b in tr.runs(reliable,sequence):
        if b-a >= max(1, round(.2*float(broadcast['fps']))):
            persistent[a:b] = True
    active = tactical['period'] > 0
    observed = np.isfinite(tactical['ball']).all(axis=1)
    missing_count = int((active & ~observed).sum())
    rows=[]
    for threshold in (.5,1.,2.):
        for motion in (False,True):
            keep = seen[mapped] & active[target] & (errors <= threshold)
            if motion:
                keep &= persistent[mapped]
            b,t = mapped[keep],target[keep]
            both = observed[t]
            residual = np.linalg.norm(tactical['ball'][t[both]]-source[b[both]],axis=1)
            recovered = int((~both).sum())
            rows.append(dict(player_error_limit_m=threshold,motion_and_persistence_gate=motion,
                             recovered_frames=recovered,missing_fraction_recovered=recovered/max(1,missing_count),
                             overlapping_ball_frames=len(residual),
                             overlap_ball_error_quantiles_m=np.quantile(residual,[.5,.9,.99]).tolist() if len(residual) else [],
                             overlap_error_over5m_fraction=float((residual > 5).mean()) if len(residual) else None))
    return dict(rows=rows,target_labels_read=False,organizer_scored=False,
                motion_gate='Both adjacent ball displacements <=65 m/s; contiguous accepted run >=0.2 s',
                interpretation='Proxy sensitivity only. Agreement where both feeds observe does not validate recovered gap coordinates. No gate selected on possession scores.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fusion-dir',type=Path,default=Path('data/experiments/broadcast_fusion'))
    p.add_argument('--out',type=Path,default=Path('data/experiments/broadcast_fusion/quality.json'))
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a completed quality audit')
    tactical=current_baseline.load_inputs('data/challenge/inputs.npz')
    broadcast=bf.load_atd('data/broadcast/BAR-ATH_broadcast_atd.xml','data/broadcast/BAR-ATH_broadcast_atd.txt',tactical['pitch'])
    result=audit(tactical,broadcast,np.load(args.fusion_dir/'alignment.npy',allow_pickle=False))
    result['source_sha256']=sha256(Path(__file__))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
