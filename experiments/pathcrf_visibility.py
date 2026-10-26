"""Descriptive H/A accuracy by ball visibility for the fixed external ablation.

These groups diagnose the frozen candidates; they do not select weights or
change the model. No organizer inputs or labels are opened.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import multisource_eval
from experiments.pathcrf_ensemble import MATCHES, METHODS, OUT, CACHE, ROOT, path_for, team_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=ROOT/'experiments/results/pathcrf_visibility.json')
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Visibility report already exists')
    rows = []
    for match in MATCHES:
        inputs, labels = multisource_eval.view(path_for(match))
        with np.load(CACHE/f'{match}_attribution.npz', allow_pickle=False) as data:
            base = data['probabilities']
        with np.load(args.out/f'{match}_team.npz', allow_pickle=False) as data:
            path = data['team']
        live = np.isin(labels, ['H', 'A'])
        seen = np.asarray(inputs.get('ball_observed', np.isfinite(inputs['ball']).all(axis=1)), dtype=bool)
        groups = {'all_graded_live': live, 'ball_observed_live': live & seen,
                  'ball_missing_live': live & ~seen}
        for method, probabilities in team_variants(base, path).items():
            guess = np.array(['H', 'A'])[probabilities[:, :2].argmax(axis=1)]
            rows.append(dict(match=match, model=method, groups={
                name: {'frames': int(mask.sum()),
                       'ha_accuracy': float(np.mean(guess[mask] == labels[mask])) if mask.any() else None}
                for name, mask in groups.items()}))
    result = dict(scope='Fixed external candidates, descriptive conditional H/A accuracy before D combination and Viterbi',
        groups='Reference-live frames surviving the existing SkillCorner visibility filter, split by input ball observation flag',
        use='Mechanism diagnostic only; no subset-based selection or weight tuning', rows=rows)
    with args.report.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for row in rows:
        print(row['match'], row['model'], row['groups'])


if __name__ == '__main__':
    main()
