"""Read-only coverage audit of frozen full H/A/D training feature matrices.

Uses no evaluation labels or scores and never changes the declared feature set.
Weighted coverage reflects the retained rows' original time/provider mass.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]


def audit(directory):
    spec_path = directory/'predeclared.json'
    spec = json.loads(spec_path.read_text())
    manifest = spec['feature_manifest']['full']
    folds = []
    for source in sorted(spec['training_matches']):
        prefix = directory/f'matrix_without_{source}_full'
        xp, yp, rp = (prefix.with_suffix(s) for s in ('.npy', '.npz', '.json'))
        receipt = json.loads(rp.read_text())
        if receipt['predeclared_sha256'] != sha256(spec_path) or receipt['x_sha256'] != sha256(xp) or receipt['y_weight_sha256'] != sha256(yp):
            raise ValueError('Frozen matrix provenance differs')
        matrix = np.load(xp, mmap_mode='r')
        with np.load(yp) as saved:
            weights = saved['weight']
        if matrix.shape != (len(weights), len(manifest)) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Invalid matrix or positive row mass')
        count = np.zeros(matrix.shape[1], np.int64)
        mass = np.zeros(matrix.shape[1], np.float64)
        low, high = np.full(matrix.shape[1], np.inf), np.full(matrix.shape[1], -np.inf)
        for start in range(0, len(matrix), 4096):
            values = matrix[start:start+4096]
            valid = np.isfinite(values)
            count += valid.sum(axis=0)
            mass += np.sum(valid*weights[start:start+4096, None], axis=0)
            low = np.minimum(low, np.min(np.where(valid, values, np.inf), axis=0))
            high = np.maximum(high, np.max(np.where(valid, values, -np.inf), axis=0))
        columns = [dict(name=m['name'], family=m['family'], finite_rows=int(count[i]),
                        finite_fraction=float(count[i]/len(matrix)), weighted_finite_fraction=float(mass[i]/weights.sum()),
                        minimum=float(low[i]) if count[i] else None, maximum=float(high[i]) if count[i] else None,
                        all_missing=bool(count[i] == 0), constant_when_observed=bool(count[i] > 0 and low[i] == high[i]))
                   for i, m in enumerate(manifest)]
        groups = []
        for family in sorted({c['family'] for c in columns}):
            selected = [c for c in columns if c['family'] == family]
            groups.append(dict(family=family, features=len(selected),
                mean_weighted_finite_fraction=float(np.mean([c['weighted_finite_fraction'] for c in selected])),
                minimum_weighted_finite_fraction=min(c['weighted_finite_fraction'] for c in selected),
                all_missing=sum(c['all_missing'] for c in selected), constant_when_observed=sum(c['constant_when_observed'] for c in selected)))
        folds.append(dict(heldout_provider=source, rows=len(matrix), features=len(manifest),
                          matrix_sha256=receipt['x_sha256'], groups=groups, columns=columns))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), audit_source_sha256=sha256(__file__),
                predeclared_sha256=sha256(spec_path), scope='Frozen training matrices only; no fitting, score reading or design changes.',
                interpretation='Finite coverage checks availability, not correctness or generalization value. Constant observed values may still supply missingness information. Weighted coverage uses equal provider/match/view and within-run sampling masses.', folds=folds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'data/experiments/joint_lgbm_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/results/joint_lgbm_feature_coverage.json')
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([dict(heldout_provider=r['heldout_provider'], rows=r['rows'], groups=r['groups']) for r in result['folds']], indent=2))
