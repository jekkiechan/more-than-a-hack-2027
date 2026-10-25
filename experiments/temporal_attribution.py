"""Fit the unchanged multi-source H/A backbone for the temporal-D ablation.

This never loads organizer inputs or labels. Cached models remain under data/.
"""
from pathlib import Path
import argparse
import json
import time
import joblib
import numpy as np
from experiments import dead_transfer, multisource_eval, refine_features, robust

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/experiments/learned_temporal')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for source, split in dead_transfer.SPLITS.items():
        path = args.out/f'{source}_attribution_train.joblib'
        if not path.exists():
            t = time.monotonic()
            joblib.dump(multisource_eval.training_rows(split['train'], 2026), path)
            print(f'ATTRIBUTION FEATURES {source} {time.monotonic()-t:.1f}s', flush=True)
        cache[source] = joblib.load(path)
    model_path = args.out/'attribution.joblib'
    if not model_path.exists():
        t = time.monotonic()
        model = multisource_eval.fit(tuple(dead_transfer.SPLITS), cache)
        model['training'] = {s: [p.stem for p in split['train']] for s, split in dead_transfer.SPLITS.items()}
        joblib.dump(model, model_path)
        print(f'ATTRIBUTION FIT {time.monotonic()-t:.1f}s', flush=True)
    del cache
    model = joblib.load(model_path)
    for source, split in dead_transfer.SPLITS.items():
        for path in split['eval']:
            out = args.out/f'{path.stem}_attribution.npz'
            if out.exists():
                continue
            inputs, labels = multisource_eval.view(path)
            x, names = refine_features.features(inputs)
            p = refine_features.probabilities_from_features(x, names, model)
            np.savez_compressed(out, probabilities=p.astype(np.float32))
            print(f'ATTRIBUTION PREDICTED {path.stem}', flush=True)
    (args.out/'attribution_manifest.json').write_text(json.dumps({
        'model_sha256': robust.sha256(model_path),
        'training': model['training'],
        'description': 'Unchanged pooled multisource_eval HGB 63 leaves/260 iterations/.08; conditional H/A fixed across every D head and provider fold.',
        'provider_holdout_scope': 'D head only; attribution remains fixed and has seen training matches from each provider.',
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
