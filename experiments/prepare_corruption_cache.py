"""Rebuild external evaluation caches needed by the light-corruption follow-up.

Restore the first experiment's frozen model archive and run its training-feature
preparation first. This helper never opens organizer inputs or reference labels.
"""
import argparse
from pathlib import Path
import warnings
import joblib
from experiments import (corruption_experiment as ce, current_baseline,
                         multisource_eval, refine_features, tracking_corruption as tc)


def run(out, source):
    spec = ce.check(out)
    original = current_baseline.load_models()['attribution']
    augmented = ce.load_model(out, out/'team_augmented.joblib')
    for path in ce.dead_transfer.SPLITS[source]['eval']:
        inputs, labels = multisource_eval.view(path)
        for view in ce.VIEWS:
            pending = out/f'pending_{path.stem}_{view}.joblib'
            target = out/f'eval_{path.stem}_{view}.joblib'
            if pending.exists() and target.exists():
                continue
            tracking = inputs if view == 'clean' else tc.corrupt(
                inputs, spec['calibration'], ce.seed(path.stem, view))
            x, names = refine_features.features(tracking)
            baseline = refine_features.probabilities_from_features(x, names, original)
            record = ce.record(tracking, labels, source, path.stem, view)
            visibility = tc.visibility(tracking)
            if not pending.exists():
                joblib.dump(dict(matrix=x, names=names, baseline=baseline, record=record,
                                 visibility=visibility), pending)
            if not target.exists():
                new = refine_features.probabilities_from_features(x, names, augmented)
                joblib.dump(dict(baseline=baseline, record=record, visibility=visibility,
                                 augmented=new), target)
            print('PREPARED', path.stem, view, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ce.DEFAULT)
    p.add_argument('--source', choices=tuple(ce.dead_transfer.SPLITS))
    args = p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for source in ([args.source] if args.source else ce.dead_transfer.SPLITS):
            run(args.out, source)


if __name__ == '__main__': main()
