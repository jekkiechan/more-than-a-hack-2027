"""Promote and run the adopted baseline package without organizer reference labels."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from experiments import challenge, learned_temporal as lt, refine_features, robust, supervised
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
MANIFEST = ROOT/'experiments/baseline_manifest.json'
MODEL_DIRECTORY = ROOT/'data/models/gru_hgb_v1'


def official_better(candidate, baseline):
    return (candidate['spell_f1'], candidate['macro_f1']) > (baseline['spell_f1'], baseline['macro_f1'])


def promote():
    if MANIFEST.exists():
        raise ValueError('Baseline already promoted; do not overwrite its provenance')
    report_path = ROOT/'experiments/results/temporal_ensemble_development.json'
    report = json.loads(report_path.read_text())
    def row(name):
        return next(r for r in report['summaries'] if r['model'] == name and r['view'] == 'provider_holdout_D')
    candidate, old = row('gru_hgb'), row('ordered_hgb')
    if not official_better(candidate, old):
        raise ValueError('GRU/HGB did not beat the previous baseline under the official headline rule')
    cohort = {name: {(r['source'], r['match']) for r in report['rows']
                     if r['model'] == name and r['fold'] != 'pooled'}
              for name in ('gru_hgb', 'ordered_hgb')}
    if not cohort['gru_hgb'] or cohort['gru_hgb'] != cohort['ordered_hgb']:
        raise ValueError('Baseline and candidate cohorts differ')
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, original in (
        ('bigru.joblib', 'pooled_bigru.joblib'),
        ('ordered_hgb.joblib', 'pooled_ordered_hgb.joblib'),
        ('attribution.joblib', 'attribution.joblib'),
    ):
        source = ROOT/'data/experiments/learned_temporal'/original
        target = MODEL_DIRECTORY/name
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError('Existing baseline model differs')
        shutil.copyfile(source, target)
        files[name] = sha256(target)
    manifest = dict(version=1, model='gru_hgb', status='Current adopted development baseline; fresh confirmation pending',
        promoted_utc=datetime.now(timezone.utc).isoformat(), replaced_baseline='ordered_hgb',
        model_directory=str(MODEL_DIRECTORY.relative_to(ROOT)), models_sha256=files,
        architecture=dict(D_members={'bigru': .5, 'ordered_hgb': .5}, team='Fixed pooled contextual HGB',
                          decoder_switch_cost=lt.COST),
        selection_rule='Spell F1 descending, macro F1 for exact ties',
        evidence=dict(report=str(report_path.relative_to(ROOT)), report_sha256=sha256(report_path),
                      view='provider_holdout_D', candidate=candidate, previous_baseline=old,
                      cohort=[list(x) for x in sorted(cohort['gru_hgb'])]),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                       (Path(__file__), Path(lt.__file__), Path(refine_features.__file__))},
        organizer_labels_used_for_promotion=False, historical_selection_records_preserved=True)
    with MANIFEST.open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print('PROMOTED GRU + HGB:',candidate['spell_f1'],candidate['macro_f1'])


def load_inputs(path):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in robust.INPUT_KEYS if key in source.files}
    for key, mask in (('ball', 'ball_observed'), ('players', 'player_observed')):
        if mask in data:
            data[key] = np.array(data[key], copy=True)
            data[key][~data[mask]] = np.nan
    return data


def load_models(directory=None, manifest_path=None):
    manifest_path = Path(manifest_path) if manifest_path else MANIFEST
    manifest = json.loads(manifest_path.read_text())
    directory = Path(directory) if directory else ROOT/manifest['model_directory']
    for name, expected in manifest['models_sha256'].items():
        if sha256(directory/name) != expected:
            raise ValueError(f'Baseline model hash mismatch: {name}')
    return {name.removesuffix('.joblib'): joblib.load(directory/name) for name in manifest['models_sha256']}


def default_output(manifest):
    return ROOT/manifest.get('default_output', f"data/experiments/current_baseline/{manifest['model']}.csv")


def dead_probability(inputs, models):
    indices = lt.sampled_indices(inputs)
    x, names = lt.player_features(inputs)
    periods = supervised.model_periods(inputs)
    record = dict(x=x[indices], names=names, period=periods[indices], indices=indices, native_n=len(x))
    pieces = [lt.upsample(lt.predict_sampled(record, models[name]), record, periods)
              for name in ('bigru', 'ordered_hgb')]
    return (pieces[0]+pieces[1])*.5


def joint_gru_probabilities(inputs, models):
    """Player-only-D joint GRU package: one network outputs full H/A/D."""
    from experiments import factorized_had, live_gap_experiment
    placeholder = np.full(len(inputs['ball']), 'X', dtype='U1')
    record = live_gap_experiment.feature_view(inputs, placeholder, 'competition', 'BAR-ATH')['joint']
    periods = supervised.model_periods(inputs)
    sampled = factorized_had.predict(record, models['joint_gru'])
    return np.column_stack([lt.upsample(sampled[:, k], record, periods) for k in range(3)])


def predict_probabilities(inputs, models, manifest=None):
    if manifest and manifest.get('architecture', {}).get('kind') == 'joint_gru':
        return joint_gru_probabilities(inputs, models)
    dead = dead_probability(inputs, models)
    team = refine_features.predict_probabilities(inputs, models['attribution'])
    return lt.refine_domain.combine(team, dead, 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('promote', 'predict'))
    parser.add_argument('--inputs', type=Path, default=ROOT/'data/challenge/inputs.npz')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--manifest', type=Path, default=MANIFEST,
                        help='Active baseline by default; historical manifests support label-free replay')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'promote':
        promote()
        return
    manifest = json.loads(args.manifest.read_text())
    output = args.output or default_output(manifest)
    if output.exists():
        raise ValueError('Refusing to overwrite a generated baseline candidate')
    inputs = load_inputs(args.inputs)
    probabilities = predict_probabilities(inputs, load_models(args.models_dir, args.manifest), manifest)
    labels = supervised.decode(probabilities, supervised.model_periods(inputs), float(inputs['fps']), lt.COST)
    labels[np.asarray(inputs['period']) <= 0] = 'X'
    challenge.write_submission(labels, output)
    output.with_suffix('.json').write_text(json.dumps(dict(
        model=manifest['model'], input_sha256=sha256(args.inputs), output_sha256=sha256(output),
        baseline_manifest_sha256=sha256(args.manifest),
        target_labels_read=False, organizer_scored=False), indent=2)+'\n')
    print('Generated and validated', manifest['model'], 'without reference labels')


if __name__ == '__main__':
    main()
