"""Fixed full-provider-holdout study of short-spell and transition objectives.

The existing slot-dropout GRU checkpoints are the matched control. Three new
loss variants are fitted per provider fold and seed; every candidate is sealed
before scoring on native and feed-like views. All frame data stays under the
ignored data/ tree and only aggregate rows are written to experiments/results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
from evaluation import possession_metrics as official_metrics

from experiments import (corruption_experiment as ce,
                         dead_transfer as dt,
                         error_analysis,
                         factorized_had,
                         gru_robustness_experiment as gr,
                         learned_temporal as lt,
                         live_gap_experiment as old,
                         metrics as metric_adapter,
                         multisource_eval as ms,
                         objective_losses,
                         objective_alignment_prep as prep,
                         refine_domain,
                         refine_features,
                         robust,
                         spell_diagnostics,
                         stress_components as sc,
                         supervised,
                         tracking_corruption as tc)
from experiments.eval_set import protocol as eval_protocol
from experiments.eval_set.protocol import sha256


ROOT = lt.ROOT
DEFAULT = ROOT / 'data/experiments/objective_alignment_v1'
RESULT = ROOT / 'experiments/results/objective_alignment_development.json'
EVAL_CACHE = ROOT / 'data/experiments/seed_gap_context_v1'
EVAL_CACHE_MANIFEST = ROOT.parent / 'mtah-data/seed_gap_context_cache_manifest.json'
EVAL_CACHE_MANIFEST_SHA256 = '11ba724c305c9dc42a65f0a62742e89256d292b8df31f45fa274a774916f4a59'
ARMS = ('control', 'spell_sqrt', 'transition', 'combined')
NEW_ARMS = ARMS[1:]
SEEDS = (2026, 2027)
VIEWS = ('clean', 'feed_like')
REPLICATES = ('corrupt_1', 'corrupt_2')
OBJECTIVE_BY_ARM = {
    'spell_sqrt': 'spell_sqrt',
    'transition': 'transition',
    'combined': 'combined',
}


def fingerprint() -> dict[str, str]:
    paths = (Path(__file__), Path(prep.__file__), Path(gr.__file__),
             Path(factorized_had.__file__), Path(objective_losses.__file__),
             Path(old.__file__), Path(lt.__file__), Path(sc.__file__),
             Path(ce.__file__), Path(error_analysis.__file__),
             Path(spell_diagnostics.__file__), Path(supervised.__file__),
             Path(ms.__file__), Path(tc.__file__), Path(metric_adapter.__file__),
             Path(refine_domain.__file__), Path(refine_features.__file__),
             Path(robust.__file__), Path(dt.__file__),
             ROOT / 'evaluation/possession_metrics.py')
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def _control_checkpoint(source: str, seed: int) -> Path:
    return gr.checkpoint(gr.DEFAULT, source, 'slot_dropout', seed)


def _control_inventory() -> dict:
    prep._reconcile_prior_source()
    frozen_path = gr.DEFAULT / 'checkpoints_frozen.json'
    frozen = json.loads(frozen_path.read_text())
    frozen_rows = {row['file']: row for row in frozen['checkpoints']}
    inventory = {}
    for source in dt.SPLITS:
        expected_sources = old.training_sources(source)
        expected_matches = [path.stem for provider in expected_sources
                            for path in dt.SPLITS[provider]['train']]
        for seed in SEEDS:
            model_path = _control_checkpoint(source, seed)
            receipt_path = model_path.with_suffix('.json')
            model = ce.load_model(gr.DEFAULT, model_path)
            row = frozen_rows.get(model_path.name)
            if row is None or row['sha256'] != sha256(model_path):
                raise ValueError(f'Archived control is not sealed: {model_path}')
            if row['receipt_sha256'] != sha256(receipt_path):
                raise ValueError(f'Archived control receipt changed: {receipt_path}')
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get('arm'), receipt.get('seed'), receipt.get('heldout')) != (
                    'slot_dropout', seed, source):
                raise ValueError(f'Unexpected archived control provenance: {receipt_path}')
            if (receipt.get('sources') != expected_sources
                    or receipt.get('training_matches') != expected_matches):
                raise ValueError(f'Archived control used a different provider fold: {receipt_path}')
            inventory[f'{source}/{seed}'] = {
                'model': str(model_path.relative_to(ROOT)),
                'sha256': sha256(model_path),
                'receipt': str(receipt_path.relative_to(ROOT)),
                'receipt_sha256': sha256(receipt_path),
                'parameters': int(model['parameters']),
                'optimizer_updates': int(model['optimizer_updates']),
                'windows_per_epoch': int(model['windows_per_epoch']),
            }
    return inventory


def _evaluation_cache_inventory(out: Path, copy_manifest: bool) -> dict:
    """Verify the exact, frozen 5-second-context features used by prior controls."""
    if not EVAL_CACHE_MANIFEST.exists():
        raise ValueError(f'Missing private evaluation cache manifest: {EVAL_CACHE_MANIFEST}')
    archive_sha = sha256(EVAL_CACHE_MANIFEST)
    if archive_sha != EVAL_CACHE_MANIFEST_SHA256:
        raise ValueError('Unexpected seed-gap-context cache manifest')
    archive = json.loads(EVAL_CACHE_MANIFEST.read_text())
    if archive.get('experiment') != 'seed_gap_context_v1' or len(archive.get('files', ())) != 176:
        raise ValueError('Unexpected evaluation cache archive contents')
    by_path = {entry['path']: entry for entry in archive['files']}
    prior_path = EVAL_CACHE / 'predeclared.json'
    prior = json.loads(prior_path.read_text())
    preparation = prep.check_preparation(out)
    prepared_inputs = preparation['data_sha256']
    for key, digest in prior['inputs_sha256'].items():
        if prepared_inputs.get(key) != digest:
            raise ValueError(f'Evaluation cache input differs from preparation declaration: {key}')
    if prior.get('views') != {'clean': ['clean'],
                              'feed_like': ['corrupt_1', 'corrupt_2']}:
        raise ValueError('Evaluation cache views differ from the frozen comparison')
    if prior.get('feed_components') != ['ball_gaps', 'coordinates', 'slot_gaps', 'id_swaps']:
        raise ValueError('Evaluation cache stress components differ from the frozen comparison')
    if prior.get('calibration') != preparation['calibration']:
        raise ValueError('Evaluation cache stress calibration differs from the frozen comparison')
    # The feature builder and the label-free stress view must match the sources
    # bound by the frozen cache declaration.
    for key in ('experiments/live_gap_experiment.py', 'experiments/refine_features.py',
                'experiments/learned_temporal.py', 'experiments/supervised.py',
                'experiments/multisource_eval.py', 'experiments/stress_components.py',
                'experiments/tracking_corruption.py', 'experiments/corruption_experiment.py'):
        if prior['source_sha256'].get(key) != sha256(ROOT / key):
            raise ValueError(f'Evaluation feature cache source differs: {key}')
    controls = _control_inventory()
    for key, entry in controls.items():
        path = entry['model']
        if prior['checkpoint_sha256'].get(path) != entry['sha256']:
            raise ValueError(f'Evaluation cache used another control checkpoint: {key}')

    cells = {}
    for source in dt.SPLITS:
        for path in dt.SPLITS[source]['eval']:
            for view, replicate in (('clean', 'clean'),
                                    ('feed_like', 'corrupt_1'),
                                    ('feed_like', 'corrupt_2')):
                filename = f'{path.stem}_{view}_{replicate}.joblib'
                archive_path = f'experiments/seed_gap_context_v1/{filename}'
                cached = EVAL_CACHE / filename
                entry = by_path.get(archive_path)
                if entry is None or not cached.exists():
                    raise ValueError(f'Missing archived evaluation feature cache: {cached}')
                if cached.stat().st_size != entry['bytes'] or sha256(cached) != entry['sha256']:
                    raise ValueError(f'Evaluation feature cache differs from private manifest: {cached}')
                cells[f'{source}/{path.stem}/{view}/{replicate}'] = dict(
                    path=str(cached.relative_to(ROOT)), bytes=entry['bytes'],
                    sha256=entry['sha256'])

    local_manifest = out / 'evaluation_cache_manifest.json'
    if copy_manifest:
        if not local_manifest.exists():
            shutil.copyfile(EVAL_CACHE_MANIFEST, local_manifest)
        elif sha256(local_manifest) != archive_sha:
            raise ValueError('Local evaluation cache manifest changed')
    return dict(archive_manifest_sha256=archive_sha,
                local_manifest=str(local_manifest.relative_to(ROOT)),
                files=cells)


def freeze(out: Path = DEFAULT) -> dict:
    """Write the immutable declaration before preparing or fitting candidates."""
    out.mkdir(parents=True, exist_ok=True)
    declaration = out / 'predeclared.json'
    if declaration.exists():
        return check(out)

    preparation = prep.check_preparation(out)
    prepared_manifest_path = out / 'prepared_training_manifest.json'
    if not prepared_manifest_path.exists():
        raise ValueError('Seal the regenerated training feature caches before objective freeze')
    prepared_manifest = prep.seal_preparation(out)
    prepared_manifest_sha = sha256(prepared_manifest_path)
    if len(prepared_manifest['caches']) != 13:
        raise ValueError('Expected exactly 13 prepared training match caches')
    parity_path = out / 'training_parity.json'
    if not parity_path.exists():
        raise ValueError('Run validate-training before freezing the objective comparison')
    training_parity = json.loads(parity_path.read_text())
    if not training_parity.get('passed'):
        raise ValueError('Prepared training records failed exact control parity')
    if (training_parity['runner_sha256'] != sha256(Path(__file__))
            or training_parity['objective_losses_sha256'] != sha256(objective_losses.__file__)
            or training_parity['factorized_had_sha256'] != sha256(factorized_had.__file__)
            or training_parity['prepared_training_manifest_sha256'] != prepared_manifest_sha):
        raise ValueError('Training parity receipt does not bind the current fit source and caches')

    calibration_path = ROOT / 'experiments/results/realistic_corruption_development.json'
    calibration_report = json.loads(calibration_path.read_text())
    stress_report_path = ROOT / 'experiments/results/stress_component_ablation.json'
    stress_report = json.loads(stress_report_path.read_text())
    control_report_path = ROOT / 'experiments/results/gru_slot_robustness_development.json'

    training_sources = {heldout: old.training_sources(heldout) for heldout in dt.SPLITS}
    training_matches = {
        heldout: [path.stem for source in training_sources[heldout]
                  for path in dt.SPLITS[source]['train']]
        for heldout in dt.SPLITS
    }
    evaluation_matches = {
        source: [path.stem for path in dt.SPLITS[source]['eval']]
        for source in dt.SPLITS
    }
    controls = _control_inventory()
    if training_parity['control_checkpoints'] != controls:
        raise ValueError('Training parity receipt used a different archived control inventory')
    evaluation_cache = _evaluation_cache_inventory(out, copy_manifest=True)

    spec = dict(
        version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        question=('Does a mild, censored short-spell loss weight, a fixed transition '
                  'surrogate, or their combination improve full-provider-holdout '
                  'possession scores over the adopted slot-dropout joint GRU?'),
        hypothesis=('The short-spell weight should improve missed brief H/A spells; '
                    'transition supervision should improve H/A turnover timing and '
                    'state boundaries. Their combined arm tests additive value.'),
        source_sha256=fingerprint(),
        data_sha256=preparation['data_sha256'],
        baseline_manifest_sha256=sha256(ROOT / 'experiments/baseline_manifest.json'),
        decision_policy_sha256=sha256(ROOT / 'experiments/model_decision_policy.json'),
        preparation_declaration_sha256=sha256(out / 'preparation_declaration.json'),
        prepared_training_manifest_sha256=prepared_manifest_sha,
        prepared_training_caches=prepared_manifest['caches'],
        training_parity_sha256=sha256(parity_path),
        training_parity=training_parity,
        training_control=dict(
            source='Archived slot_dropout arm from gru_slot_robustness_v2',
            declaration_sha256=sha256(gr.DEFAULT / 'predeclared.json'),
            frozen_manifest_sha256=sha256(gr.DEFAULT / 'checkpoints_frozen.json'),
            published_report_sha256=sha256(control_report_path),
            checkpoints=controls,
            reused_training_record_recipe='Recreated under the separately frozen objective-alignment preparation declaration.',
        ),
        evaluation_feature_cache=evaluation_cache,
        matched_scoring_reference=dict(
            report=str(stress_report_path.relative_to(ROOT)),
            report_sha256=sha256(stress_report_path),
            declaration_sha256=stress_report['spec_sha256'],
            role='Exact archived score parity check for the six slot-dropout controls '
                 'on clean and feed_like cells.'),
        calibration=calibration_report['spec']['calibration'],
        calibration_report_sha256=sha256(calibration_path),
        stress_components=['ball_gaps', 'coordinates', 'slot_gaps', 'id_swaps'],
        arms=ARMS,
        new_arms=NEW_ARMS,
        objectives=OBJECTIVE_BY_ARM,
        seeds=SEEDS,
        heldout_sources=tuple(dt.SPLITS),
        training_sources=training_sources,
        training_matches=training_matches,
        evaluation_matches=evaluation_matches,
        views=VIEWS,
        stress_replicates=REPLICATES,
        training_recipe=(
            'Adopted factorized player-only joint GRU. Reuse exact clean/legacy plus '
            '25% whole-window ball-gap and player-slot-dropout records, equal '
            'provider/match mass, train-only normalization, 5 Hz, 10 s core plus '
            '5 s context on both sides, hidden size 24, 8 epochs, AdamW .001/.0001, '
            'batch 64, gradient clip 1, direct and H/A-swapped records. Loss objective '
            'is the only change.'),
        short_spell_weight=(
            'At native frame resolution, weight only complete H/A runs. A run touching '
            'a match/time-block edge or adjacent X/unknown label is censored and stays '
            'neutral. D and unknown targets stay neutral. Per-spell factor is '
            'clip(sqrt(5 seconds / duration), 0.5, 2.0); normalize factors to mean 1 '
            'over eligible complete-spell sampled H/A rows within each record. Apply '
            'the factor only to conditional H/A classification loss; preserve the '
            'marginal D/live gate loss and each record/source training mass. H/A swap '
            'augmentation shares the same duration factors.'),
        transition_auxiliary=(
            'No added head. On adjacent sampled known targets in one model period, '
            'require every native target between their source indices to be known '
            'and in one uninterrupted full-resolution period. Use a '
            'marginal P(D) change surrogate whose P(D) comes from the player-only '
            'dead path, plus a conditional H/A switch surrogate on adjacent live '
            'targets. Each component is cost-sensitive: one half of the stable-group '
            'mean plus one half of the changed-group mean; absent groups contribute '
            'zero. The D/live component and conditional H/A component each receive a '
            'fixed half of the auxiliary loss; absent components contribute zero. '
            'Per-source/match weights preserve the existing record mass. Fixed '
            'auxiliary coefficient lambda=.1. This balanced marginal surrogate is '
            'cost-sensitive and is not a calibrated/native transition likelihood. '
            'Combined adds this term to the H/A-weighted NLL.'),
        training_labels=(
            'Use existing external training labels only. Full-resolution labels are '
            'mapped to H/A/D ids and X/unknown=-1; spell runs break at model-period '
            'and timestamp discontinuities. No evaluation labels enter either '
            'objective.'),
        primary=('Equal-provider official mean possession-spell F1, with exact ties '
                 'broken by macro F1; report every IoU .3-.7. Also report fixed '
                 'auxiliary .70 spell + .30 macro, not used for ranking.'),
        guardrails=(
            'Per seed: feed_like stress spell F1 at least matched control -0.01; '
            'each provider clean D-F1 and macro-F1 at least matched control -0.02. '
            'A clean spell gain of +0.01 is the predeclared positive screen.'),
        budget=(
            '18 new fits (3 objectives x 3 full-provider folds x 2 seeds); reuse six '
            'sealed adopted slot-dropout controls (zero new control fits). Score all '
            'four arms on five development matches: one clean view and two fixed '
            'feed_like stress replicates. Seal every model before any scoring.'),
        decoder_switch_cost=lt.COST,
        context_seconds=5,
        historical_development_exposure=True,
        organizer_labels_read=False,
        baseline_changed=False,
        fresh_confirmation=False,
        automatic_promotion=False,
        limitations=[
            'All five development matches were previously exposed; full provider holdout does not create independent confirmation.',
            'Synthetic feed_like corruption is a sensitivity probe, not evidence of real broadcast transfer.',
            'No public organizer labels or scores are used for training, selection, or tuning.',
        ],
    )
    ce.write_json(declaration, spec, exclusive=True)
    return spec


def check(out: Path = DEFAULT) -> dict:
    spec = json.loads((out / 'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprint():
        raise ValueError('Frozen source changed; create a fresh output directory')
    if spec['baseline_manifest_sha256'] != sha256(ROOT / 'experiments/baseline_manifest.json'):
        raise ValueError('Active baseline manifest changed after declaration')
    if spec['decision_policy_sha256'] != sha256(ROOT / 'experiments/model_decision_policy.json'):
        raise ValueError('Decision policy changed after declaration')
    if spec['calibration_report_sha256'] != sha256(ROOT / 'experiments/results/realistic_corruption_development.json'):
        raise ValueError('Frozen stress calibration changed')
    if spec['matched_scoring_reference']['report_sha256'] != sha256(
            ROOT / spec['matched_scoring_reference']['report']):
        raise ValueError('Archived matched scoring reference changed')
    preparation = prep.check_preparation(out)
    if spec['preparation_declaration_sha256'] != sha256(out / 'preparation_declaration.json'):
        raise ValueError('Prepared source/input declaration changed')
    prepared_path = out / 'prepared_training_manifest.json'
    if spec['prepared_training_manifest_sha256'] != sha256(prepared_path):
        raise ValueError('Prepared training feature manifest changed')
    parity_path = out / 'training_parity.json'
    if spec['training_parity_sha256'] != sha256(parity_path):
        raise ValueError('Pre-fit training parity receipt changed')
    training_parity = json.loads(parity_path.read_text())
    if (not training_parity.get('passed')
            or training_parity['runner_sha256'] != sha256(Path(__file__))
            or training_parity['objective_losses_sha256'] != sha256(objective_losses.__file__)
            or training_parity['factorized_had_sha256'] != sha256(factorized_had.__file__)
            or training_parity['control_checkpoints'] != spec['training_control']['checkpoints']):
        raise ValueError('Pre-fit training parity no longer matches the fit source')
    if spec['data_sha256'] != preparation['data_sha256']:
        raise ValueError('Objective and preparation input manifests differ')
    if spec['training_control']['declaration_sha256'] != sha256(gr.DEFAULT / 'predeclared.json'):
        raise ValueError('Archived control declaration changed')
    if spec['training_control']['frozen_manifest_sha256'] != sha256(gr.DEFAULT / 'checkpoints_frozen.json'):
        raise ValueError('Archived control checkpoint manifest changed')
    if sha256(ROOT / spec['evaluation_feature_cache']['local_manifest']) != spec[
            'evaluation_feature_cache']['archive_manifest_sha256']:
        raise ValueError('Copied evaluation feature manifest changed')
    return spec


def checkpoint(out: Path, source: str, arm: str, seed: int) -> Path:
    if arm == 'control':
        return _control_checkpoint(source, seed)
    return out / f'without_{source}_{arm}_{seed}.joblib'


def _load_control_model(spec: dict, source: str, seed: int) -> dict:
    entry = spec['training_control']['checkpoints'][f'{source}/{seed}']
    model_path, receipt_path = ROOT / entry['model'], ROOT / entry['receipt']
    if sha256(model_path) != entry['sha256'] or sha256(receipt_path) != entry['receipt_sha256']:
        raise ValueError(f'Frozen matched control changed: {model_path}')
    return ce.load_model(gr.DEFAULT, model_path)


def training_records(out: Path, heldout: str) -> list[dict]:
    spec = check(out)
    return _training_records_for_sources(out, spec['training_sources'][heldout])


def _training_records_for_sources(out: Path, sources: list[str]) -> list[dict]:
    prep.check_preparation(out)
    result = []
    for provider in sources:
        for path in dt.SPLITS[provider]['train']:
            metadata = prep.load_training_cache(path.stem, out)
            cached = {name: dict(joint=record)
                      for name, record in metadata['records'].items()}
            records = gr.paired_records(cached, path.stem, 'slot_dropout')
            # These labels are training-only full-resolution metadata. The four
            # augmented/symmetry records share references rather than copies.
            for record in records:
                record['full_y'] = metadata['full_y']
                record['full_period'] = metadata['full_period']
                record['fps'] = metadata['fps']
            result.extend(records)
            del cached, metadata
    return result


def _window_update_count(records: list[dict], seed: int) -> tuple[int, int]:
    """Reproduce the fixed training sampler's window and optimizer-step counts."""
    samples = [(i, start, size, full, use)
               for i, record in enumerate(records)
               for (start, size, full), use in zip(lt.chunks(record), record['replace_windows'])]
    valid = np.asarray([bool(np.any(record['y'][start:start+size] >= 0))
                        for record in records
                        for (start, size, _), _use in zip(lt.chunks(record), record['replace_windows'])],
                       dtype=bool)
    if len(valid) != len(samples):
        raise ValueError('Chunk and replacement-window counts differ')
    import torch
    from torch.utils.data import DataLoader, Dataset
    valid_tensor = torch.as_tensor(valid)

    class WindowIndices(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return index

    loader = DataLoader(WindowIndices(), batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    updates = 0
    for _ in range(lt.EPOCHS):
        for indices in loader:
            if valid_tensor[indices].any():
                updates += 1
    return len(samples), updates


def validate_training(out: Path = DEFAULT) -> dict:
    """Prove regenerated records preserve control normalization and fit budgets."""
    prep.check_preparation(out)
    prepared = prep.seal_preparation(out)
    if len(prepared['caches']) != 13:
        raise ValueError('Expected 13 sealed training caches before parity validation')
    control_inventory = _control_inventory()
    rows = []
    failures = []
    for heldout in dt.SPLITS:
        sources = old.training_sources(heldout)
        records = _training_records_for_sources(out, sources)
        mixed = [dict(record, x=np.where(record['replace_rows'][:, None],
                                         record['alternate'], record['x']))
                 for record in records]
        median, scale = lt.normalization(mixed)
        del mixed
        columns = factorized_had.player_columns(records[0]['names'])
        probe = factorized_had.network(len(median) * 2, columns, player_only=True)
        parameters = sum(parameter.numel() for parameter in probe.parameters())
        del probe
        counts = {seed: _window_update_count(records, seed) for seed in SEEDS}
        windows = counts[SEEDS[0]][0]
        seed_updates = {seed: counts[seed][1] for seed in SEEDS}
        for seed in SEEDS:
            control = ce.load_model(gr.DEFAULT, _control_checkpoint(heldout, seed))
            checks = dict(
                median=np.array_equal(median, control['median']),
                scale=np.array_equal(scale, control['scale']),
                feature_names=records[0]['names'] == control['feature_names'],
                player_indices=columns == control['player_indices'],
                parameters=parameters == control['parameters'],
                windows_per_epoch=windows == control['windows_per_epoch'],
                optimizer_updates=seed_updates[seed] == control['optimizer_updates'],
            )
            if not all(checks.values()):
                failures.append(dict(heldout=heldout, seed=seed,
                                     failed=[key for key, passed in checks.items() if not passed]))
            rows.append(dict(heldout=heldout, seed=seed, checks=checks,
                             parameters=int(parameters), windows_per_epoch=int(windows),
                             optimizer_updates=int(seed_updates[seed]),
                             normalization_sha256=dict(
                                 median=hashlib.sha256(np.ascontiguousarray(median).tobytes()).hexdigest(),
                                 scale=hashlib.sha256(np.ascontiguousarray(scale).tobytes()).hexdigest())))
        del records, median, scale
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        preparation_declaration_sha256=sha256(out / 'preparation_declaration.json'),
        prepared_training_manifest_sha256=sha256(out / 'prepared_training_manifest.json'),
        objective_losses_sha256=sha256(objective_losses.__file__),
        factorized_had_sha256=sha256(factorized_had.__file__),
        runner_sha256=sha256(Path(__file__)),
        control_checkpoints=control_inventory,
        rows=rows,
        failures=failures,
        passed=not failures,
        fitting_performed=False,
    )
    target = out / 'training_parity.json'
    if target.exists():
        existing = json.loads(target.read_text())
        old_compare, new_compare = dict(existing), dict(result)
        old_compare.pop('created_utc', None)
        new_compare.pop('created_utc', None)
        if old_compare != new_compare:
            raise ValueError('Training parity receipt exists and differs; preserve it and use a new study ID')
        return existing
    if failures:
        raise ValueError(f'Prepared training parity failed: {failures}')
    ce.write_json(target, result, exclusive=True)
    return result


def _evaluation_cache_path(spec: dict, source: str, match: str,
                           view: str, replicate: str) -> Path:
    key = f'{source}/{match}/{view}/{replicate}'
    entry = spec['evaluation_feature_cache']['files'].get(key)
    if entry is None:
        raise ValueError(f'No frozen evaluation features for {key}')
    path = ROOT / entry['path']
    if not path.exists() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError(f'Frozen evaluation feature cache changed: {path}')
    return path


def _load_evaluation_cache(spec: dict, source: str, match: str,
                           view: str, replicate: str) -> dict:
    path = _evaluation_cache_path(spec, source, match, view, replicate)
    cached = joblib.load(path)
    if not {'inputs', 'labels', 'record'} <= cached.keys():
        raise ValueError(f'Incomplete frozen evaluation feature cache: {path}')
    if cached['record'].get('source') != source or cached['record'].get('match') != match:
        raise ValueError(f'Evaluation feature cache has wrong match identity: {path}')
    if (len(cached['labels']) != cached['record'].get('native_n')
            or len(cached['labels']) != len(cached['inputs']['period'])):
        raise ValueError(f'Evaluation feature cache has inconsistent frame counts: {path}')
    return cached


def fit(out: Path, source: str, arm: str, seed: int) -> None:
    spec = check(out)
    if source not in dt.SPLITS or arm not in NEW_ARMS or seed not in SEEDS:
        raise ValueError('Expected one new objective, one provider holdout, and a declared seed')
    destination = checkpoint(out, source, arm, seed)
    if destination.exists():
        ce.load_model(out, destination)
        return
    if (out / 'checkpoints_frozen.json').exists():
        raise ValueError('Model fitting is closed after checkpoint sealing')
    records = training_records(out, source)
    started = time.monotonic()
    objective = OBJECTIVE_BY_ARM[arm]
    fitted = objective_losses.fit(records, player_only=True, seed=seed, objective=objective)
    ce.freeze_model(out, destination, fitted, dict(
        heldout=source, sources=spec['training_sources'][source],
        training_matches=spec['training_matches'][source], arm=arm,
        objective=objective, seed=seed, seconds=time.monotonic() - started,
        parameters=fitted['parameters'], optimizer_updates=fitted['optimizer_updates'],
        windows_per_epoch=fitted['windows_per_epoch']))
    print('FROZEN', source, arm, seed, round(time.monotonic() - started, 1), flush=True)


def seal(out: Path = DEFAULT) -> None:
    spec = check(out)
    rows = []
    for source in dt.SPLITS:
        for seed in SEEDS:
            control = _load_control_model(spec, source, seed)
            for arm in NEW_ARMS:
                path = checkpoint(out, source, arm, seed)
                fitted = ce.load_model(out, path)
                receipt = path.with_suffix('.json')
                provenance = json.loads(receipt.read_text())
                if (provenance.get('heldout'), provenance.get('arm'), provenance.get('seed')) != (
                        source, arm, seed):
                    raise ValueError(f'Unexpected candidate provenance: {receipt}')
                for key in ('parameters', 'optimizer_updates', 'windows_per_epoch'):
                    if fitted[key] != control[key]:
                        raise ValueError(f'{arm}/{source}/{seed} changed training budget: {key}')
                for key in ('median', 'scale'):
                    if not np.array_equal(fitted[key], control[key]):
                        raise ValueError(f'{arm}/{source}/{seed} changed train-only normalization: {key}')
                for key in ('feature_names', 'player_indices', 'player_only'):
                    if fitted[key] != control[key]:
                        raise ValueError(f'{arm}/{source}/{seed} changed model inputs: {key}')
                rows.append(dict(file=path.name, sha256=sha256(path),
                                 receipt_sha256=sha256(receipt)))
    destination = out / 'checkpoints_frozen.json'
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    new_checkpoints=rows,
                    reused_control_checkpoints=spec['training_control']['checkpoints'],
                    total_models=24, new_fits=18)
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing['new_checkpoints'] != rows:
            raise ValueError('Sealed checkpoints changed')
    else:
        ce.write_json(destination, manifest, exclusive=True)


def _transition_diagnostics(labels: np.ndarray, prediction: np.ndarray,
                            period: np.ndarray) -> dict:
    """Measure observed state changes only across adjacent known, continuous frames."""
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    period = np.asarray(period)
    known = np.isin(labels, supervised.LABELS)
    pairs = (known[:-1] & known[1:] & (period[:-1] == period[1:]))
    true_change = labels[:-1] != labels[1:]
    predicted_change = prediction[:-1] != prediction[1:]
    return _binary_counts(true_change[pairs], predicted_change[pairs], int(pairs.sum())) | {
        'known_adjacent_pairs': int(pairs.sum()),
        'turnover': _binary_counts(
            np.isin(labels[:-1][pairs], ('H', 'A'))
            & np.isin(labels[1:][pairs], ('H', 'A'))
            & (labels[:-1][pairs] != labels[1:][pairs]),
            np.isin(prediction[:-1][pairs], ('H', 'A'))
            & np.isin(prediction[1:][pairs], ('H', 'A'))
            & (prediction[:-1][pairs] != prediction[1:][pairs]),
            int(pairs.sum())),
    }


def _binary_counts(actual: np.ndarray, predicted: np.ndarray, population: int) -> dict:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                predicted=int(np.sum(predicted)), reference=int(np.sum(actual)),
                population=int(population))


def _timing_summary(metrics_by_tolerance: dict, fps: float, requested: tuple[float, ...]) -> dict:
    result = {}
    for seconds in requested:
        frames = max(1, round(seconds * fps))
        values = metrics_by_tolerance[frames]
        result[f'{seconds:.1f}s'] = dict(
            requested_tolerance_seconds=float(seconds),
            tolerance_frames=int(frames),
            actual_tolerance_seconds=float(frames / fps),
            precision=float(values['precision']), recall=float(values['recall']),
            f1=float(values['f1']), n_gt=int(values['n_gt']),
            n_pred=int(values['n_pred']), n_matched=int(values['n_matched']),
            n_missed=int(values['n_missed']), n_spurious=int(values['n_spurious']),
            n_ambiguous=int(values['n_ambiguous']),
            mean_latency_seconds=(float(values['mean_latency'] / fps)
                                  if values['mean_latency'] is not None else None),
            mean_abs_latency_seconds=(float(values['mean_abs_latency'] / fps)
                                      if values['mean_abs_latency'] is not None else None),
            p90_abs_latency_seconds=(float(values['p90_abs_latency'] / fps)
                                      if values['p90_abs_latency'] is not None else None),
        )
    return result


def _turnover_timing(labels: np.ndarray, prediction: np.ndarray,
                     inputs: dict) -> dict:
    """Official one-to-one timing for direct H/A changes in known runs."""
    fps = float(inputs['fps'])
    _, truth_codes, predicted_codes, _ = metric_adapter.prepare_tracking_codes(
        labels, prediction, period=inputs['period'], fps=fps,
        timestamps=inputs.get('timestamps'))
    runs = official_metrics.valid_runs(truth_codes)
    predicted_codes = official_metrics.fill_abstentions(predicted_codes, runs)
    true_turnovers = [event for event in official_metrics.transitions(truth_codes, runs)
                      if event[1] in (official_metrics.H, official_metrics.A)
                      and event[2] in (official_metrics.H, official_metrics.A)
                      and event[1] != event[2]]
    predicted_turnovers = [event for event in official_metrics.transitions(predicted_codes, runs)
                           if event[1] in (official_metrics.H, official_metrics.A)
                           and event[2] in (official_metrics.H, official_metrics.A)
                           and event[1] != event[2]]
    output = {}
    for requested in (.5, 1.0):
        tolerance = max(1, round(requested * fps))
        pairs, missed, spurious = official_metrics.match_transitions(
            true_turnovers, predicted_turnovers, tolerance, typed='pair')
        tp, fp, fn = len(pairs), len(spurious), len(missed)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        latency = np.asarray([pred[0] - truth[0] for truth, pred in pairs], dtype=float)
        output[f'{requested:.1f}s'] = dict(
            requested_tolerance_seconds=requested,
            tolerance_frames=tolerance,
            actual_tolerance_seconds=float(tolerance / fps),
            precision=float(precision), recall=float(recall), f1=float(f1),
            n_gt=len(true_turnovers), n_pred=len(predicted_turnovers),
            n_matched=tp, n_missed=fn, n_spurious=fp,
            mean_latency_seconds=float(latency.mean() / fps) if len(latency) else None,
            mean_abs_latency_seconds=float(np.abs(latency).mean() / fps) if len(latency) else None,
            p90_abs_latency_seconds=(float(np.percentile(np.abs(latency), 90) / fps)
                                     if len(latency) else None),
        )
    return output


_STRESS_ROWS = None


def _parity_reference(source: str, match: str, view: str, replicate: str,
                      seed: int) -> dict | None:
    global _STRESS_ROWS
    if _STRESS_ROWS is None:
        report = json.loads((ROOT / 'experiments/results/stress_component_ablation.json').read_text())
        _STRESS_ROWS = report['rows']
    recipe = f'slot_dropout_{seed}'
    return next((row for row in _STRESS_ROWS
                 if (row['source'], row['match'], row['view'], row['replicate'], row['recipe'])
                 == (source, match, view, replicate, recipe)), None)


def _score_row(inputs: dict, labels: np.ndarray, prediction: np.ndarray,
               probability: np.ndarray, source: str, match: str,
               view: str, replicate: str, arm: str, seed: int,
               feature_seconds: float, inference_seconds: float) -> dict:
    analysis = error_analysis.analyze(dict(inputs, labels=labels), prediction)
    diagnostic_metrics = analysis['metrics']
    scored_metrics = metric_adapter.evaluate_tracking(
        labels, prediction, period=inputs['period'], fps=float(inputs['fps']),
        timestamps=inputs.get('timestamps'))
    for key in ('macro_f1', 'spell_f1_mean', 'n_gt_spells', 'n_pred_spells'):
        if not np.isclose(diagnostic_metrics[key], scored_metrics[key], atol=1e-14, rtol=0):
            raise ValueError(f'Post-hoc diagnostic and official metric differ: {key}')
    metrics = scored_metrics
    short = analysis['by_gt_duration'][:2]
    n_short = sum(row['n_gt'] for row in short)
    n_short_matched = sum(row['n_matched_at_iou_0_5'] for row in short)
    false_spells = metrics['n_pred_spells'] - metrics['spell_f1'][.5]['n_matched']
    graded_minutes = np.isin(labels, supervised.LABELS).sum() / float(inputs['fps']) / 60
    transitions = _transition_diagnostics(labels, prediction, supervised.model_periods(inputs))
    timing = _timing_summary(metrics['transitions'], float(inputs['fps']), (.5, 1.0))
    transitions['all_state_timing'] = timing
    transitions['live_dead_timing'] = eval_protocol.summarize_d_boundaries(
        labels, prediction, period=inputs['period'], fps=float(inputs['fps']),
        timestamps=inputs.get('timestamps'))
    transitions['turnover_timing'] = _turnover_timing(labels, prediction, inputs)
    return dict(
        source=source, match=match, view=view, replicate=replicate, arm=arm, seed=seed,
        spell_f1=metrics['spell_f1_mean'], macro_f1=metrics['macro_f1'],
        weighted_70_30=.7 * metrics['spell_f1_mean'] + .3 * metrics['macro_f1'],
        d_f1=metrics['per_class']['D']['f1'], d_precision=metrics['per_class']['D']['precision'],
        d_recall=metrics['per_class']['D']['recall'], per_class=metrics['per_class'],
        spell_by_iou={str(iou): dict(f1=values['f1'], precision=values['precision'],
                                     recall=values['recall'], n_matched=values['n_matched'],
                                     n_spurious=values['n_spurious'], n_missed=values['n_missed'])
                      for iou, values in metrics['spell_f1'].items()},
        short_spell_recall=n_short_matched / n_short if n_short else None,
        short_reference_spells=n_short, short_matched_spells=n_short_matched,
        false_possessions=int(false_spells),
        false_possessions_per_minute=float(false_spells / graded_minutes) if graded_minutes else None,
        fragmented_reference_spells=analysis['overlap_structure']['n_fragmented_gt_spells'],
        n_gt_spells=metrics['n_gt_spells'], n_pred_spells=metrics['n_pred_spells'],
        graded_frames=int(np.isin(labels, supervised.LABELS).sum()),
        transitions=transitions,
        spell_diagnostics=spell_diagnostics.spell_diagnostics(
            labels, prediction, period=inputs['period'], fps=float(inputs['fps']),
            timestamps=inputs.get('timestamps')),
        feature_seconds=feature_seconds, inference_seconds=inference_seconds,
        total_seconds=feature_seconds + inference_seconds,
        probability_sha256=hashlib.sha256(np.ascontiguousarray(probability).tobytes()).hexdigest(),
    )


def _check_control_parity(row: dict, out: Path = DEFAULT,
                          probability: np.ndarray | None = None) -> dict:
    reference = _parity_reference(row['source'], row['match'], row['view'],
                                  row['replicate'], row['seed'])
    if reference is None:
        raise ValueError(f'No archived control score for {row["source"]}/{row["match"]}/'
                         f'{row["view"]}/{row["replicate"]}/{row["seed"]}')
    differences = {}
    for key in ('spell_f1', 'macro_f1', 'd_f1', 'd_precision', 'd_recall',
                'n_pred_spells', 'n_gt_spells'):
        if abs(float(row[key]) - float(reference[key])) > 1e-9:
            differences[key] = (row[key], reference[key])
    for iou, score in row['spell_by_iou'].items():
        archived = reference['spell_thresholds'][iou]
        if abs(float(score['f1']) - float(archived)) > 1e-9:
            differences[f'spell_f1@{iou}'] = (score['f1'], archived)
    if row['view'] == 'clean':
        if probability is None:
            saved_path = out / f'{row["source"]}_{row["match"]}_{row["view"]}_{row["replicate"]}_{row["arm"]}_{row["seed"]}.npz'
            if not saved_path.exists():
                raise ValueError(f'Missing saved control probabilities for parity: {saved_path}')
            with np.load(saved_path) as saved:
                probability = saved['probability']
        baseline_path = gr.DEFAULT / f'{row["source"]}_{row["match"]}_clean_slot_dropout_{row["seed"]}.npz'
        with np.load(baseline_path) as saved:
            if not np.allclose(probability, saved['probability'], atol=1e-6):
                differences['native_probability_atol_1e-6'] = 'mismatch'
    if differences:
        raise ValueError(f'Archived control parity failed: {differences}')
    return dict(source=row['source'], match=row['match'], view=row['view'],
                replicate=row['replicate'], seed=row['seed'], passed=True)


def evaluate(out: Path, source: str) -> None:
    spec = check(out)
    if source not in dt.SPLITS:
        raise ValueError('Expected a declared full-provider holdout')
    if not (out / 'checkpoints_frozen.json').exists():
        raise ValueError('Seal all 18 new candidate models before scoring')
    seal(out)
    models = {(arm, seed): _load_control_model(spec, source, seed)
              for arm in ('control',) for seed in SEEDS}
    models.update({(arm, seed): ce.load_model(out, checkpoint(out, source, arm, seed))
                   for arm in NEW_ARMS for seed in SEEDS})
    results_path = out / f'{source}_results.json'
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    parity = {
        (row['match'], row['view'], row['replicate'], row['seed']): _check_control_parity(row, out)
        for row in rows if row.get('arm') == 'control'
    }

    for match_path in dt.SPLITS[source]['eval']:
        match = match_path.stem
        for view in VIEWS:
            replicates = ('clean',) if view == 'clean' else REPLICATES
            for replicate in replicates:
                cached = _load_evaluation_cache(spec, source, match, view, replicate)
                inputs, labels, record = cached['inputs'], cached['labels'], cached['record']
                feature_seconds = 0.0
                periods = supervised.model_periods(inputs)
                for (arm, seed), fitted in models.items():
                    existing = next((row for row in rows
                                     if (row['match'], row['view'], row['replicate'], row['arm'], row['seed'])
                                     == (match, view, replicate, arm, seed)), None)
                    if existing is not None:
                        if arm == 'control':
                            key = (match, view, replicate, seed)
                            parity[key] = _check_control_parity(existing, out)
                        continue
                    started = time.monotonic()
                    sampled = factorized_had.predict(record, fitted)
                    probability = np.column_stack([
                        lt.upsample(sampled[:, index], record, periods) for index in range(3)])
                    prediction = supervised.decode(probability, periods, float(inputs['fps']), lt.COST)
                    row = _score_row(inputs, labels, prediction, probability, source, match,
                                     view, replicate, arm, seed, feature_seconds,
                                     time.monotonic() - started)
                    if arm == 'control':
                        key = (match, view, replicate, seed)
                        parity[key] = _check_control_parity(row, out, probability)
                    np.savez_compressed(out / f'{source}_{match}_{view}_{replicate}_{arm}_{seed}.npz',
                                        prediction=prediction, probability=probability)
                    rows.append(row)
                    ce.write_json(results_path, rows)
                    print('SCORED', source, match, view, replicate, arm, seed,
                          round(row['spell_f1'], 5), flush=True)

    expected_parity = sum(len(dt.SPLITS[source]['eval']) * (1 + len(REPLICATES)) for source in [source])
    expected_parity *= len(SEEDS)
    if len(parity) != expected_parity:
        raise ValueError('Control score parity record incomplete')
    ce.write_json(out / f'{source}_control_parity.json', list(parity.values()))


def _aggregate(rows: list[dict]) -> list[dict]:
    """Average stress replicates within match, matches within provider, providers equally."""
    metric_keys = ('spell_f1', 'macro_f1', 'weighted_70_30', 'd_f1', 'd_precision',
                   'd_recall', 'short_spell_recall', 'false_possessions_per_minute',
                   'n_gt_spells', 'n_pred_spells')
    iou_keys = ('0.3', '0.4', '0.5', '0.6', '0.7')

    def mean_defined(values):
        selected = [float(value) for value in values if value is not None]
        return float(np.mean(selected)) if selected else None

    summaries = []
    for arm in ARMS:
        for seed in SEEDS:
            for view in VIEWS:
                by_provider = {}
                for source in dt.SPLITS:
                    match_rows = []
                    for path in dt.SPLITS[source]['eval']:
                        cells = [row for row in rows if (row['source'], row['match'], row['view'],
                                                          row['arm'], row['seed'])
                                 == (source, path.stem, view, arm, seed)]
                        expected_replicates = 1 if view == 'clean' else 2
                        if len(cells) != expected_replicates:
                            raise ValueError(f'Incomplete match cells: {source}/{path.stem}/{view}/{arm}/{seed}')
                        match_rows.append({
                            **{key: mean_defined([row[key] for row in cells]) for key in metric_keys},
                            'spell_by_iou': {
                                iou: mean_defined([row['spell_by_iou'][iou]['f1'] for row in cells])
                                for iou in iou_keys
                            },
                        })
                    by_provider[source] = {
                        key: mean_defined([row[key] for row in match_rows])
                        for key in metric_keys
                    }
                    by_provider[source]['spell_by_iou'] = {
                        iou: mean_defined([row['spell_by_iou'][iou] for row in match_rows])
                        for iou in iou_keys
                    }
                    # Preserve provider and match counts for diagnostics.
                    by_provider[source]['matches'] = len(match_rows)
                summary = {key: mean_defined([by_provider[s][key] for s in dt.SPLITS])
                           for key in metric_keys}
                summary['spell_by_iou'] = {
                    iou: mean_defined([by_provider[source]['spell_by_iou'][iou]
                                       for source in dt.SPLITS])
                    for iou in iou_keys
                }
                summaries.append(dict(arm=arm, seed=seed, view=view,
                                     per_provider=by_provider, **summary))
    return summaries


def report(out: Path = DEFAULT) -> dict:
    spec = check(out)
    seal(out)
    rows = [row for source in dt.SPLITS
            for row in json.loads((out / f'{source}_results.json').read_text())]
    expected = {
        (source, path.stem, view, replicate, arm, seed)
        for source in dt.SPLITS for path in dt.SPLITS[source]['eval']
        for view in VIEWS for replicate in (('clean',) if view == 'clean' else REPLICATES)
        for arm in ARMS for seed in SEEDS
    }
    observed = {(row['source'], row['match'], row['view'], row['replicate'], row['arm'], row['seed'])
                for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f'Incomplete or duplicate evaluation cells: expected {len(expected)}, got {len(rows)}')
    for source in dt.SPLITS:
        for path in dt.SPLITS[source]['eval']:
            for view in VIEWS:
                replicates = ('clean',) if view == 'clean' else REPLICATES
                for replicate in replicates:
                    cells = [row for row in rows
                             if (row['source'], row['match'], row['view'], row['replicate'])
                             == (source, path.stem, view, replicate)]
                    populations = {(row['graded_frames'], row['n_gt_spells']) for row in cells}
                    if len(populations) != 1:
                        raise ValueError(f'Grading population changed across arms: {source}/{path.stem}/{view}/{replicate}')
    parity = [json.loads((out / f'{source}_control_parity.json').read_text()) for source in dt.SPLITS]
    expected_parity = {(row['source'], row['match'], row['view'], row['replicate'], row['seed'])
                       for row in rows if row['arm'] == 'control'}
    observed_parity = {(item['source'], item['match'], item['view'], item['replicate'], item['seed'])
                       for cells in parity for item in cells}
    if observed_parity != expected_parity or any(
            not item['passed'] for cells in parity for item in cells):
        raise ValueError('Control parity checks are incomplete or failed')

    summaries = _aggregate(rows)
    controls = {(row['seed'], row['view']): row for row in summaries if row['arm'] == 'control'}
    screens = []
    for arm in NEW_ARMS:
        for seed in SEEDS:
            clean = next(row for row in summaries
                         if (row['arm'], row['seed'], row['view']) == (arm, seed, 'clean'))
            stress = next(row for row in summaries
                          if (row['arm'], row['seed'], row['view']) == (arm, seed, 'feed_like'))
            base_clean = controls[(seed, 'clean')]
            base_stress = controls[(seed, 'feed_like')]
            worst_provider = min(
                clean['per_provider'][source][key]
                - base_clean['per_provider'][source][key]
                for source in dt.SPLITS for key in ('d_f1', 'macro_f1'))
            delta_clean = clean['spell_f1'] - base_clean['spell_f1']
            delta_stress = stress['spell_f1'] - base_stress['spell_f1']
            screens.append(dict(
                arm=arm, seed=seed,
                clean_spell_delta=delta_clean,
                feed_like_spell_delta=delta_stress,
                worst_provider_clean_d_or_macro_delta=worst_provider,
                passes_positive_screen=delta_clean >= .01,
                passes_guardrails=delta_stress >= -.01 and worst_provider >= -.02,
                passes_all=delta_clean >= .01 and delta_stress >= -.01 and worst_provider >= -.02,
            ))

    result = dict(
        declaration=spec,
        declaration_sha256=sha256(out / 'predeclared.json'),
        rows=rows,
        summaries=summaries,
        screens=screens,
        control_parity=parity,
        checkpoint_manifest_sha256=sha256(out / 'checkpoints_frozen.json'),
        baseline_changed=False,
        organizer_labels_read=False,
        fresh_confirmation=False,
    )
    ce.write_json(out / 'report.json', result, exclusive=True)
    ce.write_json(RESULT, result, exclusive=True)
    for row in summaries:
        print('SUMMARY', row['arm'], row['seed'], row['view'],
              *(round(row[key], 5) for key in ('spell_f1', 'macro_f1', 'd_f1')), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze-preparation', 'prepare', 'seal-preparation',
                                           'validate-training', 'freeze', 'fit', 'seal',
                                           'evaluate', 'report'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--source', choices=tuple(dt.SPLITS))
    parser.add_argument('--match')
    parser.add_argument('--arm', choices=NEW_ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if args.action == 'freeze-preparation':
            prep.freeze_preparation(args.out)
        elif args.action == 'prepare':
            if args.match:
                prep.prepare_match(args.match, args.out)
            else:
                for source in ((args.source,) if args.source else tuple(dt.SPLITS)):
                    for path in dt.SPLITS[source]['train']:
                        prep.prepare_match(path.stem, args.out)
        elif args.action == 'seal-preparation':
            prep.seal_preparation(args.out)
        elif args.action == 'validate-training':
            validate_training(args.out)
        elif args.action == 'freeze':
            freeze(args.out)
        elif args.action == 'fit':
            if args.source is None or args.arm is None or args.seed is None:
                parser.error('fit requires --source, --arm, and --seed')
            fit(args.out, args.source, args.arm, args.seed)
        elif args.action == 'seal':
            seal(args.out)
        elif args.action == 'evaluate':
            if args.source is None:
                parser.error('evaluate requires --source')
            evaluate(args.out, args.source)
        else:
            report(args.out)


if __name__ == '__main__':
    main()
