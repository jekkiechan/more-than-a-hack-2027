"""One user-requested diagnostic of the already adopted tactical package."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results'
STUDY = 'tactical_public_v1'
CSV = ROOT / 'data/experiments/tactical_public_v1/replay.csv'
MANIFEST = ROOT / 'experiments/baseline_manifest.json'
INPUT = ROOT / 'data/challenge/inputs.npz'
BASELINE = RESULTS / 'ball_exit_reconstructed_v1_public.json'
EXPECTED = '8e26f823a3e933f1feca2a64dfe6d7f728a4741548c8c7cd95d51746b6a355b9'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(name, value):
    with (RESULTS / f'{STUDY}_{name}.json').open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def bindings():
    manifest = json.loads(MANIFEST.read_text())
    directory = ROOT / manifest['model_directory']
    for name, expected in manifest['models_sha256'].items():
        if sha(directory / name) != expected:
            raise ValueError(f'Model mismatch: {name}')
    files = [MANIFEST, ROOT / 'experiments/model_decision_policy.json', INPUT, CSV, BASELINE]
    files += list((ROOT / 'experiments').rglob('*.py'))
    files += list((ROOT / 'evaluation').glob('*.py'))
    return dict(files={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                models=manifest['models_sha256'])


def seal():
    # The standard CLI validation is also run separately before this seal.
    if sha(CSV) != EXPECTED:
        raise ValueError('Default inference differs from the pre-public sealed candidate')
    if sha(INPUT) != '7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3':
        raise ValueError('Challenge inputs changed')
    write('declaration', dict(study=STUDY, utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested promotion, public score and site update on 2026-11-16',
        candidate='Already promoted tactical H/A expert from merged PR #98',
        budget='One score of the fixed candidate; no fitting, tuning, search or repeat scoring',
        public_population=dict(frames=199319, graded=7500, spells=17), bootstrap_n=1000,
        adoption='Already fixed before public reference access; retain historical guard failures',
        reference_sha256='e596a9af9626c623bb94be6fd350e12ee9a75c76a696cfadca3bc01735e92487',
        expected_csv_sha256=EXPECTED, reference_access=False, bindings=bindings()))


def score():
    declaration = RESULTS / f'{STUDY}_declaration.json'
    d = json.loads(declaration.read_text())
    if d['bindings'] != bindings() or sha(CSV) != EXPECTED:
        raise ValueError('Sealed package changed')
    if (RESULTS / f'{STUDY}_public.json').exists():
        raise ValueError('Result exists; do not repeat scoring')
    write('gate', dict(study=STUDY, claimed_utc=datetime.now(timezone.utc).isoformat(),
        declaration_sha256=sha(declaration), authorization=d['authorization']))
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    reference = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    if sha(reference) != d['reference_sha256']:
        raise ValueError('Reference mismatch; gate stays closed')
    raw = pm.evaluate(pm.label_codes(pm.read_label_csv(reference)),
                      pm.label_codes(pm.read_label_csv(CSV)), bootstrap_n=d['bootstrap_n'])
    if (raw['n_frames'], raw['n_graded'], raw['n_gt_spells']) != (199319, 7500, 17):
        raise ValueError('Unexpected public population; gate stays closed')
    candidate = _score_summary(raw)
    baseline = json.loads(BASELINE.read_text())['candidate']
    result = dict(study=STUDY, candidate=candidate, saved_baseline=baseline,
        delta={k: candidate[k]-baseline[k] for k in ('spell_f1_mean', 'macro_f1', 'd_f1')},
        declaration_sha256=sha(declaration), gate_sha256=sha(RESULTS / f'{STUDY}_gate.json'),
        adoption_precedes_public=True, selection_or_tuning=False,
        interpretation='Repeated 7500-frame/17-possession pipeline diagnostic; no continuous-match gold claim')
    write('public', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    {'seal': seal, 'score': score}[sys.argv[1]]()
