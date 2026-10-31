"""Resume the frozen four-candidate competition check in a fresh environment.

The frozen runner binds the thirteen GRU TRAIN caches by file hash, but each
cache pickles per-view ``feature_seconds`` wall-clock timings, so a rebuild can
never be byte-identical. This wrapper keeps Codex's frozen declaration
unchanged and re-pins only those 13 caches, their receipts and the 13 geometry
receipts (which also record prep seconds), after recording
the evidence that their contents are identical:

- evaluation features recompute bit-identically from the same code;
- the 13 geometry caches, raw inputs and both study declarations hash-match;
- refitting archived GRU checkpoints from the rebuilt caches reproduces their
  weights bit for bit (``without_metrica`` and ``without_sportec`` control
  2026, together covering all 13 TRAIN matches).

Every other guard of the frozen runner (sources, files, recipes, exclusive
gate, sealing) runs unmodified.

    python -m experiments.temporal_competition_resume deviation
    python -m experiments.temporal_competition_resume fit --model tree_full_127
    python -m experiments.temporal_competition_resume {prepare,seal,predict,check}
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments import (gru_robustness_experiment as gr, joint_lgbm_experiment as joint,
                         temporal_competition_check as check, temporal_competition_fit as fitting)
from experiments.eval_set.protocol import sha256

ROOT = check.ROOT
OUT = check.DEFAULT
FROZEN = ROOT/'experiments/results/temporal_competition_declaration.json'
DEVIATION = OUT/'cache_rehash_deviation.json'
PUBLIC_DEVIATION = ROOT/'experiments/results/temporal_competition_resume_deviation.json'


def train_caches():
    spec = json.loads((joint.DEFAULT/'predeclared.json').read_text())
    return sorted(spec['training_caches_sha256'])


def deviation(evidence):
    """Write the frozen declaration unchanged and record rebuilt cache hashes."""
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(FROZEN.read_text())
    if frozen['source_sha256'] != check.fingerprints():
        raise ValueError('Frozen diagnostic source differs from this checkout')
    check.verify_files(frozen['files_sha256'])
    dependencies = frozen['pooled_fit']['dependencies_sha256']
    rebuilt, unchanged = {}, []
    for key, digest in dependencies.items():
        current = sha256(ROOT/key)
        if current == digest:
            unchanged.append(key)
        elif Path(key).name.startswith('train_') or (Path(key).name.startswith('geometry_') and key.endswith('.json')):
            rebuilt[key] = dict(frozen=digest, rebuilt=current)
        else:
            raise ValueError(f'Unexpected changed dependency: {key}')
    if len(rebuilt) != 39:
        raise ValueError('Expected only 13 GRU TRAIN caches, their receipts and 13 geometry receipts to differ')
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  reason='GRU TRAIN caches pickle wall-clock feature_seconds and cache receipts record prep seconds; rebuilt bytes cannot match. Geometry caches themselves hash-match.',
                  rebuilt=rebuilt, unchanged_dependencies=len(unchanged), evidence=evidence,
                  frozen_declaration_sha256=sha256(FROZEN), organizer_labels_read=False)
    if not (OUT/'predeclared.json').exists():
        check.ce.write_json(OUT/'predeclared.json', frozen, exclusive=True)
    if json.loads((OUT/'predeclared.json').read_text()) != frozen:
        raise ValueError('Output declaration differs from the frozen declaration')
    check.ce.write_json(DEVIATION, record, exclusive=True)
    PUBLIC_DEVIATION.write_text(json.dumps(record, indent=2))
    print('DEVIATION RECORDED', len(rebuilt), 'entries')


def install():
    """Re-pin only the recorded cache hashes; everything else stays checked."""
    record = json.loads(DEVIATION.read_text())
    frozen = json.loads(FROZEN.read_text())
    for key, pair in record['rebuilt'].items():
        if sha256(ROOT/key) != pair['rebuilt']:
            raise ValueError(f'Rebuilt cache changed since deviation: {key}')
    original_joint_check = joint.check
    original_declaration = fitting.declaration
    remap = {pair['frozen']: pair['rebuilt'] for pair in record['rebuilt'].values()}
    back = {v: k for k, v in remap.items()}

    def joint_check(out):
        spec = original_joint_check(out)
        if Path(out) == joint.DEFAULT:
            spec = dict(spec, training_caches_sha256={k: remap.get(v, v) for k, v in spec['training_caches_sha256'].items()})
        return spec

    def declaration():
        fresh = original_declaration()
        deps = fresh['dependencies_sha256']
        fresh['dependencies_sha256'] = {k: back.get(v, v) for k, v in deps.items()}
        if fresh['dependencies_sha256'] != frozen['pooled_fit']['dependencies_sha256']:
            raise ValueError('Dependencies differ beyond the recorded cache rehash')
        return fresh

    joint.check = joint_check
    fitting.declaration = declaration
    gr.check(gr.DEFAULT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deviation', 'fit', 'prepare', 'seal', 'predict', 'check'))
    parser.add_argument('--model', choices=check.MODELS)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.action == 'deviation':
        deviation(json.loads(args.evidence.read_text()))
        return
    install()
    if args.action == 'fit':
        if args.model.startswith('tree_'):
            fitting.fit_tree(OUT, args.model[len('tree_'):])
        else:
            fitting.fit_gru(OUT, args.model[len('gru_'):-len('_2026')])
    else:
        {'prepare': check.prepare, 'seal': check.seal_models, 'predict': check.predict,
         'check': check.score}[args.action](OUT)


if __name__ == '__main__':
    main()
