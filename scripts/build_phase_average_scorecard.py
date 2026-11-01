"""Collect saved aggregate metrics only. Never imports a scorer or reads native labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def load(name):
    return json.loads((RESULTS/name).read_text())


def build():
    sources = ['feed_like_v2_scorecard.json', 'feed_benchmark_audit_v1.json',
        'feed_like_v2_benchmark.json', 'sportec_extra_eval_v1.json', 'feed_like_v2_public_result.json']
    cross = load(sources[0])['arms']['cand_phase_tta']
    audit = load(sources[1])['tta']['cand_phase_tta']
    v2 = load(sources[2])['arms']['cand_phase_tta']
    sportec = load(sources[3])
    public = load(sources[4])
    scorecard = dict(model='original_full pooled two-seed + equal sampling-phase probability mean',
        adoption_status='Experimental upgrade candidate; current adopted package has phase averaging disabled',
        averaging='5 phases at 25 Hz; unchanged f062 D-logit temperature4/cost3 decoder',
        original_five_match_aggregates=cross,
        original_provider_spell={v: audit['spell'][k]['providers'] for v,k in [('clean','clean'),('v1','feed_like')]},
        original_provider_macro={v: audit['macro'][k]['providers'] for v,k in [('clean','clean'),('v1','feed_like')]},
        v2_provider_spell=v2['providers'], v2_spell_per_realization=v2['per_replicate'],
        extra_sportec={v: sportec['scores'][f'cand__phase_tta__{v}'] for v in ('clean','v1','feed_like_v2')},
        public_diagnostic=public['candidate'], public_delta_vs_original_full=public['delta_vs_original_full'],
        public_csv_sha256=public['candidate_csv_sha256'],
        missing_metrics='Original five-match phase-average D metrics, per-seed metrics and per-IoU breakdowns were not published in the saved aggregate phase scorecards. Do not invent or infer them.',
        evidence_scope='Exposed development plus synthetic stress; public 7500 graded frames/17 reference spells diagnostic only',
        source_sha256={str((RESULTS/f).relative_to(ROOT)): hashlib.sha256((RESULTS/f).read_bytes()).hexdigest() for f in sources},
        fitting=False, public_rescoring=False, promotion=False)
    path = RESULTS/'original_full_phase_average_scorecard_v1.json'
    with path.open('x') as f:
        json.dump(scorecard, f, indent=2, sort_keys=True)
        f.write('\n')
    return scorecard


if __name__ == '__main__':
    build()
