# Fixed public diagnostic for the short-possession diversity candidate

Study `short_quality_public_v1`, 1 November 2026. In progress. The adopted
package is unchanged.

The user explicitly requested this public check after the completed
[external quality/diversity comparison](SHORT_QUALITY_DIVERSITY_V1.md).
The fixed candidate is `original_full`, selected before this request on the
original external feed-like benchmark. Its clean/feed-like spell F1 is
.662590/.528180 versus the adopted calibrated baseline's .658625/.518527.
The previous research leader's .527396 is a separate comparator; the adopted
baseline remains .518527. All feed-first guards pass; the historical +.01 clean
screen fails. The gain on the exposed development cohort is Sportec-driven.

## Frozen construction

The existing research models exclude entire providers. For this diagnostic,
fit exactly two pooled models with seeds 2026/2027 on the same thirteen original
TRAIN matches and 45 GAR TRAIN games. Use the frozen `OriginalDiversitySampler`,
labels, raw-feed views, losses, width96, batch32, half-original/half-GAR mixture,
2,048 updates, H/A learning rate .0003 and D learning rate .0001. Start from
the exact pooled combined-objective teachers already paired with the adopted
dense package; retain their frozen normalization. No seed or checkpoint search.

Average the two native full H/A/D probability vectors equally, then apply the
unchanged temperature-4 D-logit transform and cost-3 decoder. No component
crosses, blends or individual-seed public scores are requested. Generate all
199,319 `frame,label` rows and mark input-derived nonplaying periods X.
Inference uses tracking and unknown placeholders only. A byte-identical replay
of the adopted full CSV validates the complete input/feature/frame/decoder path
before public reference access.

## Single-use public gate

The [declaration](results/short_quality_public_v1_declaration.json) binds sources,
training data, original tracking inputs, teachers, baseline scores and the
already-known reference hash. It does not read reference labels. Seal both
new checkpoints, native probabilities, the format-valid full CSV and the
adopted replay before scoring.

Claim a distinct exclusive marker **before** reading the public reference,
including a read that subsequently fails. Validate the expected 7,500 graded
frames and 17 reference possessions, then run the official evaluator once
with 1,000 bootstrap draws. Reuse the adopted model's saved public scores;
do not rescore it. If the candidate CSV is byte-identical to the adopted
scored CSV, reuse its saved score without new reference I/O.

All prior gates remain closed. No public-driven tuning, repairs, reselection,
repeat scoring or automatic promotion follows. The five-minute window is
previously exposed diagnostic evidence, not independent or full-match accuracy.

## Verification

All 550 experiment contracts pass, including five synthetic checks of exclusive
writes, permanent closure after failed reference I/O, fit/inference rejection
after closure, artifact-tamper rejection before opening the gate, and cached
result reuse. The separate saved-artifact verifier checks model budgets and
normalizers, exact native probability averaging/decoding/CSV alignment, and
saved-score deltas. It never rereads public ground truth or recomputes its F1.

```sh
.venv/bin/python -m experiments.short_quality_public_v1 declare
.venv/bin/python -m experiments.short_quality_public_v1 fit --seed 2026
.venv/bin/python -m experiments.short_quality_public_v1 fit --seed 2027
.venv/bin/python -m experiments.short_quality_public_v1 prepare
.venv/bin/python -m experiments.short_quality_public_v1 score
.venv/bin/python scripts/verify_short_quality_public_artifacts_v1.py
```

Restore completed artifacts instead of repeating a closed fit or scoring action.
Weights, native arrays and frame outputs remain in ignored/private storage.
