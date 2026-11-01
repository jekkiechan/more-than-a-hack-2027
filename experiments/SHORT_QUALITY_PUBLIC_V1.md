# Fixed public diagnostic for the short-possession diversity candidate

Study `short_quality_public_v1`, 1 November 2026. Completed. The adopted
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
199,319 `frame,label` rows and mark frames outside the defined halves X.
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

## Results

| Protocol / metric | Adopted baseline | Fixed candidate | Delta |
| --- | ---: | ---: | ---: |
| Public spell F1 | 0.630303 | 0.611765 | -0.018538 |
| Public macro F1 | 0.864181 | 0.851742 | -0.012439 |
| Public D-F1 | 0.873502 | 0.862420 | -0.011082 |
| Public D precision | 0.854723 | 0.810293 | -0.044430 |
| Public D recall | 0.893125 | 0.921715 | +0.028591 |

| External development spell F1 | Adopted baseline | Fixed candidate |
| --- | ---: | ---: |
| Clean | .658625 | .662590 |
| Original feed-like | .518527 | .528180 |

The external clean/feed-like gains are +.003966/+.009652 against the adopted model. They remain the frozen external results; the public check does not retune or reselect that candidate. Public spell falls **.018538** and macro falls **.012439**. The active manifest/policy and all previous gates remain unchanged.

### Public spell geometry

| IoU | Baseline matched / 17 | Candidate matched / 17 | Baseline F1 | Candidate F1 |
| --- | ---: | ---: | ---: | ---: |
| 0.3 | 12 | 12 | 0.727273 | 0.705882 |
| 0.4 | 12 | 12 | 0.727273 | 0.705882 |
| 0.5 | 12 | 12 | 0.727273 | 0.705882 |
| 0.6 | 9 | 10 | 0.545455 | 0.588235 |
| 0.7 | 7 | 6 | 0.424242 | 0.352941 |

The candidate predicts 17 possessions versus the baseline’s 16 and 17 references. Both match 12 references at IoU .5; the candidate has one extra unmatched prediction. At stricter thresholds the pattern is mixed: +1 match at .6 and −1 at .7. These aggregate counts do not identify a justified frame correction. D recall rises while D precision and D-F1 fall.

The candidate’s descriptive spell bootstrap interval is [.447059, .764706], versus [.460606, .787879] for the saved baseline. These are separate intervals, not a paired significance test. Seventeen previously exposed public possessions cannot establish full-match performance.

## Completed artifact checks

Both pooled fits completed 2,048 updates, drawing 32,768 original and 32,768 GAR windows per seed. Every original TRAIN match and all 45 GAR games remain represented. The complete adopted CSV replay matches its saved hash exactly; the new full CSV passes the 199,319-row format validator. The models, source, input and predictions were sealed before the gate opened. One unique CSV was scored once.

[Public result](results/short_quality_public_v1_result.json) and [reference-free artifact verification](results/short_quality_public_v1_artifact_verification.json). The verifier checks native means, decoding, CSV, training budgets, teacher normalization and saved deltas without public reference access or metric recomputation.

The public gate is closed permanently. No new settings, seeds, repairs or organizer scores follow this diagnostic. No model promotion is performed; the adopted calibrated package and the external numerical research leader remain distinct.
