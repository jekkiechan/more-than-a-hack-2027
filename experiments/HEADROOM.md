# Validation headroom: team attribution is the current bottleneck

These diagnostics use **only J03WOY**, the original validation match, and the
previously frozen robust model trained on J03WMX, J03WN1, and J03WOH. They deliberately
use validation reference labels as oracles for analysis. They do not produce
deployable predictions, train a model, select a model, or read held-out labels.
The machine-readable aggregates are in
[`results/headroom_validation.json`](results/headroom_validation.json).

## What an oracle would fix

| Validation scenario | Frame macro F1 | Possession-spell F1 |
| --- | ---: | ---: |
| Frozen robust baseline | 0.9188 | 0.6108 |
| Oracle-perfect dead/live status; frozen team predictions | 0.9196 | 0.6090 |
| Oracle-perfect H/A team assignment; retain predicted dead/live decisions | 0.9991 | 0.9958 |
| Oracle-perfect labels throughout | 1.0000 | 1.0000 |

Frame macro F1 averages H/A/D class F1. Possession-spell F1 is the unchanged
competition average over temporal IoU thresholds 0.3, 0.4, 0.5, 0.6, and 0.7.

The first oracle corrects every dead/live error, preserving decoded H/A labels.
When the model originally predicted D on a true-live frame, its frozen
conditional H/A probabilities supply the team. The second oracle corrects H/A
confusion only where both prediction and reference are live, retaining every D
decision and every dead/live error. The scenarios therefore answer different,
explicit questions; their improvements need not add up.

Of 9,321 incorrectly labelled validation frames, 9,194 (98.6%) are H/A confusion.
Only 58 true-live frames were called dead and 69 true-dead frames were called
live. The model predicts 360 possessions against 456 reference possessions.
Correcting dead/live status alone slightly reduces spell F1 because fixing
boundaries can split or reshape intervals without fixing the team assignments.

This near-perfect dead/live performance relies on Sportec's frozen ball
coordinates during stoppages, described in [`BENCHMARK_AUDIT.md`](BENCHMARK_AUDIT.md).
It is not evidence that dead-ball detection is solved on the challenge feed.
The oracle team result shows where mistakes occur; it does not show that another
model can recover those team labels from available tracking.

## Timing resolution is not the observed ceiling

We sample the true 25 Hz labels at a fixed 5 Hz grid and expand them back to 25 Hz.
All five possible grid offsets are evaluated, separately within each half.
Reference labels are used only to construct these oracle scenarios.

| Oracle reconstruction | Frame macro F1 range | Spell F1 range |
| --- | ---: | ---: |
| Hold each sampled label until the next sample | 0.9907–0.9913 | 0.9745–0.9873 |
| Assign each frame its nearest sampled label | 0.9943–0.9947 | 0.9908–0.9943 |

These are diagnostic sampling losses, not information-theoretic bounds. An
offline model could estimate a transition between samples; a sampled label may
also omit a very short possession. Nevertheless, the gap from roughly 0.61 spell
F1 is far larger than this 5 Hz timing effect. Increasing the output rate alone
cannot plausibly resolve the current errors.

## Short possessions matter, but do not explain the whole gap

There are 456 reference possessions, with median duration 3.6 seconds. Spells
shorter than one second account for 41 possessions (9.0%) but only 0.79% of live
frames. The frozen model's recall at IoU 0.5 is:

| Reference duration | Possessions | Matched possessions | Recall |
| --- | ---: | ---: | ---: |
| Under 0.2 s | 4 | 1 | 0.250 |
| 0.2–0.5 s | 13 | 3 | 0.231 |
| 0.5–1 s | 24 | 8 | 0.333 |
| 1–2 s | 87 | 19 | 0.218 |
| 2–5 s | 163 | 78 | 0.479 |
| At least 5 s | 165 | 141 | 0.855 |

As a second oracle sensitivity check, start with perfect reference predictions
and deliberately remove short H/A runs. Repeatedly replace the shortest run
below the threshold with its longer adjacent graded run, preferring the previous
run on a tie. D runs are never removed; no operation crosses a half, time gap, or
unknown reference interval. Scores remain against the original, unmodified
reference labels.

| Remove reference possessions shorter than | Frame macro F1 | Spell F1 |
| --- | ---: | ---: |
| 0.2 s | 0.9999 | 0.9914 |
| 0.5 s | 0.9991 | 0.9719 |
| 1.0 s | 0.9953 | 0.9273 |

Deleting short possessions costs little frame F1 but appreciably harms spell F1.
Even deleting every subsecond possession under this rule leaves approximately
0.93 spell F1 when everything else is perfect. The much larger observed deficit
also involves possessions lasting several seconds. These measurements do not
establish that short spells are physically ambiguous, annotation errors, or
unrecoverable; that would require independent annotation and further evidence.

## A defensible stopping and final-test protocol

Further experiments should use the original three training matches and J03WOY
validation only, with a bounded, declared search and a fixed selection rule.
Keep these oracle arrays out of training, parameter selection, and production;
the script saves only aggregate diagnostics. A sequence of unsuccessful bounded
experiments can establish an **empirical plateau for those approaches and this
validation set**, not a proven global ceiling for possession inference.

J03WPY has already been used for the original frozen benchmark. New refinements
must not be selected on its scores. J03WR9 has also been used for the separate
PathCRF comparison. J03WQQ remains unused by our custom model development and can
serve as a final new-match check for a custom model frozen before its labels are
opened. Record code/model/configuration/data hashes first and score once.

**J03WQQ is the supplied PathCRF checkpoint's validation match.** It is therefore
not an independent final test for pretrained PathCRF, a PathCRF hybrid, or a
comparison claiming all methods are independent of that match. Such a comparison
requires another dataset or an explicitly documented new model-training split.
One additional Sportec match still cannot establish performance on a different
tracking provider, real broadcast occlusion, or the ungraded competition match.

Reproduce from the repository root:

```sh
python -m experiments.headroom
```

The script checks the frozen model hash, original training IDs, validation-data
hash, and inference/metric source hashes before computing any scenario. Its only
data-file path is the fixed validation match; no held-out match path is constructed.
