# Bidirectional player context for LightGBM and GRU fusion

Status: frozen and running; external results not yet available. Date: 2026-10-27.

This study tests whether a LightGBM dead-ball head can complement the latest
player-only D joint GRU. It preserves that GRU's conditional H/A attribution and
the decoder cost of 3. A matched feature ablation separates the value of
before/after features from changing tree regularization. No organizer labels
are read, no existing public gate is reopened, and the adopted package is not
changed.

## Frozen design

Four D heads use identical whole-match training assignments, clean and legacy
player-slot-dropout views, binary targets, and equal total weight per provider,
match and view. Native unknown targets and SkillCorner visibility grading stay
unchanged. The held-out provider is excluded from both this D head and every
learned component of its companion GRU. The five scoring matches are reused
external development data, including silver targets; this is not independent
confirmation.

Training samples every second row of the five-Hz feature grid, producing
2.5-Hz fitting rows. Temporal context is built before this thinning. Evaluation
predicts at five Hz and interpolates within continuous blocks to native frames.
All tree fits use seed 2026 and one CPU thread. Tree seed variance is not
measured. The GRU controls use both fixed seeds 2026 and 2027; averaging their
reported scores does not create an inference ensemble.

| Configuration | Features | Trees and rate | Capacity | Regularization and sampling |
| --- | --- | --- | --- | --- |
| raw_slow | Original 67 | 640 at .02 | 31 leaves, minimum 40 rows | L2 1, all rows and columns, 255 bins |
| context_slow | Expanded before/after | 640 at .02 | Identical to raw_slow | Identical to raw_slow |
| context_shallow | Expanded before/after | 480 at .03 | 15 leaves, depth 5, minimum 160 rows, Hessian minimum .01 | L1 .1, L2 5, rows .8, columns .8, 127 bins |
| context_regularized | Expanded before/after | 640 at .02 | 31 leaves, depth 7, minimum 120 rows, Hessian minimum .01 | L1 .2, L2 10, rows .8, columns .7, 127 bins |

The subsampled arms set `subsample_freq=1`; setting a row fraction alone would
not enable LightGBM bagging. Larger leaves in terms of observations, restricted
depth, L1/L2 penalties and feature sampling address the redundancy among nearby
context columns. All configurations use fixed boosting budgets without early
stopping, because a frame-loss checkpoint need not maximize decoded spell F1.
No class balancing changes the D prior or probability calibration implicitly.

## Temporal feature engineering

The original 67 player-only motion and shape features remain. Only the 18
primitive columns are expanded; already centered rolling means are not
recursively expanded. Each primitive adds an observation flag and, at one,
three and five seconds:

- The earlier value, later value, and later-minus-earlier difference.
- Separate past and future means, their difference, and standard deviations.
- The fraction of the requested past and future windows that is observed.

The two half windows exclude the target sample. Features distinguish slowing
from speeding up and sustained stillness from absent motion measurements.
Coverage uses the requested window width even at block edges, so incomplete
context is explicit. A missing half-window mean stays NaN; it is never replaced
by zero speed. Lags and summaries stop at match, half and timestamp breaks.
Nothing in the feature interface accepts labels, ball coordinates, ownership
metadata, or events. Team and player-slot invariance is inherited from the
player feature extractor.

This produces 625 columns for the current schema: 67 originals, 18 finite flags,
and 540 temporal features. Feature names are stored in each model and verified
at inference. The raw-versus-context pair is the direct feature ablation;
training gain grouped by family will be reported descriptively, without calling
it causal validation importance.

## Ensemble and selection

Let the GRU output probabilities be H, A and D, and let the LightGBM D
probability be L. The GRU's conditional live-team probability is
`q = H / (H + A)` (neutral .5 only when both live probabilities are zero).
For each fixed GRU seed, score two predeclared settings:

- D replacement: `p = L`.
- Equal D blend: `p = .5 D + .5 L`.

Both produce `[(1-p)q, (1-p)(1-q), p]`, then use the unchanged Viterbi decoder.
This isolates the D companion; it does not test a new LightGBM team classifier
or a full-vector model blend. All 12 LightGBM checkpoints and the six GRU control
checkpoints are bound by checksum receipts before any new score is read.

The budget is 12 fits and 240 paired match/view/configuration/GRU-seed/blend
rows, with 30 GRU control rows reused. Each provider contributes equally after
averaging its matches and the two stress views within match. Official spell F1
is primary, macro F1 breaks exact ties, and `.7 spell + .3 macro` is an auxiliary
column only.

For each GRU seed, the declared screen requires clean spell loss no worse than
.01 relative to its matched GRU control, stress spell gain at least .02, and no
provider's clean D F1 or macro F1 loss larger than .02. Passing this exploratory
screen does not automatically replace the adopted baseline. The report also
separates comparison with the current recipe under the matching full-provider
protocol. Feature/configuration selection on these reused matches still needs
independent confirmation.

Diagnostics include every IoU's TP/FP/FN, short-spell recall, fragmentation and
merging geometry, recall at graded-run edges, complete-spell boundary error,
D precision/recall, and reference/predicted spell counts. These diagnose the
unchanged official metric rather than redefine it.

## Reproduction and artifact state

The input arrays are the hash-verified 18 external matches in
`experiments/eval_set/manifest.json`. Restore the private cache first. The shared
GRU control runner and evaluation caches live in
`data/experiments/gru_slot_robustness_v2`; no missing checkpoint is silently
substituted. The new output directory is
`data/experiments/temporal_lgbm_v1`.

```sh
python -m experiments.temporal_lgbm_experiment freeze
python -m experiments.temporal_lgbm_experiment prepare
# Each of the three providers and four configurations is fitted exactly once.
python -m experiments.temporal_lgbm_experiment fit --source sportec --config raw_slow
python -m experiments.temporal_lgbm_experiment seal
# Requires the completed GRU control evaluation caches and probabilities.
python -m experiments.temporal_lgbm_experiment evaluate
python -m experiments.temporal_lgbm_experiment report
```

At declaration time, the six feature/fusion contract tests pass. Full experiment
results, remote artifact availability and the resulting decision will be filled
in after the frozen run finishes. No competition CSV or score is produced by
this study.
