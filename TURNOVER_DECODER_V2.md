# Turnover decoder v2: sample-weight scale diagnostic

V2 changes one numerical choice in the frozen v1 turnover-head fit: it
multiplies every source/match-balanced sample weight by the number `N` of
directed TRAIN edge rows, changing normalized weights from sum 1 to mean 1.
It keeps the same three provider folds, records, targets, features, weighted
transforms, weighted natural event priors, `C=0.1`, solver, max iterations,
cost formula, bounds and 15 evaluation cells. The independent reviewer
revalidated the frozen declaration and its v1 provenance chain before any v2
fit. All three v2 heads were sealed before scoring.

## Why change the weight scale

Scikit-learn's L2 logistic objective uses the sum of weighted row losses. With
v1's normalized weights, their sum was 1, so `C=0.1` multiplied a single
effective unit of loss while the L2 term remained unchanged. Multiplying all
weights by `N` is mathematically equivalent to keeping normalized weights and
using effective `C=N×0.1`; it preserves the relative source/match masses. V2
continues to use the original normalized weights for weighted feature
imputation/scaling and for the natural event prior, so those quantities are
unchanged. This is a post-v1 regularization-units diagnosis, not a C or
threshold sweep. The declaration records the previously observed v1 outcome
and the fact that these five development matches were already exposed.

The heads use the exact v1 predictions, folds and H/A edge definitions. As in
v1, training predictions from the frozen backbone are in-sample for TRAIN
matches; the held-out provider remains excluded from every learned component.
No GRU was refit, no new baseline was trained, and there was no organizer
scoring.

## Training diagnostics

TRAIN event AUC and log-loss are weighted by the unchanged normalized
source/match masses. They are descriptive in-sample head diagnostics, not
selection metrics.

| Held-out fold | Directed rows (`N`) | Weight mean v1 → v2 | Coef L2 norm v1 → v2 | TRAIN AUC v1 → v2 | TRAIN log-loss v1 → v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sportec | 260,156 | 0.00000384 → 1.000000 | 0.002234 → 3.655320 | .857473 → .880898 | .041225 → .028490 |
| Metrica | 283,012 | 0.00000353 → 1.000000 | 0.002364 → 4.444062 | .844260 → .878845 | .053551 → .038971 |
| SkillCorner | 156,444 | 0.00000639 → 1.000000 | 0.002989 → 3.988563 | .905449 → .928002 | .053940 → .030957 |

All three L-BFGS fits converged in 25–29 iterations. The natural weighted
event prevalences and 50/50 training source masses are identical to v1. The
increase in coefficient norms and training discrimination confirms the v1
head was strongly dominated by L2 at its original weight scale; it does not
by itself establish held-out usefulness.

## Frozen evaluation results

The evaluation uses the same 15 cells and exact cost-3 control as v1. Values
are equal-provider means; `false extra` and `merged` count spells per match.

| View / decoder | Spell F1 | Macro F1 | Short recall (<3 s) | False extra | Merged | IoU .3 | IoU .4 | IoU .5 | IoU .6 | IoU .7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean, cost 3 | .633205 | .854226 | .384161 | 125.556 | 30.556 | .723165 | .695803 | .646311 | .587839 | .512906 |
| Clean, v2 | .631627 | .853869 | .405888 | 132.444 | 29.000 | .722549 | .695019 | .644219 | .583517 | .512831 |
| Feed-like, cost 3 | .498830 | .811387 | .203371 | 137.889 | 43.667 | .610564 | .570382 | .503104 | .438757 | .371340 |
| Feed-like, v2 | .495856 | .811366 | .211930 | 141.389 | 43.056 | .608449 | .565011 | .500896 | .434785 | .370138 |

Clean spell F1 changed by −0.001578, failing the predeclared +0.01 screen.
Feed-like spell F1 changed by −0.002974, above its −0.01 floor. Provider clean
macro-F1 deltas were +0.000483 / −0.000569 / −0.000984 and D-F1 deltas were
0.000456 / 0.000021 / 0.000585 for Sportec / Metrica / SkillCorner; those
guardrails passed. Per-provider clean spell-F1 changes were +0.000776,
−0.003744 and −0.001766. Feed-like spell-F1 fell for all providers: −0.005305,
−0.002172 and −0.001445.

Short-spell recall increased by +0.021727 clean and +0.008559 feed-like. The
tradeoff was worse overall spell overlap: all five aggregate clean IoUs fell
by −0.000616 / −0.000784 / −0.002092 / −0.004322 / −0.000075, and all five
feed-like IoUs fell by −0.002115 / −0.005371 / −0.002208 / −0.003972 /
−0.001203. False extra spells increased by 6.889 clean and 3.500 feed-like
per match; merged references decreased by 1.556 and 0.611. The component-seed
diagnostics show the same short-recall increase but mixed spell-F1 changes;
the actual two-seed ensemble remains the primary result.

The edge-cost output now moves materially away from cost 3. Across eligible
directed intervals, costs spanned the declared 1.5–4.5 bounds. Bound clipping
affected 173,678/291,668 (59.5%) clean directed edges and 448,222/583,336
(76.8%) feed-like edges. Directional interval-average costs were 4.037/4.013
clean and 4.223/4.207 feed-like for H→A/A→H. The v2 head creates many more
switch opportunities, recovering short references while adding false spells
and lowering IoU.

The numerical correction addresses why v1's fitted head was nearly inert, but
does not improve the task metric screen. Keep the exact cost-3 baseline. The
cohort is five already exposed development matches; Metrica and SkillCorner
labels are event-derived silver, and feed-like corruption is synthetic. This
result does not estimate performance on new matches and does not support
promotion.

## Reproduction and artifacts

```sh
/workspace/venv/bin/python -m unittest discover -s experiments/tests -p 'test_turnover_decoder_v2.py' -v
/workspace/venv/bin/python -m experiments.turnover_decoder_v2 freeze
/workspace/venv/bin/python -m experiments.turnover_decoder_v2 fit
/workspace/venv/bin/python -m experiments.turnover_decoder_v2 seal
/workspace/venv/bin/python -m experiments.turnover_decoder_v2 score
```

The v2 declaration, fit/score receipts and artifact receipt live under
`experiments/results/`. The heads, seal, per-cell arrays and complete run are
local-only under `data/experiments/turnover_decoder_v2_unit_weights/`.
The independent replay receipt verified all 30 saved cost-3/candidate decode
rows, exact v1 baseline-array parity, unchanged candidate emissions and
provider/view aggregates within `1e-12`; it did not refit or rerun inference.
