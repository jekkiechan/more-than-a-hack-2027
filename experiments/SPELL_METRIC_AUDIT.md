# Possession spell metric audit and optimization directions

Keep the official spell objective unchanged. The current scorer and external
adapter correctly preserve unknown ground truth, match teams, and separate
possession intervals at half and timestamp breaks. The most useful improvement
is to retain the five IoU score breakdowns and coverage diagnostics in every new
model comparison. A single spell mean, frame F1, or possession count cannot
identify the error mechanism.

This audit adds executable metric contracts and aggregate diagnostics. It also
analyzes **saved external predictions** of the original unrestricted joint GRU
and matched baseline. All 30 archived score/count rows reproduce within 1e-12;
input hashes match the archived declaration. No models were fitted and no
organizer labels were read. This is posthoc development analysis of previously
exposed matches, not a new model selection or confirmation experiment.

## Exact ranking objective

The implementation is [the unchanged official scorer](../evaluation/possession_metrics.py),
SHA256 `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`.
Its semantics agree with [upstream at f8f2629](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/evaluation/possession_metrics.py).

1. Ground-truth X breaks the timeline into graded runs. Predictions outside
   those runs have no effect. Within each run, predicted X is forward-filled;
   leading X is backfilled from the first known prediction. An entirely
   abstained run remains X. Frame scores still penalize raw predicted X.
2. A possession is a maximal H or A interval, using half-open frame indices
   `[start, stop)`. D is not a possession, but separates adjacent possessions.
3. Only same-team intervals can match. Temporal IoU is their overlapping
   duration divided by their union duration.
4. At each threshold 0.3, 0.4, 0.5, 0.6 and 0.7, candidate pairs are sorted by
   descending IoU, then reference index and prediction index. Greedy assignment
   consumes both sides once. Equality at the threshold qualifies.
5. With G reference spells, P predictions and T matches, precision is T/P,
   recall T/G, and F1 is `2T/(G+P)`. Zero-denominator cases return zero.
   The headline is the arithmetic mean of the five F1 values. It is not mAP,
   duration-weighted F1, or the mean IoU of matched intervals.

The external [adapter](metrics.py) inserts ungraded separators at period and
timestamp discontinuities, preserving native X. Its shared preparation helper
now also supplies the new diagnostics; this refactor does not alter scoring.
Dataset aggregation remains mean within provider, then equal provider mean.
Pooling all spell counts would produce a different objective and provider weight.

## What the archived predictions show

These are **full H/A plus D provider holdouts**, with one Sportec, one Metrica
and three SkillCorner development matches. Stress averages the same two fixed
corruptions per match. The original unrestricted joint GRU below is distinct
from the later player-only D factorized GRU in
[DEADBALL_TRANSFER](DEADBALL_TRANSFER.md).

| View and model | Mean spell F1 | F1 at 0.3 | F1 at 0.5 | F1 at 0.7 | Under 3 second recall at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean baseline | 0.5338 | 0.6485 | 0.5472 | 0.3949 | 29.17% |
| Clean original joint GRU | 0.6272 | 0.7152 | 0.6327 | 0.5233 | 37.97% |
| Stress baseline | 0.4129 | 0.5319 | 0.4196 | 0.2828 | 15.21% |
| Stress original joint GRU | 0.3976 | 0.5177 | 0.3960 | 0.2725 | 16.06% |

The clean improvement is larger at IoU 0.7 (+0.1284) than at 0.3 (+0.0667).
It includes stronger interval alignment, not merely more low-quality detections.
The fraction of reference spells overlapped by multiple same-team predictions
falls from 5.60% to 1.49%. This overlap definition includes any positive overlap;
it describes fragmentation geometry without proving a particular cause.

Under stress the original GRU loses to baseline at all five thresholds. At 0.5,
precision is 0.4691 versus 0.4766 and recall is 0.3719 versus 0.3957. The slightly
better short-spell recall does not rescue the whole objective. This rules out
using short-spell recall alone as a replacement selection criterion. These
patterns cannot be attributed to the later player-only D model, whose separate
saved report establishes the stronger D-overprediction stress failure.

### A substantial coverage limitation

In SkillCorner, **69.86% of reference spells touch an edge of a graded run**,
averaged across its three matches, compared with 0.22% for Sportec and 1.05% for
Metrica. Such edges can be created by unknown-label margins and visibility
filtering, as well as actual recording boundaries. They are not necessarily the
start or end of the complete football possession.

The official score remains valid for its observed population, but high spell
F1 on that population cannot establish complete-possession timing through
broadcast cuts. Retain the mask for comparability; report edge-touching and
fully observed cohorts separately. Do not fill unknown labels to manufacture
complete intervals. New independent labels should cover stoppages, restarts
and real occlusions with clearly recorded annotation uncertainty.

## Remaining metric and reporting gaps

| Gap | Consequence | Action |
| --- | --- | --- |
| Recent experiment rows retain the mean and selected 0.5 diagnostics, but omit the full five-threshold curve | A boundary-alignment gain and a detection-count gain can look identical | Save per-IoU precision, recall, F1 and TP/FP/FN for every arm and view |
| Per-frame likelihood is the training loss | Long stable stretches dominate gradients while each short possession counts once at evaluation | Select complete decoded pipelines by official spell F1; test duration or boundary loss changes only as separate matched ablations |
| Greedy matching is not maximum-cardinality matching | A high-IoU pair can prevent two lower-IoU feasible pairs | Preserve official assignment and expose unmatched spells with eligible overlaps; never substitute Hungarian or maximum-cardinality matching for ranking |
| Predicted X is filled for temporal metrics | Spell F1 alone does not penalize all abstention errors | Keep abstention counts and frame metrics visible; abstention is not a robustness repair |
| Reference-run edges censor interval endpoints | Timing errors at clipped edges can be misleadingly small | Exclude edge-touching spells from matched-boundary timing and report excluded counts |
| The official transition matcher has no run identifier | Transitions from different runs can match across short X gaps or inserted one-frame separators | Treat these secondary transition diagnostics cautiously; a separate within-run transition diagnostic is preferable before using it for optimization. Primary spell overlap cannot cross the disjoint runs |
| Official bootstrap resamples matched reference flags with prediction count fixed | It does not reproduce variable false-positive counts, assignment changes or unseen-provider uncertainty | Use paired whole-match/provider summaries for candidate differences; report seed ranges and the limited independent sample count |

The official `macro_f1` function still has a stale ranking sentence in its
docstring; the scorer module, ranking runner and project decision policy clearly
rank by spell F1. No official file was edited to repair documentation.

## Revised research priorities

1. **Repair the latest GRU's stress D precision while protecting clean spell
   gains.** The latest constrained model's clean/stress spell scores are
   0.6093/0.2991 versus matched baseline 0.5338/0.4129, averaging two separate
   seeds. Its Sportec/Metrica stress D recall near 0.99/0.98 accompanies precision
   near 0.63/0.47. Test restored player-slot-dropout training with the same data,
   normalization, decoder and full-pipeline holdouts; inspect interval recall
   and false D fragmentation as well as frame D F1.
2. **Make LightGBM an interval-aware complementary model.** Engineer separate
   past/future values, local slopes, before-after contrasts and persistence at
   multiple scales; preserve missingness and reset at input breaks. Compare
   restricted leaf/depth sizes, minimum leaf support, subsampling and shrinkage
   on whole matches. Select by decoded spell F1, then test a frozen GRU mixture
   against both standalone models on identical clean/stress cells.
3. **Measure threshold and duration behavior before changing the decoder.**
   If the 0.3 tier is healthy but 0.7 weak, test boundary-local features or timing
   refinements. If precision and predicted counts deteriorate, inspect excess
   short spells; if recall collapses, inspect erased or merged possessions.
   Do not reduce spell count blindly: smoothing also erases legitimate short
   possessions. Preserve the current decoder in the initial model comparisons
   so causes remain identifiable.
4. **Resolve the labeling and confirmation gap.** Obtain an independent,
   rights-cleared broadcast match with complete H/A/D intervals and real
   occlusions. Reused provider holdouts, synthetic stress and the organizer's
   17-spell diagnostic do not establish full-match generalization.

The adopted baseline remains unchanged. These directions supersede applying
the old broad claim that dead-ball detection is never the bottleneck to every
new model. The latest constrained GRU specifically demonstrates that it can be.

## Artifacts and checks

- [Aggregate archived audit](results/spell_metric_audit.json) includes all 30
  cells, dataset and prediction hashes, exact score parity, and the summaries.
- [New diagnostics](spell_diagnostics.py) retain the full IoU curve, duration
  and team recall under the global matching, interval overlap geometry,
  graded-edge counts and matched complete-spell boundary errors.
- [Metric contract tests](tests/test_spell_diagnostics.py) cover wrong team,
  short-spell weighting, one-frame fragmentation, exact IoU equality, greedy
  assignment conflicts, X handling, all-D and empty-spell cases, scorer parity,
  subgroup matching and boundary censoring. All 22 focused metric tests pass.
- [Reproduction script](spell_audit_saved.py) reads only the declared external
  archive, checks input hashes and archived score/count parity, and refuses to
  overwrite a completed audit. It does not refit or choose a model.

```sh
python -m unittest experiments.tests.test_metrics experiments.tests.test_spell_diagnostics -v
python -m experiments.spell_audit_saved --out data/experiments/spell_metric_audit_repeat.json
```

Whole-suite validation and the concurrently run new model results are reported
in the overall research update. The synthetic contracts verify metric behavior;
they are not evidence of improved predictive performance.
