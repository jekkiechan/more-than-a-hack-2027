# Does context-conditioned ball-gap placement make feed_like optimistic?

**Answer: no.** Evaluation view only; no retraining, no decoder change, no organizer
labels or ground truth read, nothing scored on the competition match. Adopted model
`joint_gru_combined_ensemble_v1` (provider-holdout checkpoints, seeds 2026/2027, cost-3 Viterbi),
five exposed development matches. Declaration frozen before any score:
`data/experiments/context_gaps_v1/declaration.json` (private). Aggregate results:
[results/context_gaps.json](results/context_gaps.json). Code: `context_gaps.py`.

## Method
- **Hazard.** Gradient-boosted classifier (depth 3, 100 iterations, `min_samples_leaf` 500) on
  competition-feed tracking only. Target: ball missing on the next frame. 11 features (ball speed,
  acceleration, direction change over 0.4 s, nearest-player distance, players within 3/8 m, ball x/y,
  distance to touchline and goal line, nearest-player speed). All inputs are corner-origin metres
  (competition pitch 105.3x68, providers 105x68; checked min/max), windows in seconds so 10 Hz
  SkillCorner is comparable. 92,796 eligible frames, 149 onsets.
- **Durations.** Empirical competition durations (cap 30 s), three pools by predicted-hazard tercile.
- **Views.** `context_gaps` = `stress_components.corrupt(coordinates, slot_gaps, id_swaps)` with the same
  seeds as feed_like `corrupt_k`, then context-simulated ball gaps (seeds `ce.seed(match,'ctx_k')`).
  `random_gaps_matched` = same per-period gap count and durations, uniform onsets.
- **Scalar.** One hazard multiplier per match (deviation from "single scalar": a global one gave
  0.31-0.64 coverage per match), fitted so synthetic gap intervals cover 0.4069 of active frames, the
  quantity the feed_like renewal process targets. Native provider gaps are overlaid, so union missing
  is higher (context 0.59, feed_like 0.55 mean over views including native).

## Hazard
AUC 0.907 on the second half (0.876 reversed split). Almost all of it comes from three features:
permutation AUC drop ball speed 0.164, direction change 0.178, acceleration 0.098; every other feature
is below 0.01. **The direction is opposite to the hypothesis**: onsets concentrate when the ball is slow
(onset rate 2.15x average in the slowest speed quintile, 0.20x in the fastest; direction change 2.6x vs
0.1x). Slow ball plus zero direction change could be dead balls or the tracker losing a stationary ball;
tracking alone cannot say which. Near a goal line 1.98x, near a touchline 1.64x, in the lowest quintiles.

## Realism (label-free)
| | gaps/min | covered fraction | duration q10/50/90 (s) |
| --- | ---: | ---: | --- |
| real feed | 1.42 | 0.407 | 0.47 / 10.7 / 30 (capped) |
| feed_like (holdout) | 1.89 | 0.408 | 0.36 / 10.6 / 26.2 |
| context_gaps (holdout) | 1.80 | 0.401 | 5.2 / 10.6 / 30 |

context_gaps lacks the 12% of real gaps under 1 s (hazard mass sits in the low tercile, whose pool
has few short gaps). Discriminator (real onsets vs synthetic, 5-fold CV AUC, 0.5 = indistinguishable):
on non-hazard descriptors 0.831 context, 0.835 feed_like, 0.816 random_matched, but the same
descriptors give 0.564 within the competition feed, so this number is dominated by provider shift and
does not rank the views. On the hazard features only: context 0.748, feed_like 0.914,
random_matched 0.913 (partly by construction, since the hazard was fitted on them).

## Scores (equal-provider mean; replicates, then matches, then providers)
| View | spell F1 | IoU .3 / .4 / .5 / .6 / .7 | macro F1 | D-F1 | under-3 s recall | pred / GT spells |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| clean | 0.6332 | .723 / .696 / .646 / .588 / .513 | 0.8542 | 0.8491 | 0.384 | 375 / 404 |
| feed_like | 0.4988 | .611 / .570 / .503 / .439 / .371 | 0.8114 | 0.8446 | 0.203 | 318 / 404 |
| random_gaps_matched | 0.5105 | .619 / .579 / .516 / .453 / .386 | 0.8126 | 0.8453 | 0.254 | 325 / 404 |
| context_gaps | 0.5114 | .619 / .574 / .520 / .457 / .387 | 0.8106 | 0.8451 | 0.253 | 322 / 404 |

Spell F1 by provider (Sportec / Metrica / SkillCorner): clean .643/.797/.459; feed_like .473/.602/.421;
random_matched .482/.624/.426; context .517/.592/.425. Replicate noise is large (J03WOY ctx_1 .547 vs
ctx_2 .487), so differences below about 0.03 are not resolved.

## Transitions hidden inside synthetic gaps (fraction of true boundaries; both adjacent frames missing)
| View | covered fraction | H<->A | live<->D (Sportec / Metrica) | all, pooled per cell |
| --- | ---: | ---: | --- | ---: |
| feed_like | 0.408 | 0.413 | .444 / .438 | 0.421 |
| random_gaps_matched | 0.400 | 0.397 | .341 / .438 | 0.397 |
| context_gaps | 0.400 | 0.399 | .399 / .297 | 0.394 |

Counts: H<->A 344 / 280 / 596 and live<->D 223 / 200 / 0 (Sportec / Metrica / SkillCorner; SkillCorner has
no graded D). Enrichment over coverage is 0.98 for context_gaps, 0.99 random, 1.03 feed_like.

## Decision (pre-declared rule)
context_gaps spell F1 is 0.0126 **higher** than feed_like (rule needed at least 0.02 lower), and it hides
transitions at a **lower** rate (0.394 vs 0.421). Placement alone (context vs random_matched) is -0.0009.
**feed_like placement is adequate**; do not replace it. The learned hazard gives no evidence that real
losses preferentially hide possession changes. The hazard learned a slow-ball signature, not the
fast-ball or crowd signature hypothesised.

## Limits
- Hazard fitted on 149 onsets from one match; the competition's ball behaviour may not transfer to the
  providers (their ball is smoother; most simulated onsets land in the lowest-hazard tercile).
- Features describe the ball track before loss, which cannot separate real occlusion from tracker
  behaviour. No ground truth was used, so "coincides with possession changes" is tested only on
  provider labels.
- context_gaps lacks short (<1 s) gaps. Per-match scalar and the feed_like coordinate seeds are
  choices made before scoring and recorded in the declaration.
- Five previously exposed matches, two replicates: this is development evidence, not adoption evidence.

```sh
python -m experiments.context_gaps prepare   # hazard, scalars, gap sets, realism
python -m experiments.context_gaps freeze
python -m experiments.context_gaps score --match metrica_3   # each match
python -m experiments.context_gaps report
```
