# Feed-like benchmark audit

Study `feed_benchmark_audit_v1`, 1 November 2026. A post-hoc audit with no fitting,
no organizer labels and no public scoring. The adopted model is unchanged.

## Question

The `original_full` candidate leads the original feed-like benchmark (.528180 vs
adopted .518527) but scored lower on the 17-reference public window (.611765 vs
.630303). Two questions follow. Is the candidate's development gain real? And what
could move original feed-like spell F1 toward .55 without losing robustness?

## Summary

1. **Decoder and arm-combination headroom is nearly gone.** Choosing the best of 54
   calibrated-Viterbi settings *separately for each cell, using the labels* reaches only
   .5332 feed-like (+.005). Choosing the best of 17 archived arms per cell reaches .5358
   (+.0076). An equal blend of all 17 arms scores .5259. All 17 arms make 79.7% of the
   candidate's feed-like frame errors, and a majority of arms make 94.5%. Any stacking
   over these arms is bounded by this shared error.
2. **The original benchmark puts ball gaps where the real feed does not.** In the
   organizer tracking, 40.7% of in-play frames lack the ball. The adopted model predicts
   dead ball on 83.9% of those frames and on 18.7% of frames where the ball is visible.
   Its D head uses player features only. Gaps of 15 s or longer hold 75% of the missing
   time, and the model calls about 93% of that time dead. The synthetic feed-like views
   place gaps independently of game state. On Sportec, true D is 46% of frames whether
   or not the ball is missing.
3. **Under gaps matched to the challenge, scores are much higher and the candidate still
   leads.** With 4 new realizations per match, the candidate beats the adopted model in
   **12 of 12** realization-level comparisons across three gap regimes.
4. **Realization noise is as large as the remaining gap to .55.** For the same candidate,
   the four random-gap realizations range from .5307 to .5434 (phase 0). The earlier
   fresh pair scored .547221 and the original pair .528180. One phase-TTA realization
   reaches .5526.
5. **Sampling-phase test-time augmentation is a small, consistent, free gain.** Swapping
   the teams at test time is an exact no-op, because the models are already
   swap-equivariant.
6. **The public drop is about one spell.** At IoU .3–.5 both CSVs match 12 of 17
   reference spells. The candidate predicts one more spell (17 vs 16), gains one match
   at .6 and loses one at .7. Both public 90% bootstrap intervals are about .32 wide.

## Paired development comparison

This is a within-match block bootstrap: 2-minute blocks, 2,000 draws, original
archived cells.

| Comparison | View | Delta | 90% interval | P(>0) |
| --- | --- | ---: | --- | ---: |
| Candidate − adopted | clean | +.0040 | [−.0067, +.0147] | .73 |
| Candidate − adopted | feed-like | +.0097 | [+.0011, +.0180] | .96 |
| Candidate − leader | clean | +.0054 | [−.0018, +.0132] | .88 |
| Candidate − leader | feed-like | +.0008 | [−.0050, +.0065] | .58 |

These intervals ignore between-match, realization and selection (winner's-curse)
uncertainty. Treat them as lower bounds on the total uncertainty.

## Oracle headroom (candidate, analysis only)

| Label oracle applied to decoded output | Clean spell | Feed-like spell |
| --- | ---: | ---: |
| None (candidate) | .6626 | .5282 |
| Perfect dead/live; model team on recovered live frames | .7408 | .5889 |
| Paste true spells shorter than 1 s | .7288 | .6089 |
| Paste true spells shorter than 3 s | .8222 | .7504 |
| Correct team only inside true spells shorter than 3 s | .7641 | .6789 |
| Paste true spells of 3 s or longer | .7948 | .7476 |
| Perfect team wherever both prediction and truth are live | .8528 | .8281 |

Team attribution is still the dominant lever, concentrated in short spells. Dead/live
errors are worth about +.06 under feed-like stress. The earlier Sportec-only headroom
study found almost no dead/live headroom on clean frozen-ball data.

## Challenge ball-gap profile (label-free)

This uses the organizer tracking inputs and the adopted model's sealed label-free CSV
only. The organizer label array is never read.

| Statistic | Challenge |
| --- | ---: |
| Ball missing, in-play frames | .407 |
| P(missing given predicted D) / P(missing given predicted live) | .755 / .120 |
| P(predicted D given missing) / P(predicted D given visible) | .839 / .187 |
| Gap runs (all in-play) | 150 |
| Gap time in runs ≥15 s, and its predicted-D coverage | 75% / ~93% |
| Missing-and-predicted-live runs, median length | 133, 1.8 s |

Predicted D is a model output, not truth. The D head sees players only, but a model
that leans toward D in gaps would inflate these estimates. On the original synthetic
views, the model's predicted-D rate given a missing ball is close to the true rate
(Sportec .51–.55 vs .44–.46). That bias is therefore modest.

## Challenge-matched views

New views are built from the clean development inputs. The non-ball defects match the
feed-like recipe: coordinates, slot gaps and ID swaps. Three ball-gap regimes follow:

* **random:** the existing renewal process, placed independently of game state.
* **matched:** each true dead run is blanked at the challenge D rate. Short live gaps are
  sampled from the challenge's live-gap durations until 12% of live frames are missing.
* **matched_live2x:** the same, at twice the live rate.

Dev labels only place the simulated gaps. Inference never receives labels. Each regime
has 4 realizations per match. Scores are equal-provider means.

| Regime (miss given D / miss given live) | Adopted | Raw-feed control | Candidate | Adopted + phase TTA | Candidate + phase TTA |
| --- | ---: | ---: | ---: | ---: | ---: |
| random (.60 / .41) | .5178 | .5293 | .5361 | .5273 | .5426 |
| matched (.82 / .10) | .6171 | .6222 | .6248 | .6266 | .6349 |
| matched_live2x (.82 / .17) | .6108 | .6142 | .6196 | .6130 | .6222 |

Candidate minus adopted, per realization:

| Regime | Realizations 1–4 |
| --- | --- |
| random | +.0172, +.0294, +.0079, +.0186 |
| matched | +.0047, +.0070, +.0130, +.0062 |
| matched_live2x | +.0096, +.0113, +.0090, +.0055 |

Macro F1 moves in the same direction in every regime (candidate .8641 vs adopted .8636
matched; .8252 vs .8206 random). SkillCorner is the exception: the candidate trails the
adopted model by .009 under matched gaps (.477 vs .486). Sportec and Metrica carry the
gain, the same provider pattern as on the original benchmark.

## Test-time views (original archived cells)

| Arm | Clean spell | Feed-like spell | Feed macro |
| --- | ---: | ---: | ---: |
| Candidate | .6626 | .5282 | .8254 |
| Candidate + team swap | .6626 | .5282 | .8254 |
| Candidate + 5 Hz phase TTA | **.6676** | **.5316** | .8269 |
| Adopted | .6586 | .5185 | .8233 |
| Adopted + 5 Hz phase TTA | .6589 | .5231 | .8246 |

The models sample features at 5 Hz from 25 Hz tracking. Phase TTA averages the
probabilities from all 5 sampling offsets (2 for 10 Hz SkillCorner) before the unchanged
decoder. It is non-negative in every development view and regime here, across both
models. The gain ranges from +.0003 to +.0102 and costs 5× inference. Phase 0 replays
the archived record and probabilities exactly. The adopted phase-0 output reproduces the
archived f062 labels with zero mismatches.

## Interpretation and decision

* Development evidence favors the candidate over the adopted model in every
  corruption regime tested. The public difference is consistent with one spell of
  noise and should not decide adoption. The remaining risks are SkillCorner, the
  winner's-curse selection among about 10 arms, and the lack of independent matches.
* Reaching .55 on the two original realizations alone is a weak target. Single
  realizations move the same model by about ±.01. The original realizations also
  overweight live-play ball blackout relative to the challenge feed.
* Decoder and arm-stacking work has small measured ceilings (+.005 / +.0076 per-cell
  oracles). Larger gains need new information about team attribution in short spells.
  That means new supervision or inputs, not recombining the current models.
* No promotion. The phase-TTA and matched-benchmark findings are proposals for a
  separately declared study.

## Recommended next experiments

1. Declare a **multi-realization selection benchmark**: at least 4 realizations each of
   random and challenge-matched gaps. Pre-register it before scoring any new arm. The
   runner's `views` step provides the generator.
2. Add **phase TTA** to the candidate inference path as a fixed, label-free change.
   Verify it once on all regimes, with no further tuning.
3. Run a **training-time matched-gap augmentation**: replace random ball gaps with
   state-dependent gaps in the raw-feed training views. This needs the GAR TRAIN records,
   which are not on any private archive branch. Restoring them from the gated SoccerNet
   source is the blocker.
4. **Short-spell team attribution** remains the largest oracle lever (+.15 feed-like for
   <3 s spells). Prioritize externally corroborated short supervision, as STATUS already
   says.

## Reproduction

Check out three private archives: short-quality-diversity, feed-spell-push and
dense-targeted. Pull their LFS objects for the predictions, models, teachers and
`seed_gap_context_v1` inputs. Then run:

```sh
MTAH_DATA_DIR=../mtah-data MTAH_FSP_DIR=../mtah-fsp MTAH_DT_DIR=../mtah-dt \
  .venv/bin/python -m experiments.feed_benchmark_audit_v1 all
```

Aggregate results: [`results/feed_benchmark_audit_v1.json`](results/feed_benchmark_audit_v1.json).
Views, probabilities and per-cell scores stay in ignored `data/experiments/feed_benchmark_audit_v1/`.
The full rerun takes about 70 minutes on 4 CPUs.
