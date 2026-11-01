# Challenge-calibrated feed simulation (`feed_like_v2`)

Studies `feed_like_v2_benchmark`, `feed_like_v2_train` and `feed_like_v2_eval`,
1 November 2026. No organizer labels are read and nothing is scored on the public
window. No model is promoted here.

> **Naming.** "Adopted" below means `joint_gru_sn_dense_calibrated_ensemble_v1`, the
> package adopted when this study started. The user later promoted `original_full`
> (`joint_gru_sn_dense_original_diversity_ensemble_v1`, PR #70); it appears below as the
> "`original_full` candidate". The organizer predicted-state statistics use the earlier
> package's sealed label-free CSV.

## Why a new feed view

The [benchmark audit](FEED_BENCHMARK_AUDIT_V1.md) found that the original synthetic
`feed_like` view loses the ball independently of game state, while the organizer feed
loses it mostly in dead play. This study profiles the organizer tracking more fully,
builds a simulator calibrated to it, and re-ranks models on the result. The comparison
uses tracking statistics only; organizer labels are never read.

## What the organizer feed looks like

Every statistic is computed from organizer tracking inputs plus the adopted model's
sealed label-free CSV, which supplies the dead/live state. The same statistics are
computed on simulated views using the adopted model's own predictions, so both sides
are measured the same way.

* **Ball loss follows game state.** The ball is missing on 75.5% of predicted-dead
  frames and 12.0% of predicted-live frames. In 81% of dead runs of at least 3 s the
  ball goes missing, typically right at the start of the run (median 0.02 s), and stays
  missing for about 66% of the run.
* **Most live-play ball loss is a dead gap spilling over its edge.** Two-thirds of
  live-missing runs touch a dead run. Isolated live gaps occur 0.76 times per live minute.
* **Player dropouts are rare, long and happen at the edge of the pitch.** There are
  17.4 dropouts per dead minute and 6.3 per live minute. The median gap is 8 s, and no
  gap is shorter than 2.04 s, which suggests the provider interpolates short gaps.
  Dropout hazard is about 10× higher at or beyond the touchline: 13.4% of dropouts begin
  with the player outside the pitch, versus 0.9% of observed positions. Dropouts do not
  depend on distance to the ball, so this is not a camera field-of-view effect.
* **Not a defect:** single-frame ball jumps of more than 1 m (49 per minute) are sustained
  fast movement such as kicks, not outliers. No frame sits more than 1 m from a 9-frame
  median. This is a smoothing difference between providers, so it is not simulated.

## Simulator

[`feed_like_v2.py`](feed_like_v2.py) keeps the frozen camera drift, local noise and
jumps from `stress_components`. It replaces the state-blind defects with four
calibrated ones:

1. **Dead-play ball loss.** Each true dead run is blanked with the calibrated
   probability, onset and end-margin quantiles.
2. **Ball gaps around stoppages.** Live gaps extend across dead-run edges at the
   calibrated rate, and isolated live gaps are added only up to the calibrated rate, net
   of the provider's native gaps.
3. **Player dropouts.** Dropout hazard depends on game state and distance to the pitch
   boundary, with duration quantiles by state, again added only up to the calibrated rate.
4. **Identity switches without gaps.** Switches happen where two same-team tracks pass
   within 1 m, at the original swap frequency, and leave no gap.

Development and training labels only *place* defects by state, as the real feed's loss
depends on state. Features and inference never see labels. Two scales (dead-play
dropout ×1.25, isolated live gaps ×0.5) were fixed after two single-realization fidelity
checks against the organizer profile, before any v2 model scores existed. The
calibration is in [`results/feed_like_v2_calibration.json`](results/feed_like_v2_calibration.json).

## Fidelity

Median relative error across 15 profile statistics on the full-pitch providers
(Sportec, Metrica): **v1 0.72 → v2 0.09**. Each v2 value is the mean of 4 realizations;
v1 uses the two original replicas.

| Statistic | Organizer | v1 Sportec / Metrica | v2 Sportec / Metrica |
| --- | ---: | ---: | ---: |
| Ball missing, predicted dead | .755 | .428 / .936 | .667 / .907 |
| Ball missing, predicted live | .120 | .415 / .385 | .106 / .070 |
| Isolated live gaps per live minute | .76 | 1.24 / 1.71 | .71 / 1.13 |
| Live-missing runs touching dead | .669 | .592 / .499 | .700 / .528 |
| Player dropouts per minute, dead / live | 17.4 / 6.3 | 19.7 / 20.2 · 19.9 / 20.1 | 16.6 / 6.0 · 16.5 / 6.2 |
| Player gap median (s) | 8.0 | 1.5 / 1.5 | 8.0 / 8.2 |
| Player gaps shorter than 1 s | 0% | 33% / 32% | 0% / 0% |
| Dropouts starting outside the pitch | 13.4% | 1.0% / 0.8% | 10.2% / 8.3% |
| Dead minus live visible players | −2.0 | 0.0 / 0.0 | −1.1 / −1.2 |

Three residual gaps remain:

* The dead-play visibility drop is about half the organizer's.
* Metrica's native ball blanking during dead play keeps its predicted-dead ball loss high.
* SkillCorner is broadcast tracking with heavy native loss, so v2 adds nothing on top of
  it and it cannot match a full-pitch feed. It stays in the equal-provider mean, and the
  full-pitch mean is also reported.

## Re-ranking under v2

Five development matches, 4 realizations each, the unchanged f062 decoder,
equal-provider means.

| Arm | v2 spell | Full-pitch spell | v2 macro | Per realization |
| --- | ---: | ---: | ---: | --- |
| Adopted | .5836 | .6298 | .8513 | .5941 .5864 .5728 .5812 |
| Adopted + phase TTA | **.5880** | **.6365** | **.8525** | .6006 .5919 .5785 .5811 |
| `original_full` candidate | .5785 | .6257 | .8488 | .5836 .5782 .5731 .5791 |
| Candidate + phase TTA | .5836 | .6333 | .8500 | .5849 .5850 .5757 .5886 |
| Raw-feed (v1 training views) | .5784 | .6228 | .8495 | .5878 .5783 .5687 .5786 |
| v2-trained (v2 training views) | .5827 | .6284 | .8515 | .5931 .5850 .5737 .5788 |

* **The candidate no longer leads.** On Metrica it loses in 4 of 4 realizations (−.013 to
  −.038) and on SkillCorner in 11 of 12. On Sportec it gains in 4 of 4 (+.007 to +.020).
  It finds more short spells (Metrica short-spell recall .321 vs .252) but over-splits
  possessions when players drop out (Metrica 337 predicted spells vs 288, against 382
  references). This matches the direction of its lower public score.
* **Phase TTA helps the adopted model** (+.0044 on v2). It is non-negative on every
  benchmark tested here and in the audit.
* **Training on v2 views beats its matched control.** Against the raw-feed model with v1
  views, the v2-trained model gains +.0043 on v2 (4 of 4 realizations, +.0002 to +.0067).
  It still trails the adopted model by .0009.

## Scorecard: every arm on every view

Equal-provider spell F1 / macro F1. Clean and v1 use the original archived cells (v1 has
2 replicas). v2 uses 4 realizations, and the full-pitch column averages Sportec and
Metrica only. Public scores are the existing one-time 17-spell diagnostics, reused and
not rescored; the other arms were never scored on the public window.
[Scorecard JSON](results/feed_like_v2_scorecard.json).

| Arm | Clean | v1 feed-like | v2 | v2 full-pitch spell | Public (17 spells) |
| --- | --- | --- | --- | ---: | --- |
| Calibrated dense (previous adopted) | .6586 / .8714 | .5185 / .8233 | .5836 / .8513 | .6298 | .6303 / .8642 |
| + phase TTA | .6589 / .8709 | .5231 / .8246 | **.5880** / .8525 | **.6365** | — |
| `original_full` (now adopted) | .6626 / .8702 | .5282 / .8254 | .5785 / .8488 | .6257 | .6118 / .8517 |
| + phase TTA | **.6676** / .8713 | **.5316** / .8269 | .5836 / .8500 | .6333 | — |
| Raw-feed (v1 training views) | .6547 / .8686 | .5252 / .8269 | .5784 / .8495 | .6228 | — |
| + phase TTA | .6607 / .8704 | .5268 / .8271 | .5824 / .8499 | .6289 | — |
| v2-trained (v2 training views) | .6499 / .8693 | .5169 / .8224 | .5827 / .8515 | .6284 | — |
| + phase TTA | .6562 / .8706 | .5226 / .8233 | .5856 / .8527 | .6332 | — |

Phase TTA raises spell F1 for every arm on every view.

## v2-trained model on the older benchmarks

| Benchmark | Adopted | Raw-feed (v1 views) | v2-trained |
| --- | ---: | ---: | ---: |
| Original clean | .6586 | .6547 | .6499 |
| Original feed-like (v1, 2 replicas) | .5185 | .5252 | .5169 |
| Audit random-gap views (4 realizations) | .5178 | .5293 | .5203 |
| Audit matched ball-gap views (4 realizations) | .6171 | .6222 | .6179 |

Training on a corruption mostly buys robustness to *that* corruption. The raw-feed
recipe's gains on v1 do not transfer to the more realistic v2. The reverse is also
true: v2 training does not help on v1.

## Verification

* The SoccerNet-GAR TRAIN source (7 files, 5.04 GB) matches the frozen SHA-256 values.
  The rebuilt dense records reproduce the documented 45 games, 11,249 records and
  5,063,339 native frames.
* **Training parity:** refitting the archived raw-feed `leave_out_metrica_2026` model
  from its archived v1 views reproduces every parameter to within 5.1 × 10⁻⁶ (23 arrays).
  Byte hashes differ. The v2 fits change only the training views.
* Inference replays archived probabilities exactly (see the audit).
* Synthetic contract tests: `experiments/tests/test_feed_like_v2.py`.

Receipts: [`calibration`](results/feed_like_v2_calibration.json),
[`benchmark`](results/feed_like_v2_benchmark.json),
[`training`](results/feed_like_v2_train.json),
[`older benchmarks`](results/feed_like_v2_eval.json). Views, checkpoints and per-cell
scores stay in ignored `data/experiments/feed_like_v2_*`.

## Decision and limitations

* **The v1-feed advantage behind the `original_full` promotion does not hold on v2.** The
  previous calibrated-dense package, especially with 5 Hz phase TTA, ranks higher on the
  more realistic feed and also on the 17-spell public window. Whether to revert is the
  user's decision. Phase TTA is label-free and adds no learned parameters, but needs a
  separate declared integration and confirmation before any promotion.
* **Use `feed_like_v2` (4+ realizations) as the stress guardrail instead of the two v1
  replicas.** The .55 target was set on v1. On v2 the previous calibrated-dense package already
  scores .584–.588, and v1 rankings did not survive the move to a more realistic feed.
* **Circularity:** calibration conditions on the adopted model's predicted dead state,
  which may favor models that call dead play like it does. The placement itself uses true
  state on development data.
* **Same five exposed matches:** v2 is a better proxy, not independent confirmation.
  Full-match organizer performance remains unknown.

Reproduce:

```sh
export MTAH_DATA_DIR=../mtah-data MTAH_FSP_DIR=../mtah-fsp MTAH_DT_DIR=../mtah-dt
python -m experiments.feed_like_v2_benchmark calibrate
python -m experiments.feed_like_v2_benchmark views      # adop/cand/raw (+ phase TTA)
python -m experiments.soccernet_dense_labels --splits train --output data/experiments/dense_label_weight_v1/records
python -m experiments.feed_like_v2_train prepare
python -m experiments.feed_like_v2_train parity
python -m experiments.feed_like_v2_train fit --fold leave_out_metrica --seed 2026   # ... all 6
V2_ARMS=v2train python -m experiments.feed_like_v2_benchmark views
python -m experiments.feed_like_v2_benchmark fidelity && python -m experiments.feed_like_v2_benchmark report
```
