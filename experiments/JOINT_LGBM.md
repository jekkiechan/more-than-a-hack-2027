# Full H A D LightGBM features and GRU ensembles

Status: declared and frozen on 2026-10-27; feature preparation and training in
progress. Every LightGBM model directly predicts **H, A and D**. Its ensemble
averages the complete three-class probability vector with a GRU before applying
the shared decoder. The adopted baseline remains unchanged.

## What is being tested

The first three arms isolate feature engineering at identical capacity. The
fourth tests larger trees on exactly the same complete feature set.

| Arm | Features | Leaves | Maximum depth | Learning rate | Boosting rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy_63 | 204 | 63 | 8 | .01 | 640 |
| geometry_63 | 227 | 63 | 8 | .01 | 640 |
| full_63 | 1003 | 63 | 8 | .01 | 640 |
| full_127 | 1003 | 127 | 10 | .01 | 640 |

All four use minimum 120 rows per leaf, minimum Hessian .01, L1 .2, L2 10,
column fraction .7, row fraction .8 with bagging every iteration, 127 bins,
seed 2026 and one CPU thread. Each multiclass round produces three class trees,
so 640 rounds permits 1,920 trees. This is a bounded initial training budget;
it is not mathematically equivalent to 1,280 binary rounds, and does not claim
convergence or optimal capacity. The full training multi-logloss trajectory is
saved to diagnose whether the chosen budget may still be underfitting. No
validation-driven early stopping or extra arm is permitted in this study.

## New possession evidence

The original 204 columns already contain 137 team/ball geometry and contact
features plus 67 player-motion features. Nearest-player distances, instantaneous
closing rates, relative speed, nearest-player Gaussian control evidence and
centered means are therefore existing inputs.

The geometry arm adds 23 different measurements:

- **Control over all observed players.** Each team's best and second-best
  distance-and-relative-motion control scores, their separation and valid
  candidate fraction. A player who is closest geometrically need not have the
  strongest motion agreement with the ball. Cross-team strength, signed margin
  and ambiguity summarize whether control evidence favors one team.
- **Projected closest approach.** For each team, minimum predicted miss distance,
  time to that closest approach within two seconds, distance improvement and
  candidate coverage. These use relative position and velocity, rather than
  treating current closing speed alone as an interception prediction.
- **Collective stopping and restarting.** Median signed player acceleration,
  braking fraction, accelerating fraction and observation coverage distinguish
  synchronized slowing from restarting. Unknown observations stay unknown.

These are tracking-derived physical proxies, not possession labels. New
geometry uses native observation masks and valid native derivatives, rejects
implausible velocities/accelerations, is invariant to roster-slot order, and
never carries derivatives across an input discontinuity.

The full arm adds 14 reliable contact-anchor features. A candidate anchor needs
an observed ball, finite relative motion, best-control score at least .2 and
absolute normalized team margin at least .2, sustained for at least .2 seconds.
Strictly earlier and later anchors are searched within the same continuous
input block and capped at 10 seconds. Their team evidence, ages, validity,
agreement and change can inform a long ball gap without filling ball positions.
The thresholds are fixed physical heuristics, not selected from labels.

The remaining 762 features describe time order:

- For 31 curated primitive channels, a finite-value flag; earlier/later values
  and their difference at .4, 1 and 3 seconds; and separate past/future means,
  standard deviations, coverage and mean differences over 1 and 3 seconds.
  These contribute 744 columns. Already centered means are not blindly expanded.
- Team-level control persistence and observation coverage in past/future
  windows of 1, 3 and 5 seconds contribute 18 columns. These use team evidence,
  not persistent player-slot identity.

The actual manifest is **204 + 23 + 14 + 744 + 18 = 1,003 columns**. Missing
half-window means remain NaN and coverage records incomplete support. All
context resets at match, half and timestamp breaks. Every raw team swap occurs
before nonlinear engineering; the explicit feature manifest separately records
paired, signed and even transformations, including the fact that a signed
margin's standard deviation must not change sign.

## Training and evaluation protocol

All models use the existing whole-match provider holdouts. Sportec, Metrica or
SkillCorner is excluded from every learned component of the corresponding
LightGBM and GRU pipeline. The five evaluation matches remain previously
exposed development data; Metrica/SkillCorner targets include silver labels and
SkillCorner's visibility filter and X margins are preserved.

Training reuses the exact joint-GRU clean and coarse-ball feature records, with
its same 25% whole-core ball-gap replacement and direct/H-A-swapped copies.
Nonlinear features are computed separately on each complete observation view
before selecting replacement cores, so a splice cannot create artificial
before/after transitions. Geometry is recomputed only for the new raw features;
the existing 204 columns are reused from their verified caches.

To bound tree compute while preserving short transitions, a training-only
sampler retains the first and last three 5-Hz samples of every contiguous known
label run and samples its interior every fourth sample. Tiny runs are retained.
Within-run Voronoi weights represent the original known 5-Hz frames assigned to
each selected row; they preserve each class's pre-thinning mass exactly. Unknown
targets have zero mass. This does not use labels at inference. Provider, match
and observation view each retain their declared equal total weight.

| Excluded provider | Original known 5-Hz training rows across four copies | Retained rows | Original class mass |
| --- | ---: | ---: | --- |
| Sportec | 717784 | 265056 | Exactly preserved |
| Metrica | 831828 | 307396 | Exactly preserved |
| SkillCorner | 565556 | 186484 | Exactly preserved |

Disk-backed matrices bound memory and reuse the exact full-feature matrix for
the 63- and 127-leaf fits. The declared budget is 12 fits and 300 new evaluation
cells: four configurations × 15 match/view pairs × five recipes. The recipes are
standalone trees and fixed 50/50 full-vector blends with the control and
slot-dropout GRUs, each at seeds 2026 and 2027. The 60 GRU rows are reused. Separate
GRU seed scores are not an inference seed ensemble; tree seed variance is not
measured.

All 12 tree checkpoints and all 12 GRU checkpoints are sealed before tree
scoring. Shared clean/stress inputs, labels, clocks and native probabilities
are independently checked and hashed. The decoder stays at cost 3. Official
spell F1 is primary, macro F1 breaks exact ties, and the 70/30 summary is an
auxiliary column. Each recipe is screened against the matched full-provider
adopted recipe: clean spell gain at least .01, no provider clean D-F1/macro loss
larger than .02, and stress spell loss no larger than .01. Passing this reused
development screen does not automatically promote a model.

Diagnostics include each IoU's spell counts, short and censored-edge recall,
fragmentation/merging overlap geometry, complete-spell boundary timing, D
precision/recall, training feature gain and the training-loss trajectory.
Full multiclass trees can learn ball-source shortcuts; their D prediction is
not restricted to player-only evidence. Synthetic stress performance does not
establish independent broadcast robustness.

## Reproduction and artifacts

The frozen output is `data/experiments/joint_lgbm_v1`. Restore the verified
external arrays and completed shared GRU caches/checkpoints first. The new
runner and models are `joint_lgbm_experiment.py`, `joint_lgbm.py` and
`possession_geometry.py`.

```sh
python -m experiments.joint_lgbm_experiment freeze
python -m experiments.joint_lgbm_experiment prepare
for source in sportec metrica skillcorner; do
  for config in legacy_63 geometry_63 full_63 full_127; do
    python -m experiments.joint_lgbm_experiment fit --source "$source" --config "$config"
  done
done
python -m experiments.joint_lgbm_experiment seal
python -m experiments.joint_lgbm_experiment seal-inputs
python -m experiments.joint_lgbm_experiment evaluate
python -m experiments.joint_lgbm_experiment report
```

The 60 `source_match_view_config_probability.npz` files preserve standalone
native H/A/D probabilities, sampled probabilities and sample indices, making
future ensembles reproducible. All 300 saved predictions have array and file
hashes; combined probability hashes permit independent verification of every
full-vector blend. Models, matrices, tracking and frame outputs remain private.
Only aggregate reports are published. No organizer diagnostic is run.

Before fitting, all 19 geometry/full-class contracts and the 124-test repository
suite passed. Results, measured runtime, guardrails and remote artifact receipts
will be recorded when this frozen experiment completes.
