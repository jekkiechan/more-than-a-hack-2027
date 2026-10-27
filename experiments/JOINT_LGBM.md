# Full H A D LightGBM features and GRU ensembles

Completed on 2026-10-27: **12 full H/A/D LightGBM models and all 300 declared
standalone/GRU ensemble evaluations**. Ordered temporal features improve stress
performance, and the wider 127-leaf models improve over Full 63. The strongest
stress blend substantially repairs the GRU’s synthetic-stress loss, but every
recipe fails an adoption guardrail because of remaining D or stress failures.
**Retain the adopted baseline.** These trees predict all three classes directly;
their ensembles average complete H/A/D probability vectors before decoding.

## Matched results

All 300 declared tree/ensemble cells completed. Scores average matches within
provider and then weight the three providers equally. The evaluation cohort is
one Sportec match, one Metrica match and three SkillCorner matches; each has a
clean view and two fixed stress views. The [aggregate report](results/joint_lgbm_development.json)
contains every cell, declaration, guardrail and diagnostic. No organizer labels
were scored. These are reused external development matches, not confirmation data.

Rows marked “mean” average the separate 2026/2027 GRU pipeline scores. They do
not average the two GRUs at inference. Each actual blend is fixed 50/50 between
one full H/A/D tree model and one GRU. The 70/30 column is auxiliary; ranking uses
unrounded clean official spell F1, with macro F1 only for exact ties.

| Model / recipe | Clean spell | Stress spell | Clean macro | Clean 70/30 | Clean D precision | Clean D recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adopted recipe | 0.5338 | 0.4129 | 0.8300 | 0.6227 | 0.8939 | 0.7920 |
| GRU control mean | 0.6092 | 0.2991 | 0.8410 | 0.6788 | 0.8833 | 0.7950 |
| GRU slot-dropout mean | 0.6132 | 0.3141 | 0.8489 | 0.6839 | 0.8904 | 0.8070 |
| legacy_63 standalone | 0.5399 | 0.4085 | 0.7957 | 0.6166 | 0.9888 | 0.6527 |
| geometry_63 standalone | 0.5365 | 0.4055 | 0.7992 | 0.6153 | 0.9888 | 0.6632 |
| full_63 standalone | 0.5497 | 0.4285 | 0.7979 | 0.6241 | 0.9851 | 0.6500 |
| full_127 standalone | 0.5555 | 0.4345 | 0.7995 | 0.6287 | 0.9862 | 0.6529 |
| legacy_63 + control mean | 0.6220 | 0.3893 | 0.8469 | 0.6895 | 0.9564 | 0.7561 |
| legacy_63 + slot_dropout mean | 0.6119 | 0.3954 | 0.8537 | 0.6844 | 0.9617 | 0.7696 |
| geometry_63 + control mean | 0.6220 | 0.3858 | 0.8484 | 0.6899 | 0.9566 | 0.7600 |
| geometry_63 + slot_dropout mean | 0.6129 | 0.3942 | 0.8550 | 0.6856 | 0.9618 | 0.7734 |
| full_63 + control mean | 0.6095 | 0.4139 | 0.8421 | 0.6793 | 0.9532 | 0.7472 |
| full_63 + slot_dropout mean | 0.6042 | 0.4171 | 0.8487 | 0.6776 | 0.9620 | 0.7593 |
| full_127 + control mean | 0.6148 | 0.4177 | 0.8433 | 0.6834 | 0.9563 | 0.7470 |
| full_127 + slot_dropout mean | 0.6101 | 0.4226 | 0.8498 | 0.6820 | 0.9644 | 0.7592 |

### What the matched ablations establish

The 23 instantaneous geometry additions alone do not provide a consistent
benefit: standalone clean/stress spell F1 decreases by 0.0033/0.0030 versus
legacy features, and the blend changes are small and mixed. The full temporal
package matters more. At identical 63-leaf capacity, it improves standalone
clean/stress spell F1 by **0.0131/0.0230** versus geometry and improves stress
spell F1 in all four blends. Its blend clean scores decline, so this is a
robustness tradeoff rather than an unconditional feature win. Anchors, ordered
context and persistence were added together; this comparison cannot assign
the gain uniquely to one of those families.

Increasing to 127 leaves and depth 10 improves clean and stress spell F1 in
all five recipes versus Full 63. Standalone gains are **0.0058/0.0060**.
This supports the requested wider/deeper hypothesis within the measured
0.01-learning-rate, 640-round budget. Leaves and depth changed together; a
higher learning rate and 1,280-round multiclass models were not compared.

The full127 + slot-dropout GRU blend raises mean stress spell F1 from
**0.3141 to 0.4226** versus its corresponding GRU, while clean changes from
**0.6132 to 0.6101**. Both seeds improve stress substantially. This is useful
complementarity, but it still fails the clean SkillCorner D guardrail. The
numerical clean leader is geometry63 + control2026 at 0.6257; its stress score
is only 0.3718, and it fails both stress and provider-D requirements.

### Seed results and every guardrail

All 20 configurations/recipes fail at least one predeclared adoption guardrail.
The required deltas against the matched adopted recipe are clean spell ≥ +0.01,
stress spell ≥ −0.01, and worst provider clean D-F1/macro delta ≥ −0.02.
The last column below is the minimum provider D-F1 or macro delta; its sign
and magnitude matter even when aggregate spell F1 improves.

| Tree | Recipe / GRU seed | Clean spell | Stress spell | Clean macro | Clean 70/30 | Worst provider D/macro delta | Screen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| legacy_63 | standalone | 0.5399 | 0.4085 | 0.7957 | 0.6166 | -0.3454 | Fail |
| legacy_63 | control_2026 | 0.6247 | 0.3725 | 0.8513 | 0.6927 | -0.0619 | Fail |
| legacy_63 | control_2027 | 0.6192 | 0.4062 | 0.8426 | 0.6862 | -0.1136 | Fail |
| legacy_63 | slot_dropout_2026 | 0.6101 | 0.3910 | 0.8515 | 0.6825 | -0.0570 | Fail |
| legacy_63 | slot_dropout_2027 | 0.6137 | 0.3997 | 0.8559 | 0.6863 | -0.0219 | Fail |
| geometry_63 | standalone | 0.5365 | 0.4055 | 0.7992 | 0.6153 | -0.3249 | Fail |
| geometry_63 | control_2026 | 0.6257 | 0.3718 | 0.8533 | 0.6940 | -0.0556 | Fail |
| geometry_63 | control_2027 | 0.6184 | 0.3999 | 0.8435 | 0.6859 | -0.1105 | Fail |
| geometry_63 | slot_dropout_2026 | 0.6110 | 0.3919 | 0.8529 | 0.6835 | -0.0478 | Fail |
| geometry_63 | slot_dropout_2027 | 0.6149 | 0.3965 | 0.8571 | 0.6876 | -0.0158 | Fail |
| full_63 | standalone | 0.5497 | 0.4285 | 0.7979 | 0.6241 | -0.3729 | Fail |
| full_63 | control_2026 | 0.6163 | 0.4055 | 0.8464 | 0.6853 | -0.0859 | Fail |
| full_63 | control_2027 | 0.6026 | 0.4223 | 0.8379 | 0.6732 | -0.1304 | Fail |
| full_63 | slot_dropout_2026 | 0.6067 | 0.4128 | 0.8457 | 0.6784 | -0.0733 | Fail |
| full_63 | slot_dropout_2027 | 0.6018 | 0.4213 | 0.8516 | 0.6767 | -0.0380 | Fail |
| full_127 | standalone | 0.5555 | 0.4345 | 0.7995 | 0.6287 | -0.3687 | Fail |
| full_127 | control_2026 | 0.6225 | 0.4072 | 0.8474 | 0.6900 | -0.0859 | Fail |
| full_127 | control_2027 | 0.6071 | 0.4283 | 0.8392 | 0.6767 | -0.1301 | Fail |
| full_127 | slot_dropout_2026 | 0.6088 | 0.4164 | 0.8462 | 0.6800 | -0.0714 | Fail |
| full_127 | slot_dropout_2027 | 0.6113 | 0.4289 | 0.8534 | 0.6840 | -0.0421 | Fail |

| GRU alone | Clean spell | Stress spell | Clean macro | Clean 70/30 |
| --- | ---: | ---: | ---: | ---: |
| control_2026 | 0.6083 | 0.2767 | 0.8408 | 0.6780 |
| control_2027 | 0.6102 | 0.3215 | 0.8413 | 0.6795 |
| slot_dropout_2026 | 0.6088 | 0.3180 | 0.8434 | 0.6792 |
| slot_dropout_2027 | 0.6176 | 0.3101 | 0.8543 | 0.6886 |

Two GRU seeds expose sensitivity but are not a confidence interval. The trees
use one fixed seed, and their seed variance is unmeasured.

### The main remaining gap: dead-play recall

The SkillCorner result demonstrates why spell F1 alone cannot certify a useful
H/A/D model. Full127 standalone improves its provider spell F1 from the adopted
recipe’s **0.4225 to 0.4322**, while D-F1 collapses from **0.7022 to 0.3335**
and D recall from **0.5909 to 0.2035**. Its D precision is 0.9794: it recognizes
few stoppages, but those few are usually correct. D is not a separately matched
possession-spell class in the official objective; the [metric audit](SPELL_METRIC_AUDIT.md)
explains this gap. Full multiclass trees have access to ball evidence, so they
do not inherit the GRU’s player-only D restriction.

| Provider | Recipe | Clean spell | Clean macro | D F1 | D precision | D recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sportec | Adopted recipe | 0.5332 | 0.8500 | 0.9098 | 0.9297 | 0.8907 |
| sportec | Full127 standalone | 0.5466 | 0.8840 | 0.9565 | 0.9835 | 0.9310 |
| sportec | Full127 + slot GRU mean | 0.6060 | 0.8888 | 0.9451 | 0.9638 | 0.9274 |
| metrica | Adopted recipe | 0.6457 | 0.8912 | 0.8893 | 0.8840 | 0.8946 |
| metrica | Full127 standalone | 0.6875 | 0.9046 | 0.9020 | 0.9957 | 0.8243 |
| metrica | Full127 + slot GRU mean | 0.7644 | 0.9256 | 0.9128 | 0.9688 | 0.8630 |
| skillcorner | Adopted recipe | 0.4225 | 0.7488 | 0.7022 | 0.8680 | 0.5909 |
| skillcorner | Full127 standalone | 0.4322 | 0.6099 | 0.3335 | 0.9794 | 0.2035 |
| skillcorner | Full127 + slot GRU mean | 0.4598 | 0.7351 | 0.6455 | 0.9605 | 0.4872 |

The stronger robustness blend still has SkillCorner D-F1 0.6308/0.6602 for
seeds 2026/2027, below the 0.7022 baseline by 0.0714/0.0421. Both exceed the
allowed 0.02 loss. Averaging the seeds cannot erase either failed guardrail.

### Spell diagnostics

Adding the full feature package to the standalone 63-leaf tree raises clean
IoU-0.7 spell F1 from 0.4009 to 0.4250; wider trees raise it to 0.4315.
However, short-spell recall at IoU 0.5 falls from 0.2791 with geometry alone
to 0.2547 with Full 63, recovering only to 0.2675 with Full 127. Better strict
overlap does not mean the short-possession problem is solved.

The full127 + slot-GRU blend has mean clean IoU-0.7 F1 0.5001 versus 0.4917
for its GRU alone, but short-spell recall falls from 0.3356 to 0.3183. Across
the five clean matches, 1,963 reference spells are graded. The blend predicts
2,033/1,997 spells versus 1,968/1,854 for its corresponding GRU seeds; these
are counts, not pooled-score rankings or counts of false positives.

All duration recall diagnostics retain the official global greedy matching.
Fragment/merge diagnostics describe overlapping geometry, not mutually exclusive
causes. Complete-boundary timing excludes censored graded-run edges; SkillCorner
visibility and silver labels limit what these aggregate gains establish.

### Next research decision

Retain the adopted baseline. Keep the full H/A/D probability artifacts for
ensembling research, and prioritize restoring cross-provider D recall while
preserving the temporal bundle’s stress benefit. A future declared complete
H/A/D tree model could restrict its D component to player evidence while using
rich contact/context evidence for team attribution. Alternatively, transfer the
curated contact-memory features into the existing constrained GRU. Either needs
a matched test on additional held-out matches. More leaves or rounds alone
do not address the demonstrated D failure; 1,280 rounds remain a separate
unmeasured compute hypothesis.

## Frozen comparison design

The first three arms isolate feature engineering at identical capacity. The
fourth tests larger trees on exactly the same complete feature set.

| Arm | Features | Leaves | Maximum depth | Learning rate | Boosting rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy_63 | 204 | 63 | 8 | .01 | 640 |
| geometry_63 | 227 | 63 | 8 | .01 | 640 |
| full_63 | 1003 | 63 | 8 | .01 | 640 |
| full_127 | 1003 | 127 | 10 | .01 | 640 |

All four set `min_child_samples=120`, minimum Hessian .01, L1 .2, L2 10,
column fraction .7, row fraction .8 with bagging every iteration, 127 bins,
seed 2026 and one CPU thread. Each multiclass round produces three class trees,
so 640 rounds permits 1,920 trees. This is a bounded initial training budget;
it is not mathematically equivalent to 1,280 binary rounds, and does not claim
convergence or optimal capacity. The full training multi-logloss trajectory is
saved to diagnose whether the chosen budget may still be underfitting. No
validation-driven early stopping or extra arm is permitted in this study.
The [capacity rationale and structural audit](LGBM_CAPACITY_REASONING.md) explain
why feature count alone cannot determine a useful leaf or depth limit.

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

For player $i$, let $r_i$ be ball position minus player position, $v_i$
be ball velocity minus player velocity, and $d_i=\lVert r_i\rVert$.
The control proxy is

$$c_i=\exp[-\tfrac12(d_i/1.8)^2]\exp[-\tfrac12(\lVert v_i\rVert/5)^2].$$

The team score takes the maximum over every valid observed player. Its
normalized home margin is $(c_H-c_A)/(c_H+c_A+0.001)$ and ambiguity is
$2\min(c_H,c_A)/(c_H+c_A+0.001)$. Both teams must have valid evidence;
a missing opponent does not imply uncontested control.

For valid relative speed above 0.5 m/s, closest approach uses
$\tau_i=\operatorname{clip}(-r_i\cdot v_i/\lVert v_i\rVert^2,0,2)$ and miss
distance $\lVert r_i+\tau_i v_i\rVert$. Each team contributes the minimum
miss distance, its time and improvement over current distance. Deterministic
tie handling preserves roster permutation invariance. This constant-velocity
projection is a short-horizon cue, not a simulated interception outcome.

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

## Feature coverage audit

A [read-only audit of the prepared training matrices](results/joint_lgbm_feature_coverage.json)
found no entirely missing feature in any fold. There were 6, 13 and 6 columns
constant wherever observed in the Metrica-, SkillCorner- and Sportec-excluded
folds respectively. Weighted instantaneous home control availability was
68.6%, 57.2% and 54.2%; prior reliable-contact evidence was available on
83.3%, 70.0% and 77.0%, and future evidence on 82.4%, 69.8% and 76.1%.
Ordered context means had weighted finite coverage of 87.6%, 82.5% and 81.0%.
These checks establish that the engineered inputs are populated and that
contact anchors extend evidence through observation gaps. They do not establish
predictive accuracy or causal feature importance.

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

## Compute cost and learned use of the features

All 12 models completed their fixed 640 rounds and contain 1,920 class trees.
Summed per-fit elapsed time was 7.00 hours; up to four one-thread workers ran
concurrently, and the complete fit phase took about 2 hours 10 minutes. Receipts
include matrix preparation/checking where applicable, so the following times
are operational costs rather than isolated algorithm benchmarks.

| Arm | Metrica excluded, minutes | Sportec excluded, minutes | SkillCorner excluded, minutes |
| --- | ---: | ---: | ---: |
| Legacy 63 | 17.9 | 9.0 | 9.0 |
| Geometry 63 | 18.2 | 13.8 | 8.9 |
| Full 63 | 64.7 | 52.8 | 31.0 |
| Full 127 | 76.7 | 72.2 | 45.8 |

Training multi-logloss continued declining over the final 100 rounds in every
fit, by 0.0108–0.0153. Full 63 ended at 0.1144–0.2257 and Full 127 at
0.0893–0.1856 across folds. Wider trees fit the training labels more closely;
this alone does not establish a held-out benefit or prove that additional
rounds would improve spell F1. A 1,280-round full H/A/D budget remains unmeasured.

The [independent tree-structure audit](results/joint_lgbm_capacity_audit.json)
confirmed that every tree reaches its depth cap. Additional capacity is used,
but the five-training-match SkillCorner holdout has thinner support: its wider
model's median leaf contains 253 rows and 24.3% of leaves contain at most 150.
Only 21.6% of that fold's D trees reach 127 leaves, compared with about 87% of
H/A trees. LightGBM's `min_child_samples` is an approximate size constraint:
1.29–12.50% of leaves across these models have fewer than 120 actual rows.
These are correlated training-row counts, not independent examples;
capacity needs validation separately from whether a configured limit binds.

The following shares normalize split gain within each model, then average the
three provider folds equally. They describe what training splits used, not
held-out causal importance; correlated features can substitute for one another.

| Arm | Legacy | New instantaneous geometry | Reliable anchors | Ordered context | Control persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full 63 | 41.0% | 0.2% | 17.7% | 32.7% | 8.4% |
| Full 127 | 40.5% | 0.2% | 17.6% | 34.2% | 7.6% |

The trees materially use contact memory and ordered temporal evidence. The
small direct geometry share does not mean geometry is unused: anchors and
persistence are themselves derived from its control evidence. The declared
feature ablation measures the complete added packages; it does not isolate
anchors from ordered context individually.

## Reproduction and artifacts

The frozen output is `data/experiments/joint_lgbm_v1`. Restore the verified
external arrays and completed shared GRU caches/checkpoints first. The new
runner and models are `joint_lgbm_experiment.py`, `joint_lgbm.py` and
`possession_geometry.py`.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
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
suite passed. All 12 checkpoints, 60 standalone probability artifacts and 300
predictions are complete and sealed. The [independent verification receipt](results/temporal_followup_verification.json)
passed checks of all 300 full-vector probability blends, 15 representative
decodes, exact cohorts/summaries/guardrails and the model/input seals. Together
with the GRU study, it covers 24 models and 360 new evaluation rows. Private
artifact provenance is recorded in the [follow-up artifact receipt](results/temporal_followup_artifacts.json).
The complete 738-file archive passes local size/SHA256 checks, but its upload
is blocked by automatic approval review pending explicit user authorization
for the private destination and payload. Earlier GRU and partial tree uploads
were remotely verified; the complete archive is not yet remotely restorable.
The adopted baseline manifest remains unchanged.
