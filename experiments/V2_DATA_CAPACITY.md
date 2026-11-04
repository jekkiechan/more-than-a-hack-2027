# V2 data and capacity study

Status: complete; both independent audits passed. Training ran on two CPU
workers from 4 November 2026 at 12:04 UTC. No promotion decision was made.

Date: 4 November 2026. The active package remains
`joint_gru_sn_dense_original_diversity_phase_ensemble_v1`. The eligible PR82
v2 spell target is **0.5927491913246999**. The previous weight-edit readout
reached 0.594110 v2 spell on the exposed development cohort, but its result
does not replace the adopted package or waive any of this study's clean/v1/v2
retention checks. The frozen training protocol is on source commit `98ba97c`.
Its declarations are [training](results/v2_data_capacity_v1_declaration.json)
and [evaluation](results/v2_data_capacity_eval_declaration.json); execution
amendments are the [execution wrapper](results/v2_data_capacity_eval_execution_amendment.json),
[recovery wrapper](results/v2_data_capacity_eval_recovery_v2_amendment.json) and
[v3 receipt wrapper](results/v2_data_capacity_eval_execution_v3_amendment.json).

## Question and fixed training recipes

Can added recurrent capacity or a balanced mix of the existing v1 and v2
training views match or exceed the eligible PR82 v2 spell score while avoiding
any aggregate regression against the adopted package on clean, v1 and v2 spell,
macro, D-F1, D precision, D recall and short-spell recall?

The comparison has three recipes:

| Recipe | Width | Training views |
| --- | ---: | --- |
| `width96_control` | 96 | Frozen PR82 four-view v2 recipe |
| `width192_capacity` | 192 | Same PR82 recipe and training budget |
| `mixed_v1_v2_width96` | 96 | Equal v1/v2 family mass; four v2 realizations represented evenly |

Each recipe has six provider-held-out fits: two fixed seeds (2026 and 2027) for
each of the Metrica, SkillCorner and Sportec exclusions, for 18 fits total.
Every evaluated provider is excluded from model fitting, its teacher and
normalization. The eligible pool contains 13 original-provider TRAIN matches
and 45 GAR TRAIN games with generated silver labels. The original/GAR training
mixture is fixed at 50/50. Each fit uses batch size 32 and 2,048 updates, with
the PR82 sampler, training targets and other optimizer settings held fixed.
No new gold labels are added.

The trusted teacher metadata has 408 input features across 204 feature names.
The width-96 control is intended to isolate the shared trainer adapter from
the frozen width-96 implementation by exact two-step synthetic replay of its
loss trace and model state. The conditional H/A recurrent component has
291,842 parameters at width 96 and 694,274 at width 192 (about 2.38 times as
many); the D-head architecture is held fixed. Width 192 changes H/A capacity.
The mixed recipe changes training-view family mass while keeping width and the
rest of the recipe fixed. The v1 and v2 families are synthetic views of the
same eligible training matches; this comparison does not add independent
matches. The budget matches optimizer updates, not floating-point operations,
so the wider arm costs more compute; actual fit-time ranges appear below.

## Matched full-pipeline evaluation

The cohort is the same five previously exposed development matches, evaluated
on 35 cells: five clean, ten fixed v1 (two realizations per match) and twenty
fixed v2 (four realizations per match). All native phases are included: five
phases for 25 Hz matches and two for 10 Hz matches. Each model's candidate
probabilities are averaged over the two fixed seeds within each native phase,
then native phases are averaged before decoding. Metrics use the official
possession evaluator. Aggregate order is realizations within match, matches
within provider and providers equally.

For each trained recipe, the fixed pipeline comparisons are:

1. The new model alone with H/A transition cost 3.
2. A fixed 50/50 blend with the archived calibrated model, decoded at H/A cost
   1.5 (the PR82 comparison).
3. A fixed 50/50 blend with the adopted model, decoded at H/A cost 3.
4. New conditional H/A predictions at cost 3 under the frozen adopted D mask.

The fourth arm has a matched archived adopted H/A-with-the-same-D-mask control,
which isolates the conditional H/A decoder change. That archived control is a
diagnostic reference, not a fourth newly trained candidate. The evaluator
also replays archived adopted and calibrated controls before candidate
scoring; the adopted replay must match saved results within the predeclared
1e-8 tolerance. The evaluator declaration binds inputs, models, source,
baseline manifest and policy hashes before replay and candidate scoring. The
controls-only replay has completed on all 35 cells and passed all 24 declared
official aggregate checks, with a maximum absolute difference of
2.220446049250313e-16. This replay uses saved, sealed
probabilities and official metric recomputation; it did not refit models or
score candidates.

The hard success rule has zero tolerance: for each of clean, v1 and v2, all six
quality metrics—spell F1, macro F1, D-F1, D precision, D recall and short-spell
recall at IoU 0.5 for spells under three seconds—must be greater than or equal
to the matched adopted control. V2 spell F1 must also be at least
**0.5927491913246999**. Equality passes; any lower value fails. No rounded
value is used for the decision. Reference spell counts must remain invariant;
predicted spell counts are reported as diagnostics. Report provider results and
every failed aggregate guard. Provider diagnostics do not introduce an
unregistered provider-level tolerance.

## Results and decision — 4 November 2026

**No upgrade.** The frozen 12 candidate arms produced **0/12 arms passing all 19 strict checks** (six quality metrics in each of clean, v1 and v2, plus the exact PR82 v2 spell floor). The width-96 PR82-style blend reproduces the eligible target exactly at **0.5927491913246999** v2 spell F1, but fails the adopted clean and v1 spell floors and additional v1 guards. Width 192 scores 0.592533455 v2 spell (−0.000215736 versus its matched width-96 recipe); the v1/v2 mix scores 0.591496403 (−0.001252788). Neither reaches the PR82 floor. No default or manifest changes.

The tables report equal-provider means. Each metric tuple is ordered spell F1 / macro F1 / D-F1 / D precision / D recall / short-spell recall (IoU 0.5, under 3 seconds). Values are rounded to six decimals for display; gates use saved unrounded values. `70/30` is the fixed descriptive value `0.7 × spell + 0.3 × macro`, not the ranking metric.

### clean

| Arm | Spell / macro / D-F1 / D-precision / D-recall / short recall | 70/30 |
| --- | --- | ---: |
| Adopted original_full | 0.667556 / 0.871271 / 0.870926 / 0.884844 / 0.861920 / 0.406351 | 0.728670 |
| `width96_control/pr82_blend_ha1_5` | 0.665355 / 0.872661 / 0.873282 / 0.885562 / 0.865543 / 0.414599 | 0.727547 |
| `width96_control/adopted_blend_ha3` | 0.666418 / 0.871161 / 0.870513 / 0.878770 / 0.866512 / 0.403753 | 0.727841 |
| `width96_control/standalone_ha3` | 0.658713 / 0.869752 / 0.869869 / 0.874274 / 0.869340 / 0.397005 | 0.722024 |
| `width96_control/adopted_dead_new_ha3` | 0.663647 / 0.870809 / 0.870926 / 0.884844 / 0.861920 / 0.403306 | 0.725795 |
| `width192_capacity/pr82_blend_ha1_5` | 0.664877 / 0.872135 / 0.872916 / 0.884918 / 0.865333 / 0.426684 | 0.727054 |
| `width192_capacity/adopted_blend_ha3` | 0.663617 / 0.870947 / 0.870602 / 0.879021 / 0.866422 / 0.397960 | 0.725816 |
| `width192_capacity/standalone_ha3` | 0.660439 / 0.869132 / 0.869350 / 0.873159 / 0.869387 / 0.406256 | 0.723047 |
| `width192_capacity/adopted_dead_new_ha3` | 0.664566 / 0.870302 / 0.870926 / 0.884844 / 0.861920 / 0.409384 | 0.726286 |
| `mixed_v1_v2_width96/pr82_blend_ha1_5` | 0.664859 / 0.873055 / 0.873749 / 0.887544 / 0.864574 / 0.411931 | 0.727318 |
| `mixed_v1_v2_width96/adopted_blend_ha3` | 0.667526 / 0.871345 / 0.870751 / 0.881642 / 0.864300 / 0.401777 | 0.728672 |
| `mixed_v1_v2_width96/standalone_ha3` | 0.663835 / 0.870689 / 0.870200 / 0.878284 / 0.865939 / 0.399874 | 0.725891 |
| `mixed_v1_v2_width96/adopted_dead_new_ha3` | 0.666433 / 0.871162 / 0.870926 / 0.884844 / 0.861920 / 0.405693 | 0.727852 |

### feed_like_v1

| Arm | Spell / macro / D-F1 / D-precision / D-recall / short recall | 70/30 |
| --- | --- | ---: |
| Adopted original_full | 0.531576 / 0.826853 / 0.860563 / 0.843006 / 0.881217 / 0.220289 | 0.620159 |
| `width96_control/pr82_blend_ha1_5` | 0.518637 / 0.823302 / 0.860769 / 0.841770 / 0.882945 / 0.226049 | 0.610036 |
| `width96_control/adopted_blend_ha3` | 0.527070 / 0.826242 / 0.860522 / 0.837749 / 0.887000 / 0.219061 | 0.616822 |
| `width96_control/standalone_ha3` | 0.521399 / 0.821115 / 0.858934 / 0.831403 / 0.890574 / 0.215829 | 0.611314 |
| `width96_control/adopted_dead_new_ha3` | 0.523939 / 0.822897 / 0.860563 / 0.843006 / 0.881217 / 0.216427 | 0.613626 |
| `width192_capacity/pr82_blend_ha1_5` | 0.525421 / 0.823866 / 0.860657 / 0.841189 / 0.883354 / 0.233145 | 0.614955 |
| `width192_capacity/adopted_blend_ha3` | 0.529979 / 0.826213 / 0.860569 / 0.837606 / 0.887179 / 0.223648 | 0.618849 |
| `width192_capacity/standalone_ha3` | 0.522531 / 0.820653 / 0.858776 / 0.831480 / 0.890047 / 0.223899 | 0.611967 |
| `width192_capacity/adopted_dead_new_ha3` | 0.523978 / 0.821918 / 0.860563 / 0.843006 / 0.881217 / 0.222219 | 0.613360 |
| `mixed_v1_v2_width96/pr82_blend_ha1_5` | 0.523583 / 0.825423 / 0.861150 / 0.844466 / 0.880818 / 0.227005 | 0.614135 |
| `mixed_v1_v2_width96/adopted_blend_ha3` | 0.530668 / 0.827329 / 0.861000 / 0.840937 / 0.884302 / 0.215222 | 0.619666 |
| `mixed_v1_v2_width96/standalone_ha3` | 0.527105 / 0.825780 / 0.861120 / 0.838656 / 0.886863 / 0.212930 | 0.616707 |
| `mixed_v1_v2_width96/adopted_dead_new_ha3` | 0.528884 / 0.825814 / 0.860563 / 0.843006 / 0.881217 / 0.216069 | 0.617963 |

### feed_like_v2

| Arm | Spell / macro / D-F1 / D-precision / D-recall / short recall | 70/30 |
| --- | --- | ---: |
| Adopted original_full | 0.583571 / 0.849975 / 0.866138 / 0.873034 / 0.862980 / 0.284153 | 0.663492 |
| `width96_control/pr82_blend_ha1_5` | 0.592749 / 0.854043 / 0.868707 / 0.873520 / 0.867179 / 0.295024 | 0.671137 |
| `width96_control/adopted_blend_ha3` | 0.589337 / 0.851604 / 0.867361 / 0.867379 / 0.870846 / 0.285615 | 0.668017 |
| `width96_control/standalone_ha3` | 0.590072 / 0.852075 / 0.867232 / 0.862076 / 0.875652 / 0.284361 | 0.668673 |
| `width96_control/adopted_dead_new_ha3` | 0.590141 / 0.851413 / 0.866138 / 0.873034 / 0.862980 / 0.289240 | 0.668523 |
| `width192_capacity/pr82_blend_ha1_5` | 0.592533 / 0.853827 / 0.868594 / 0.873937 / 0.866697 / 0.293171 | 0.670922 |
| `width192_capacity/adopted_blend_ha3` | 0.589113 / 0.851335 / 0.866823 / 0.866715 / 0.870321 / 0.287247 | 0.667779 |
| `width192_capacity/standalone_ha3` | 0.587105 / 0.851759 / 0.866837 / 0.862669 / 0.874297 / 0.280236 | 0.666502 |
| `width192_capacity/adopted_dead_new_ha3` | 0.586186 / 0.851065 / 0.866138 / 0.873034 / 0.862980 / 0.285111 | 0.665650 |
| `mixed_v1_v2_width96/pr82_blend_ha1_5` | 0.591496 / 0.854169 / 0.868842 / 0.876079 / 0.865086 / 0.293279 | 0.670298 |
| `mixed_v1_v2_width96/adopted_blend_ha3` | 0.586693 / 0.850738 / 0.866638 / 0.869532 / 0.867293 / 0.283417 | 0.665907 |
| `mixed_v1_v2_width96/standalone_ha3` | 0.589051 / 0.851950 / 0.867762 / 0.867949 / 0.870958 / 0.285747 | 0.667920 |
| `mixed_v1_v2_width96/adopted_dead_new_ha3` | 0.588519 / 0.851010 / 0.866138 / 0.873034 / 0.862980 / 0.289535 | 0.667266 |

### Same-pipeline changes versus the width-96 control

Spell-F1 deltas below compare each capacity/data recipe with the width-96 control using the same decoder and blend. They are paired on the same 35 cells and do not change the strict adopted-package gate.

| Recipe | Pipeline | Clean Δ | V1 Δ | V2 Δ |
| --- | --- | ---: | ---: | ---: |
| width192_capacity | pr82_blend_ha1_5 | -0.000478 | +0.006784 | -0.000216 |
| width192_capacity | adopted_blend_ha3 | -0.002801 | +0.002909 | -0.000225 |
| width192_capacity | standalone_ha3 | +0.001726 | +0.001131 | -0.002967 |
| width192_capacity | adopted_dead_new_ha3 | +0.000919 | +0.000039 | -0.003955 |
| mixed_v1_v2_width96 | pr82_blend_ha1_5 | -0.000496 | +0.004946 | -0.001253 |
| mixed_v1_v2_width96 | adopted_blend_ha3 | +0.001108 | +0.003598 | -0.002644 |
| mixed_v1_v2_width96 | standalone_ha3 | +0.005122 | +0.005705 | -0.001022 |
| mixed_v1_v2_width96 | adopted_dead_new_ha3 | +0.002786 | +0.004946 | -0.001623 |

The four fixed pipeline arms trade outcomes across views. The new-H/A arm under the adopted D mask retains the adopted D-F1, D precision and D recall exactly because the D mask is frozen. It still changes H/A spell, macro and short-spell results and does not meet the joint gates. The archived adopted-H/A-with-the-same-D-mask control reproduces the adopted H/A result under that decoder, confirming the decoder comparison is isolated.

### Provider spell-F1 deltas versus adopted

These are per-provider spell deltas for every candidate arm, ordered Metrica / SkillCorner / Sportec within each cell. They show the development-cohort provider spread; the complete per-provider six-metric aggregates remain in the result JSON.

| Arm | Clean Δ spell | V1 Δ spell | V2 Δ spell |
| --- | --- | --- | --- |
| `width96_control/pr82_blend_ha1_5` | -0.0122 / +0.0038 / +0.0018 | -0.0106 / -0.0078 / -0.0205 | +0.0223 / -0.0014 / +0.0067 |
| `width96_control/adopted_blend_ha3` | -0.0053 / -0.0011 / +0.0031 | -0.0040 / -0.0017 / -0.0079 | +0.0108 / -0.0020 / +0.0085 |
| `width96_control/standalone_ha3` | -0.0182 / -0.0016 / -0.0068 | -0.0128 / -0.0047 / -0.0130 | +0.0135 / -0.0045 / +0.0105 |
| `width96_control/adopted_dead_new_ha3` | -0.0084 / +0.0007 / -0.0040 | -0.0031 / -0.0029 / -0.0169 | +0.0158 / -0.0043 / +0.0082 |
| `width192_capacity/pr82_blend_ha1_5` | -0.0107 / +0.0013 / +0.0014 | +0.0056 / -0.0063 / -0.0177 | +0.0206 / -0.0029 / +0.0092 |
| `width192_capacity/adopted_blend_ha3` | -0.0048 / -0.0044 / -0.0026 | +0.0044 / +0.0000 / -0.0092 | +0.0099 / -0.0042 / +0.0109 |
| `width192_capacity/standalone_ha3` | -0.0171 / -0.0031 / -0.0011 | -0.0057 / -0.0033 / -0.0181 | +0.0125 / -0.0088 / +0.0068 |
| `width192_capacity/adopted_dead_new_ha3` | -0.0068 / +0.0000 / -0.0022 | +0.0040 / -0.0035 / -0.0233 | +0.0135 / -0.0080 / +0.0023 |
| `mixed_v1_v2_width96/pr82_blend_ha1_5` | -0.0138 / +0.0058 / -0.0000 | -0.0066 / -0.0055 / -0.0119 | +0.0178 / -0.0011 / +0.0071 |
| `mixed_v1_v2_width96/adopted_blend_ha3` | -0.0016 / -0.0004 / +0.0019 | +0.0011 / -0.0010 / -0.0029 | +0.0066 / -0.0015 / +0.0043 |
| `mixed_v1_v2_width96/standalone_ha3` | -0.0076 / +0.0010 / -0.0045 | -0.0044 / -0.0024 / -0.0066 | +0.0112 / -0.0010 / +0.0063 |
| `mixed_v1_v2_width96/adopted_dead_new_ha3` | -0.0017 / +0.0028 / -0.0045 | +0.0012 / -0.0001 / -0.0092 | +0.0121 / -0.0016 / +0.0043 |

The width-96 PR82 blend's v2 spell gains over adopted come from Metrica and Sportec (+0.0223 and +0.0067); SkillCorner is slightly lower (−0.0014). On v1, all three providers regress. Across arms, provider gains are mixed; no aggregate win removes those within-provider tradeoffs.

### Strict guard outcomes and seed diagnostics

| Arm | Passed / 19 | Failed checks |
| --- | ---: | ---: |
| `width96_control/pr82_blend_ha1_5` | 15/19 | 4 |
| `width96_control/adopted_blend_ha3` | 7/19 | 12 |
| `width96_control/standalone_ha3` | 7/19 | 12 |
| `width96_control/adopted_dead_new_ha3` | 12/19 | 7 |
| `width192_capacity/pr82_blend_ha1_5` | 14/19 | 5 |
| `width192_capacity/adopted_blend_ha3` | 9/19 | 10 |
| `width192_capacity/standalone_ha3` | 7/19 | 12 |
| `width192_capacity/adopted_dead_new_ha3` | 14/19 | 5 |
| `mixed_v1_v2_width96/pr82_blend_ha1_5` | 14/19 | 5 |
| `mixed_v1_v2_width96/adopted_blend_ha3` | 9/19 | 10 |
| `mixed_v1_v2_width96/standalone_ha3` | 8/19 | 11 |
| `mixed_v1_v2_width96/adopted_dead_new_ha3` | 12/19 | 7 |

Every arm fails at least four checks. The PR82-target misses are the most direct capacity/data-mix failures: width 192 and mixed v1/v2 fall short of the exact v2 threshold, while width 96 reaches it but fails clean spell, v1 spell, v1 macro and v1 D precision. Other failed checks include the zero-tolerance v1/clean quality floors; no post-result threshold changes were made.

Component-seed diagnostics are the new standalone model's fixed-seed results, not the probability-ensemble scores in the tables:

| Recipe | Seed | Clean spell | V1 spell | V2 spell |
| --- | ---: | ---: | ---: | ---: |
| width96_control | 2026 | 0.652388 | 0.515325 | 0.584968 |
| width96_control | 2027 | 0.658551 | 0.511432 | 0.585065 |
| width192_capacity | 2026 | 0.649449 | 0.509858 | 0.577999 |
| width192_capacity | 2027 | 0.660843 | 0.516304 | 0.586419 |
| mixed_v1_v2_width96 | 2026 | 0.655868 | 0.515253 | 0.581508 |
| mixed_v1_v2_width96 | 2027 | 0.657151 | 0.521197 | 0.585698 |

### Separately declared uniform-ensemble follow-up — 4 November 2026

The original 12-arm study remains complete and negative. A final bounded follow-up under `v2_capacity_ensemble_v1` asked whether a fixed uniform mean across the three already-trained recipe experts improves the result. It averages the three saved experts equally within each of the four previously declared pipeline types. There were no new fits, inference, seeds, model weights or decoder choices. The new declaration bound the original audit and all 140 input caches before these four outputs were scored. This is additional development on the same 35 cells from five exposed matches, not fresh confirmation.

The follow-up also fails: **0/4 arms pass all 19 strict checks**. Its best v2 spell is the uniform calibrated blend at **0.592584638**, which is **0.000164553 below** the exact PR82 threshold. It remains below adopted clean spell, and every follow-up arm has clean-spell, v1-spell and v1-macro regressions versus adopted. Across both stages, **0/16 candidate pipelines qualify as an upgrade**. The full metric tuples use the same order and 70/30 definition as the original tables.

| Follow-up arm | Clean: spell / macro / D-F1 / D-precision / D-recall / short recall | 70/30 | V1: spell / macro / D-F1 / D-precision / D-recall / short recall | 70/30 | V2: spell / macro / D-F1 / D-precision / D-recall / short recall | 70/30 | Checks passed |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| `uniform_standalone_ha3` | 0.661013 / 0.870031 / 0.870199 / 0.876042 / 0.868350 / 0.402802 | 0.723718 | 0.523809 / 0.823127 / 0.860160 / 0.834412 / 0.889695 / 0.217997 | 0.613605 | 0.590815 / 0.851984 / 0.867129 / 0.863725 / 0.873808 / 0.287631 | 0.669165 | 7/19 |
| `uniform_pr82_blend_ha1_5` | 0.666629 / 0.873083 / 0.873731 / 0.886540 / 0.865453 / 0.418678 | 0.728565 | 0.523551 / 0.824093 / 0.860664 / 0.842196 / 0.882272 / 0.232298 | 0.613714 | 0.592585 / 0.854025 / 0.868574 / 0.874095 / 0.866428 / 0.293284 | 0.671017 | 14/19 |
| `uniform_adopted_blend_ha3` | 0.665730 / 0.871455 / 0.870659 / 0.880707 / 0.864938 / 0.403495 | 0.727447 | 0.528934 / 0.826490 / 0.860852 / 0.838664 / 0.886620 / 0.218820 | 0.618201 | 0.587409 / 0.851078 / 0.867088 / 0.867782 / 0.869887 / 0.284398 | 0.666510 | 9/19 |
| `uniform_adopted_dead_new_ha3` | 0.664078 / 0.870737 / 0.870926 / 0.884844 / 0.861920 / 0.407646 | 0.726075 | 0.526251 / 0.824372 / 0.860563 / 0.843006 / 0.881217 / 0.216557 | 0.615688 | 0.590482 / 0.851527 / 0.866138 / 0.873034 / 0.862980 / 0.290813 | 0.668796 | 13/19 |

Provider spell-F1 deltas versus adopted, in Metrica / SkillCorner / Sportec order:

| Follow-up arm | Clean Δ | V1 Δ | V2 Δ |
| --- | --- | --- | --- |
| `uniform_standalone_ha3` | -0.0133 / -0.0011 / -0.0052 | -0.0087 / -0.0013 / -0.0133 | +0.0149 / -0.0040 / +0.0108 |
| `uniform_pr82_blend_ha1_5` | -0.0101 / +0.0050 / +0.0023 | +0.0004 / -0.0063 / -0.0182 | +0.0216 / -0.0019 / +0.0073 |
| `uniform_adopted_blend_ha3` | -0.0048 / -0.0024 / +0.0017 | -0.0001 / -0.0015 / -0.0063 | +0.0070 / -0.0023 / +0.0068 |
| `uniform_adopted_dead_new_ha3` | -0.0046 / +0.0003 / -0.0061 | +0.0006 / +0.0002 / -0.0168 | +0.0164 / -0.0033 / +0.0076 |

The PR82-style uniform blend improves v2 spell for Metrica and Sportec (+.0216 and +.0073) while SkillCorner declines (−.0019), and v1 spell falls for all three providers. Provider spread therefore does not change the strict aggregate decision. No new per-seed ensemble claim is made; the original member-seed diagnostics above are reused only as diagnostics of the three individual experts.

The uniform result is saved in [`v2_capacity_ensemble_v1.json`](results/v2_capacity_ensemble_v1.json), with its [pre-score declaration](results/v2_capacity_ensemble_v1_declaration.json). The [independent audit](results/v2_capacity_ensemble_v1_verification.json) replayed 140 arm/cell scalar combinations, recomputed uniform probabilities and all 19 strict checks, and confirmed no inference was repeated. It also confirmed that the original 0/12 result remains intact. The verifier required schema-lookup corrections; those were confined to audit code, and the scalar result hashes and scores were unchanged. The earlier verifier failures and logs are preserved. This was the final bounded check. No further ensemble weighting or capacity/data-mix search is planned for this exposed cohort.

### Execution, verification and limits

All 18 declared fits completed, each with 2,048 updates. Width-96 control fits took 294.0–307.3 seconds; width-192 fits took 465.0–467.7 seconds; mixed-view fits took 294.0–298.4 seconds, excluding preparation. All 18 model artifact structures and all six sampler/exposure parity groups passed; the mixed recipe's 52 match-fit balance checks passed. This confirms the planned data exposure and training completion, not independent model generalization. New model artifacts remain local-only.

The scalar result is saved in [`v2_data_capacity_eval.json`](results/v2_data_capacity_eval.json), bound to the frozen declaration, seal and controls replay. Its [independent audit](results/v2_data_capacity_eval_verification.json) verified all 35 cells and recomputed aggregate metrics and zero-tolerance gates. No public labels, GAR TEST, or organizer scoring were used. These are development results on five previously exposed matches and synthetic feed-like views only; there is no independent continuous-match gold confirmation.

The package remains unchanged. The .60 stretch target is unmet by the best result by 0.0072508087. The wider arm costs about 1.5× fit time at equal update count and loses v2 spell against the matched width-96 blend; the mixed-view recipe also loses v2 spell. The frozen adopted-D mask preserves D metrics but leaves H/A quality below the strict all-view requirements. These results do not support promotion or a further selection on this cohort.

Repository source and results remain local. An automatic approval review rejected the GitHub push and public draft PR before any remote change, stating that authorization to conduct the research did not authorize public disclosure of code, provenance and results to this fork. No push or PR was created.

The research Site update is also prepared locally at source commit
`6d9e0436e7ca18209863f40a348143c154f3c5e9`. Site metadata was refreshed for
the owner, custom audience, access revision 2 and existing external viewer;
the live Site remains version 30. Its source push, native save and deployment
were rejected before remote mutation because the new non-public results would
be shared with that external viewer without explicit authorization for this
payload and destination. Static checks passed; browser QA was unavailable. Both
publications are blocked pending user approval for those destinations.
The local delivery state is recorded in the
[delivery receipt](results/v2_data_capacity_delivery.json).

## Scope, commands and outstanding work

This is development evidence on five exposed matches and synthetic feed-like
views. It cannot establish independent continuous-match gold performance,
real-broadcast robustness, GAR TEST performance or organizer performance.
There is no public or GAR scoring, no public-label access, and no promotion.
The adopted package, manifests and prior frozen reports remain unchanged.

The training declaration and preparation are frozen. The final full suite
passes 711 experiment tests in 21.520 seconds and 26 root tests; the
documentation check finds 132 Markdown files and
1,075 local links/anchors with no errors. The bundled
`evaluation/results/baseline_raw.csv` passes the
199,319-row format check only. The pre-fit validation commands were:

```sh
.venv/bin/python -m unittest discover -s experiments/tests -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_docs.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/baseline_raw.csv
```

These checks do not score candidates. Execution used
`MTAH_DATA_DIR=/workspace/mtah-dt` with the configured SoccerNet D/RF roots and
GAR adapter. The training command is:

```sh
.venv/bin/python -m experiments.v2_data_capacity fit-all --workers 2
```

The evaluation source required three execution wrappers to bind the configured
artifact roots, preserve the frozen 35-cell control emissions and write valid
receipts. The first serial feature-freeze attempt ran for 14 CPU minutes and
was cancelled before writing a declaration or scoring candidates; its
cancellation receipt is in ignored local data. The first controls replay
completed all 35 cells but its receipt failed because it used the wrong parent
study root. A recovery attempt preserved its partial JSON receipt unchanged
(42,723 bytes; SHA-256
`9d583026fbfc4b58c2a8e3b760310b9c29b7f4610be392a94c669672559e0422`) after
native-array serialization failed. No native NPZ probability caches were
overwritten. The successful v3 controls replay rebuilt official scalar metrics
from those sealed probabilities and wrote
[`v2_data_capacity_eval_control_replay_v3.json`](results/v2_data_capacity_eval_control_replay_v3.json).
These amendments changed receipt paths, root binding, canonical reduction
order and JSON serialization only; model arms, inputs, metrics, thresholds and
candidate scoring policy are unchanged. Earlier failed receipts and logs
remain preserved locally.

The controls-only execution command sequence was:

```sh
.venv/bin/python -m experiments.v2_data_capacity_eval_execution freeze-core
.venv/bin/python -m experiments.v2_data_capacity_eval_execution freeze-amendment
.venv/bin/python -m experiments.v2_data_capacity_eval_recovery freeze
.venv/bin/python -m experiments.v2_data_capacity_eval_execution_v3 freeze
.venv/bin/python -m experiments.v2_data_capacity_eval_execution_v3 replay-controls
```

Candidate scoring ran after the seal and completion of all 18 fits. The
evaluator commands were:

```sh
.venv/bin/python -m experiments.v2_data_capacity_eval_execution_v3 seal
.venv/bin/python -m experiments.v2_data_capacity_eval_execution_v3 run
.venv/bin/python -m scripts.verify_v2_data_capacity
```

Both independent audits passed, along with the final 711-test experiment suite
and 26 root tests. The verified readouts remain development evidence only; the
adopted package remains unchanged.
