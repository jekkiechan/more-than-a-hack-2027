# Iterative v2 model experiments

Study date: 6 November 2026. Status: completed and independently verified. The adopted package remains
`joint_gru_sn_dense_original_diversity_phase_ensemble_v1`.

## Decision

**The >0.60-with-no-regression goal is not met; retain the adopted package.**
Three rounds test twelve candidate pipelines using twelve new continuation
fits. The numerical leader, the continuation/calibrated log-probability blend,
reaches **0.592638 v2 spell F1**, up **0.009067** from adopted and **0.007362**
below 0.60. It improves spell and macro F1 on all three views while retaining
D metrics exactly, but under-three-second recall regresses by **0.002091 /
0.002937 / 0.001759** on clean/v1/v2. Zero of twelve pipelines passes the full
success rule. All results are development point estimates. Independent official replay
passed for all three rounds.

The final blends provide a better balanced spell/macro tradeoff than the
round-one combinations, but none supplies a retained upgrade. The boundary
loss is worse than its paired existing-objective control. The 512-update
continuation budget and the tested loss do not establish that further training
or additional data are intrinsically ineffective. Stop this declared search;
the next evidence priority is independently annotated continuous-match data,
with any future training objective tested under a new bounded declaration.

## Question and prior evidence

Can combining archived models and continuing training on the additional
SoccerNet-GAR supervision raise matched v2 spell F1 above **0.60**, with no
regression in the other protected metrics?

This study builds on open PRs #82–84 at `30ddcbc`. Their best v2 point estimate,
0.594110, did not retain all clean/v1 metrics. Larger H/A capacity and a mixed
v1/v2 training recipe also failed the strict retention rule. Those historical
results and failures remain unchanged. Luna assisted with restoration,
experimental implementation, contract tests, and independent verification.

The [parent plan](results/v2_iterative_v1_plan.json) limits the work to three
separately frozen rounds: four archived-model combinations, twelve paired
continuation fits, and a final bounded combination follow-up. Each round has its
own declaration, sources, artifact hashes and results.

## Data and evaluation

The development cohort is the same five previously exposed matches: Sportec
J03WOY, Metrica Game 3 and SkillCorner 1996435, 1996436 and 2006229. Clean input,
two fixed historical v1 realizations and four fixed v2 realizations produce
35 matched cells. Replicas are averaged within match, then matches within
provider, then providers equally. Each prediction averages the two fixed seeds
within each native sampling phase, then averages phases before decoding.
Individual-seed scores are reported separately; their metric mean is not the
ensemble score.

All models, teachers, normalizers and original training examples exclude the
held-out provider. The additional training corpus is the archived 45-game GAR
TRAIN set, combined with the original 13 TRAIN matches. These GAR records carry
generated dense silver labels and discontinuous segments. This study reuses
existing additional supervision; it does not acquire new independently labelled
matches. Native unknowns, missingness, half/time breaks and team symmetry are
preserved.

The evaluator must reproduce all 24 adopted aggregate metrics within the
predeclared 1e-8 replay tolerance before any candidate scoring. Success uses
**zero regression tolerance**: spell F1, macro F1, D-F1, D precision, D recall
and under-three-second spell recall must each be at least the matched adopted
value on clean, v1 and v2. V2 spell must be strictly greater than 0.60. Ranking
uses unrounded v2 spell, with macro F1 breaking exact ties. The 70/30 combined
score is descriptive. Reference spell counts must remain invariant; predicted
counts and provider spread are diagnostics.

## Round one: combine archived models

Four fixed pipelines combine adopted original-full and archived calibrated
probabilities: a uniform categorical blend with the joint cost-3 decoder;
a uniform conditional H/A blend under the adopted D mask with H/A cost 3;
the same conditional blend at cost 1.5; and an equal conditional logit blend
under the adopted D mask at cost 3. The fixed-D adopted-only conditional
replay is a diagnostic, since conditional and joint H/A decoding need not
produce identical labels.

Round one’s numerical leader is the fixed-D logit blend: v2 spell **0.588294**,
up **0.004723** from adopted. Provider spell deltas are +0.013863 Metrica,
+0.004281 SkillCorner and -0.003976 Sportec. It regresses clean/v1 spell and
short-spell recall in every view: clean -0.022854, v1 -0.007254 and v2 -0.015989.
All three providers lose v2 short recall. The equal-provider mean predicted
spell count falls from 336.1 to 320.5 while the reference mean stays 404.3;
these weighted counts differ from the unweighted cohort total of 1,963.
Lowering the conditional H/A cost to 1.5 recovers most v2 short recall but drops
v2 spell to 0.585504. Thus the archived combinations show a boundary tradeoff,
not a retained improvement. The two leader seed diagnostics, 0.583611 and
0.584801 v2 spell, are not paired gains against corresponding adopted seeds;
the saved adopted metric reference is the pooled ensemble.

## Round two: paired boundary-loss continuation

Two recipes, three provider exclusions and seeds 2026/2027 give twelve fits.
Each starts from the corresponding verified adopted H/A checkpoint, retains
its exact normalizer and D weights, and runs 512 updates with batch size 32,
a 50/50 original/GAR mixture and learning rate 0.0003. Both recipes use the
shared distinct-possession sampler and the same four round-robin v2 TRAIN
views. The paired arms must have identical source/game/window draw schedules.

The control retains the existing dense duration-weighted H/A cross-entropy
and live-transition objective. The treatment adds a signed H/A logit-margin
loss with coefficient 0.1 and margin 1. It covers the two rows adjacent to a
known H/A transition plus one neighboring model sample on either side. Shared
valid-pair masks, period identity and known H/A targets exclude false edges
across gaps, unknowns, dead play or padding. Labels enter this loss only while
fitting. Inference receives tracking-derived features alone.

Each recipe is evaluated on its own and as a fixed equal conditional-H/A blend
with the adopted model, always retaining the adopted decoded D mask and H/A
cost 3. Preserving D probabilities alone would not guarantee unchanged decoded
D labels, which is why these arms explicitly retain the decoded mask.

Round two favors the existing objective over boundary margin. The control plus
adopted blend reaches **0.668249 / 0.534925 / 0.585615** clean/v1/v2 spell,
gains of +0.000694 / +0.003349 / +0.002044. Short-spell recall also improves in
all three views (+0.003150 / +0.012026 / +0.005823), and all D metrics remain
unchanged. Its only quality failure is clean macro F1, **-0.000111**; v2 also
remains below 0.60. This small failure is preserved under the predeclared
zero-tolerance rule.

The boundary-margin blend lowers spell relative to its paired control on all
three views (-0.000396 / -0.002576 / -0.003038). It recovers slightly more short
spells on v1/v2 but loses macro F1 as well. Standalone continuation is weaker
than blending with the adopted expert. Thus the additional loss does not
establish a retained benefit; the existing-objective continuation is selected
for the bounded final combinations.

## Round three: bounded combinations after round two

The stronger standalone continuation recipe on unrounded v2 spell (macro F1
breaks an exact tie) is selected from round two. Four final pipelines combine
its saved conditional H/A probabilities with the archived calibrated model,
with or without the adopted model as a third equally weighted expert. Arithmetic
and log-probability means are each tested once. All four preserve the adopted
decoded D mask and H/A cost 3. There are no extra fits, inference passes or tuned
blend weights. The selection is explicitly post-result development work; it
is not independent confirmation. Individual-seed diagnostics in rounds two and
three hold the adopted ensemble D mask fixed.

Round three selects `ce_control` and its two-expert log-probability blend as
the numerical leader. V2 spell gains occur on every provider: **+0.024345
Metrica / +0.000995 SkillCorner / +0.001860 Sportec**. Most of the aggregate
gain therefore comes from Metrica. Short recall falls on all three providers
under v2. The v1 spell mean also hides provider regressions: Metrica +0.021301,
SkillCorner -0.002216 and Sportec -0.017522. Clean Sportec spell falls 0.001356.
The final three-expert means reduce the short-recall losses and retain v2
short recall, but still lose clean/v1 short recall and remain below 0.60.
These provider tradeoffs further limit a broad improvement claim.

## Execution and verification

An initial slow attempt was stopped before any baseline cell or candidate score
was emitted. Its interruption receipt and 34 completed training-view artifacts
were preserved. The remaining preparation and control inference use a declared
runtime amendment: vectorized NumPy row quantiles, tested bitwise against the
original implementation on synthetic edge cases and an 800-frame native feature
matrix. Metric aggregation sorts cell keys to prevent worker-completion order
from creating a floating-point difference in strict zero-tolerance checks. The
model, loss, sampler, features and scoring definitions are unchanged by that
amendment. See the [runtime parity receipt](results/v2_iterative_v1_runtime_parity.json)
and [execution amendment](results/v2_iterative_v1_execution_amendment.json).

The [training seal](results/v2_iterative_train_v1_seal.json) records twelve
completed fits and paired sampling/exposure equality. The
[control replay](results/v2_iterative_v1_control_replay.json) passes all 24
saved adopted aggregate comparisons. Every clean or individual corruption
realization contains 1,963 reference spells across the five matches. Independent
verification reconstructs the combinations from sealed emissions, replays the
official scorer, recomputes provider and seed summaries, and checks every strict
gate. Weights, emissions and native predictions remain in ignored local data;
aggregate JSON and hashes are committed for review.
The twelve research checkpoints and their sidecars also have a local 17.6 MB
bundle, with every archive member checked against its sealed hash; see the
[bundle receipt](results/v2_iterative_v1_checkpoint_bundle.json). The bundle
excludes data and teacher dependencies and has not been remotely uploaded or
independently restored. Individual fitting time ranges from 98.69–172.95 seconds
for the control and 100.44–176.49 seconds for boundary margin on this CPU; these
timings exclude data loading and are descriptive, not a controlled benchmark.

A new two-step synthetic trainer-parity test initially failed bitwise equality.
The independent CE implementation reuses the logits slice for both CE and the
transition term; the shared trainer creates a separate slice node. A temporary
in-memory probe matching the latter restored exact weight parity. The actual
runs preserve their frozen implementation. Their paired control and treatment
share that implementation, and the observed difference from the shared trainer
is below 3e-7 in state values after two synthetic updates. The test uses that
numerical tolerance; evaluation retention remains exactly zero-tolerance. See
[the original diagnostic](results/v2_iterative_v1_training_numerics.json) and
[the isolating slice probe](results/v2_iterative_v1_training_numerics_slice_probe.json).
This short smoke check does not establish byte-identical long training runs.

The independent verifier also required three schema fixes: recursive comparison
of per-IoU metrics, a zero D column in conditional H/A reconstruction, and the
extra seed dimension in round-three summaries. These changes affect only the
auditor, not candidate results or strict tolerances. The
[round-three amendment](results/v2_iterative_v1_round3_verifier_amendment.json)
preserves the failed tail comparison after all prediction, official metric,
pooled aggregate and gate checks had passed. The corrected seed comparator
passes a [targeted real-data check](results/v2_iterative_v1_round3_seed_tail_check.json);
the complete round-three audit subsequently passed without changing candidate
results or tolerances.

All three [independent verification receipts](results/v2_iterative_v1_verification.json)
pass for the complete 35-cell cohort. Exact verifier source versions are retained
in the [source inventory](results/v2_iterative_verifier_sources/inventory.json);
the final verification summary also binds the completed round-three receipt to
its matching source snapshot. [Code validation](results/v2_iterative_v1_code_validation.json)
passes 752 experiment tests, 26 root tests, the file-backed fit contract and the
bundled 199,319-row CSV format check. That CSV check is format-only and does
not score organizer labels. Native outputs and research weights remain local.

### Reproduction

Restore the exact private inputs and adopted/calibrated folds listed by the
[baseline declaration](results/v2_iterative_v1_declaration.json) and the
[training declaration](results/v2_iterative_train_v1_declaration.json). Use the
pinned experiment environment. Set `MTAH_DATA_DIR` and `MTAH_DT_DIR` to the
dense archive, `MTAH_SQD_DIR` to the short-quality archive, and
`MTAH_RF_ARCHIVE_DIR` to the restored feed-robustness artifacts. Run with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

The completed execution sequence is listed below; the preserved interrupted
attempt is described above. Completed scoring stages are one-shot:

```sh
.venv/bin/python -m experiments.v2_iterative_eval freeze
.venv/bin/python -m experiments.v2_iterative_execution freeze
.venv/bin/python -m experiments.v2_iterative_execution prepare
.venv/bin/python -m experiments.v2_iterative_execution controls
.venv/bin/python -m experiments.v2_iterative_train_run freeze
.venv/bin/python -m experiments.v2_iterative_train_run fit --workers 2
.venv/bin/python -m experiments.v2_iterative_eval freeze-round1
.venv/bin/python -m experiments.v2_iterative_eval run-round1
.venv/bin/python -m experiments.v2_iterative_followup freeze
.venv/bin/python -m experiments.v2_iterative_followup run
.venv/bin/python -m experiments.v2_iterative_round3 freeze
.venv/bin/python -m experiments.v2_iterative_round3 run
```

The declaration and scoring stages refuse reuse of completed IDs. Do not delete
receipts to rerun them. After restoring their sealed artifacts, the read-only
verification command is:

```sh
.venv/bin/python -m scripts.verify_v2_iterative --round all
```

## Limitations and delivery

These are further development experiments on selected-on matches. The v2
simulator places corruption using development/training states and has a
historical model-conditioned calibration limitation. Four realizations reduce
corruption-draw sensitivity; they do not create independent-match evidence.
No organizer-reference scoring, GAR VALID/TEST selection, pooled deployment
fitting or promotion is part of this experiment. Weights and native outputs
remain in ignored local storage; remote availability is recorded separately.

## Completed scorecard

Equal-provider official scores. All displayed values are rounded; decisions use the saved unrounded values. The strict retention check covers six quality metrics on each of clean, v1 and v2.

| Round / pipeline | Clean spell | v1 spell | v2 spell | v2 macro | Retains all metrics | >.60 and retention |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 / Adopted original-full | 0.667556 | 0.531576 | 0.583571 | 0.849975 | Yes | No |
| 1 / Archived team blend · cost 1.5 | 0.668546 | 0.529174 | 0.585504 | 0.851543 | No | No |
| 1 / Archived team blend · cost 3 | 0.667181 | 0.529196 | 0.587916 | 0.851881 | No | No |
| 1 / Archived logit blend | 0.667181 | 0.529633 | 0.588294 | 0.851888 | No | No |
| 1 / Joint archived blend | 0.664688 | 0.527796 | 0.587677 | 0.851829 | No | No |
| 2 / Boundary + adopted | 0.667854 | 0.532349 | 0.582577 | 0.849784 | No | No |
| 2 / Boundary margin | 0.661316 | 0.526371 | 0.577860 | 0.849522 | No | No |
| 2 / Control + adopted | 0.668249 | 0.534925 | 0.585615 | 0.850181 | No | No |
| 2 / Continuation control | 0.664697 | 0.529823 | 0.583169 | 0.850046 | No | No |
| 3 / Three-model log | 0.668616 | 0.532460 | 0.590751 | 0.851622 | No | No |
| 3 / Three-model mean | 0.669328 | 0.532460 | 0.591259 | 0.851658 | No | No |
| 3 / Continuation + calibrated log | 0.670707 | 0.532096 | 0.592638 | 0.852359 | No | No |
| 3 / Continuation + calibrated mean | 0.668642 | 0.532384 | 0.592428 | 0.852363 | No | No |

### Numerical leader: protected-metric deltas

| View | Spell | Macro | D F1 | D precision | D recall | Short recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | +0.003152 | +0.000645 | +0.000000 | +0.000000 | +0.000000 | -0.002091 |
| feed_like_v1 | +0.000521 | +0.000059 | +0.000000 | +0.000000 | +0.000000 | -0.002937 |
| feed_like_v2 | +0.009067 | +0.002385 | +0.000000 | +0.000000 | +0.000000 | -0.001759 |

### Numerical leader: provider and seed diagnostics

| View | Metrica spell | SkillCorner spell | Sportec spell | Seed 2026 spell | Seed 2027 spell | 70/30 auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.847868 | 0.486129 | 0.678125 | 0.671493 | 0.665373 | 0.731070 |
| feed_like_v1 | 0.650584 | 0.448784 | 0.496921 | 0.532242 | 0.525926 | 0.620541 |
| feed_like_v2 | 0.705251 | 0.485134 | 0.587529 | 0.591820 | 0.586762 | 0.670554 |

Two fixed seeds are sensitivity diagnostics, not a confidence interval; neither seed is selected. Provider weighting gives each of the three providers equal mass. The auxiliary score does not override failed checks.

### Every failed check

- **Archived team blend · cost 1.5:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v1/spell_f1_vs_adopted`, `feed_like_v2/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Archived team blend · cost 3:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `clean/spell_f1_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/spell_f1_vs_adopted`, `feed_like_v2/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Archived logit blend:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `clean/spell_f1_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/spell_f1_vs_adopted`, `feed_like_v2/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Joint archived blend:** `clean/d_precision_vs_adopted`, `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `clean/spell_f1_vs_adopted`, `feed_like_v1/d_f1_vs_adopted`, `feed_like_v1/d_precision_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/spell_f1_vs_adopted`, `feed_like_v2/d_f1_vs_adopted`, `feed_like_v2/d_precision_vs_adopted`, `feed_like_v2/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Boundary + adopted:** `clean/macro_f1_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v2/macro_f1_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`, `feed_like_v2/spell_f1_vs_adopted`.
- **Boundary margin:** `clean/macro_f1_vs_adopted`, `clean/spell_f1_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v1/spell_f1_vs_adopted`, `feed_like_v2/macro_f1_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`, `feed_like_v2/spell_f1_vs_adopted`.
- **Control + adopted:** `clean/macro_f1_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Continuation control:** `clean/macro_f1_vs_adopted`, `clean/spell_f1_vs_adopted`, `feed_like_v1/macro_f1_vs_adopted`, `feed_like_v1/spell_f1_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`, `feed_like_v2/spell_f1_vs_adopted`.
- **Three-model log:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Three-model mean:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Continuation + calibrated log:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.
- **Continuation + calibrated mean:** `clean/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v1/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/short_spell_recall_iou_0_5_lt3s_vs_adopted`, `feed_like_v2/spell_f1_strictly_above_0_60`.

The [machine-readable readout](results/v2_iterative_v1_readout.json) contains every protected metric; the round result files also retain provider, individual-seed and spell-count diagnostics.

## Published delivery

Code, aggregate results and independent verification are on
[PR #85](https://github.com/jekkiechan/more-than-a-hack-2027/pull/85).
The [research dashboard](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/experiments.html#v2-iterative)
is published as **version 32**. Desktop and mobile checks pass for Overview and
Experiments, including all 36 view/metric combinations and thirteen comparison
rows. The existing restricted audience is unchanged at access revision 2.
The [delivery receipt](results/v2_iterative_v1_delivery.json) records the native
successful deployment, source commit, artifact metadata and browser checks.
Input-tar and service-archive hashes are recorded separately; no byte-for-byte
remote archive restore is claimed. New model weights and native predictions
remain local-only. The adopted package and decision policy remain unchanged.
