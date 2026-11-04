# V2 data and capacity study

Status: running; training began 4 November 2026 at 12:04 UTC. The 18-fit job is
underway on two CPU workers. No candidate score or promotion decision is
available.

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
so the wider arm costs more compute; measured fit times will be reported after
training.

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

## Scope, commands and outstanding work

This is development evidence on five exposed matches and synthetic feed-like
views. It cannot establish independent continuous-match gold performance,
real-broadcast robustness, GAR TEST performance or organizer performance.
There is no public or GAR scoring, no public-label access, and no promotion.
The adopted package, manifests and prior frozen reports remain unchanged.

The training declaration and preparation are frozen. The final-source
experiment suite has 702 passing tests and the root suite has 26 passing tests;
the documentation check finds 132 Markdown files and 1,056 local links/anchors
with no errors. The bundled `evaluation/results/baseline_raw.csv` passes the
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

Candidate scoring is not part of the controls replay. The remaining evaluator
commands are `...v2_data_capacity_eval_execution_v3 seal` and `... run`, after
all 18 fits complete. At this update, the two workers are active on the first
fold and no fit completion count, candidate metric, gate outcome or result
table is available.

After scoring, this report will record completed fit counts and times, final
source and verification commands, results for all arms, provider deltas, failed
guards, interpretation, limitations and local artifact state. The experiment
index and current research status will be updated with that readout. The
adopted package remains unchanged while the run is in progress.
