# Iterative v2 model experiments

Study date: 6 November 2026. Status: running. The adopted package remains
`joint_gru_sn_dense_original_diversity_phase_ensemble_v1`.

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
