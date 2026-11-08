# Postprocessing the approved 0.593619 frontier

Status: both rounds independently verified; no retained improvement,
8 November 2026. Keep the approved .593619 frontier.

The user clarified that iteration must start from the approved
`joint_gru_sn_dense_spell_frontier_ensemble_v1`, not the historical predecessor.
The target control is **.670136549956 clean / .532128306932 v1 /
.593618719890 v2 spell F1**. The earlier
[historical pilot](COMPLEMENTARY_SPELL_V1.md) does not establish improvement
over this model.

## First-round result

**Keep the approved frontier after this round.** None of nine candidates passes
all frozen guards. [All 350 saved predictions](results/frontier_postprocess_v1_readout.json)
replay independently, including 105 boundary raw-run-order checks. The
[full results](results/frontier_postprocess_v1_results.json) retain every
per-cell metric and failed candidate.

| Arm | Clean spell | V1 spell | V2 spell | Failed retention checks |
| --- | ---: | ---: | ---: | --- |
| Current frontier | .670137 | .532128 | .593619 | Control |
| Seed 2026 weight .25 | .668344 | .529795 | .590199 | All spell/short-recall views; v1 macro |
| Seed 2026 weight .75 | .669187 | .534317 | .592114 | Clean/v2 spell and short recall |
| Boundary cap .2 s | .670137 | .532128 | .593619 | No strict v2 improvement |
| Boundary cap .4 s | .670195 | .532320 | .593497 | V2 spell |
| Boundary cap .8 s | .670299 | .532352 | .593593 | V2 spell |
| H/A cost 2 | .669670 | .531519 | .588565 | All spell views |
| H/A cost 4 | .667543 | .531413 | .591926 | All spell and short-recall views |
| Disagreement strength .5 | .670402 | .532067 | .593372 | V1/v2 spell and short recall |
| Disagreement strength 1 | .670422 | .532584 | .593689 | V1/v2 short recall |

Equal seed weighting remains strongest on v2 among the three current-frontier
ratios tested. The numerical leader improves v2 spell by only **.000070220**;
its v1 short recall falls from .220366965 to .220008543, and v2 short recall
from .288603888 to .287519500. Its clean/v1/v2 macro values are
.871819916/.826205750/.852193020, all within the declared tolerance. Boundary
cap .8 s improves v1/v2 short recall to .221583360/.289635634, but loses
.000025242 v2 spell. The earlier predecessor boundary gain therefore does
not transfer to the current frontier as a retained standalone improvement.

Control macro F1 is .871864381/.826154300/.852203729. The auxiliary 70/30
spell/macro summary is .730654899/.620336105/.671194223 for control and
.730841111/.620670760/.671240164 for disagreement 1; it is not the ranking
criterion and does not override failed short-recall guards.

| V2 provider | Current spell | Disagreement 1 | Delta | Boundary .8 s | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metrica | .705783 | .706454 | +.000670 | .705637 | -.000147 |
| SkillCorner | .483972 | .484172 | +.000200 | .484043 | +.000071 |
| Sportec | .591101 | .590441 | -.000660 | .591101 | +.000000355 |

All candidates preserve the exact D/X masks, reference spell counts and graded
frames. D precision/recall remain .884844/.861920 clean, .843006/.881217 v1 and
.873034/.862980 v2. Summed scored predicted-spell counts across each view's
cells are 1,799/3,187/6,916 for control, 1,795/3,186/6,907 for disagreement 1,
and 1,800/3,187/6,914 for boundary .8 s. Boundary relocation preserves raw run
order; clipping to the fixed reference-valid runs can change scored spell
counts. The independent verifier checks those distinctions explicitly.

## Second round: one fixed composition

The single second-round question was whether applying the already fixed .8 s
boundary relocation after the already fixed strength-1 disagreement rule
combines their complementary effects. The separate
[post-result declaration](results/frontier_postprocess_round2_declaration.json)
adds no parameter search and retains the current-frontier guards. All 35
baseline and both standalone predictions/metric rows must replay before the
combined candidate is scored. The first-round failures remain unchanged.

The [combined result](results/frontier_postprocess_round2_results.json) does
not resolve the tradeoff:

| View | Spell F1 | Delta | Macro F1 | Short recall | Short-recall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | .670526595 | +.000390045 | .871822164 | .408597177 | .000000000 |
| V1 | .532656372 | +.000528065 | .826240320 | .221583360 | +.001216394 |
| V2 | .593532431 | -.000086289 | .852240820 | .288551246 | -.000052641 |

V2 provider spell deltas are +.000663 Metrica, +.000272 SkillCorner and
-.001194 Sportec. The 70/30 summaries are .730915266/.620731556/.671144948.
V2 spell and short recall fail; the other retention checks pass.
[Independent verification](results/frontier_postprocess_round2_readout.json)
replays all 35 new predictions and 105 baseline/standalone controls (140 official
scorecards), including all 35 gated-run-order checks. Scored predicted-spell
counts are 1,796/3,186/6,905 clean/v1/v2, with reference and grading populations
unchanged. V2 short matches increase by one in the pooled cell totals while
equal-provider short recall declines; the predeclared equal-provider measure,
not the pooled count, determines the guard.

After this follow-up, stop this local postprocessing search and retain the
approved package. Neither the nine isolated candidates nor this one composition
supports an upgrade from .593619. This is a measured plateau for the tested
mechanisms on this exposed cohort, not proof of an optimal model. New continuous
annotations or emissions that recover missing Sportec turnovers are more useful
next directions than finer decoder/mixture grids on these same five matches.

## Recover the current comparator first

The approved model pools CE stage-1024, calibrated, and predecessor conditional
H/A probabilities at .5/.375/.125, with equal paired seeds, native phase
averaging, H/A transition cost 3, and the predecessor's fixed decoded D mask.
The required research checkpoints and prediction caches were local-only in the
previous environment. Restore archived inputs and weights before reconstructing
any missing components.

The exact CE checkpoints were unavailable. A separately declared reconstruction
used the original numerical code: six provider-excluded CE fits of 512 updates
at learning rate .0003, followed by 512 updates at .0001 with the original
reset-seed draw schedule. The archived predecessor weights, paired teachers,
normalization, original TRAIN records, four fixed v2 TRAIN views, and 45 GAR
TRAIN games remain matched. Reconstruction is not an assertion of original
checkpoint byte identity.

The [restore receipt](results/frontier_reconstruction_v1_restore.json) verifies
130 selected source artifacts (3,362,311,524 bytes). All 52 regenerated v2
TRAIN caches and their 52 sidecars match the original declaration byte for
byte. The [training declaration](results/frontier_recovery_v1_training_declaration.json)
binds the original numerical source and input inventory. A supplemental source
receipt binds five additional transitive modules before fitting.

An initial two-worker attempt exhausted memory before writing any checkpoint.
A frozen execution amendment then ran twelve fresh subprocesses serially.
All twelve completed. Independent inspection verifies actual model/sidecar
hashes, paired teacher and warm-start lineage, unchanged D parameters and
normalization, and exact sampling, view draws and exposure counts. The largest
recorded mean-loss difference from the original runs is 1.6764e-8. None of the
twelve serialized checkpoint hashes matches its historical file; the artifacts
are explicitly reconstructed. Only the following metric replay can establish
comparability for this experiment.

Before candidate scoring, the [control-only preflight](results/frontier_postprocess_v1_preflight.json)
reproduced every saved selected-frontier metric row across all 35 development
cells, including counts and grading populations. Aggregate clean/v1/v2 spell
matches **.6701365499557103/.532128306931943/.5936187198896373** exactly.
The [independent training audit](results/frontier_recovery_v1_independent_training_audit.json)
and native cache source closure also pass. This is a matched replay of the
current recipe using reconstructed checkpoint files, not a claim that their
serialized bytes equal the missing originals.

## Fixed candidate mechanisms

- Seed-2026 weights .25 and .75, retaining the three expert-family weights;
  .50 is the current control.
- Posterior-median H/A boundary relocation with .2, .4, and .8-second caps,
  applied separately to the current equal-seed control.
- H/A switch costs 2 and 4, compared with the current cost 3.
- Two fixed cross-expert disagreement gates that raise switch cost locally
  where the three current experts disagree; strengths .5 and 1.

These are nine candidates plus the unchanged control, without a cross-product.
The [frozen declaration](results/frontier_postprocess_v1_declaration.json) binds
the precise numerical settings, source, model/input hashes, and cohort before
candidate scoring. Luna agents implement
the reconstruction, candidate runner, disagreement rule, and independent audit.

After cache preparation, freeze the evaluation manifest and declaration without
scoring any candidates:

```sh
python -m experiments.frontier_postprocess_v1 freeze \
  data/experiments/frontier_postprocess_v1/evaluation_manifest.json \
  --native-manifest data/experiments/frontier_recovery_v1/frontier_native_probability_manifest.json \
  --declaration data/experiments/frontier_postprocess_v1/declaration.json
```

The separate control-only preflight then replays every saved frontier cell
before `run` can score the ten declared arms. `run` requires the exact
declaration and a passed preflight receipt; do not invoke it until the
predeclared cache and declaration have been reviewed.

## Comparison with Claude's concurrent work

On 8 November the requested branch became visible at
[`claude/fervent-curie-fcqsta`, `61f8182`](https://github.com/jekkiechan/more-than-a-hack-2027/tree/61f8182f86f2faca7873edd4db4c060a93285ec2).
Its [197-arm study](https://github.com/jekkiechan/more-than-a-hack-2027/blob/61f8182f86f2faca7873edd4db4c060a93285ec2/experiments/V2_POSTPROC_V1.md)
reports an exact six-decimal frontier replay and no retained candidate. Its
numerical v2 maximum, .594992, loses clean spell and short recall. It covers
mixture/cost grids, extra continuation seeds, stage snapshots, alternative D
masks, short-run insertion and calibrated warm starts. We inspected its saved
evidence without modifying that branch or rescoring its outputs.

The current cost arms provide overlapping controls; boundary relocation and
expert-disagreement gating remain distinct mechanisms. This evidence argues
against another broad cost or family-weight grid. Claude's turnover-count
diagnostics suggest a deficit in Sportec emissions; they do not establish that
any particular new feature or training source would solve it.

The 50/50 ratios refer to two different choices: GAR/original training-objective
weight and equal averaging of the two inference seeds. The earlier
[extra-data continuation](V2_EXTRA_DATA.md) tested .625 GAR weight and failed
retention despite a small v2 gain. That does not prove .5 optimal; it makes a
repeat of the same ratio sweep less useful. Current-frontier seed weights are
tested separately here.

## Evaluation and decision

Use the same five exposed development matches, clean inputs, two historical v1
realizations and four fixed v2 realizations. Average realizations within match,
matches within provider, then providers equally. Use the official metrics.
Preserve the current control's D/X mask and time/period boundaries, and retain
the exact reference and grading populations. Changing H/A switch costs may
change predicted spell counts; report those changes rather than assuming they
are invariant.

Rank by v2 spell F1, with macro F1 as an exact tie-breaker. A retained candidate
must improve v2 spell, retain clean/v1 spell and short-spell recall in all views,
and keep each view's macro F1 within .001 of the current frontier. Report
provider spread and all failed checks. No organizer or GAR evaluation is part
of this comparison, and no automatic model promotion follows.

The development matches have repeated prior exposure and v2 stress is
model-conditioned. A result above .593619 is a matched development improvement,
not independent continuous-match gold confirmation or a performance ceiling.

Metadata clarification: R1 inherited the native cache's `evidence_scope` text,
including “no new candidate scores.” That phrase describes cache preparation;
R1 subsequently scored the 350 declared outputs documented above. The frozen
record is preserved, and R2 uses an explicit composition-study scope string.

## Validation

The final experiment suite ran 893 tests: 887 passed, one skipped, and five errored
because the inherited frontier-public tests require absent historical
`public_candidate.csv` / `public_prediction_seal.json` artifacts. No new
experiment test failed. All 26 root tests, the exact file-backed fitting
contract, and the bundled example CSV format check pass. The focused current
runner, decoder, native-mask and readout tests pass (18 tests), as do eight
second-round contract tests. The inherited failing test file is itself bound by
historical evidence; it was not rewritten to bypass the missing artifacts.
These checks do not score organizer labels.

## Delivery

Code and aggregate evidence are being delivered in
[draft PR #88](https://github.com/jekkiechan/more-than-a-hack-2027/pull/88), which
includes the still-open PR #87 ancestry. No model promotion or closed public
gate change is part of this work. The research checkpoint and native-probability
archive will use a new branch in the existing private cache repository; the
delivery receipt will distinguish remote availability from these local results.
