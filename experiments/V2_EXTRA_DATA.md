# Extra GAR data: targeted continuation

Study `v2_extra_data_v1` · 7 November 2026 · complete; independently audited.

The user requested further tests after approving the frozen
[spell-frontier blend](V2_SPELL_FRONTIER.md) for promotion. This study uses that
blend as its development control. The previous public diagnostic is closed;
no organizer reference access or public scoring is authorized by this study.
The separate promotion retains the exact previously approved weights.

## Results and decision

**Retain the approved frontier blend. None of the three continuation arms passes all retention checks.**

| Arm | Clean spell | V1 spell | V2 spell | V2 macro | Failed retention checks |
| --- | ---: | ---: | ---: | ---: | --- |
| Approved frontier control | 0.670137 | 0.532128 | 0.593619 | 0.852204 | — |
| fresh_control | 0.670446 | 0.532146 | 0.594751 | 0.852458 | feed_like_v1/short_spell_recall_iou_0_5_lt3s |
| gar_quality50 | 0.669714 | 0.532168 | 0.593404 | 0.851999 | clean/spell_f1; feed_like_v2/spell_f1 |
| gar_weight625 | 0.670696 | 0.531067 | 0.595084 | 0.852542 | feed_like_v1/short_spell_recall_iou_0_5_lt3s; feed_like_v1/spell_f1 |

### Isolating each data intervention

Deltas below compare with the new 256-update fresh continuation control. They isolate the data change at a matched update budget; the table above compares against the approved frontier blend.

| Data change | Δ clean spell | Δ v1 spell | Δ v2 spell |
| --- | ---: | ---: | ---: |
| gar_weight625 | +0.000250 | -0.001079 | +0.000333 |
| gar_quality50 | -0.000732 | +0.000023 | -0.001347 |

### Provider and seed spread

| Arm / view / provider | Spell | Δ spell vs frontier | Δ macro vs frontier | Short recall |
| --- | ---: | ---: | ---: | ---: |
| fresh_control / clean / Metrica | 0.848418 | +0.000000 | +0.000058 | 0.611111 |
| fresh_control / clean / SkillCorner | 0.484478 | +0.000409 | -0.000180 | 0.284803 |
| fresh_control / clean / Sportec | 0.678442 | +0.000519 | +0.000378 | 0.329949 |
| fresh_control / feed_like_v1 / Metrica | 0.647072 | +0.002765 | +0.001116 | 0.293651 |
| fresh_control / feed_like_v1 / SkillCorner | 0.447913 | -0.000742 | +0.000562 | 0.227133 |
| fresh_control / feed_like_v1 / Sportec | 0.501452 | -0.001970 | +0.000228 | 0.134518 |
| fresh_control / feed_like_v2 / Metrica | 0.709676 | +0.003893 | +0.000932 | 0.361111 |
| fresh_control / feed_like_v2 / SkillCorner | 0.483039 | -0.000933 | -0.000122 | 0.273781 |
| fresh_control / feed_like_v2 / Sportec | 0.591537 | +0.000437 | -0.000048 | 0.230964 |
| gar_quality50 / clean / Metrica | 0.848968 | +0.000550 | +0.000071 | 0.611111 |
| gar_quality50 / clean / SkillCorner | 0.481212 | -0.002858 | -0.000810 | 0.286953 |
| gar_quality50 / clean / Sportec | 0.678961 | +0.001039 | +0.000477 | 0.329949 |
| gar_quality50 / feed_like_v1 / Metrica | 0.646547 | +0.002239 | +0.000303 | 0.301587 |
| gar_quality50 / feed_like_v1 / SkillCorner | 0.448717 | +0.000062 | +0.000203 | 0.233620 |
| gar_quality50 / feed_like_v1 / Sportec | 0.501241 | -0.002181 | +0.000415 | 0.134518 |
| gar_quality50 / feed_like_v2 / Metrica | 0.704659 | -0.001124 | -0.000008 | 0.357143 |
| gar_quality50 / feed_like_v2 / SkillCorner | 0.482737 | -0.001235 | -0.000485 | 0.277025 |
| gar_quality50 / feed_like_v2 / Sportec | 0.592815 | +0.001715 | -0.000121 | 0.232234 |
| gar_weight625 / clean / Metrica | 0.848418 | +0.000000 | +0.000058 | 0.611111 |
| gar_weight625 / clean / SkillCorner | 0.484190 | +0.000120 | -0.000273 | 0.284803 |
| gar_weight625 / clean / Sportec | 0.679481 | +0.001558 | +0.000379 | 0.329949 |
| gar_weight625 / feed_like_v1 / Metrica | 0.644424 | +0.000116 | +0.000234 | 0.293651 |
| gar_weight625 / feed_like_v1 / SkillCorner | 0.446784 | -0.001871 | +0.000176 | 0.226057 |
| gar_weight625 / feed_like_v1 / Sportec | 0.501993 | -0.001429 | +0.000878 | 0.134518 |
| gar_weight625 / feed_like_v2 / Metrica | 0.710363 | +0.004580 | +0.000966 | 0.363095 |
| gar_weight625 / feed_like_v2 / SkillCorner | 0.483754 | -0.000219 | +0.000071 | 0.275430 |
| gar_weight625 / feed_like_v2 / Sportec | 0.591136 | +0.000035 | -0.000021 | 0.228426 |

| Arm / view | Seed 2026 spell | Seed 2027 spell | Ensemble spell |
| --- | ---: | ---: | ---: |
| fresh_control / clean | 0.669825 | 0.666463 | 0.670446 |
| fresh_control / feed_like_v1 | 0.533152 | 0.524541 | 0.532146 |
| fresh_control / feed_like_v2 | 0.590169 | 0.588712 | 0.594751 |
| gar_quality50 / clean | 0.671140 | 0.667570 | 0.669714 |
| gar_quality50 / feed_like_v1 | 0.533179 | 0.523141 | 0.532168 |
| gar_quality50 / feed_like_v2 | 0.590665 | 0.588110 | 0.593404 |
| gar_weight625 / clean | 0.671799 | 0.665581 | 0.670696 |
| gar_weight625 / feed_like_v1 | 0.534454 | 0.524804 | 0.531067 |
| gar_weight625 / feed_like_v2 | 0.591354 | 0.588999 | 0.595084 |

Seed diagnostics hold the predecessor ensemble D mask fixed. The headline is an inference probability ensemble, not an average of seed scores. All ensemble and seed predictions retain the exact D mask. The scorecard includes D F1/precision/recall, short recall and predicted/reference spell counts for every cell.

### Verification and artifacts

The stage-1024 control replays with zero difference across ten representative seed/phase arrays, 35 scored cells and 27 aggregate/published-value checks. Training receipts record 8,192 windows per fit. The training audit checks whole-provider exclusion, checkpoint/teacher lineage, normalization and D equality, exact control/weight-arm windows, and paired original-window/game exposures for quality sampling. The saved-output audit independently reconstructs all phase/seed mixtures and scores every candidate and seed prediction.

- [Training declaration](results/v2_extra_data_v1_training_declaration.json), [18-fit inventory and full-seal hash](results/v2_extra_data_v1_training_inventory.json) and [training audit](results/v2_extra_data_v1_training_audit.json).
- [Control replay](results/v2_extra_data_v1_preflight.json), [evaluation declaration](results/v2_extra_data_v1_declaration.json), [complete scorecard](results/v2_extra_data_v1.json) and [independent verification](results/v2_extra_data_v1_verification.json).
- [Decision readout](results/v2_extra_data_v1_readout.json).

Research weights, full sampling receipts and frame outputs remain in ignored local storage. The unchanged full training seal is locally retained at its frozen path; only its aggregate inventory and hash are published. No new remote weight upload or public scoring is claimed.

## Additional exposure and bounded follow-up

The quality arm increased replacement draws from roughly 981–1,031 to
2,030–2,067 of 4,096 GAR windows per fit. The existing qualified catalog has
403 of 1,427 short spans; the control already nominated 402–403, versus all
403 for the quality arm. This mostly increases emphasis on known spans,
rather than adding new supervision. Original-window sequences and game
exposures remain paired; control and loss-weight arms use identical windows.

A separate [four-mixture plan](results/v2_extra_data_mix_v1_plan.json) tests
quality fractions 0.25, 0.50 and 0.75 with fresh continuation, plus an equal
quality/weight-arm blend. It combines cached CE probabilities before the
unchanged outer H/A mixture. No new fitting, inference or decoder search is
allowed. This follows complementary development failures, not a public or
GAR result. Official decoded metrics are nonlinear, so score interpolation
does not establish a feasible blend. Results and audit follow below when complete.

The independent verifier's first pass reproduced all cell/seed scores and
aggregates, then failed its own eligibility-schema comparison. The
[correction receipt](results/v2_extra_data_v1_audit_correction.json) preserves
the attempt: only the verifier's metadata/boolean handling changed. The
frozen evaluator, candidate outputs, thresholds and results did not change;
the complete corrected audit passed.

## Question and declared design

Can a targeted change in how existing GAR TRAIN data contributes improve the
frontier blend while retaining clean, v1 and v2 spell F1? Prior scaling,
capacity, v1/v2 mixing and longer continuation results remain preserved.
This is a shorter, paired continuation with two isolated data interventions.

| Arm | GAR loss share | GAR quality replacement rate |
| --- | ---: | ---: |
| Fresh continuation control | 0.500 | 0.25 |
| More GAR objective weight | 0.625 | 0.25 |
| More corroborated short GAR possessions | 0.500 | 0.50 |

All 18 fits start from the exact balanced stage-1024 checkpoint: three excluded
providers × two fixed seeds × three arms. Each gets 256 updates at learning rate
0.0001, ending at stage 1280. Batches retain 16 original-provider and 16 GAR
windows; the objective weight is not a sampling fraction. Original short-span
replacement remains 0.25. Fresh schedules offset the original distinct-span
sampler and trainer draw RNG by 104729, with paired Torch/dropout seeds.
The original clean mass and four fixed v2 TRAIN views, teacher, normalization,
D parameters, targets and architecture are unchanged.

The eligible pool is the same 13 original TRAIN matches (whole-provider
exclusion in each fold) and 45 GAR TRAIN games. Corroboration uses the existing
TRAIN-only generated-label and geometric checks. It changes sampling, not
labels or inference. These are additional draws from existing silver data,
not new annotations. GAR VALID/TEST are excluded from fitting.

## Comparison and selection

The deployed recipe stays fixed: CE 0.5, calibrated 0.375 and predecessor 0.125,
arithmetic conditional H/A pooling and H/A transition cost 3, with the exact
predecessor decoded D mask. Average the two seeds within each native phase,
then phases before decoding. Score the same 35 development cells: five clean,
ten v1 and twenty v2. Average realizations within match, matches within
provider, and providers equally. Per-seed scores are diagnostics, not separate
candidate pipelines.

The control has clean/v1/v2 spell F1 **0.670137/0.532128/0.593619**. Each candidate
must equal or improve all three spell means, strictly improve v2, retain
short-spell recall in every view, and retain macro F1 within 0.001 per view.
D masks and reference counts must be identical. Rank eligible candidates by
unrounded v2 spell F1, macro F1 for exact ties, then ID. The aspiration remains
v2 spell F1 above 0.60. No automatic promotion of a new arm follows.

Control replay precedes fitting/scoring. Candidate weights, evaluation code,
cohort identities and recipe are sealed before candidate evaluation. Reports
will distinguish the paired continuation effect from each data intervention,
including every failed guard and provider/seed spread.

## Evidence and limitations

- [Frozen plan](results/v2_extra_data_v1_plan.json).
- [Previous frozen selection](results/v2_spell_frontier_v1_selection.json).

These five matches have been repeatedly exposed to development. The v2 views
are synthetic and model-conditioned. Any gain here needs further confirmation;
GAR silver segments and the reused 17-possession public window do not establish
independent continuous-match gold performance. The exact previously approved
package's GAR TEST check is a separate deployment diagnostic, not a source of
training choices for this study.
