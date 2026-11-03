# A 0.60 spell-F1 target on the v2 feed simulation

Date: 3 November 2026. Status: completed; retain the adopted package.

**Best eligible v2 spell F1: 0.592749**, versus the adopted package's
0.583571 (+0.009178). Clean spell is 0.665355 versus 0.667556 (-0.002201).
The 0.60 target remains unmet by 0.007251. The winner is a fixed 50/50
probability blend of the new four-view model and archived calibrated model,
with native phase averaging and H/A transition cost 1.5. All common retention
checks pass; the +0.01 clean improvement screen still fails. These are exposed
development results, not a new deployment or independent confirmation.

The user requested a new attempt to improve `feed_like_v2` spell F1 to 0.60,
using the latest data and Luna implementation agents. This report covers
bounded consensus, calibration, training and decoder-transfer comparisons,
plus a final fixed refinement of H/A transition cost.
The target is a development objective, not a guarantee or an independent
estimate of organizer-match performance.

## Evidence and eligible data

The current original-full phase-average package has saved clean/v2 spell F1
**0.667556 / 0.583571**. Calibrated dense with phases has saved
**0.658880 / 0.588020**. The exact archived provider-excluded checkpoints,
teachers and development tracking inputs are restored and checked against
their private archive manifests before replay.

The new PCBAS evidence supplies sparse action anchors with no ball tracking,
not continuous possession supervision. The [GAR no-ball bridge](GAR_NOBALL_BRIDGE_V1.md)
attributes the main PCBAS error pattern to player-only inference and does not
prioritize PCBAS team-head training. The earlier SkillCorner expansion failed
its clean improvement gates. These findings inform the present recipe; they
do not justify treating those sources as new dense gold labels.

The training pool remains the eligible 13 original-provider TRAIN matches and
45 SoccerNet-GAR TRAIN games. GAR labels are generated silver labels. No
organizer reference labels, PCBAS reserves, GAR VALID or GAR TEST games are
used for fitting or candidate selection in these studies.

## Fixed comparisons

All comparisons use the same five previously exposed development matches,
full H/A and D provider exclusions, native unknown targets and time breaks,
clean inputs and four fixed `feed_like_v2` realizations. Average realizations
within match, then matches within provider, then providers equally. Native
phase averaging uses five phases at 25 Hz and two at 10 Hz. Official spell F1
is primary, macro F1 breaks exact ties, and the fixed 70/30 score is auxiliary.

- **Phase-decoder consensus** compares the two archived model families using
  mean emissions, decoded-state plurality, inverse-entropy weighted votes and
  an 80% agreement gate, plus one fixed 50/50 family blend: nine arms.
- **V2 decoder calibration** compares 109 fixed arms on those archived
  emissions. D temperature is 2/4/8; D and H/A transition costs are
  1.5/3/6; H/A temperature is 0.65/1. The grid includes exact f062
  (D temperature 4, H/A temperature 1, both costs 3) and a fixed family blend.
  This is selection on exposed development data.
- **V2 original-possession training** replaces the original-full recipe's
  single v1 augmented original-provider view with four fixed v2 realizations.
  Draw-time round robin preserves the existing clean/augmented record mass
  and the original/GAR sampler schedule. The short-possession diversity
  sampler, GAR records, fold-teacher normalization, width 96, batch 32,
  50/50 original/GAR mixture, learning rates, losses and 2,048 updates stay
  fixed. Three provider exclusions and two fixed seeds produce six fits.
- **Training evaluation** has three declared arms: the new full model;
  its conditional H/A probabilities crossed with archived calibrated D
  probabilities within seed; and a 50/50 new/calibrated probability blend.
  All use native phase averaging and unchanged f062. Component-seed results
  are descriptive diagnostics, with no seed selection.

The training comparison changes augmentation generation and its number of
fixed realizations together. It cannot isolate the effect of four views
versus one. The historical one-view v2-trained result used a different sampler
and is a separate diagnostic, not its matched control.

## Replay and execution records

Candidate scoring requires a controls-only official replay of both archived
families, reproducing saved clean/v2 spell and macro F1 within 1e-8. Source,
calibration, models, teachers, input files and caches are hash-bound. Training
and evaluation declarations precede fits; a separate six-model seal precedes
new-model scoring. The completed replay reproduces all eight saved aggregates
with **zero error** on all 25 cells, including 100 component-seed diagnostics.

The first consensus runner failed before caching emissions because it imported
the wrong tracking helper. Its source and declaration are preserved. The
reviewed runtime wrapper corrects that import and adds bound prepared-cell
caches. Its first execution used system Python without PyTorch and stopped
before scoring; replay resumed with the repository's pinned virtual environment.
These failed attempts are recorded separately, without changing their frozen
declarations or opening any organizer scoring gate.

## Results and decision

All five comparisons are complete. None reaches 0.60. The active package,
policy, archived guard failures and closed organizer scoring gates remain
unchanged.

| Fixed arm | Clean spell F1 | V2 spell F1 | V2 macro F1 |
| --- | ---: | ---: | ---: |
| Adopted original-full phase mean | 0.667556 | 0.583571 | 0.849975 |
| Calibrated phase mean | 0.658880 | 0.588020 | 0.852534 |
| Calibrated decoded-state vote | 0.657054 | 0.589139 | 0.852957 |
| Original-full decoded-state vote | 0.668127 | 0.584648 | 0.850136 |
| Fixed 50/50 phase-emission blend | 0.664688 | 0.587677 | 0.851829 |
| Calibrated phase mean, H/A transition cost 1.5 | 0.666601 | 0.589658 | 0.853511 |
| New four-view full model, f062 | 0.658713 | 0.590072 | 0.852075 |
| New conditional H/A with calibrated D, f062 | 0.661772 | 0.590081 | 0.850852 |
| New/calibrated 50/50 blend, f062 | 0.663694 | 0.591393 | 0.852927 |
| New/calibrated blend, H/A cost 1.5 — eligible winner | 0.665355 | 0.592749 | 0.854043 |
| New/calibrated blend, H/A cost 2 — fails clean floor | 0.662538 | 0.592926 | 0.853786 |

The numerical consensus leader gains 0.001119 against its matched calibrated
control, but loses 0.010501 clean spell versus the adopted package. A separate,
explicitly post-hoc common readout applies the calibration study's conservative
guards to all nine consensus arms: aggregate clean spell within 0.005 of the
adopted package, each provider's clean macro/D-F1 within 0.02, and v2 spell
within 0.01. Under those checks the fixed family blend is the best eligible
consensus-study arm. These checks do not replace the frozen study's original
family-relative provider diagnostics or the historical failed clean screen.

The [independent readout verification](results/v2_spell_target_readout_verification.json)
recomputes all saved aggregate/provider means within 1e-14, verifies 25/25 cells
and nine arms per cell, and confirms unchanged reference spell counts. The
[full result](results/v2_spell_target_runtime.json) retains all arms and metrics;
the [preflight](results/v2_spell_target_runtime_preflight.json) retains controls
and component-seed diagnostics.

The [109-arm calibration result](results/v2_calibration_target.json) selects
`adop_dt4_dead3_ha1.5_ht1` as both numerical and eligible leader. Relative to
the matched calibrated phase-mean control, only H/A transition cost changes
from 3 to 1.5: clean spell gains 0.007721 and v2 spell gains 0.001638. Relative
to the adopted original-full package, this changes both model family and
decoder; v2 spell gains 0.006087 while clean spell falls 0.000955. All declared
clean/provider guards pass. The new +0.01 clean-gain diagnostic fails, and
the historical failed screen remains unchanged.

The selected configuration scores 0.583672 and 0.578551 v2 spell on component
seeds 2026 and 2027. The 0.589658 ensemble averages probabilities before
decoding; it is not an average of those seed scores. Independent reduction of
all 109 arms' saved cell metrics matches their aggregate/provider means within
1e-12.

All six new fits completed their fixed 2,048 updates, with measured training
wall times of 375.1–409.6 seconds per fit, excluding preparation and loading.
An independent six-fit audit checks model/teacher hashes, held-out-provider
exclusion, 32,768 draws from each source per fit, original sampler parity and
per-match realization balance within one draw. The
[training artifact receipt](results/v2_short_original_diversity_artifacts.json)
records local verification of all 52 prepared views and six models.

The [three-arm new-model result](results/v2_training_eval.json) completes all
25 cells and records 75 ensemble rows plus 150 component-seed rows. The fixed
50/50 new/calibrated blend is strongest at 0.591393 v2 spell, improving the
adopted package by 0.007822 with a clean change of -0.003862. It passes the
common checks against the adopted original-full package. The new full model
and conditional H/A cross fail the common clean-spell floor. The frozen
evaluator separately reports calibrated-baseline guards; all three pass those
less restrictive clean comparisons. The
[independent readout](results/v2_training_eval_readout_verification.json)
keeps these two reference choices separate and verifies all aggregates within
1e-14, fixed reference counts and all 160 prediction caches.

Evaluation required mechanical runtime amendments for checkpoint-name aliases
and scalar types in cache identity metadata. The latter followed a partial
ten-cell attempt without an aggregate result. All prediction arrays and model
hashes were preserved, all 160 caches were independently validated, and the
complete fixed cohort was replayed. No decoder or model setting changed from
those partial results; the original runners, declarations and failure records
remain available.

After selecting the old-model calibration configuration, a separate fixed
transfer study freezes that one setting on all three new-model arms before
the new-model aggregate results exist. It adds no grid, fit or seed selection.
This is further development on the same exposed cohort, not fresh evidence.
The [transfer result](results/v2_calibration_transfer.json) completes all three
arms and selects the 50/50 blend at 0.592749 v2 spell. Its
[independent verification](results/v2_calibration_transfer_readout_verification.json)
checks the full cohort, reference counts, reduction and guards.

The calibration optimum used the lowest H/A cost in its grid. A final
[configuration plan](results/v2_calibration_ha_boundary_plan.json) therefore
fixes costs 0.5/0.75/1/1.5/2 on the archived calibrated recipe and the three
new-model recipes, holding D temperature 4, D cost 3 and H/A temperature 1.
Its 20 arms ran because the fixed transfer did not reach 0.60. This plan was
recorded before the transfer aggregate existed; no further cost choices follow
the result. It remains selection on the same exposed development matches.

The [final boundary result](results/v2_calibration_ha_boundary_v1.json) does
not improve the eligible winner. H/A cost 2 produces the numerical maximum,
0.592926, but its clean spell 0.662538 is 0.00001735 below the fixed 0.662556
floor. That guard is not relaxed. Selected-setting seed v2 spell scores are
0.586460 and 0.585790; the probability ensemble is 0.592749. The
[independent reduction](results/v2_calibration_ha_boundary_independent_aggregate_verification.json)
matches aggregate/provider values within 2.22e-16.

The final runner required seed-diagnostic tuple/mapping adapters after its
pooled scoring pass. The corrected execution persisted and hash-bound all
500 candidate scores before reduction, reused those scores for the seed
readout, and changed no arm, prediction or selection rule. One initial
pre-score recovery receipt was replaced to add orchestration metadata before
scoring; its original bytes were not retained. This preservation exception is
explicit in the [provenance note](results/v2_calibration_ha_boundary_recovery_provenance_note.json).
It does not change the original study declaration, source, inputs or scores.

## Validation and next evidence

The final local suite passes 639 experiment tests and 26 root tests. The
bundled 199,319-row CSV passes format validation; no new organizer CSV or public
reference score is produced. Documentation links and whitespace are checked
with the delivered report and index updates. Independent readouts verify each
completed cohort and its reductions; the controls-only replay has zero error.

This delivery retains the current default. A deployment candidate would need
its own pooled training/package and exact-package GAR comparison; the six
new fits here are provider-excluded research models. Future GAR TEST reuse
must be recorded as additional exposure. Independent annotated continuous
matches remain the stronger confirmation need; the existing new PCBAS
anchors cannot supply continuous possession spell labels or observed balls.

## Scope and delivery

V2 remains a synthetic, model-conditioned calibration with known fidelity
limitations. Repeated selection on the five development matches does not add
independent matches. This work makes no GAR TEST or continuous-match gold
confirmation claim.

New training views, checkpoints and native predictions stay in ignored local
storage. Aggregate reports and provenance receipts are suitable for the
research repository. Remote availability of older restored archives does not
establish remote availability of the new artifacts.

Code and aggregate evidence are delivered in
[PR #82](https://github.com/jekkiechan/more-than-a-hack-2027/pull/82).
The [research progress site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
is published at version 29 after desktop/mobile layout and chart-control checks.
The existing restricted audience is unchanged at access revision 2. The
[delivery receipt](results/v2_spell_target_delivery.json) records source,
archive, native publication and validation metadata. The first automatic
publication review rejected the action; a fresh owner/allowlist check resolved
the concern and the same native tool published the saved version successfully.
