# A 0.60 spell-F1 target on the v2 feed simulation

Date: 3 November 2026. Status: running; adoption unchanged.

The user requested a new attempt to improve `feed_like_v2` spell F1 to 0.60,
using the latest data and Luna implementation agents. This report covers three
bounded studies and one shared evaluation of the resulting training recipe.
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

The calibration and training comparisons are running. The completed consensus
study does not reach 0.60. The active package, policy, archived guard failures
and closed organizer scoring gates remain unchanged.

| Fixed arm | Clean spell F1 | V2 spell F1 | V2 macro F1 |
| --- | ---: | ---: | ---: |
| Adopted original-full phase mean | 0.667556 | 0.583571 | 0.849975 |
| Calibrated phase mean | 0.658880 | 0.588020 | 0.852534 |
| Calibrated decoded-state vote | 0.657054 | 0.589139 | 0.852957 |
| Original-full decoded-state vote | 0.668127 | 0.584648 | 0.850136 |
| Fixed 50/50 phase-emission blend | 0.664688 | 0.587677 | 0.851829 |

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

## Scope and delivery

V2 remains a synthetic, model-conditioned calibration with known fidelity
limitations. Repeated selection on the five development matches does not add
independent matches. This work makes no GAR TEST or continuous-match gold
confirmation claim.

New training views, checkpoints and native predictions stay in ignored local
storage. Aggregate reports and provenance receipts are suitable for the
research repository. Remote availability of older restored archives does not
establish remote availability of the new artifacts.
