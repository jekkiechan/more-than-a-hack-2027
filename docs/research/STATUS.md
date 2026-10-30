# Current research state

Updated 2026-10-30 after the user-authorized SoccerNet-capacity promotion;
the working baseline is `joint_gru_soccernet_capacity_ensemble_v1`.
Earlier study records below remain historical. The former combined-objective
package came from [PR #29](https://github.com/jekkiechan/more-than-a-hack-2027/pull/29)
and remains the unchanged D-teacher source. Package and restore details live in
[BASELINE](../../experiments/BASELINE.md).
Check live PR state before resuming. This is the current coordination summary;
package details belong in [BASELINE](../../experiments/BASELINE.md).

## SoccerNet-GAR H/A continuation — 30 October 2026

[Stage 1](../../experiments/SOCCERNET_EXTENSION.md) completed 18 full-provider
H/A-only warm-start fits. The actual two-seed GAR50 probability ensemble
scores .631957/.495121 clean/feed-like, against baseline .633205/.498830 and a
matched old-data-only control .629253/.495753. GAR50 gains .002703 clean over
old-only but loses .001248 clean and .003709 feed-like to baseline; it fails
the predeclared +.01 clean improvement screen. Provider clean D/macro and
feed-like spell guards pass. Keep `joint_gru_combined_ensemble_v1` with no
promotion at the stage-1 decision. This remains stage 1's frozen outcome; the
later capacity package was promoted under a separately authorized exception.

The requested larger-capacity continuation is complete; its results and the
single post-selection public diagnostic are recorded below. The five
development matches and 17-spell public window are previously exposed, so the
public result is only a small pipeline check. SoccerNet uses sparse
PASS/CROSS/SHOT center-team silver labels, not dense possession labels. GAR
valid/test data were excluded from training and evaluation; the organizer reference was scored once only
after external selection. Artifacts remain in the local ignored data
directory; no private upload was made. Read the capacity report for numeric
parity, verification and amendment details.

### SoccerNet capacity/data-scaling continuation — external and public checks complete

The frozen `soccernet_capacity_v1` study widened only conditional H/A GRUs and
used 42,919 sparse event-team anchors from 45 disjoint 2022 World Cup TRAIN
fixtures; 60 conflicting event-team centers remain unknown. The official
eligible event counts are 40,745 PASS, 1,514 CROSS, and 720 SHOT (94.80% PASS).
Training preserves the baseline normalizer, D path, and original-data
spell/transition objectives. The selected width-96 two-seed ensemble scores
.641790/.502586 clean/feed-like, versus baseline .633205/.498830. A matched
equal-budget old-only control scores .633652/.494188; the SoccerNet result
gains .008138/.008398 over that control. All declared provider/feed-like
guards pass, but width 96 misses the separate +.01 clean improvement screen
by .001415. The frozen study did not automatically promote the model; the user
later authorized adoption with this exception disclosed.

The external selection and six-fit old-only attribution control are complete.
The one-time post-selection public diagnostic scores .569697 mean spell F1,
.861288 macro F1, and .855417 D-F1 versus the saved baseline .557576/.854088/
.854810. Both candidates predict 16 spells against 17 references over 7,500
graded frames; the mean spell-F1 bootstrap intervals overlap broadly. The
public window is a small noisy pipeline check, not confirmation. Its score was
already known when the user authorized the later promotion, but it was not a
model-selection or tuning input. The frozen study did not adopt width 96
because it misses the +.01 clean screen; the later user approval explicitly
accepted that exception.

All final independent receipts pass: coarse 195-row replay, selected-final
180-row replay, matched control 45-row replay, pooled weight seals, and exact
public prediction/CSV replay. The public verifier opened no reference file and
did not rescore metrics; the separate one-time public score gate did access
the reference once. The final suite has 353 experiment tests and 18 root tests;
file-backed fit parity and CSV format checks pass. The
[report](../../experiments/SOCCERNET_CAPACITY.md) records methods, uncertainty,
the runtime-only metadata/parity amendments, and checker correction. The local
artifact inventory hashes 1,187 files (about 1.96 GB) across stage 1 and this
capacity continuation; it is not a remote restoration archive. Source
snapshots, weights, and verification artifacts remain local, with no private
archive upload.

## Smart Tagging — 29 October 2026

The [fixed Smart Tagging study](../../experiments/SMART_TAGGING_VALIDATION.md)
completed a genuine-XML audit, label-free predictions and one separately
user-requested public diagnostic. Its bounded missing-ball H/A prior lowers
spell F1 **.557576 → .537143** while macro rises **.854088 → .866346**.
Predicted possessions rise 16 → 18 against 17 references; at IoU .5 matched
references stay ten and spurious predictions rise six → eight. Native P(D)
is bitwise unchanged; public D metrics are also unchanged. Keep
`joint_gru_combined_ensemble_v1`; no tuning or adoption follows.

The public window contains only 7,500 graded frames/17 possessions. No external
Smart Tagging accuracy comparison or new feed-like evaluation ran, and
full-match performance remains unknown. Genuine tags supply directional prior
evidence on 32.98% of missing-ball frames after taper/conflict handling. An
independent match with same-system tags and blind continuous possession labels
is needed for calibration or credible efficacy evidence.

Luna subagents implemented/reviewed the adapter and audit, found pre-reference
checker-only issues, and independently verified the final prior/decoder replay
without rescoring labels. Original/final predictions are byte-identical. The
[single public gate](../../experiments/results/smart_tagging_public_check_started.json)
is closed; historical gates and active manifests/policy remain unchanged.
291 experiment contracts, 18 root tests, file-backed fit parity and format
checks pass. New artifacts are locally verified, without a new private upload.
[Verification](../../experiments/results/smart_tagging_verification.json) and
[artifact inventory](../../experiments/results/smart_tagging_artifacts.json)
record exact scope and availability.

## SkillCorner removal — 29 October 2026

The [direct source-removal study](../../experiments/SKILLCORNER_REMOVAL.md) completes six matched-budget fits without the original eight SkillCorner training matches. On the unchanged original five-match full-provider benchmark, clean/feed-like spell declines **.633205/.498830 → .625786/.496092**. Both seeds decline in both views. SkillCorner's three primary evaluation cells are exact parity; the changed Sportec/Metrica mean declines .011128/.004106.

A separate provider-represented Sportec/Metrica supplement improves after removal (+.013053 clean / +.012420 feed-like), largely from Sportec. Metrica gains only .002720 clean and loses .007082 feed-like. The result shows a transfer/specialization tradeoff, not a demonstrated competition gain from deletion. Only the frozen primary clean-spell screen fails; the other thirteen guards pass. Retain the active dataset, combined-objective ensemble and policy; no automatic promotion or organizer scoring occurs.

The six sealed checkpoints pass independent budget/model/normalizer checks. All 283 repository contracts, exact file-backed fit parity and the bundled submission format pass. Independent external verification completed in a documented replay-prefix/aggregation sequence; all thirty cells, 180 rows and twenty-four summaries pass. Its checker-only corrections and executed-source-byte limitation are audited. The local archive is SHA-256 verified after restoring 370 files; no new private upload was attempted. The detailed report owns its availability.

## SkillCorner expansion — 29 October 2026

The user-authorized [matched-budget data study](../../experiments/SKILLCORNER_EXPANSION.md) completes six fits with six additional SkillCorner training matches. On the identical seven-match cohort, full-provider clean/feed-like spell is .641468/.510206 → .643523/.502104; pooled clean/feed-like is .696722/.554822 → .686765/.546531. Both frozen clean screens fail; retain the active thirteen-match training set and combined-objective ensemble. All other feed-like and provider clean frame-quality guards pass. The nine acquired matches remain experimental; reserved 2017461 stays unscored. The separately requested fixed public diagnostic also loses spell F1: .557576 → .548571, despite macro .854088 → .874121. It follows the sealed external decision and causes no selection or tuning.

The new labels are provider/event silver, and the six training matches add 4.66 usable hours under the existing visibility mask. Sparse retained native ball gaps limit competition representativeness. Full methods, paired seed/provider rows and verification evidence belong in the study report. The new private archive is frozen locally; automatic approval review blocked upload to `jekkiechan/mtah-data` pending explicit destination-specific authorization. No remote restore is claimed.

## Current adopted baseline

`joint_gru_soccernet_capacity_ensemble_v1` uses width-96 pooled conditional H/A
heads and the unchanged combined-objective seed-2026/2027 hidden-24 D teachers,
followed by the unchanged cost-3 Viterbi decoder. Its external clean/feed-like
spell scores are .641790/.502586 versus .633205/.498830 for the prior package;
matched equal-budget old-only is .633652/.494188. The frozen study missed the
+0.01 clean screen by .001415, while passing provider D/macro and feed-like
guards. The user authorized adoption afterward with that exception disclosed.
The public diagnostic was already known at adoption but was not selection
evidence; its score is .569697 versus .557576 on only 7,500 frames and 17
references. No fresh confirmation or full-match result is established. See the
[promotion guide](../../experiments/BASELINE.md) and
[capacity report](../../experiments/SOCCERNET_CAPACITY.md).
The promotion has local restore only: the generic private-cache default still
restores the former package, and no remote cache/archive was updated.

The [former combined-objective package](../../experiments/OBJECTIVE_ENSEMBLE.md)
is archived. Its fitted D teachers, old manifest, policy, and historical
adoption record remain unchanged.

### Former package: deployment and public check

The former package's two pooled deployment fits used the same 13 TRAIN matches
and exact normalization/schema/budget parity. Its 199,319-row label-free CSV
is format-validated and reproduces through the baseline entrypoint. This was a
development adoption on exposed matches and synthetic stress, not independent
confirmation; the prior loss study's failed +.01 both-seed screen remains
unchanged.

Its separately requested fixed public diagnostic was run **after** that
adoption: both ensembles scored **.557576 spell**, with combined macro **.854088**
versus prior **.862375**. D-F1 improved .848746 → .854810. This does not
confirm a public-window gain; only 7,500 graded frames and 17 possessions are
available. No tuning followed that diagnostic. The [saved public
result](../../experiments/results/combined_objective_public_check.json) and
exclusive gate remain final. Full-match performance is unknown. Restore
details: [BASELINE](../../experiments/BASELINE.md).

## Spell-F1 follow-up — 29 October 2026

The [current-baseline opportunity audit](../../experiments/SPELL_OPPORTUNITY_AUDIT.md)
reproduces all fifteen saved full-provider cells. Under-three-second spells are
about 40% of reference possessions but 61% of clean misses; their IoU-.5 recall
is .384 clean and .203 feed-like. Wrong-team errors remain a promising target,
especially in synthetic long ball gaps. Native clean long-live-gap support
remains only 108 frames, and SkillCorner grading-edge censoring still limits
boundary conclusions.

The completed [window-overlap study](../../experiments/WINDOW_OVERLAP.md)
uses zero new fits and one fixed five-second shifted tiling. Clean/feed-like
spell is .633939/.498330 versus the adopted .633205/.498830: +.000734/-.000500.
It fails the declared +.01 clean screen. Independent replay verifies all fifteen
candidate/control cells and the aggregate. Retain the baseline; do not sweep
offsets or weights on these results.

The [contact-memory study](../../experiments/CONTACT_MEMORY.md) completed six
small adapters on frozen combined-objective backbones and fifteen ensemble
cells. Clean/feed-like spell is .628430/.499523: -.004775/+.000694 versus the
baseline. It fails even its original positive-clean-gain screen, as well as
the separately recorded +.01 high-value screen. Short-spell recall rises
slightly, but overall possession recovery falls on every clean provider.
Archived member/ensemble P(D) remains exact; decoded D metrics can still change.
The numerical replay-guard correction is documented separately and changes no
models, predictions, decoder settings, or criteria. No candidate is adopted.

The [independent-data search](../../experiments/INDEPENDENT_DATA_SEARCH.md)
acquired and hash-verified small SoccerTrack v2 match-117093 samples from its
official public Drive mirror. Full panorama video and player tracks are
available, but continuous H/A/D labels are absent and the ball coordinates are
interpolated from event annotations. This is a new event-silver/annotation-pilot
route, not native broadcast confirmation. PFF access remains pending; legacy
full-match SNGAR requires access. No organizer labels, scoring gates or working
model files changed in this follow-up.

Both candidate studies have independent official-score replay receipts and
226 passing contracts. The [shared private archive](../../experiments/results/spell_f1_followup_artifacts.json)
was uploaded after explicit user authorization. A [fresh remote restore](../../experiments/results/spell_f1_followup_remote_verification.json) verified all 295 files and hashes at private commit `7231018`. The earlier automatic approval rejection is retained as history.

## Short-spell continuation — 29 October 2026

The [recoverability audit](../../experiments/SHORT_SPELL_RECOVERABILITY.md)
separates complete under-three-second misses from grading-edge spells. Among
complete short misses, 15% clean / 10% feed-like have correct-team raw emissions
that cost 3 entirely suppresses; 40% / 39% have insufficiently aligned intervals,
and 45% / 51% have no local correct-team top-1 emission. Raw argmax raises
complete-short recall but lowers overall spell F1 to .561/.419. This is a
saved-output diagnostic, not an oracle gain estimate or a decoder candidate.

The [native-rate emission study](../../experiments/NATIVE_SHORT_SPELL.md)
completed twelve frozen-backbone residual fits: native 10/25 Hz versus matched
5 Hz, two seeds and three whole-provider exclusions. Native clean/feed-like
spell is .629784/.498723; matched 5 Hz is .630100/.499242, against the saved
baseline .633205/.498830. Native timing therefore shows no aggregate advantage
in this instantaneous linear-residual recipe. Both arms fail the +.01 clean
screen; provider guards pass. A reviewed pre-fit correction fixed only an
active-provider mass assertion and preparation scheduling, preserving frozen
source, data, weights and fitting choices.

The [turnover-edge decoder](../../TURNOVER_DECODER.md) completed three heads
and fifteen paired cells with unchanged emissions. V1 weights summed to one,
so its fixed L2 penalty kept costs near 3: clean spell was unchanged and
feed-like improved only .000166. The separately declared
[mean-one-weight follow-up](../../TURNOVER_DECODER_V2.md) changed only the
regularization units and completed three new heads before scoring. Short-spell
recall rises .021727/.008559, but false extra spells rise 6.889/3.500 per match,
and spell F1 falls to .631627/.495856. Both recipes fail the clean screen.
No cost, threshold or regularization sweep follows these results.

All three fitted studies have saved-array independent replay receipts with
zero mismatches. The full repository suite passes 248 tests; the file-backed
fit contract and bundled CSV format check also pass. No organizer labels or
scoring gates were accessed. Keep `joint_gru_combined_ensemble_v1`; none of
these candidates is adopted. The [private archive receipt](../../experiments/results/short_spell_recovery_remote_verification.json)
verifies all 420 files (452,899,231 bytes) from a fresh remote clone at private
commit `0ff246b`. Earlier per-study local-only receipts are preserved as history.

## Historical architecture comparison

The completed [temporal architecture comparison](../../experiments/TEMPORAL_ARCHITECTURE.md)
tested the former `joint_gru_slotdrop_v1` recipe as its GRU reference, alongside
BiLSTM, Transformer and TCN, under a fixed budget: 18 challenger fits and 200
cells across two seeds and three full-provider holdouts. Its GRU clean/feed-like
means of .6132/.4798 summarize two separate fits, not a prediction ensemble.
All challengers failed the declared both-seed screen, supporting retention of
the GRU family for that recipe and budget. The then-adopted
`joint_gru_seed_ensemble_v1` scored .6265/.4874 in the separate later
[seed-ensemble study](../../experiments/SEED_GAP_CONTEXT.md); it was not tested
in the architecture comparison. Do not read the historical GRU mean as a score
or matched comparison for the adopted ensemble.

The report and aggregate/receipt JSON files are published here. Its frozen
implementation remains only in local working-branch snapshots `c31afbc`
(implementation) and `9f023de` (comparison result); no code, weights or frame
outputs are included in this documentation publication.

## Completed architecture-ensemble comparison

The [zero-fit architecture-ensemble study](../../experiments/TEMPORAL_ARCHITECTURE_ENSEMBLES.md)
scored 900 cells across eleven pooled blends and same-seed diagnostics with no
new fits. All eleven pooled blends fail the declared screen; retain the then-adopted
original-loss two-seed GRU. The closest blend, GRU+TCN, scored .626189/.483176 clean/feed-like
versus .626531/.487419 for the adopted GRU. GRU+BiLSTM+TCN scored
.624938/.487455: feed-like was effectively flat while primary clean spell
declined. Same-seed tables are diagnostics, not eligibility results. These are
reused development matches, not fresh confirmation; no promotion or organizer
scoring occurred.

Independent verification passed all 900 member-mean probability checks,
cost-3 decodes and official metric/diagnostic replays, as well as 25 native
views, 200 member artifacts, 50 original GRU controls and 15 pooled-GRU
reference cells. It reproduced 108 summaries and all 11 eligibility screens;
all 2,101 manifest entries validated. The 175 existing contract tests and 9
ensemble tests passed. The implementation remains on local working-branch
commit `ce0a159`; no model code, weights, probability arrays or frame-level
outputs are part of this documentation change. The [verification receipt](../../experiments/results/temporal_architecture_ensembles_verification.json)
and [artifact inventory](../../experiments/results/temporal_architecture_ensembles_artifacts.json)
record the replay and manifest results.

[Experiment progress Page](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9)
shows dated milestones and interactive charts; update it when promising work completes.

## Latest evidence

| Scope / evidence | Finding | Decision |
| --- | --- | --- |
| Completed [combined-objective ensemble comparison](../../experiments/OBJECTIVE_ENSEMBLE.md), full provider holdouts | Fixed saved-probability blend improved clean/feed-like spell .626531/.487419 → .633205/.498830; its practical guards passed. SkillCorner spell declined about .007 in both views. Independently verified; two pooled deployment fits and entrypoint replay completed. | Adopted at the time and later superseded by the user-authorized SoccerNet-capacity package. Its own fixed public diagnostic and decision remain historical. |
| Completed [seed ensemble / gap-context study](../../experiments/SEED_GAP_CONTEXT.md), full provider holdouts | Actual ensemble .6265/.4874; 30/30 parity. Live feed-like gaps >=10s have 24.3% wrong team versus 5.1% false D; native clean support is only 108 live frames. Longer context helps local recovery but not spell consistently. | User adopts the original-context ensemble; fifteen-second context remains experimental. Independent labeled broadcast confirmation is pending. No organizer scoring. |
| Previous adopted recipe (`gru_hgb_ballgap25_v1`), historical **D-only provider holdout, pooled H/A**; [baseline](../../experiments/BASELINE.md) | Clean spell/macro .5740/.8439; composite-stress .4315/.7866. Previous .5746/.8449 and .4205/.7699. | Adopted robustness tradeoff, not a clean-score win. |
| Merged [tracking pilot](../../experiments/TOP_THREE_PILOT.md), PR #15 | 80 pooled development evaluations; matched gap/suppression arms lose overall spell. Fusion recovers 4,419 missing-ball frames (6.84%), with >10% overlapping ball positions disagreeing by >5 m. | No preprocessing/fusion adoption. |
| Completed [confidence abstention](../../experiments/CONFIDENCE_ABSTENTION.md); **D-only provider holdout, pooled H/A** | 135 evaluations; 15/15 baseline parity. Primary .60 X gate: clean spell .57402→.58240, stress .43152→.42608; macro .84394→.82607 / .78659→.73476. All eight gates fail guardrails. X/hold differences occur on SkillCorner native unknown-label gaps. | Retain baseline; no organizer scoring or promotion. |
| Draft [joint D repair](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/DEADBALL_TRANSFER.md), PR #16; **full H/A+D provider holdouts** | Baseline clean/stress spell .5338/.4129. Player-only D joint GRU .6093/.2991, averaging two separate seeds. Clean SkillCorner D collapse repaired; both seeds fail stress guardrail. | Retain baseline. Separate-seed means are not an ensemble. |
| Draft [completed public diagnostic](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/JOINT_COMPETITION_CHECK.md), PR #16 | Unchanged pooled seed-2026 model .6194 spell/.8574 macro/.8506 D; saved baseline .5257/.8588/.8568. Only 7,500 frames and 17 possessions. | Descriptive check; gate closed. No promotion, tuning or repeat scoring. |
| Completed [matched GRU slot-dropout study](../../experiments/GRU_SLOT_ROBUSTNESS.md), two seeds / full provider holdouts | Mean clean spell .6092 → .6132; mean stress .2991 → .3141. Stress improves seed 2026 and declines seed 2027; both candidate seeds remain below matched baseline .4129. | Robustness remains unresolved; retain baseline and evaluate the frozen full H/A/D tree ensembles. |
| Completed [full H/A/D LightGBM study](../../experiments/JOINT_LGBM.md), 12 fits / 300 cells | Ordered temporal features improve stress spell; 127 leaves/depth 10 improve on 63/depth 8. Full127 standalone clean/stress .5555/.4345; its dropout-GRU blends average .6101/.4226 across two separate seeds. All 20 recipes fail a provider guardrail. | Retain baseline. SkillCorner clean D F1 is .3335 standalone and .6308/.6602 in the promising blends, versus baseline .7022. Spell gains alone do not protect dead-ball recall. |
| Completed [spell metric audit](../../experiments/SPELL_METRIC_AUDIT.md), saved external predictions | All 30 baseline/original-joint score rows reproduce. Clean joint gain is strongest at strict IoU; stress loses at every threshold. 69.86% of SkillCorner reference spells touch grading-window edges. | Keep official scoring; report all five IoUs, duration recall, fragmentation and censored boundaries. |

The .5740 and .5338 baselines use different component holdouts; do not
subtract them as a model delta. All five external development matches have
prior exposure. The follow-up imports the required PR #16 implementation as
research dependencies; this is not model adoption. Older diagnostics remain closed.

The [spell F1 explainer](https://chatgpt.com/space/page_4198931ce5288191bc4f646773074e10)
contains exact scoring rules, worked examples and an interactive timeline.

## Next work

The next recommended investment is a [blind, continuously labeled ten-minute
SoccerTrack panorama pilot](../../experiments/SHORT_SPELL_NEXT.md), while pursuing
authorized true-broadcast access. A new short-possession emission/interval
hypothesis needs a distinct mechanism and stronger interval labels. Instantaneous native geometry
and learned H/A transition costs are now tested and fail the clean screen.
The completed contact-memory, window-overlap and turnover-decoder packages do
not justify more local parameter variants on these exposed matches.
The data pilot needs new H/A/D/X annotation and measured-ball provenance; its
event-interpolated ball coordinates are not independent input evidence.

The earlier proposals below remain historical leads, not a new ranking after
this follow-up. Freeze a [declaration](templates/EXPERIMENT.md) before fitting;
do not infer that a proposed study has already run.

| Priority / state | Next action | Required evidence |
| --- | --- | --- |
| 1 — Ready to design | Diagnose SkillCorner dead-ball under-recall in the complete H/A/D tree predictor and separate the useful temporal groups: reliable anchors, ordered context and persistence. The full127/dropout-GRU blends are promising robustness candidates, not adopted models. | Attribute source/ball-visibility effects before another capacity increase. Every tree arm must still output full H/A/D. Freeze matched feature/architecture comparisons, preserve both seeds and provider guards, and inspect D recall alongside all five spell IoUs. |
| 2 — Proposed | Isolate native player-frame blanking in the constrained GRU; compare otherwise identical composite views with that component omitted. [Diagnosis and bounded design](../../experiments/GRU_SLOT_ROBUSTNESS.md#proposed-native-frame-blanking-experiment). | Support loss rises sharply under stress, but causality is unproven. Proposed six new fits / 70 new cells; preserve both-seed clean/provider/stress guardrails and test false-D rates on reference-live frames. Not yet declared or run. |
| 3 — Data blocked | Obtain an independent labeled broadcast match; follow [private issue #4](https://github.com/jekkiechan/mtah-data/issues/4) for rights, blind annotation, tracking/alignment and sealed confirmation. | PFF access was requested; response not established here. Audio/video or model-generated labels do not supply independent H/A/D ground truth. |
| 4 — Ready to design | Calibrate pilot player defects and obtain independent ball-fusion confidence before integration. | Separate label-free calibration/audit partitions, preserved native masks, position-quality evidence and external possession evaluation. Coverage alone is insufficient. |
| 5 — Deferred | Boundary/audio/decoder work or broad tuning only after checking completed studies. Fixed confidence abstention/holding at .50/.60/.70/.80 is now tested and fails clean/stress guardrails. | A distinct hypothesis and matched external controls; use independent, more completely labeled broadcast data before generalizing native-X benefits or tuning more gates. |

New comparative work should hold out the provider from every learned H/A and D
component and normalization. The historical policy file remains unchanged for
frozen package reproducibility; its older D-only preferred view does not support
a full-pipeline transfer claim. Spell F1 remains primary, macro F1 the exact-tie
break, and the fixed 70/30 summary auxiliary. No global weighted score can hide
a failed provider or corruption guardrail.

## Open work and artifacts

The user-requested [seed/gap-context study](../../experiments/SEED_GAP_CONTEXT.md)
is complete: six matched fits, 75 score cells, exact normalization/budget parity,
15 exact probability ensembles and 45 independent official score replays.
[Complete private artifact receipt](../../experiments/results/seed_gap_context_artifacts.json)
records remote verification. Next action is genuine labeled broadcast long-gap
evidence, rather than another context sweep on this exposed synthetic benchmark.

Reusable preparation infrastructure (2026-10-28):
[`preparation.py`](../../experiments/preparation.py) resolves the existing
whole-match/provider manifest and caches unnormalized joint features by input,
code, dependency and view hashes. The
[preparation skill](../../.agents/skills/possession-preparation/SKILL.md) owns
usage and limits. Synthetic contracts cover provider isolation, grading,
time breaks and cache integrity. Private-array preparation and runtime savings
have not been measured here. Next new runner can import this API; frozen study
runners, declarations, baseline and scoring gates are unchanged.

The user-requested [competition check of four temporal candidates](../../experiments/TEMPORAL_COMPETITION_CHECK.md) is complete and its gate is closed: dropout GRU .5818, robustness blend .5818, clean leader blend .5500, Full127 .5371 spell F1. It was resumed with a [recorded cache-rehash deviation](../../experiments/results/temporal_competition_resume_deviation.json). The [stress-component ablation](../../experiments/STRESS_COMPONENT_ABLATION.md) and the promotion are in PR #20.


The completed confidence-abstention cache is uploaded on private
`mtah-data` branch `experiments/confidence-abstention-v1` at `f240858`;
all 213 files were independently downloaded and hash-verified. The
[receipt](../../experiments/results/confidence_abstention_artifacts.json)
records the restore manifest and byte-identical aggregate replay. No model
checkpoint or baseline adoption changed.

| PR / inspected head | State at review |
| --- | --- |
| [#18](https://github.com/jekkiechan/more-than-a-hack-2027/pull/18), `possession/temporal-spell-robustness` | Completed research in a draft PR: metric audit, interactive explainer, 12 GRU fits / 60 cells and 12 full H/A/D tree fits / 300 cells. Independent verification checks all 360 cells, source/input/model seals, full-vector blends and guardrails. All 135 tests pass after integrating the independently completed confidence-abstention study from main. GRU remote restore is verified. The complete 738-file archive is locally hash-verified; its upload is blocked by automatic approval review pending explicit user authorization. [Artifact receipt](../../experiments/results/temporal_followup_artifacts.json). No adoption. |
| [#16](https://github.com/jekkiechan/more-than-a-hack-2027/pull/16), `possession/live-gap-had-gru`, `8c704aa` | Still draft. Original private archive restored and all 143 manifest files verified, including 18 external arrays. Follow-up player-only D checkpoints remain absent from the remote branch; new controls are reproduced explicitly rather than claimed restored. |
| [#13](https://github.com/jekkiechan/more-than-a-hack-2027/pull/13), `possession/gap-results`, `33c03ca` | Draft reports/aggregates only; experimental implementation excluded. |
| [#11](https://github.com/jekkiechan/more-than-a-hack-2027/pull/11), `possession/spell-balanced-attribution`, `b93ce36` | Draft attribution study; no adoption. |
| [#8](https://github.com/jekkiechan/more-than-a-hack-2027/pull/8), `possession/robust-pipeline`, `7433fef` | Older draft contextual pipeline; historical comparator. |

At handoff update the review date, branch/commit, outcome, artifact availability,
closed-check receipt location and next action. Do not merge or promote unrelated
draft work as part of documentation maintenance.
