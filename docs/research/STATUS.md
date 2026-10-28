# Current research state

Reviewed 2026-10-28 (promotion); earlier review 2026-10-27: fork main `5005115`, PR #16 at `8c704aa`, and the
`possession/temporal-spell-robustness` follow-up.
Check live PR state before resuming. This is the current coordination summary;
package details belong in [BASELINE](../../experiments/BASELINE.md).

## Adopted baseline

`joint_gru_slotdrop_v1` (promoted 2026-10-28): pooled slot-dropout joint GRU,
player-only D head, seed 2026, Viterbi cost 3. External full-provider-holdout
clean spell .6088/.6176 versus the previous baseline's .5338; feed-like stress
.4742/.4853 versus .4328. It passes every guardrail once feed-like stress replaces
composite stress, a change decided after the composite results were known
([ablation](../../experiments/STRESS_COMPONENT_ABLATION.md)). Public five-minute
check .5818 spell / .8689 macro versus .5257 / .8588. Single seed, so full-match
performance is unknown. Package details: [BASELINE](../../experiments/BASELINE.md).
The previous `gru_hgb_ballgap25_v1` is archived under `experiments/baselines/`.

## Running objective comparison

The user-requested [objective-alignment study](../../experiments/OBJECTIVE_ALIGNMENT.md)
tests capped short-possession weighting, a cost-sensitive transition auxiliary,
and their fixed combination against the adopted slot-dropout GRU recipe. All
13 training feature caches are sealed and six control normalization/budget
checks pass. Eighteen new full-provider-holdout fits (two seeds) are running;
no comparative scores, organizer scoring, or adoption change yet. The separate
[PR22](https://github.com/jekkiechan/more-than-a-hack-2027/pull/22) context/seed-ensemble
study remains distinct.

## Latest evidence

| Scope / evidence | Finding | Decision |
| --- | --- | --- |
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

These are proposals. Freeze a [declaration](templates/EXPERIMENT.md) before
fitting; do not infer that a proposed study has already run.

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
