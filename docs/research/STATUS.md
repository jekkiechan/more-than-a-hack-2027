# Current research state

Reviewed 2026-10-27: fork main `5005115`, PR #16 at `8c704aa`, and the
`possession/temporal-spell-robustness` follow-up.
Check live PR state before resuming. This is the current coordination summary;
package details belong in [BASELINE](../../experiments/BASELINE.md).

## Adopted baseline

`gru_hgb_ballgap25_v1`: fixed 50/50 GRU/ordered-HGB D heads, contextual HGB
teams with 25% ball-gap training replacement, Viterbi cost 3. The
[manifest](../../experiments/baseline_manifest.json) binds the package and
explicit user-authorized robustness tradeoff. Independent broadcast confirmation
is pending; no organizer submission or measured full-match performance exists.
The user's .90 target on both competition metrics remains unestablished.

## Latest evidence

| Scope / evidence | Finding | Decision |
| --- | --- | --- |
| Adopted recipe, historical **D-only provider holdout, pooled H/A**; [baseline](../../experiments/BASELINE.md) | Clean spell/macro .5740/.8439; composite-stress .4315/.7866. Previous .5746/.8449 and .4205/.7699. | Adopted robustness tradeoff, not a clean-score win. |
| Merged [tracking pilot](../../experiments/TOP_THREE_PILOT.md), PR #15 | 80 pooled development evaluations; matched gap/suppression arms lose overall spell. Fusion recovers 4,419 missing-ball frames (6.84%), with >10% overlapping ball positions disagreeing by >5 m. | No preprocessing/fusion adoption. |
| Draft [joint D repair](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/DEADBALL_TRANSFER.md), PR #16; **full H/A+D provider holdouts** | Baseline clean/stress spell .5338/.4129. Player-only D joint GRU .6093/.2991, averaging two separate seeds. Clean SkillCorner D collapse repaired; both seeds fail stress guardrail. | Retain baseline. Separate-seed means are not an ensemble. |
| Draft [completed public diagnostic](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/JOINT_COMPETITION_CHECK.md), PR #16 | Unchanged pooled seed-2026 model .6194 spell/.8574 macro/.8506 D; saved baseline .5257/.8588/.8568. Only 7,500 frames and 17 possessions. | Descriptive check; gate closed. No promotion, tuning or repeat scoring. |
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
| 1 — Running | [Restore player-slot dropout](../../experiments/GRU_SLOT_ROBUSTNESS.md) in the latest joint GRU with matched controls. In parallel, build a full **H/A/D multiclass LightGBM** with matched geometry and before/after feature ablations, fixed full-vector GRU blends, and a larger-tree arm. The earlier D-only tree studies were superseded by the user's explicit target correction before validation scoring. | Frozen full-pipeline provider holdouts, clean/stress guardrails, both fixed GRU seeds, complete paired cohorts and per-IoU diagnostics. |
| 2 — Data blocked | Obtain an independent labeled broadcast match; follow [private issue #4](https://github.com/jekkiechan/mtah-data/issues/4) for rights, blind annotation, tracking/alignment and sealed confirmation. | PFF access was requested; response not established here. Audio/video or model-generated labels do not supply independent H/A/D ground truth. |
| 3 — Ready to design | Calibrate pilot player defects and obtain independent ball-fusion confidence before integration. | Separate label-free calibration/audit partitions, preserved native masks, position-quality evidence and external possession evaluation. Coverage alone is insufficient. |
| 4 — Evidence led | Choose boundary or decoder changes from the five-IoU curve and split/merge diagnostics after the current model comparisons. | Preserve the official greedy matcher and native X masks. Earlier short-spell, duration and decoder studies remain relevant; do not repeat failed arms unchanged. |

New comparative work should hold out the provider from every learned H/A and D
component and normalization. The historical policy file remains unchanged for
frozen package reproducibility; its older D-only preferred view does not support
a full-pipeline transfer claim. Spell F1 remains primary, macro F1 the exact-tie
break, and the fixed 70/30 summary auxiliary. No global weighted score can hide
a failed provider or corruption guardrail.

## Open work and artifacts

| PR / inspected head | State at review |
| --- | --- |
| [#18](https://github.com/jekkiechan/more-than-a-hack-2027/pull/18), `possession/temporal-spell-robustness` | Draft follow-up. Metric audit and explainer completed. GRU: 12 declared fits / 60 cells, training running. Corrected full H/A/D LightGBM: four frozen feature/capacity arms (12 fits / 300 cells), with 204/227/1,003 features; preparation and training running. Earlier D-only trees are scope-superseded and unscored; their receipts remain for provenance. Initial code CI passed; corrected full H/A/D source passes all 124 local tests. |
| [#16](https://github.com/jekkiechan/more-than-a-hack-2027/pull/16), `possession/live-gap-had-gru`, `8c704aa` | Still draft. Original private archive restored and all 143 manifest files verified, including 18 external arrays. Follow-up player-only D checkpoints remain absent from the remote branch; new controls are reproduced explicitly rather than claimed restored. |
| [#13](https://github.com/jekkiechan/more-than-a-hack-2027/pull/13), `possession/gap-results`, `33c03ca` | Draft reports/aggregates only; experimental implementation excluded. |
| [#11](https://github.com/jekkiechan/more-than-a-hack-2027/pull/11), `possession/spell-balanced-attribution`, `b93ce36` | Draft attribution study; no adoption. |
| [#8](https://github.com/jekkiechan/more-than-a-hack-2027/pull/8), `possession/robust-pipeline`, `7433fef` | Older draft contextual pipeline; historical comparator. |

At handoff update the review date, branch/commit, outcome, artifact availability,
closed-check receipt location and next action. Do not merge or promote unrelated
draft work as part of documentation maintenance.
