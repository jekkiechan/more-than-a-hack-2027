# Current research state

Reviewed 2026-10-27: fork main `1b3fc4a`, plus drafts through PR #16.
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

The .5740 and .5338 baselines use different component holdouts; do not
subtract them as a model delta. All five external development matches have
prior exposure. PR #16 code is draft-only. Older diagnostics remain closed.

## Next work

These are proposals. Freeze a [declaration](templates/EXPERIMENT.md) before
fitting; do not infer that a proposed study has already run.

| Priority / state | Next action | Required evidence |
| --- | --- | --- |
| 1 — Ready to design | Address PR #16's player-corruption D overprediction. Isolate restored legacy D player-slot-dropout training from inference-time tracking repair; test one mechanism at a time. | Matched full-pipeline provider holdouts, frozen clean/stress guardrails, per-provider D precision/recall and fixed seeds. Public labels cannot choose repairs. |
| 2 — Data blocked | Obtain an independent labeled broadcast match; follow [private issue #4](https://github.com/jekkiechan/mtah-data/issues/4) for rights, blind annotation, tracking/alignment and sealed confirmation. | PFF access was requested; response not established here. Audio/video or model-generated labels do not supply independent H/A/D ground truth. |
| 3 — Ready to design | Calibrate pilot player defects and obtain independent ball-fusion confidence before integration. | Separate label-free calibration/audit partitions, preserved native masks, position-quality evidence and external possession evaluation. Coverage alone is insufficient. |
| 4 — Deferred | Boundary/audio/decoder work or broad tuning only after checking the completed studies in the index. | A distinct hypothesis and matched external controls; do not repeat failed arms unchanged. |

## Open work and artifacts

| PR / inspected head | State at review |
| --- | --- |
| [#16](https://github.com/jekkiechan/more-than-a-hack-2027/pull/16), `possession/live-gap-had-gru`, `8c704aa` | Draft joint/D-repair/check. Earlier ~44 MB archive reported uploaded; additional ~3.7 MB repair archive and pooled diagnostic artifacts not remotely confirmed. Verify availability before claiming restore works. |
| [#13](https://github.com/jekkiechan/more-than-a-hack-2027/pull/13), `possession/gap-results`, `33c03ca` | Draft reports/aggregates only; experimental implementation excluded. |
| [#11](https://github.com/jekkiechan/more-than-a-hack-2027/pull/11), `possession/spell-balanced-attribution`, `b93ce36` | Draft attribution study; no adoption. |
| [#8](https://github.com/jekkiechan/more-than-a-hack-2027/pull/8), `possession/robust-pipeline`, `7433fef` | Older draft contextual pipeline; historical comparator. |

At handoff update the review date, branch/commit, outcome, artifact availability,
closed-check receipt location and next action. Do not merge or promote unrelated
draft work as part of documentation maintenance.
