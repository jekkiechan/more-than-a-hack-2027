# Guidance and lessons

Reviewed 2026-10-27. Past-chat transcripts/search were unavailable. Sources
below are repository records and PR descriptions through #16; inspected PR
discussions were empty. A recorded user request is attributed to its record,
not to a direct chat review. Results-derived procedures are our interpretation.

| Source / type | Durable lesson | Where applied |
| --- | --- | --- |
| User correction, 2026-10-30, direct guidance; [data access](../../experiments/SOCCERNET_DATA_ACCESS.md) | SoccerNet-GAR is available from Hugging Face. Check the pinned `OpenSportsLab/SoccerNet-GAR` tracking branch before asking for a local data archive. Raw dataset access and recovery of our trained checkpoints are separate. | AGENTS; SoccerNet data access |
| [Decision policy](../../experiments/model_decision_policy.json), recorded guidance | Spell first, macro for exact ties; fixed 70/30 is an additional summary, not competition ranking. | Readout skill |
| [PR #10](https://github.com/jekkiechan/more-than-a-hack-2027/pull/10), recorded choice | User chose simpler GRU/HGB although the three-way blend is numerically higher. Explain simplicity as a tradeoff; do not relabel the numerical winner. | Readout skill |
| [Promotion manifest](../../experiments/baseline_manifest.json), recorded request | User explicitly adopted ball-gap robustness after seeing diagnostics. Preserve the old clean-rule winner and disclose prior public exposure. | Readout skill; BASELINE |
| [PR #16 diagnostic declaration](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/JOINT_COMPETITION_CHECK.md), recorded request | User requested competition scoring and suggested preprocessing. A hypothesis is not measured evidence; the requested check does not select repairs or override failed stress guardrails. | Both skills |
| [D-only evaluation](../../experiments/LEARNED_TEMPORAL.md) vs [full-pipeline study](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/LIVE_GAP_HAD_GRU.md), methodological lesson | Same match names do not imply the same protocol. Audit H/A, D, normalization, pretrained exposure, views and provider weighting before computing deltas. | Both skills |
| [Tracking pilot](../../experiments/TOP_THREE_PILOT.md), results-derived lesson | Better coverage or gap-frame accuracy can coexist with worse spell F1. Check quality, boundaries, short spells and provider spread; do not assume interpolation or fusion helps. | Experiment skill |
| [D repair](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/DEADBALL_TRANSFER.md), results-derived lesson | A clean improvement can hide a severe stress regression. Separate training augmentation from inference repair and report failed guardrails before recommending adoption. | Both skills |
| [BASELINE restore](../../experiments/BASELINE.md), operational lesson | Verify cached artifact hashes and LFS contents before retraining. Local verification, remote upload and independent restore are different accomplishments. | Experiment skill; STATUS |
| This cleanup request, direct guidance | Act through PR publication without another approval round; prune unnecessary context. Use a short entrypoint, task-triggered skills, one current status and links to historical evidence. | AGENTS; documentation map |

Add only durable corrections supported by a source. Put changing scores and
one-run details in reports. Amend an existing skill before creating overlapping
ones, and remove superseded instructions instead of accumulating addenda.
