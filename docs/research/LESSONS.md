# Guidance and lessons

Reviewed 2026-10-31. These are source-backed repository/PR lessons, not a
claim to have reviewed past-chat transcripts. Direct user guidance is attributed
to its recorded source; results-derived procedures are our interpretation.

| Source / type | Durable lesson | Where applied |
| --- | --- | --- |
| User correction, 2026-10-30, direct guidance; [data access](../../experiments/SOCCERNET_DATA_ACCESS.md) | SoccerNet-GAR is available from Hugging Face. Check the pinned `OpenSportsLab/SoccerNet-GAR` tracking branch before asking for a local data archive. Raw dataset access and recovery of our trained checkpoints are separate. | Restore reference; SoccerNet data access |
| [Decision policy](../../experiments/model_decision_policy.json), recorded guidance | Spell first, macro for exact ties; fixed 70/30 is an additional summary, not competition ranking. | Readout reference |
| [PR #10](https://github.com/jekkiechan/more-than-a-hack-2027/pull/10), recorded choice | User chose simpler GRU/HGB although the three-way blend is numerically higher. Explain simplicity as a tradeoff; do not relabel the numerical winner. | Readout reference |
| [Ball-gap promotion](../../experiments/results/baseline_ballgap25_promotion.json), recorded request | User explicitly adopted ball-gap robustness after seeing diagnostics. Preserve the old clean-rule winner and disclose prior public exposure. | Readout reference; BASELINE |
| [PR #16 diagnostic declaration](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/JOINT_COMPETITION_CHECK.md), recorded request | User requested competition scoring and suggested preprocessing. A hypothesis is not measured evidence; the requested check does not select repairs or override failed stress guardrails. | Experiment and readout references |
| [D-only evaluation](../../experiments/LEARNED_TEMPORAL.md) vs [full-pipeline study](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/LIVE_GAP_HAD_GRU.md), methodological lesson | Same match names do not imply the same protocol. Audit H/A, D, normalization, pretrained exposure, views and provider weighting before computing deltas. | Experiment and readout references |
| [Tracking pilot](../../experiments/TOP_THREE_PILOT.md), results-derived lesson | Better coverage or gap-frame accuracy can coexist with worse spell F1. Check quality, boundaries, short spells and provider spread; do not assume interpolation or fusion helps. | Experiment reference |
| [D repair](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/DEADBALL_TRANSFER.md), results-derived lesson | A clean improvement can hide a severe stress regression. Separate training augmentation from inference repair and report failed guardrails before recommending adoption. | Experiment and readout references |
| [BASELINE restore](../../experiments/BASELINE.md), operational lesson | Verify cached artifact hashes and LFS contents before retraining. Local verification, remote upload and independent restore are different accomplishments. | Restore reference; experiment reference |
| [Earlier cleanup guide](https://github.com/jekkiechan/more-than-a-hack-2027/blob/96314f8f4489ae356afea03111fa4b0dbfd40ec5/AGENTS.md), recorded guidance; current cleanup request | Publish authorized PRs without another confirmation; keep short entrypoints, task-triggered skills and one current status. | AGENTS; documentation map |
| [Previous evidence-refresh checklist](https://github.com/jekkiechan/more-than-a-hack-2027/blob/96314f8f4489ae356afea03111fa4b0dbfd40ec5/AGENTS.md), recorded corrections | Refresh current main and relevant PRs before model-choice advice. Distinguish recipes, stress views, separate-seed means, measured ensembles and adoption. | Readout reference |
| [Scope-superseded D-only studies](EXPERIMENTS.md#temporal-robustness-follow-up), recorded scope correction | Confirm component versus full H/A/D scope before fitting; a D-only architecture comparison cannot answer a joint-pipeline question. | Experiment and readout references |
| [Matched follow-up](../../experiments/SOCCERNET_FOLLOWUP.md) and [later exact restore](../../experiments/results/baseline_soccernet_capacity_remote_artifacts.json), operational lesson | A reconstructed reference and an exact deployment restore are different artifacts. Later deployment recovery does not recover all research folds. | Restore reference |
| [Window-overlap failures](../../experiments/WINDOW_OVERLAP.md), [contact replay correction](../../experiments/results/contact_memory_runtime_correction.json), operational lesson | Exercise replay on the execution path before the full run; preserve failed runs and record explicit numerical/checker corrections. | Experiment reference |
| [Dense promotion](../../experiments/results/baseline_sn_dense_promotion.json) and [private delivery PR #7](https://github.com/jekkiechan/mtah-data/pull/7), recorded state | Adoption can precede delivery to private main. Inspect delivery branches and use versioned manifests; an approved screen exception does not change the failed study screen. | Restore and readout references |

| [Fresh Luna navigation runs](NAVIGATION.md), observed usability lesson | A new agent read the entire historical index for current tasks, producing truncated output. Route by task and search the relevant index section; grouping skills alone does not prove lower reading cost. | AGENTS; research skill router; navigation evaluation |

| [Local harness verification](NAVIGATION.md#harness-verification), observed environment failure | System Python passed focused preparation tests but lacked full-suite PyTorch/PyArrow. Use the interpreter with pinned dependencies; a small suite passing does not verify the full environment. | AGENTS checks; CI |

| [Fresh-agent hash misstatement](NAVIGATION.md#fresh-agent-smoke-test), observed reporting error | The planner's `manifest_sha256` identifies the source manifest; it is not the plan fingerprint. Name each hash by what it actually seals. | Preparation reference |

Add only durable corrections supported by a source. Put changing scores and
one-run details in reports. Amend an existing workflow reference before creating overlapping
ones, and remove superseded instructions instead of accumulating addenda.
