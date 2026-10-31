# Research documentation

Read [STATUS](STATUS.md) to resume, then find the needed report in
[EXPERIMENTS](EXPERIMENTS.md). Load a skill only when its trigger applies.

| Location | Owns | Update when |
| --- | --- | --- |
| [AGENTS](../../AGENTS.md) | Short routing and durable constraints | An operating rule changes |
| [experiments/AGENTS](../../experiments/AGENTS.md) | Compressed lineage, vocabulary, settled directions, code map | A study completes or a module's role changes |
| [STATUS](STATUS.md) | Current adoption, next actions, blockers and draft state | Work completes, priorities change or a PR merges |
| [EXPERIMENTS](EXPERIMENTS.md) | Study IDs, outcomes, protocols and evidence links | A study is declared or reported |
| [BASELINE](../../experiments/BASELINE.md) and its manifest | Package versions, promotion evidence, restore commands | A baseline is explicitly adopted |
| [LESSONS](LESSONS.md) | Sources for reusable guidance and skills | A correction or repeated failure warrants a rule |
| [Experiment template](templates/EXPERIMENT.md) | Declaration and final readout | Copy for a new study |
| `experiments/*.md` and `experiments/results/` | Detailed methods and saved aggregate evidence | That study produces evidence |

Keep one current summary in STATUS; entrypoints link to it. Do not copy new
scoreboards into every README. Record code merged, results published, artifacts
uploaded/restored and model adopted as separate states. Pin draft-only evidence
to a commit; check live PR state when resuming.

Add one index row per bounded study, then update its outcome and STATUS in the
same PR. Leave a concrete next action or blocker. Remove completed proposals
from the active queue. Use Git history for superseded handoffs; do not create
another archive copy. Preserve frozen manifests, receipts, source hashes and
result JSON; correct explanatory prose with dated notes instead of rewriting
old selection decisions.

Skills contain reusable procedures, not model versions or score snapshots.
Load only the report needed for the task. The
[pre-cleanup AGENTS](https://github.com/jekkiechan/more-than-a-hack-2027/blob/1b3fc4a4ed1868bf8c7ca26c99ec443947cf3458/AGENTS.md)
remains in history.
