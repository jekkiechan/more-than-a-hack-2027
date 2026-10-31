# Documentation map

A new contributor starts at [root AGENTS](../../AGENTS.md), reads
[STATUS](STATUS.md), then follows the task's skill and relevant index row.
The [root README](../../README.md) also links the upstream challenge guide.

| Location | Owns | Update when |
| --- | --- | --- |
| [AGENTS](../../AGENTS.md) | Task routing, durable constraints, delivery/checks | An operating rule changes |
| [experiments/AGENTS](../../experiments/AGENTS.md) | Local editing and frozen-evidence rules | Experiment workflow changes |
| [experiments/README](../../experiments/README.md) | Vocabulary and code map | A shared module's role changes |
| [STATUS](STATUS.md) | Current adoption, next actions, blockers and open delivery | Priorities/adoption change or relevant work merges |
| [EXPERIMENTS](EXPERIMENTS.md) | One study entry, outcome, protocol/evidence links and settled directions | A study is declared or reported |
| [BASELINE](../../experiments/BASELINE.md) | Active package, promotion evidence, restore/inference commands and archived package pointers | A package is adopted or delivered |
| [LESSONS](LESSONS.md) | Source-backed corrections and the skills that encode them | A durable lesson changes a procedure |
| [.agents/skills/](../../.agents/skills/) | Task-triggered reusable procedures | A correction warrants a reusable step |
| [Experiment template](templates/EXPERIMENT.md) | Declaration and final readout structure | Reporting requirements change |
| `experiments/*.md` and `experiments/results/` | Detailed methods, scorecards and saved evidence | That study produces evidence |

## Keep one owner for each fact

Keep changing adoption/priorities in STATUS, package details in BASELINE and
scorecards in the study report. Entrypoints link to those owners. Skills contain
procedures, never current model versions, priorities or score snapshots. Add a
lesson with its evidence source; amend an existing skill before creating a new
overlapping one. Remove completed proposals from the current queue.

Record code merged, results published, artifacts uploaded/restored and model
adopted as separate states. Date PR snapshots and pin draft-only evidence to a
commit; check live state when resuming. A later verified receipt supersedes an
availability note without rewriting it.

## Preserve evidence and validate edits

Use Git history for superseded handoffs and setup notes; do not create another
archive copy. Keep unique historical reports discoverable through the index.
The original Sportec report is [PRIMARY_TRACKING](../../experiments/PRIMARY_TRACKING.md).
Root turnover reports stay in place because frozen runner provenance references
their paths. Upstream challenge/rule/data/evaluation documents remain authoritative.

Do not rewrite frozen declarations, manifests, policies, receipts, source hashes,
result JSON or hash-bound Markdown (including `results/sn_dense_v1_progress.md`).
Use a dated explanatory note or new receipt for corrections. Documentation
maintenance never promotes a model or reopens a scoring gate.

For docs-only edits, check local links and heading anchors, current-state claims
against the active manifest/policy and PR evidence, and the changed-file list for
frozen evidence. Confirm a moved report retains its content. CI runs the
synthetic/contract suites and bundled CSV format check; never score organizer
labels to verify docs.
