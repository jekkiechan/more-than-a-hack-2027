# Documentation map

Start at [AGENTS](../../AGENTS.md). Choose the relevant destination below;
there is no required tour of every document.

| Question | Owner |
| --- | --- |
| What is adopted, next, blocked or awaiting delivery? | [STATUS](STATUS.md) |
| How do I restore/run the package or find its evidence? | [BASELINE](../../experiments/BASELINE.md), active manifest and policy |
| How do I perform this research task? | [One research skill](../../.agents/skills/possession-research/SKILL.md), then its applicable reference |
| Where is the implementation / what do terms mean? | [Research code map and vocabulary](../../experiments/README.md) |
| What was tried / what should not be repeated? | [Searchable experiment index](EXPERIMENTS.md#find-a-study), then a study report |
| Which past correction produced a procedure? | [LESSONS](LESSONS.md) |
| What must a new study declare/report? | [Experiment template](templates/EXPERIMENT.md) |
| Why this layout / does navigation work? | [Navigation evaluation](NAVIGATION.md) |

## Maintenance rules

- AGENTS owns universal constraints, repo commands and task routing. Do not
  duplicate them in another AGENTS or README. Put task-specific procedures in
  the research skill's four direct references; read only those needed.
- STATUS owns changing adoption/priorities/blockers, BASELINE owns the package
  and restore instructions, reports own detailed scorecards. Entrypoints link
  to each owner. Skills have no current package names or score snapshots.
- Add one index row per bounded study and update it and STATUS when work
  completes. Remove completed proposals from the active queue. Date/pin PR
  snapshots and check live heads when resuming.
- Record code merged, results published, uploaded, restored and adopted as
  separate states. Use a later receipt for changed artifact availability.
- Use Git history for superseded handoffs; retain unique reports and link them
  from the index. Keep source-bound paths such as the root turnover reports.
- Frozen-evidence restrictions live in AGENTS. This includes JSON and sealed
  progress Markdown, not just code. A prose correction goes beside the record
  or in a new receipt. Upstream challenge documents remain authoritative.

## Checks

Run `python scripts/check_docs.py` for local links, heading anchors and skill
routing. CI runs this alongside existing checks. Verify current-state claims
against manifest/policy and relevant PR evidence; inspect the changed-file list
for frozen evidence. No organizer scoring is needed. For navigation changes,
repeat the fresh-agent tasks documented in NAVIGATION and record observed
answers, command outcomes and limitations; a link checker cannot grade reasoning.
