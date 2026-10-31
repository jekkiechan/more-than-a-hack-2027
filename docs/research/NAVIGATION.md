# Navigation design and evaluation

Reviewed 2026-10-31. This evaluates contributor navigation and the local harness;
it is not a possession-model experiment and reads no organizer labels.

## Guidance consulted

- [Codex AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md):
  project instructions are discovered by scope; reserve them for durable repo
  guidance. Avoid a second routing file with the same job.
- [Codex skills](https://learn.chatgpt.com/docs/build-skills): keep a skill focused
  on a job, with clear inputs/outputs and tested trigger behavior.
- [Claude skill authoring](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices):
  expose details progressively, link references directly from SKILL, provide
  navigation for longer files, and evaluate representative tasks before/after.
- [Claude Code practices](https://code.claude.com/docs/en/best-practices): keep
  always-loaded instructions short, include useful exact check commands, and
  distinguish written guidance from checks the harness actually enforces.

## Consolidation decisions

| Area | Decision and reason |
| --- | --- |
| Four possession skills | One `possession-research` skill for one repo-specific research lifecycle; restore, preparation, experiment and readout become four directly linked references. Load only requested stages. This is our application of the guidance, not a vendor mandate to merge skills. |
| Nested experiments AGENTS | Remove it after moving its unique frozen-evidence rule into root AGENTS. It mostly repeated root rules and sent readers through another routing page. |
| Agent guide vs directory README | Root AGENTS owns constraints/commands/routing; experiments README owns vocabulary/code map. Do not duplicate workflow menus in both. |
| Experiment index | Keep one searchable index, add section navigation, and make reading it conditional on historical evidence/new-study work. Its 29 KB full text was unnecessary for the benchmark tasks. |
| Current package vs sealed progress wording | Explain beside BASELINE that the sealed report's “published adopted ensemble” means the previous baseline at study time. Preserve the hash-bound progress report. |
| STATUS, BASELINE, index, reports | Keep separate: current coordination, package commands/evidence, discovery and detailed records change for different reasons. Merging them would force unrelated history into common tasks. |
| Historical reports, frozen JSON and source snapshots | Preserve unique evidence, provenance and source-bound paths. Similar-looking historical instructions describe different frozen studies; deduplicating them can break reproducibility. |
| Upstream challenge docs | Preserve authoritative rules/data/evaluation/submission documents; these are externally defined contracts. |

The grouped skill has one discovery description and a small routing body, rather
than one giant procedure. All four references link directly from SKILL; no
reference chain is required to discover a stage. If a stage later becomes a
standalone reusable workflow with a distinct trigger or users, split it again.
Fewer skill names alone is not evidence of lower runtime/context cost.

## Fresh-agent smoke test

The [reusable scenarios and rubric](evals/navigation.json) cover restore,
readout and provider-isolated preparation. Each run uses a new Luna agent with
no inherited chat; the prompt names the tasks and root AGENTS, not answer paths
or the proposed skill structure. The agent executes the real read-only split
planner and the focused synthetic preparation tests. No model training/scoring,
organizer references, private downloads or writes are allowed.

Baseline: commit `3150116` (content merged as PR #44). Candidate: the working
tree for the grouped-skill change, based on `2032335`. Same model and high
reasoning; the candidate prompt additionally asks for start/end timestamps and
an explicit file list to improve measurement. Ancillary evaluation documentation
and the checker were authored while the candidate ran; relevant routing and
procedure content stayed the same apart from nonsemantic line wrapping and
restoring a BASELINE heading anchor. Answers are graded by the parent
against 12 checks in the rubric, not by the tested agent itself.

| Observation | Four-skill baseline | Grouped-skill candidate |
| --- | --- | --- |
| Parent-graded task checks | 12/12 | 12/12 |
| Real split preview | 5 train, 3 selection; provider excluded | Same |
| Focused synthetic preparation suite | 8/8 passed | 8/8 passed |
| Distinct files read (self-reported) | 13 | 17 |
| Requested reading (rough agent estimate) | ~800 lines / ~40 KB | ~1,700 lines / ~80 KB |
| Tool calls (self-reported, including poll) | 9, corrected from initial 8 | 10 |
| Measured elapsed | Unavailable | 106 seconds |

Both runs solved restore, readout and preparation without prohibited operations.
Baseline loaded the full index and its output was truncated; candidate avoided
that read. Candidate also verified more authoritative files (manifest/policy and
promotion/public receipts), so it read more total material. **This does not show
a speed or token improvement from grouping.** Reading estimates are approximate,
not provider token telemetry; there is one run per layout, no baseline clock and
unequal evidence coverage. The candidate's extra hash labelled “plan hash” was
the manifest SHA field; the rubric did not request that hash, and the returned
plan/counts were correct. This is a reporting precision issue, not a plan failure.

Baseline flagged the progress report's historical adoption wording; candidate
had the explicit BASELINE clarification and did not repeat that ambiguity.
Candidate loaded exactly restore/preparation/readout, leaving the experiment
reference unopened for this three-stage task.

A third fresh Luna agent ran the preparation case alone: 5/5 case checks and
8/8 focused tests passed in seven shell invocations. It read only three guidance
files (AGENTS, SKILL and preparation reference, about 11 KB), then inspected the
first 70 lines each of the planner and its tests. The planner/tests loaded the
manifest programmatically. It did not read STATUS, BASELINE, the historical
index, restore/readout/experiment references or this evaluation. This verifies
selective stage navigation on a real task, without claiming a measured speedup
against an unrun isolated-stage baseline.

The small sample supports functional navigation checks, not a claim of general
agent performance. No actual private restore/inference, skill auto-activation
or large model experiment was tested; commands for those were evaluated as
read-only plans. Frozen evidence/model code remained unchanged.

## Repeat and interpret

1. Start a fresh agent at the intended checkout, without this report/rubric in
   its context. Give the setup and three case prompts from the JSON. Preserve
   task constraints and use the same model/effort for a comparison.
2. Keep relevant routing/procedures fixed while each run works. Capture its actual commands,
   answers, failures, files read and tool counts. Grade each rubric check and
   explicitly identify prohibited operations if any.
3. Run `python scripts/check_docs.py`, the real planner and the focused
   preparation tests. These deterministic checks complement agent reasoning;
   they do not prove it. The checker covers repo-local link targets/Markdown
   heading anchors, skill name/description and direct reference discoverability.
4. After routing changes, repeat the cases. Test an isolated stage as well if
   verifying selective reference loading; a multi-stage prompt should load
   multiple references. Never infer skill auto-activation from a prompt that
   explicitly supplies AGENTS: this setup tests repo-guided discovery.

The checker intentionally skips remote URL health and Markdown links in fenced
examples; it is a standard-library check for the Markdown subset used here,
not a full Markdown renderer or a frozen-evidence authorizer. CI runs it on each
PR, with focused checker tests and the existing synthetic suites.

## Harness verification

The new checker found two existing `BASELINE.md#private-cache-restore` links
that pointed to a removed heading. Restoring that heading fixes both without
editing the source-bound study reports. Checker contracts exercise missing files,
missing/duplicate headings, fenced examples, remote links, encoded/angle paths,
reference definitions, skill frontmatter and direct reference discoverability.

Local verification: 427 experiment tests and 26 root tests (including 8 new
checker tests) pass; file-backed fit parity and bundled CSV format checks pass.
The first broad-suite attempt used system Python without PyTorch/PyArrow and
failed dependency imports. The repo `.venv` already had CPU PyTorch 2.8.0; after
installing the pinned PyArrow 20.0.0 there, the full suite passed. No dependency
file changed. Both Luna preparation runs used system Python and needed neither
neural dependency for their 8 focused tests.
