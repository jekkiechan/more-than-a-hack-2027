# Working in experiments

Repo-wide [AGENTS](../AGENTS.md) applies. Read only the task-relevant sections of
[the research code guide](README.md): vocabulary and code map live there.
Current adoption and priorities live in [STATUS](../docs/research/STATUS.md);
prior outcomes and settled directions live in [EXPERIMENTS](../docs/research/EXPERIMENTS.md).

- Before new work, locate its nearest prior study in the index. Read the report
  and saved diagnostics before proposing another sweep on an exposed cohort.
- Import shared libraries from the code map. Preserve frozen study runners,
  declarations, manifests, receipts and source hashes; create a new runner/ID
  for a new hypothesis. A new directory does not create fresh evidence.
- Use the [experiment skill](../.agents/skills/possession-experiment/SKILL.md)
  before fitting or scoring and the
  [preparation skill](../.agents/skills/possession-preparation/SKILL.md)
  before copying feature/split logic. Unsupported features need a new versioned
  recipe and parity contracts.
- Reports under `results/` can be hash-bound evidence too (for example the dense
  study progress report). Do not edit them to tidy prose. Link to their record.
