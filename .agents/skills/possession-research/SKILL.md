---
name: possession-research
description: Restore possession artifacts, prepare provider splits/features, run bounded experiments, or compare model evidence in this repo. Use for these research tasks, including model-choice chat; skip ordinary docs edits.
---

# Possession research workflow

Choose the requested stage and read only its reference. Read multiple references
when the task crosses stages; do not load the whole package by default.

| Task / input | Reference | Expected output |
| --- | --- | --- |
| Missing datasets, weights or matching source | [Restore](references/restore.md) | Exact artifact/manifest, verified restore or access blocker |
| Whole-match/provider fold or reusable feature cache | [Preparation](references/preparation.md) | Validated plan/recipe, training-only normalization, cache receipt when prepared |
| New hypothesis, fitting or evaluation | [Experiment](references/experiment.md) | Bounded declaration, matched controls, frozen evidence and result |
| Scores, model choice, adoption advice or study readout | [Readout](references/readout.md) | Protocol-matched comparison, guardrails, adoption/evidence status |

## Select evidence without loading all history

- For current-model or restore questions, read [BASELINE](../../../experiments/BASELINE.md)
  and the active [manifest](../../../experiments/baseline_manifest.json) /
  [policy](../../../experiments/model_decision_policy.json). Check
  [STATUS](../../../docs/research/STATUS.md) for priorities and delivery blockers.
- Before current-results advice, refresh relevant current-main/PR evidence as
  described in Readout. A purely local split-plan task needs no remote PR scan.
- For prior studies or a new hypothesis, search the
  [index](../../../docs/research/EXPERIMENTS.md) by mechanism/study ID and read
  the relevant section/report. Do not read the full historical index by default.
- Find shared modules in the [code map](../../../experiments/README.md#code-map).
  Repo constraints and checks remain in [AGENTS](../../../AGENTS.md).
- Procedures do not authorize new fitting, organizer scoring, upload or model
  promotion beyond the user's task. Preserve frozen evidence and closed gates.

Correction provenance: [LESSONS](../../../docs/research/LESSONS.md).
