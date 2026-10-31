---
name: possession-experiment
description: Design or run a bounded possession-model, tracking-repair, corruption, or transfer comparison. Use before fitting or new evaluation; not for documentation-only edits or reading saved scores.
---

# Bounded possession experiment

1. Read current STATUS and the relevant experiment-index row. Restore the
   matched baseline/cache using the [restore skill](../possession-restore/SKILL.md)
   and verify source, input and checkpoint hashes.
   Inspect actual LFS objects, not pointer files. Distinguish missing access
   from a model failure; do not retrain merely because a cache was not restored.
2. Copy [the declaration template](../../../docs/research/templates/EXPERIMENT.md).
   State one falsifiable question; isolate architecture, augmentation,
   preprocessing and decoder changes with matched controls. Confirm whether the
   user intends a D-only component or full H/A/D pipeline before allocating fits;
   a component study cannot answer a full-pipeline request. Check completed
   studies first. Freeze arms, budgets, seeds and clean/stress guardrails.
3. Record all match IDs, provider exclusions and prior researcher/checkpoint
   exposure. Exclude the provider from every learned component and normalization
   for a full-pipeline claim. Derived views/windows stay with their source match.
   Historical matches remain development evidence even after a new holdout split.
4. Audit each source's H/A/D semantics and missingness shortcuts. Preserve
   native masks, unknown targets, grading population, half/time boundaries and
   team symmetry. Recompute features after raw tracking corruption. State any
   training-label-conditioned augmentation explicitly; evaluation/inference
   cannot use those states. Synthetic stress is not real broadcast confirmation.
5. Test contracts before fitting: target independence of inference, masks/time
   breaks, team swaps, cohort completeness, baseline parity and gate integrity.
   Freeze source/config/input hashes and models before scoring. Do not edit
   completed declarations or run extra arms after seeing results under that ID.
   Before a full run, replay a representative saved control with the actual
   execution path. Declare numerical tolerances before scoring. If a replay
   guard or assertion is wrong, preserve the failed run and record a bounded
   correction; never loosen it silently after seeing candidate scores.
6. Run the declared external comparison to completion. Verify paired cohort cells,
   unchanged grading/spell counts and saved baseline parity. Summarize seed
   spread; a mean of separate runs is not an inference ensemble.
7. Use the readout skill. Save aggregate reports and artifact receipts, update
   EXPERIMENTS and STATUS, then publish the authorized PR. Record actual remote
   artifact availability and the next action/blocker.

For any separately requested organizer pipeline diagnostic, generate and
format-validate the full CSV without reference access, seal source/model/input/
prediction hashes, and claim a distinct exclusive gate before scoring. Reuse
saved baseline scores. Read the saved report after completion; no repeat
scoring, public-driven tuning or automatic promotion follows.

Sources and rationale: [LESSONS](../../../docs/research/LESSONS.md).
