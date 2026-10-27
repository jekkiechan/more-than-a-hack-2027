---
name: possession-readout
description: Compare possession experiment scores, explain wins or regressions, write a research readout, or recommend baseline adoption using saved evidence and explicit protocol boundaries.
---

# Evidence-backed possession readout

1. Read the saved aggregate evidence and declaration; do not rescore to answer
   a reporting question. Check the active baseline manifest and reporting policy.
2. Establish comparability before subtracting scores: same matches, grading,
   views/seeds, provider weights, learned-component holdouts, decoder and metric.
   Separate pooled, D-only holdout and full H/A+D holdout. If mismatched, report
   separate tables and the missing matched comparison.
3. Rank on unrounded official spell F1, macro F1 for exact ties. Show both
   components and the fixed 70/30 auxiliary summary; do not rank by the summary.
   Explain simplicity or robustness choices as explicit tradeoffs.
4. Lead with the decision, then matched clean/stress deltas and failed guardrails.
   Include provider spread, D precision/recall, boundary/short-spell diagnostics,
   reference/predicted spell counts and seed uncertainty where measured. Do not
   attribute a whole-pipeline change to architecture alone.
5. Label evidence honestly: development, previously exposed, synthetic stress,
   requested public diagnostic or independent confirmation. The organizer
   window has only 17 possessions. Separate bootstrap intervals are not paired
   significance; neither proves full-match accuracy. Native X margins may make
   some boundaries ungradable. Say what was not measured.
6. Separate numerical leader, adopted working baseline and frozen historical
   selection. A failed stress guardrail remains failed after a better public
   point estimate. Change adoption only with an explicit decision and manifest/
   provenance update; preserve historical winners and closed gates.
7. End with the concrete next experiment or blocker and artifact availability.
   Update the study's index row and STATUS. Keep the full methods/scorecard in
   the report, not in AGENTS or every README.

Use the [template](../../../docs/research/templates/EXPERIMENT.md) for a
standalone report. Guidance provenance: [LESSONS](../../../docs/research/LESSONS.md).
