---
name: possession-readout
description: Compare possession experiment scores, explain wins or regressions, write a research readout, or recommend baseline adoption using saved evidence and explicit protocol boundaries.
---

# Evidence-backed possession readout

1. Refresh current evidence before answering, including chat advice. Check current
   fork `main`, STATUS, BASELINE, the active manifest/policy and recent relevant
   merged/open PR heads and reports. Distinguish adopted, completed experimental,
   running and proposed work. Saved Pages and historical verdicts may lag; do not
   call a method untested from an old summary. If current sources conflict,
   inspect the promotion receipt and recent commits and disclose unresolved gaps.
   Read saved aggregate evidence and declarations; do not rescore to report.
2. Establish comparability before subtracting scores: same matches, grading,
   views/seeds, provider weights, learned-component holdouts, decoder and metric.
   Name the exact stress view: `feed_like` and `composite` differ. Preserve a
   composite failure and disclose a post-result guardrail change. Distinguish
   model family, training recipe and pipeline; an earlier failed GRU recipe is
   not evidence against a newer recipe. A D-only architecture comparison does
   not establish the best joint H/A/D architecture. Separate pooled, D-only
   holdout and full H/A+D holdout. If mismatched, report
   separate tables and the missing matched comparison.
3. Rank on unrounded official spell F1, macro F1 for exact ties. Show both
   components and the fixed 70/30 auxiliary summary; do not rank by the summary.
   Explain simplicity or robustness choices as explicit tradeoffs.
4. Lead with the decision, then matched clean/stress deltas and failed guardrails.
   Include provider spread, D precision/recall, boundary/short-spell diagnostics,
   reference/predicted spell counts and seed uncertainty where measured. Do not
   attribute a whole-pipeline change to architecture alone. Verify ensemble
   evidence on the current head: averaging separate seed scores is not averaging
   predictions. A newer measured ensemble may be experimental while adoption
   remains single-seed; cite each one's report and evaluation scope.
5. Label evidence honestly: development, previously exposed, synthetic stress,
   requested public diagnostic or independent confirmation. The organizer
   window has only 17 possessions. Separate bootstrap intervals are not paired
   significance; neither proves full-match accuracy. Native X margins may make
   some boundaries ungradable. Say what was not measured.
6. Separate numerical leader, adopted working baseline and frozen historical
   selection. A failed stress guardrail remains failed after a better public
   point estimate. Change adoption only with an explicit decision and manifest/
   provenance update; preserve historical winners and closed gates. Describe
   the best supported current choice separately from superiority established by
   a matched architecture comparison. A user-approved screen exception changes
   adoption, not the original study's failed decision.
7. End with the concrete next experiment or blocker and artifact availability.
   Update the study's index row and STATUS. Keep the full methods/scorecard in
   the report, not in AGENTS or every README.

Use the [template](../../../docs/research/templates/EXPERIMENT.md) for a
standalone report. Guidance provenance: [LESSONS](../../../docs/research/LESSONS.md).
