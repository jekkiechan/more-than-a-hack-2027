# Feed-like training views for the calibrated dense ensemble

Status: preparation. Study `feed_robustness_v1`, 2026-11-01, based on fork
main `7b60056`. The stretch target is 0.55 feed-like official spell F1 against
the promoted calibrated ensemble's 0.518527 (clean 0.658625).

## Declared comparison

The hypothesis is that realistic raw-tracking defects during dense-model
training improve transfer to feed-like tracking. The existing training uses
coarse/slot-dropout views and ball-gap alternatives; the recent 522-configuration
[postprocessing study](FEED_SPELL_PUSH.md) did not test this training change.
Earlier composite-augmentation outcomes used different models and defects,
including whole-frame sampling loss absent from this feed-like view.

Replace only the second original-provider training view and its alternative
with a raw feed-like corruption. Keep the first clean view and its ball-gap
alternative, every target, sampled row, window, H/A swap and sampling group.
Features are recomputed after ball gaps, coordinate noise/drift/jumps, player
slot gaps and same-team identity swaps. No regional or whole-frame dropout is
introduced. Training defects use one new fixed seed per match. Native missing
observations, timestamps, half boundaries and time discontinuities are preserved.
Targets never control the corruption.

The unchanged D2 trainer fits six models: three complete provider exclusions
and seeds 2026/2027. Each starts from its exact held-out teacher, uses frozen
normalization, width 96, 2,048 updates, batch 32, learning rates 0.0003 for H/A
and 0.0001 for D, and a 50/50 original-provider/SoccerNet batch mixture. The
45 dense SoccerNet TRAIN games and their silver supervision remain unchanged.
The new view occupies half the original-provider records, approximately one
quarter of each complete batch in expectation. The same samples train both
H/A and D paths.

Compare the archived baseline, the new two-seed probability ensemble, and one
fixed equal blend of their native probabilities. Apply temperature 4 to pooled
D logits followed by the unchanged cost-3 decoder to every arm. Report each
component seed as well. There is no decoder, duration, blend-weight or
training-budget search. All six checkpoints must be sealed before candidate
evaluation.

## Evaluation and safeguards

Use the existing five development matches: Metrica 3, Sportec J03WOY, and
SkillCorner 1996435/1996436/2006229. Each has its clean view and two frozen
feed-like corruptions. H/A, D and normalization exclude the evaluated provider.
Aggregate replicas within match, matches within provider, then providers equally.
Rank by unrounded official feed-like spell F1, then macro F1 for exact ties.
Report the auxiliary 70/30 spell/macro summary, without using it to select.

Require clean aggregate spell at least baseline minus 0.005, each provider's
clean macro and D-F1 at least baseline minus 0.02, and each provider's feed-like
spell at least baseline minus 0.01. Report the historical +0.01 clean screen
separately. Success at 0.55 is a stretch, not a promised outcome. A negative
result ends this bounded comparison.

Before fitting, verify archived artifacts, reproduce all 45 calibrated control
cells to 1e-12, and replay a real restored model through the execution path
(maximum probability difference 2e-6, zero changed decoded labels). Verify
the new training wrapper reproduces the original samples exactly with the
replacement disabled. Preserve masks, reference counts and grading populations
in every comparison. New model and native prediction files stay in ignored
`data/experiments/feed_robustness_v1/`.

If an eligible arm improves feed-like spell F1, test that fixed selection and
the baseline on two newly prepared corruption realizations per development
match. Freeze their preparation separately before scoring and do not reselect
on that check. This tests corruption sensitivity, not independent-match transfer.

All five matches have prior research exposure and some targets are event-derived
silver labels. No organizer labels, GAR VALID/TEST labels, public-scoring gate,
or model promotion is part of this study.

## Reproduction

Restore the exact dense archive at private commit `64b6eaafef0f25bd3229aae2021efc9e13a8a1c9`
using `dense_targeted_followups_cache_manifest_v1.json`, plus the exact raw
original-provider arrays, teacher checkpoints and training/evaluation caches
bound by `objective_alignment_cache_manifest.json`. Verify each required file;
an LFS pointer is not an artifact. The declaration records the complete selected
dependency hashes and the known reconstructed-D2 provenance.

```sh
.venv/bin/python -m experiments.feed_robustness_v1 prepare
.venv/bin/python -m experiments.feed_robustness_v1 freeze
.venv/bin/python -m experiments.feed_robustness_v1 preflight
.venv/bin/python -m experiments.feed_robustness_v1 fit --fold leave_out_metrica --seed 2026
# Complete the other five declared fold/seed pairs, then:
.venv/bin/python -m experiments.feed_robustness_v1 seal
.venv/bin/python -m experiments.feed_robustness_v1 evaluate
```

Completed declarations, locks and results must not be overwritten or reused.
