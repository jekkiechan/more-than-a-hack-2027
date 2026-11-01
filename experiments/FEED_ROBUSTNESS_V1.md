# Feed-like training views for the calibrated dense ensemble

Status: completed; stretch unmet; baseline unchanged.
Study `feed_robustness_v1`, 2026-11-01, based on fork
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

## Conditional component follow-up

Before any candidate scores were available, `feed_head_transfer_v1` declared
two fixed follow-ups if the full augmentation study misses 0.55: augmented
conditional H/A with baseline D, and baseline conditional H/A with augmented D.
Cross components within each matching seed, average native probabilities, and
apply the same frozen calibrated decoder. This isolates whether one output
component benefits while the other regresses. It adds zero fits and 90 scored
cells, with no calibration or blend-weight tuning. The controls, primary metric
and guards remain unchanged. The fixed whole-model blend and the component
crosses are different experiments, with separate declarations and scoring locks.

```sh
.venv/bin/python -m experiments.feed_head_transfer_v1 declare
# After the parent comparison and its independent replay complete:
.venv/bin/python -m experiments.feed_head_transfer_v1 run
```

## Initial results

All six fits completed at 2,048 updates. The 45 control cells reproduce the
promoted calibrated pipeline exactly; real restored-model inference differs
by at most 2.38 × 10⁻⁷ in probabilities and changes zero decoded labels.

| Fixed arm | Clean spell F1 | Feed-like spell F1 | Guards |
| --- | ---: | ---: | --- |
| Promoted baseline | 0.658625 | 0.518527 | Pass |
| Raw-feed augmented ensemble | 0.654740 | 0.525240 | Pass |
| Fixed equal whole-model blend | 0.660076 | 0.520468 | Pass |
| Augmented H/A, baseline D | 0.656137 | 0.523298 | Pass |
| Baseline H/A, augmented D | 0.656566 | 0.517831 | Pass |

The full augmented ensemble gains +0.006713 feed spell and loses 0.003885
clean spell, inside the declared clean tolerance. Feed gains occur in both
seeds and all three providers. Neither component swap beats the complete
augmented model. The +0.01 historical clean-gain screen fails for every arm;
these are feed-first experimental candidates, not a promotion.

The separate replay verifies 135 parent score rows and 90 component-transfer
rows, native probability ensembles/blends, unchanged grading, and exact
training-sample exposure parity with the old D2 fits.

## Fixed diversity continuation

After this positive original-cohort result, `feed_seed_diversity_v1` tests two
additional dense-training seeds, 2028/2029. They reuse provider-excluded
teachers 2026/2027 respectively; they are not newly trained teacher seeds.
The same augmentation, examples, objectives, normalization and 2,048-update
budget apply to six new fits. The sole primary candidate is the equal native
probability average of all four augmented members. There is no seed/subset
selection or mixture-weight search. Compare with both the two-member candidate
and promoted baseline, retaining the original guards. All additional models
must be sealed before scoring.

This continuation was declared before reading fresh-corruption scores. Evaluate
the fixed four-member rule on the frozen fresh views regardless of its original
score, without reselection. New corruption realizations on exposed matches
remain weaker evidence than new independently labeled matches.

## Preserved fresh-view runtime correction

The first fresh-view evaluation stopped before scoring any candidate or writing
prediction arrays: `preparation.build_record` stores feature names as a NumPy
array, while the frozen predictor compares lists. The original runner,
preparation hashes, failure log and scoring lock are preserved. A separately
recorded runtime adapter converts only feature-name metadata to an equal Python
list at the feature-only boundary. It changes no numerical features, targets,
models, decoder, arms or score population. A contract confirms array/list
equivalence and exclusion of target fields. The same correction is declared
before the diversity continuation's fresh evaluation.

Evidence: [parent declaration](results/feed_robustness_v1_declaration.json),
[parent aggregate](results/feed_robustness_v1_aggregate.json),
[parent verification](results/feed_robustness_v1_verification.json),
[component declaration](results/feed_head_transfer_v1_declaration.json),
[component aggregate](results/feed_head_transfer_v1_aggregate.json),
[component verification](results/feed_head_transfer_v1_verification.json),
[fresh-view declaration](results/feed_robustness_confirmation_v1_declaration.json),
[bounded execution amendment](results/feed_confirmation_runtime_v1_parent_amendment.json),
[diversity declaration](results/feed_seed_diversity_v1_declaration.json).

## Completed diversity and fresh checks

| Fixed rule | Original clean | Original feed-like | Fresh feed-like |
| --- | ---: | ---: | ---: |
| Promoted baseline | 0.658625 | 0.518527 | 0.524936 |
| Two augmented seeds | 0.654740 | 0.525240 | 0.533871 |
| Four augmented seeds | 0.656688 | 0.524077 | 0.535179 |

Four-seed guards pass: **True**. Improves the two-seed candidate: **False**. The original-benchmark 0.55 stretch is **not reached**. Best eligible arm is Two augmented seeds, 0.525240. No model promotion occurs.

All twelve fits completed at the declared budget. The fresh two-seed improvement is +0.008936 relative to its matched baseline; these are new synthetic realizations on the same previously exposed matches, not independent-match confirmation. The additional four-seed rule was fixed before fresh results and evaluated regardless of its original result.

The independent verifiers replay 135 parent, 90 component-transfer, 60 fresh parent and 55 diversity score cells. They check official scores, equal-provider aggregation, unchanged native grading, component algebra and original-view probability ensemble parity. Fresh four-seed scores and grading are replayed; its probability averaging is not separately reconstructed by the verifier.

## Diagnostics and next decision

The initial two-seed augmentation raises feed macro F1 from 0.823343 to 0.826937, while D-F1 declines from 0.859048 to 0.857715 (D precision 0.843498, recall 0.875295). Short-spell recall changes only from 0.198820 to 0.199402. Predicted spell count is 282.778 versus 404.333 reference spells under equal-provider weighting. This points to substantial remaining short-possession recovery work; the robustness improvement does not solve that failure mode.

The strongest next hypothesis is better short-possession supervision and coverage, tested with an explicit sampling/exposure audit before fitting. Prior 2× weighting barely exposed the relevant examples, and adding training seeds alone is a different intervention. Independent densely labeled matches would give stronger evidence than further tuning these five development matches.

Validation: 531 tests pass, plus integrity, file-backed training-contract, bundled CSV-format and documentation checks. No organizer references were accessed or scored.

Evidence: [fresh aggregate](results/feed_robustness_confirmation_v1_aggregate.json), [fresh replay](results/feed_confirmation_independent_verification_v1.json), [diversity aggregate](results/feed_seed_diversity_v1_aggregate.json), [diversity fresh result](results/feed_seed_diversity_v1_fresh.json), [diversity replay](results/feed_diversity_independent_verification_v1.json).

## Provider and seed spread

| Provider | Baseline feed | Two-seed augmented feed | Four-seed augmented feed |
| --- | ---: | ---: | ---: |
| Metrica | 0.606749 | 0.617628 | 0.618407 |
| SkillCorner | 0.454592 | 0.457600 | 0.456015 |
| Sportec | 0.494242 | 0.500494 | 0.497808 |

| Augmented training seed | Clean spell | Feed spell |
| --- | ---: | ---: |
| 2026 | 0.645040 | 0.518567 |
| 2027 | 0.648017 | 0.516938 |
| 2028 | 0.645852 | 0.518603 |
| 2029 | 0.642903 | 0.511507 |

These individual-score spreads are descriptive, not confidence intervals. The ensemble is evaluated from averaged probabilities rather than averaged metrics.

Four-seed historical +0.01 clean screen: **False**.

## Verified delivery

Research [PR #62](https://github.com/jekkiechan/more-than-a-hack-2027/pull/62)
is merged with passing CI. An independent private archive clone with its own
LFS store fetched and restored all **408 files / 1,672,495,783 bytes**, with exact
size and SHA-256 checks. This covers the new artifacts and source overlay;
original dependencies remain those in the declaration and reproduction section.
Archive identifiers and restore instructions remain in the private archive.

The [delivery receipt](results/feed_robustness_v1_delivery.json) records the
verification totals and publication. Research site version 17 is
[published](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/experiments.html#feed-robustness)
with unchanged restricted access (revision 2). Desktop/mobile layouts, chart
view switching and live HTTP checks pass. The adopted package and organizer
scoring gates remain unchanged.
