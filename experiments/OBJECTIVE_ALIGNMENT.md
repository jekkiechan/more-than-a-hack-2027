# Possession-aware training objectives

Status: running. All thirteen feature caches are sealed; all six matched
control normalization/budget checks pass. The eighteen new fits are underway.
Date: 2026-10-28. Branch: `possession/objective-alignment`, based on `fa61bd3`.

The user requested a matched trial of short-possession weighting and transition
supervision, using Luna subagents at xhigh. The study isolates the training loss
of the adopted slot-dropout, player-only-D joint GRU. It does not change the
working baseline or use organizer reference labels.

## Question and fixed comparison

Does a mild possession-aware training objective improve official decoded spell
F1 while retaining frame quality and feed-like tracking robustness?

| Arm | Changed mechanism |
| --- | --- |
| Control | Restored adopted-recipe provider-holdout checkpoints, seeds 2026/2027 |
| Spell | Capped square-root duration weights on conditional H/A training loss |
| Transition | Fixed-weight probability-based dead/live and H/A change auxiliary |
| Combined | Both fixed mechanisms |

The comparison retains five seconds of context on each side of ten-second
output cores, the same features, hidden-24 factorized bidirectional GRU,
player-only D path, eight epochs, batch size 64, AdamW settings, training-only
normalization, augmentation/window choices, team symmetry and Viterbi cost 3.
The separate [PR22 context study](https://github.com/jekkiechan/more-than-a-hack-2027/pull/22)
tests seed averaging and longer context; those changes are not mixed into this
loss comparison.

Three complete provider holdouts and two fixed seeds give six restored control
models and eighteen new fits. Every learned H/A, D and normalization component
excludes the held-out provider. The thirteen external TRAIN matches supply
training targets. The five previously exposed development matches supply
clean and fixed `feed_like` evaluation views, with stress replicates averaged
within match, matches within provider, then providers equally. These are
development findings, not independent confirmation.

## Why the losses differ

The existing categorical likelihood decomposes into a D/live term and a
conditional H/A term. Spell weighting changes only the conditional H/A term,
so it does not directly change dead/live class mass. The fixed raw multiplier
is `clip(sqrt(5 seconds / duration), 0.5, 2)`. Durations come from native,
uncorrupted external training labels before 5 Hz sampling. Unknown labels and
period/time discontinuities split runs. Censored runs retain neutral weight;
eligible live multipliers are normalized to preserve their training mass.

The transition auxiliary derives change probabilities from adjacent marginal
probabilities: one term for D/live changes from the player-only D probability,
and one for H/A changes conditional on both reference endpoints being live.
The coefficient is .1, with fixed half shares for each component and fixed
half shares for stable/change groups inside each component; an absent group
or component contributes zero. Training-only group weights retain provider
and match balance. This is a cost-sensitive adjacent-marginal independence
surrogate, not a calibrated transition likelihood or a differentiable
implementation of official one-to-one spell matching. It adds no inference
head or parameters. Pair masks exclude unknown native intervals and period/time
breaks; each edge belongs to its left output-core frame.

## Evaluation and safeguards

Rank by unrounded official spell F1 averaged across IoU .3/.4/.5/.6/.7; macro
F1 breaks exact ties. Report the fixed 70/30 summary as an auxiliary column.
Compare each seed against its corresponding restored control. The screen is
clean spell improvement of at least .01 in both seeds, feed-like spell delta
at least -.01, and each provider's clean macro/D-F1 delta at least -.02.
Also report D precision/recall, all five spell thresholds, short-spell recall,
missed/spurious spells, fragmentation and transition timing where graded.
Unknown-label censorship remains part of the fixed evaluation population.

The runner freezes source/configuration/input/cache/checkpoint hashes before
fitting and seals all candidate models before scoring. It verifies archived
control parity and does not tune objective weights, select seeds, sweep the
decoder, or reopen organizer scoring gates. All models and frame outputs stay
in ignored `data/` or the private cache. Adoption is a separate decision.

## Validation and artifacts

The original 145 synthetic/contract tests pass in Python 3.12.14 with the
repository's pinned dependencies and CPU PyTorch 2.8.0. The private
`gru_slot_robustness_cache_manifest.json` (253 files) and
`seed_gap_context_cache_manifest.json` (176 files) were restored and verified
by size and SHA-256. The archives do not include training feature caches;
those are regenerated in the new study directory from the exact original raw
view/seed recipe, under a separately frozen preparation declaration. Historical
directories remain read-only. The embedded historical source inventory differs
only in the three later-promoted baseline files; all feature/training sources
match. The eight new loss contracts pass, including exact zero-coefficient
control state/history/prediction parity. Protocol tests and the immutable full
declaration will be recorded before training. Results and completion status
will be added after the declared run.

## Frozen evidence

- [Study declaration](results/objective_alignment_declaration.json).
- [Preparation declaration](results/objective_alignment_preparation.json).
- [Exact training parity](results/objective_alignment_training_parity.json).

All 156 synthetic/contract tests pass, and the bundled 199,319-row submission
passes the official format validator. Comparative scores are pending.
