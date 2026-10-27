# Dead-ball information restriction: frozen follow-up

Keep `gru_hgb_ballgap25_v1`. This follow-up addresses the joint GRU's dead-ball
failure from [the completed experiment](LIVE_GAP_HAD_GRU.md). The hypothesis
was formed after reading all five development matches; provider holdout does
not remove that researcher exposure. A positive result is another development
finding, not proof of generalization or permission to promote a replacement.

## Hypothesis and diagnostic

The existing baseline prevents ball features from reaching its D models. The
joint GRU removed that restriction. Sportec freezes the ball during dead play,
which can supply an easy provider-specific cue. A training-only masking audit
of the already fitted joint models found mean P(D) on true dead frames changed
from .994 to .842 on Sportec and .959 to .742 on Metrica when all team-feature
values/masks were withheld. SkillCorner changed .864 to .851. On live frames,
P(D) increased .004 to .104, .004 to .122, and .022 to .180 respectively.

These are means across training matches/folds, not held-out scores. Removing
an entire feature family is out of distribution and cannot by itself establish
causality. The paired architecture ablation below tests the information-path
hypothesis. The original experiment's label and feature contracts remain intact.

## Predeclared comparison

Exactly two variants, two fixed fit seeds (2026 and 2027), and three complete
provider holdouts: 12 checkpoints. All checkpoints must be sealed before any new
evaluation. No optimizer sweep, early stopping, threshold adjustment, decoder
change, best-seed selection, or result-driven follow-on fit is permitted in this
run. Report both seeds and their mean/range; do not ensemble them.

Both models have one shared bidirectional GRU with hidden size 24 and three
output coordinates. They form a normalized H/A/D distribution as
`P(H/A) = (1 - P(D)) * P(H/A | live)`. Two shared-weight passes compute the live
team logits and D logit. This changes the parameterization from the original
three-way softmax but retains the same categorical likelihood, without class
weights. The two new variants have identical capacity, initialization, data
order, training inputs, augmentation, update counts and compute structure:

- **factorized_all:** both passes receive all existing features.
- **factorized_player_D:** only the D pass has all non-player feature values
  **and finite-value masks** removed. Its inputs are the existing strict
  player-only `D_` feature block. H/A still receives the complete feature union.

The restriction guarantees P(D) cannot depend on ball/team features at inference;
it does not guarantee decoded D labels are invariant because the fixed decoder
also uses H/A evidence. Weights remain shared, so H/A gradients can influence
learned player representations. The matched unrestricted control distinguishes
restriction effects from the output-factorization change. Archived joint-GRU
and current-recipe results are also shown, without retraining those comparators.

Training retains the earlier joint-GRU recipe exactly: clean/coarse-ball views,
25% whole-window ball-gap replacement, H/A swaps, equal source/match loss mass,
training-only normalization, eight epochs, batch 64, AdamW .001/.0001, 5 Hz,
10-second cores with five seconds of context per side. No new synthetic
corruption or labels are introduced. Evaluation inputs and native grading masks
are the same clean and two fixed label-free stress views. All components and
normalization exclude the held-out provider; windows never cross match/period
boundaries. The original Viterbi cost remains 3.

## Reporting and stopping rule

Use official spell F1 as primary ranking, retain macro F1 and D F1/precision/
recall, and report each provider and seed, short-spell recall, false possessions,
fragmentation and runtime. D improvements cannot hide losses on other providers.
The conservative development screen requires **both seeds** to satisfy:

- Equal-provider clean spell F1 at least +.01 above the current recipe.
- No provider's clean D F1 or macro F1 more than .02 below the current recipe.
- Equal-provider stress spell F1 no more than .01 below the current recipe.

These predeclared margins are decision aids, not significance tests. Five reused
matches, three providers and two seeds cannot quantify deployment generalization.
No training continues after the readout; failed variants remain in the report.
Fresh independently labeled broadcast confirmation remains tracked in
[private data issue #4](https://github.com/jekkiechan/mtah-data/issues/4).

## Reproduction

Restore the completed live-gap experiment's private archive and its 18 external
arrays. Run `python -m experiments.live_gap_experiment prepare` if its 13 ignored
training-feature caches are not present; the old frozen sources must match.
Then, with the same pinned temporal environment:

```sh
python -m experiments.deadball_transfer_experiment freeze
# Repeat for every provider, arm and fixed seed (12 fits):
python -m experiments.deadball_transfer_experiment fit --source sportec --arm factorized_player_D --seed 2026
python -m experiments.deadball_transfer_experiment seal
python -m experiments.deadball_transfer_experiment evaluate
python -m experiments.deadball_transfer_experiment report
python -m unittest discover -s experiments/tests -v
```

Use a fresh `--out` directory for an independent repeat; completed reports and
checkpoint seals are immutable. Raw inputs, fitted models and predictions stay
in ignored/private storage. No organizer evaluation is part of this program.
