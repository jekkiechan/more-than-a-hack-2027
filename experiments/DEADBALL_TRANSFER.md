# Dead-ball information restriction: completed follow-up

Keep `gru_hgb_ballgap25_v1`. This follow-up addresses the joint GRU's dead-ball
failure from [the completed experiment](LIVE_GAP_HAD_GRU.md). The hypothesis
was formed after reading all five development matches; provider holdout does
not remove that researcher exposure. A positive result is another development
finding, not proof of generalization or permission to promote a replacement.

The observed clean SkillCorner collapse is repaired in both seeds: D F1 rises
from .1773 in the original joint GRU to .7123/.6986, near the baseline's .7022.
SkillCorner macro F1 recovers from .5479 to .7527/.7509. The matched unrestricted
factorized control reaches only .1997/.3743 D F1, so factorization alone does not
explain the recovery. This supports the player-only D restriction on these
development matches, without establishing fresh-provider generalization.

Across all providers, the constrained model averages .6093 clean spell F1,
.8411 macro F1 and .8303 D F1. Both seeds clear the clean +.01 spell screen and
keep each provider's clean D/macro within the frozen .02 tolerance of baseline.
**Both fail the stress guardrail**: spell F1 is .2767/.3214 versus .4129 baseline.
Keep the baseline; no additional development fits or threshold adjustments
followed this readout. A later explicit user request authorized one unchanged,
pooled seed-2026 fit and a separate competition pipeline check: .6194 spell /
.8574 macro / .8506 D F1 versus the saved baseline .5257/.8588/.8568. See
[the completed diagnostic](JOINT_COMPETITION_CHECK.md). It does not override
the external stress failure or change the baseline.

Under corruption the constrained model overpredicts dead play on Sportec and
Metrica. D precision averages .6291/.4741 while recall is .9886/.9799; baseline
precision is .8027/.8002. SkillCorner stress D F1 remains near baseline, so the
clean recovery does not imply uniform corruption robustness. The unchanged
joint training recipe lacks the baseline D models' player-slot-dropout view.
Restoring that existing robustness mechanism is a concrete hypothesis for a
separate predeclared study, not an untested fix claimed by this result. Fresh
independent broadcast confirmation is still needed before adoption.

## Completed development readout

All 12 checkpoints were frozen before scoring; all 60 new evaluations completed.
The 30 archived current-recipe/original-joint comparisons are reused unchanged.
New-arm figures below average seeds 2026 and 2027; these are separate models,
not an ensemble. Provider means give each provider equal weight. These are full
H/A-and-D provider holdouts, not scores from the pooled deployed package.

| Model | Clean spell | Clean macro | Clean D F1 | Stress spell | Stress macro | Stress D F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current recipe | 0.5338 | 0.8300 | 0.8338 | 0.4129 | 0.7779 | 0.8168 |
| Original joint GRU | 0.6272 | 0.7883 | 0.6704 | 0.3976 | 0.6722 | 0.5846 |
| Unrestricted factorized | 0.6132 | 0.7900 | 0.6901 | 0.4017 | 0.6946 | 0.6336 |
| Player-only D factorized | 0.6093 | 0.8411 | 0.8303 | 0.2991 | 0.6464 | 0.7073 |

Auxiliary policy summary (70% spell / 30% macro); this does not rank models:

| Model | Clean 70/30 | Stress 70/30 |
| --- | ---: | ---: |
| Current recipe | 0.6227 | 0.5224 |
| Original joint GRU | 0.6755 | 0.4800 |
| Unrestricted factorized | 0.6663 | 0.4896 |
| Player-only D factorized | 0.6788 | 0.4033 |

### Individual seeds and frozen screen

| Model | Seed | Clean spell | Clean macro | Clean D F1 | Stress spell | Passes all guardrails |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unrestricted factorized | 2026 | 0.6191 | 0.7837 | 0.6680 | 0.3875 | No |
| Unrestricted factorized | 2027 | 0.6073 | 0.7963 | 0.7121 | 0.4159 | No |
| Player-only D factorized | 2026 | 0.6083 | 0.8408 | 0.8316 | 0.2767 | No |
| Player-only D factorized | 2027 | 0.6103 | 0.8415 | 0.8290 | 0.3214 | No |

### Clean provider breakdown

| Provider | Model | Spell mean [seed range] | Macro mean | D F1 mean [seed range] | D recall mean |
| --- | --- | ---: | ---: | ---: | ---: |
| sportec | Current recipe | 0.5332 [0.5332, 0.5332] | 0.8500 | 0.9098 [0.9098, 0.9098] | 89.1% |
| sportec | Original joint GRU | 0.6629 [0.6629, 0.6629] | 0.8995 | 0.9458 [0.9458, 0.9458] | 91.6% |
| sportec | Unrestricted factorized | 0.6627 [0.6550, 0.6704] | 0.8869 | 0.9319 [0.9209, 0.9430] | 89.9% |
| sportec | Player-only D factorized | 0.6202 [0.6099, 0.6306] | 0.8630 | 0.9020 [0.9002, 0.9038] | 89.6% |
| metrica | Current recipe | 0.6457 [0.6457, 0.6457] | 0.8912 | 0.8893 [0.8893, 0.8893] | 89.5% |
| metrica | Original joint GRU | 0.8066 [0.8066, 0.8066] | 0.9176 | 0.8880 [0.8880, 0.8880] | 80.0% |
| metrica | Unrestricted factorized | 0.7616 [0.7458, 0.7774] | 0.8959 | 0.8512 [0.8413, 0.8612] | 74.6% |
| metrica | Player-only D factorized | 0.7565 [0.7461, 0.7669] | 0.9085 | 0.8834 [0.8787, 0.8881] | 89.2% |
| skillcorner | Current recipe | 0.4225 [0.4225, 0.4225] | 0.7488 | 0.7022 [0.7022, 0.7022] | 59.1% |
| skillcorner | Original joint GRU | 0.4121 [0.4121, 0.4121] | 0.5479 | 0.1773 [0.1773, 0.1773] | 9.8% |
| skillcorner | Unrestricted factorized | 0.4154 [0.4096, 0.4212] | 0.5873 | 0.2870 [0.1997, 0.3743] | 17.2% |
| skillcorner | Player-only D factorized | 0.4511 [0.4482, 0.4541] | 0.7518 | 0.7055 [0.6986, 0.7123] | 59.8% |

### Paired effect of restricting D inputs

| Provider | Clean D-F1 delta, 2026 / 2027 | Clean spell delta, 2026 / 2027 | Stress D-F1 delta, 2026 / 2027 |
| --- | ---: | ---: | ---: |
| sportec | -0.0392 / -0.0207 | -0.0399 / -0.0451 | -0.1069 / -0.0496 |
| metrica | +0.0175 / +0.0468 | -0.0314 / +0.0211 | -0.1248 / -0.1937 |
| skillcorner | +0.5127 / +0.3244 | +0.0386 / +0.0329 | +0.5444 / +0.3725 |

Differences are constrained minus unrestricted, paired by fit seed. They isolate
the D information restriction within the factorized model; they do not estimate
uncertainty over unseen matches or providers.

### Behavior and cost

| Model | Short-spell recall | False possessions / min | Seconds / match |
| --- | ---: | ---: | ---: |
| Current recipe | 29.2% | 2.917 | 52.9 |
| Original joint GRU | 38.0% | 2.753 | 22.9 |
| Unrestricted factorized | 37.4% | 2.799 | 25.1 |
| Player-only D factorized | 35.0% | 2.106 | 24.8 |

Each new model has 62,643 parameters and one shared GRU weight set. Computing
both factorized paths and H/A swap symmetry takes four sequence passes. Timings
include features and fixed decoding; baseline timings are archived from the
same environment in the earlier run, not concurrent randomized benchmarks.
Every prediction and frame-level result remains in private/ignored storage.

All source/data/checkpoint receipts and paired update/window counts were verified.
81 contract/synthetic tests, bundled format validation and implementation CI passed.
Two initial constrained Sportec jobs hit the memory limit and were restarted
with the same frozen seeds after reducing concurrency; no retry checkpoint was
selected by score. This operational retry did not change training or evaluation.

Aggregate evidence: [full results](results/deadball_transfer_development.json),
[run audit](results/deadball_transfer_audit.json), and
[training-only shortcut diagnostic](results/dead_shortcut_training_audit.json).
The follow-up archive is verified locally in `/workspace/mtah-live-gap`, with
`deadball_transfer_cache_manifest.json` (including the prior-run dependencies).
Its additional upload to private `mtah-data` branch `experiments/live-gap-had-v1`
is pending separate approval; the first experiment's approved upload is complete.

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

This isolates the D information path within the joint training recipe. It does
not restore the baseline D models' separate player-slot-dropout training view.
That remains a known difference from the adopted pipeline and a limitation of
any claim that this experiment reproduces all of its robustness mechanisms.

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

Once the follow-up upload is approved, restore the complete archive from private
`mtah-data` branch `experiments/live-gap-had-v1` (including the prior experiment
and 18 external arrays):

```sh
python scripts/competition_cache.py fetch --manifest deadball_transfer_cache_manifest.json --data-dir /path/to/code/data
```

For a fresh repeat in the code checkout, regenerate the 13 ignored training
feature caches if absent; their timing metadata can change cache file hashes,
so freeze a new output directory. Use the same pinned temporal environment:

```sh
python -m experiments.live_gap_experiment prepare
python -m experiments.deadball_transfer_experiment freeze --out data/experiments/deadball_transfer_repeat
for provider in sportec metrica skillcorner; do
  for arm in factorized_all factorized_player_D; do
    for seed in 2026 2027; do
      python -m experiments.deadball_transfer_experiment fit --source "$provider" --arm "$arm" --seed "$seed" --out data/experiments/deadball_transfer_repeat
    done
  done
done
python -m experiments.deadball_transfer_experiment seal --out data/experiments/deadball_transfer_repeat
python -m experiments.deadball_transfer_experiment evaluate --out data/experiments/deadball_transfer_repeat
python -m experiments.deadball_transfer_experiment report --out data/experiments/deadball_transfer_repeat
python -m unittest discover -s experiments/tests -v
```

Use a fresh `--out` directory for an independent repeat; completed reports and
checkpoint seals are immutable. Raw inputs, fitted models and predictions stay
in ignored/private storage. No organizer evaluation is part of this program.
