# Realistic tracking corruption experiment

This fixed external-development experiment tests whether replacing the existing
coarse augmentation improves the adopted GRU plus ordered HGB baseline. It
retains the model capacities, eight GRU epochs, 50/50 D blend, conditional H/A
combination, and Viterbi switch cost 3. It does not promote a replacement or
read the organizer's public possession labels.

Four arms isolate the change: unchanged baseline, augmented team attribution
only, augmented D heads only, and augmentation of both. Each training model
receives one clean copy and one augmented copy. For H/A, the composite copy
replaces the old Markov ball-gap copy; for D, it replaces whole-period player
slot dropout. Team swapping and source weighting follow the existing fitters.

The organizer's unlabeled tracking supplies only aggregate ball-gap statistics.
Ball absence is 40.7% of active-half frames, with a median missing run of 10.7 s.
Gap draws use the frozen duration histogram and a 30 s cap. Visible-player
quantiles are 18/21/22/22/22 at the 10th/25th/50th/75th/90th percentiles. These
statistics do not identify why observations disappear, and visibility statistics
are descriptive rather than fitted dropout targets.

The fixed composite recipe also includes temporary spatially correlated player
loss, temporary individual player gaps, smooth shared coordinate drift, small
smooth object-specific errors, occasional shared jumps, same-team track-slot
reassignment after a gap, and a varying 2–8% sample-loss probability. Coordinate
error magnitudes and event rates are assumptions; no coordinate reference is
available to estimate them. All changes occur before feature extraction, retain
NaNs/observation masks, and reset at half/time discontinuities. Frame grids,
timestamps, original target labels and original grading masks remain fixed.

The five whole-match development cases are unchanged: Sportec J03WOY, Metrica
game 3, and SkillCorner 1996435/1996436/2006229. Historical Sportec tests are
excluded. Both the pooled view and provider holdouts of D training are measured;
the team backbones remain pooled in all folds. Provider holdout therefore does
not describe an unseen-provider test of the complete model.

Each model is evaluated on the same clean inputs and two deterministic composite
corruptions per match. The two seeds are averaged within each match, then matches
within each provider, then the three providers equally. Clean and corrupted
scores remain separate. The official spell F1 is primary, with macro F1 for
exact ties; the established 70% spell / 30% macro summary is descriptive only.
The report retains every match/seed/arm result and paired whole-match bootstrap
deltas. These reused matches and synthetic corruptions cannot establish improved
competition performance or replace independent broadcast validation.

## Results

All 120 evaluations completed. The unchanged baseline reproduces the archived
pooled and provider-held-out clean spell, macro and D F1 summaries to 1e-12.
The following uses the preferred D-provider-holdout view, with H/A pooled.
Corrupted scores average the two seeds within each match before provider means.

| Training change | Clean spell F1 | Clean macro F1 | Corrupted spell F1 | Corrupted macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Unchanged GRU + HGB | .5746 | .8449 | .4205 | .7699 |
| Team attribution only | .5659 | .8437 | .4429 | .7895 |
| D heads only | .5718 | .8414 | .4370 | .7814 |
| Both | .5617 | .8399 | .4564 | .7993 |

The composite recipe improves stress robustness but costs native-input spell
accuracy. Augmenting both components adds 3.59 percentage points of corrupted
spell F1, with gains on Sportec (+5.06), Metrica (+4.15) and SkillCorner (+1.55).
Its clean loss is 1.29 points. Team-only augmentation adds 2.24 corrupted spell
points and improves conditional H/A accuracy on reference-live frames from
80.10% to 83.23%; it loses 0.88 clean spell points. D-only augmentation has the
smallest clean loss, 0.28 points, and adds 1.65 corrupted spell points, with no
stress spell gain on SkillCorner.

The pooled deployment-family comparison agrees on the main tradeoff: both-head
augmentation changes clean spell/macro from .5952/.8687 to .5876/.8669 and
corrupted spell/macro from .4307/.7810 to .4702/.8247. Pooled D-only training is
nearly tied on clean spell (.5953), but its clean gain does not survive the
D-provider holdouts. The adopted baseline therefore remains unchanged.

The conditional paired-match bootstrap interval for the both-head corrupted
spell gain is [+3.21, +3.79] percentage points. This conditions on these already
used matches: one Sportec and one Metrica match cannot estimate provider
uncertainty, and the intervals do not establish independent significance.
The evaluation also shares its corruption family with training; two unseen
seeds do not validate a new failure mechanism or real broadcast transfer.

The [full report](results/realistic_corruption_development.json) includes all
match/seed rows, clean and corrupted summaries, D boundaries, and paired deltas.
[Artifact receipts](results/realistic_corruption_artifacts.json) bind all 15
checkpoints to the frozen source/configuration and record package versions.
The 35-file, 13.8 MB archive is preserved privately in `mtah-data` under
`experiments/realistic_corruption_v1/`, with `corruption_cache_manifest.json`.
Until its draft PR is merged, use private cache branch
`experiments/realistic-corruption` when restoring this archive.
The existing cache fetch/restore script verifies each archived file. No raw
corrupted tracking or feature matrix is published.

The next bounded experiment should reduce the corruption training weight and
separate long ball gaps from player/coordinate corruption, retaining the old
augmentation as a control. This batch does not identify which corruption is
helpful or which causes the clean loss. Freeze those comparisons and training
budgets before another external evaluation; prioritize a fresh labeled
broadcast match for confirmation. No organizer recheck was performed.

## Reproduce

Restore external NPZs, organizer input arrays and the baseline package from the
private cache as described in [BASELINE.md](BASELINE.md). Install the pinned
experiment requirements plus CPU PyTorch 2.8.0. All fitted models, prepared
features, corrupted tracking and per-frame predictions remain under ignored
`data/experiments/realistic_corruption_v1/`. Only aggregate reports are tracked.

```sh
python -m experiments.corruption_experiment freeze
python -m experiments.corruption_experiment prepare
python -m experiments.corruption_experiment fit-team
python -m experiments.corruption_experiment fit-d --fold pooled --arm augmented
for source in sportec metrica skillcorner; do
  python -m experiments.corruption_experiment fit-d --fold without_$source --arm baseline
  python -m experiments.corruption_experiment fit-d --fold without_$source --arm augmented
done
python -m experiments.corruption_experiment prepare-eval
for fold in pooled without_sportec without_metrica without_skillcorner; do
  python -m experiments.corruption_experiment evaluate --fold $fold
done
python -m experiments.corruption_experiment report
python -m unittest discover -s experiments/tests -v
python -m unittest experiments.eval_set.test_protocol -v
```

The freeze records code, data, baseline and input hashes, the recipe and all
comparison arms before fitting or evaluation. Modified experiment sources
require a distinct output directory. Model receipts bind checkpoint hashes to
that freeze, and reports refuse incomplete cohorts or existing report files.
The pooled unchanged arm loads the exact adopted checkpoints; provider-held-out
baseline D heads are refitted using their original protocol. No public-window
check marker, promotion manifest or prior candidate is changed.
