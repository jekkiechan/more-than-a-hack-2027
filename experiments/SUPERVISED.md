# Supervised tracking experiment

`supervised.py` combines geometry/motion features, histogram gradient-boosted
trees, and a Viterbi decoder with a penalty for changing state. The primary
benchmark is the externally sourced Sportec dataset, with complete matches
assigned to training, validation, and test before fitting any model.

## Inputs and features

The input contract is the shared NPZ format: player and ball positions in metres,
team membership, period, timestamps, frame rate, pitch dimensions, and optionally
whether the ball was directly observed. Player identities and event annotations
are unnecessary.

The 65 features describe:

- each team's closest player distances and counts near the ball;
- ball speed, observation gaps, and distance to the pitch boundary;
- the last and next geometrically close player approaches, at three radii;
- median and 75th-percentile player speed, stationary/running fractions, and their
  temporal averages; displacements implying speeds above 12 m/s are excluded;
- short centred temporal averages and nearest-team votes.

All interpolation, feature windows, and decoding reset at period boundaries and
timestamp discontinuities. Positions need no persistent player identity, although
velocity estimates can still be affected by small identity-switch displacements.
Centred features use future context, so this is an offline method.

Source possession, owner, ball-in-play/status, event type, and target labels are
**not input features**. A check confirms that changing labels leaves the feature
matrix unchanged. A second check confirms home/away swapping is equivariant.

## Training and selection

Training samples approximately four frames per second from the training matches.
Team-swap augmentation exchanges home and away and preserves dead-ball labels.
Inference also averages a prediction with its team-swapped counterpart.

The candidate set compares multiclass models with 15 leaves/100 boosting
iterations and 31 leaves/160 iterations, plus a two-stage model (dead/live followed
by home/away), with 31 leaves/160 iterations in each stage. Each is compared with
switch costs of 0, 3, 8, 16, and 32. Viterbi costs are scaled to the frame rate.
Selection maximizes whole-match validation spellF1 among candidates attaining
macroF1 of at least 0.90; if none do, it maximizes macroF1. Best-macro and best-spell
models are also retained. Test data are not loaded during this selection. The
chosen model and config are saved before any requested test run.

```bash
python experiments/supervised.py
```

This default command trains and validates on the primary Sportec split and does
**not** evaluate test data. Outputs are in the ignored
`data/experiments/supervised/` directory: `model.joblib`, `frozen_config.json`, and
`validation.json`. Alternative extrema are `best_macro_model.joblib` and
`best_spell_model.joblib`. The root experiment coordinator controls the held-out test.

The callable interface is:

```python
from experiments.supervised import load, fit, predict
model = fit(training_npz_paths)
labels = predict(load(match_npz_path), model)
```

Scoring uses `experiments.metrics.evaluate_tracking`, which retains the official
metric implementation and handles frame rate and period/time gaps consistently.

## Validation results (not held-out test results)

Training matches: J03WMX, J03WN1, J03WOH. Validation match: J03WOY.
The held-out J03WPY match was not loaded or evaluated by this experiment agent.
The root coordinator evaluates the frozen model separately.

| Model | Macro F1 | Possession-spell F1 |
| --- | ---: | ---: |
| Multiclass 31-leaf HGB, no decoding | 0.9154 | 0.2882 |
| Multiclass HGB + Viterbi cost 3 (selected) | 0.9191 | 0.6022 |
| Multiclass HGB + Viterbi cost 8 (best macro) | 0.9191 | 0.5848 |
| Two-stage HGB + Viterbi cost 3 | 0.9175 | 0.5909 |
| Selected model, ball-missing stress | 0.7448 | 0.3156 |

The selected model uses 31 leaves, 160 iterations, and switch cost 3. Its clean
validation accuracy is 0.9349; per-class F1 is home 0.8773, away 0.8807, and dead
0.9992. It predicts 367 possession spells against 456 reference spells.
The stress test masks the ball in label-independent Markov blocks with expected
41% missingness, mean duration 1 second, seed 2026. It was applied after selection
and was not used to choose the model or settings.

These results exceed 0.90 **validation macroF1**, but not 0.90 possession-spell F1.
They do not establish either threshold on unseen matches or challenge footage.

## Scope and limitations

Sportec has explicit H/A/D reference labels. SkillCorner is an auxiliary source
with provider-generated H/A labels and unknown X frames, but no explicit dead-ball
target. Performance on its labelled frames cannot establish three-state possession
performance, especially during unlabelled/contested intervals.

The method has not used the hackathon match's public labels for fitting, parameter
selection, or evaluation. External Sportec tracking is cleaner than the hackathon
tracking, so any external score does not establish performance on the hackathon's
missing-ball and identity-switch conditions.

The data audit found an especially important shortcut: Sportec repeats the last
ball XY position throughout dead-ball periods in the source XML itself. A model
using movement can therefore recognize many D frames almost perfectly from a
provider-specific processing artifact. The high D score contributes materially
to the 0.919 macroF1, while H/A scores are below 0.90. The missing-ball stress
result further demonstrates the transfer limitation. No source BallStatus,
BallPossession, speed, or height field is supplied to this model as a feature.

The run uses NumPy, SciPy, scikit-learn, joblib, and threadpoolctl, with one compute
thread per estimator to share CPU with the parallel experiments.
