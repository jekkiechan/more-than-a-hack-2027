# Geometry-emission hidden Markov model

`temporal.py` implements a separate probabilistic approach to possession:

1. Build symmetric features from ball/player coordinates and velocities: nearest
   three player distances per team, relative velocity of the closest player,
   nearest player speed, ball speed, observation coverage, outside-pitch distance,
   and player motion. Collective-motion features include median speed and running
   fraction, excluding implausible track jumps. Short ball gaps up to 0.5 seconds
   are interpolated within a period only. Longer gaps remain missing.
2. Learn a full-covariance Gaussian mixture for each observed target class.
   Training uses team-swapped examples so the model cannot rely on home advantage.
3. Estimate a transition matrix on the training matches. Decode the whole sequence
   with Viterbi, resetting at period changes and timestamp discontinuities.
4. Choose the mixture count and transition weight using the validation match only.
   The selection rule maximizes spell F1 among configurations meeting 0.90 macro
   F1; if none qualify, it chooses highest macro F1. Save the selected configuration
   before evaluating the untouched test match.

When the ball is missing for longer than the interpolation limit, the emission
model marginalizes out ball-related features and uses only collective player
motion. It does not pretend that missing ball coordinates are a stationary ball.

No possession-owner, live/dead, event, or label field enters the input features.
No challenge public labels are read by this experiment. Labels are read only by
the external training and evaluation functions. Model outputs and trained models
are stored under gitignored `data/experiments/`.

## Interface and reproduction

```python
from experiments.temporal import predict
labels = predict(data, {"model_path": "path/to/gmm.joblib", "transition_scale": 4.0})
```

The `data` dictionary follows the shared NPZ contract: physical ball/player
coordinates in metres, team membership, period, frame rate, timestamps, and pitch
dimensions. `predict` does not need or inspect its `labels` key.

```bash
# Primary three-class training and validation. The default never opens test labels.
python experiments/temporal.py --split data/external_audit/sportec/split.json \
  --out data/experiments/temporal_sportec
```

The module limits numerical-library threading to one thread. Its dependency set
is NumPy, SciPy, scikit-learn, joblib, and threadpoolctl, alongside the shared
official-metric wrapper.

## Primary three-class development result

The Sportec split is fixed by match ID: training `J03WMX`, `J03WN1`, `J03WOH`;
validation `J03WOY`; test `J03WPY`. The targets derive from the explicit supplied
BallStatus and BallPossession fields. Those fields are excluded from features.

Training tried 2, 4, and 8 Gaussian components per class with six transition weights.
A further validation-only check separated team-turnover and live/dead transition
weights; it did not improve the selected result. The frozen selection is therefore
the simpler **4-component model with transition weight 4**.

| Evaluation | Macro F1 (H/A/D) | Spell F1 | Frame accuracy |
| --- | ---: | ---: | ---: |
| Sportec validation | 0.8972 | 0.4968 | 0.9169 |
| Same validation, synthetic ball gaps | 0.8570 | 0.4099 | 0.8788 |

The stress run was performed after freezing the configuration. It uses the shared
target-independent random-gap generator with expected missingness 41%, mean gap
one second, and seed 2026; realized missingness was 40.32%. This is an artificial
robustness check and does not replicate difficult real occlusions.

The model did **not** reach 0.90 macro F1 on validation. Its clean validation
class F1 scores were H 0.8410, A 0.8541, and D 0.9964. The very strong D result
has an important domain limitation: the original Sportec source freezes ball XY
coordinates during dead-ball periods. That processing artifact provides a much
easier signal than the hackathon's uncorrected, frequently missing ball tracking.
No equivalent hackathon performance is claimed.

The temporal agent did not inspect the primary test labels. Frozen configuration,
model/code hashes, validation trials, and stress metrics are under
`data/experiments/temporal_sportec/`. The coordinating evaluation can score the
frozen model once on the held-out match; no tuning follows that score.

## Auxiliary development smoke test

SkillCorner supplied possession labels are algorithmic silver labels and cover
only home/away ownership. Null ownership stays X; it is not treated as dead ball.
This experiment is therefore **not a three-class H/A/D benchmark**.

An earlier 17-feature prototype was exercised on the preselected SkillCorner
validation match `1925299`, reaching masked spell F1 0.5083 and supported H/A macro
F1 0.8418. This historical smoke test predates the collective-motion additions;
it is not the frozen primary model, not a three-class result, and not evidence of
reaching 0.90 F1. SkillCorner test data was not evaluated.

## Checks and limitations

Synthetic checks passed for feature independence from labels, exact team-swap
equivariance, finite features through missing-ball gaps, period-reset detection,
and recovering a clear possession switch with Viterbi.

This model assumes that useful physical patterns transfer between matches.
Mixture densities can overstate confidence for unfamiliar tracking distributions;
emission differences are clipped to limit a single outlier's influence. A
first-order HMM implies geometric state durations and has no explicit rule for
referee stoppages. If training targets lack D, the model cannot learn dead ball
and makes no claim to solve that part of the challenge.
