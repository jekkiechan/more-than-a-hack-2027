# Physical-feature refinement

This experiment extends the frozen supervised feature implementation with
ball–player relative motion and contact evidence. It tests two model capacities,
using the same clean-plus-masked training protocol as the robustness experiment.
The earlier frozen models and their source files are unchanged.

## Bounded protocol

Only training matches J03WMX, J03WN1, J03WOH and validation match J03WOY can be
loaded by the training command. The script rejects other train/validation splits
and has no test option. It does not load the competition ground truth or any of
J03WPY, J03WR9, or J03WQQ.

New features add:

- Relative ball/player velocity, radial closing speed, velocity alignment, player
  speed along ball motion, and player acceleration for each team's nearest two
  players.
- Short temporal averages of relative motion; local ball acceleration and turn
  angles; ball motion toward the closest pitch boundary.
- Per-team smooth contact evidence combining proximity with shared movement,
  and proximity with ball acceleration.
- Mean motion of teammates within 3 m and 8 m of the ball, plus team speed
  quantiles available even during ball-observation gaps.

Velocity is lightly smoothed over a centered 0.16-second neighborhood. Implausible
player/ball velocities are excluded. All temporal operations respect periods and
time discontinuities. Every team-specific feature has a paired H_/A_ name so the
existing swap operation and inference symmetrization remain exact. Feature
computation explicitly allowlists tracking inputs and cannot read labels, owner,
or ball-status fields.

Training includes one clean and one independently masked view of each match, each
with home/away swap augmentation. Missingness has expected rate 41%, mean block
length one second, training seeds 2026–2028, and validation seed 2026. The sampling
stride is 6 at 25 Hz, matching the prior requested 4 Hz baseline (effective
4.167 Hz). Reference labels affect only supervised targets and valid training-row
selection, never observation masking.

Two predeclared capacities are tested: 31 leaves / 160 iterations and 63 leaves /
260 iterations. Both use multiclass histogram gradient boosting, learning rate
0.08, minimum leaf size 35, L2 regularization 1, random state 17, no early stopping,
and one CPU thread. Decoder penalties are 0, 1, 3, and 8.

Selection maximizes clean validation possession-spell F1 among configurations
whose clean frame macro F1 is at least 0.90; if none qualify, it maximizes clean
macro F1. Synthetic-gap metrics are supplementary and never affect selection.

## Run and inference API

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/refine_features.py
```

Use `--check` to run the small target-independence, team-swap, masked-swap, and
known-motion checks without loading any match. Frozen results cannot be
overwritten; use a different `--out` for a separately declared experiment.

```python
import joblib
from experiments.refine_features import predict, predict_probabilities

model = joblib.load("data/experiments/refine_features/model.joblib")
probabilities = predict_probabilities(tracking_inputs, model)  # columns H, A, D
labels = predict(tracking_inputs, model)
```

The model requires this new feature API; passing it to
`supervised.predict_probabilities` would fail the feature-schema check.

## Results

The run completed in 613.3 seconds on one CPU thread, fitting 283,164 augmented
training rows with 137 features. The selected configuration is **63 leaves,
260 iterations, switch cost 3**.

| Leaves / iterations | Switch cost | Clean macro F1 | Clean spell F1 | Masked macro F1 | Masked spell F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 31 / 160 | 0 | 0.930596 | 0.306064 | 0.904047 | 0.238480 |
| 31 / 160 | 1 | 0.933089 | 0.642716 | 0.908458 | 0.561419 |
| 31 / 160 | 3 | 0.931357 | 0.656595 | 0.908803 | 0.601757 |
| 31 / 160 | 8 | 0.928819 | 0.631771 | 0.906695 | 0.575505 |
| 63 / 260 | 0 | 0.933764 | 0.294480 | 0.908615 | 0.224197 |
| 63 / 260 | 1 | 0.936495 | 0.644161 | 0.911685 | 0.552101 |
| **63 / 260** | **3** | **0.936455** | **0.671795** | **0.913057** | **0.606404** |
| 63 / 260 | 8 | 0.932220 | 0.647595 | 0.910791 | 0.593140 |

Compared with the prior gap-augmented model (clean macro/spell
0.918769 / 0.610784), the selected refinement gains 0.017687 macro F1 and
0.061011 spell F1. Under the identical synthetic mask, gains are 0.014804 macro F1
and 0.046106 spell F1. Holding the original 31-leaf / 160-iteration capacity fixed,
the physical features already gain 0.045810 clean spell F1; the larger capacity
adds a further 0.015200. This separates the feature benefit from the capacity
benefit and supports testing a different training objective before another size
sweep. It does not establish a general performance ceiling.

Frozen artifacts under ignored `data/experiments/refine_features/`:

- `model.joblib`, SHA-256
  `dc2eb2800e078c4c86a90c28b4c35dfa8662f06b275a5e2b6da692261003ea1a`.
- `frozen_config.json` contains source/data/model hashes and the selection policy.
- `validation.json` contains the complete compact metrics for all eight candidates.
- `J03WOY_clean_probabilities.npz` and `J03WOY_masked_probabilities.npz` contain
  selected probabilities (float32 columns H/A/D), period, timestamps, and fps.
  They contain no reference labels.

The separately owned [combined decoder experiment](refine_combined.py) applies
one already selected team/dead penalty pair to these frozen probability arrays.
Its reported clean spell F1 of 0.673046 adds only 0.001251 over the uniform penalty,
so most of that composed model's gain comes from the physical features and model
capacity. See that experiment's artifacts for its independent freeze/verification.

## Scope and limits

This is further selection on an already used validation match. It does not create
new independent evidence or justify tuning on previously opened test matches.
The synthetic gaps do not reproduce broadcast occlusion, identity swaps, camera
motion, or calibration errors. New motion features may be particularly sensitive
to noisier tracking, so masked validation and deployment diagnostics matter even
when clean validation improves.
