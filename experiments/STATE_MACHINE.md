# Geometry and temporal state-machine experiment

This approach uses player and ball coordinates, team membership, period/time
boundaries, pitch dimensions, and sampling rate. It never consumes the source
possession owner, ball status, speed, height, or target labels as features.

`experiments.state_machine.predict(data, config)` returns H/A/D/X labels from the
shared tracking NPZ schema. It needs NumPy and SciPy and runs on CPU.

## Method

1. Optionally interpolate short ball-observation gaps, without crossing period or
   timestamp discontinuities.
2. Compute nearest home/away-player distances to the ball. Only players within a
   selected control radius provide evidence. Smooth relative-distance evidence
   over a centered window.
3. Require sustained opponent evidence before changing possession. Once confirmed,
   place the transition at the beginning of the supporting evidence. This is an
   offline method that uses future context.
4. Detect candidate stoppages from an out-of-bounds or stationary ball. After a
   stoppage, require ball movement before accepting a team possession. These
   physical cues are highly dataset-dependent.
5. Carry the current state through unobserved intervals; reset all state at period
   or timestamp discontinuities. A block with no evidence stays X.

## Development protocol

The primary external source is Sportec's CC BY 4.0 open dataset. The split was fixed
before model fitting: train J03WMX/J03WN1/J03WOH, validation J03WOY, test J03WPY.
The frozen configuration was chosen using only external training and validation.
No hackathon public ground truth was used for training, parameter choice, or
model comparison.

Dead-ball geometry thresholds were selected on the three training matches. A
144-configuration temporal search on validation was followed by a 120-configuration
refinement of smaller control distances. The selection criterion was the highest
spell F1 among configurations with validation macro F1 at least 0.90, or the highest
macro F1 if none qualified. This is a fairly broad search against one validation
match; validation performance is not an independent generalization estimate.

Commands, after the datasets have been prepared:

```bash
python experiments/state_machine.py tune \
  --split data/external_audit/sportec/split.json \
  --data data/external_audit/sportec/processed \
  --out data/experiments/state_machine/sportec --dead-ball
python experiments/state_machine.py tune \
  --split data/external_audit/sportec/split.json \
  --data data/external_audit/sportec/processed \
  --out data/experiments/state_machine/sportec --dead-ball --refine
```

The selected configuration is stored locally in
`data/experiments/state_machine/sportec/frozen_config.json`:

```json
{
  "interpolate_s": 0.0,
  "smooth_s": 0.7,
  "control_radius_m": 1.0,
  "margin": 0.0,
  "confirmation_s": 0.7,
  "dead_ball": true,
  "stationary_s": 0.2,
  "stationary_radius_m": 0.01,
  "restart_speed_m_s": 1.5
}
```

Its recorded file SHA-256 is
`4c330b0d19efd9b36d3dbd923c35d4f474eb95c95bc2e2db94190b4c40c7cd4f`.

## Validation results

Scores use the unchanged competition scorer through
`experiments.metrics.evaluate_tracking`, which separates periods and time gaps
and uses the dataset's actual sampling rate. These numbers cover one full external
validation match, not the hackathon game or the reserved test match.

| Method | Accuracy | H/A/D macro F1 | Spell F1 | Team accuracy on actual live frames |
| --- | ---: | ---: | ---: | ---: |
| Nearest visible player | 0.4032 | 0.3502 | 0.0998 | 0.7533 |
| Nearest player + same stationary/outside dead-ball cue | 0.8647 | 0.8322 | 0.1219 | 0.7490 |
| Selected temporal state machine | 0.9225 | 0.9048 | 0.5801 | 0.8559 |
| Same frozen model, synthetic missing-ball blocks | 0.8597 | 0.8411 | 0.4825 | 0.7783 |

The synthetic stress mask is independent of labels, uses seed 2026, targets a 41%
missing rate with mean gap length one second, and realizes 40.32% missingness on
this validation match. It is a robustness probe, not a faithful model of real
camera occlusion. The stress result did not choose the parameters.

The selected model's clean validation class F1 values are H 0.8569, A 0.8651,
D 0.9924. It predicts 410 possession spells against 456 reference spells.

## Critical limitation: dead-ball coordinates are frozen upstream

The Sportec raw XML repeats the last live ball XY position throughout dead-ball
periods. This was independently verified against raw files; the converter does
not insert that behavior. Source speed and height, which also have artifacts,
are excluded from all inputs.

The very high D score exploits the frozen-coordinate convention through a
stationarity feature. Consequently validation macro F1 above 0.90 is evidence on
this particular external data representation, **not evidence of 0.90 performance
on noisy Metrica tracking**. The live-team result, geometric ablation, and masked
result show the remaining difficulty. This model does not achieve 0.90 spell F1
on validation.

An auxiliary SkillCorner experiment used provider-generated H/A labels, leaving
unknown labels as X. That dataset has no D targets and cannot establish three-class
performance. Its initial validation search reached spell F1 0.4954 and two-class
macro F1 0.8467; its reserved test match was not used for development.

## Checks and artifacts

Synthetic checks verify retrospective turnover placement, missing-ball carry,
reset at period boundaries, and invariance to changing the target-label array.
The input and output code is syntax-checked. Search records, per-match predictions,
and diagnostics stay under the gitignored `data/experiments/state_machine/` tree.
Primary held-out test evaluation is coordinated by the parent experiment runner
after all methods' configurations are frozen.
