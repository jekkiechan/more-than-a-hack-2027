# Deeper LightGBM with smaller learning rate

Status: frozen, awaiting training and complete external results.
Date: 2026-10-27.

This extension tests the requested increase in LightGBM leaves and depth,
combined with a smaller learning rate. It was declared while the original
[temporal feature study](TEMPORAL_LGBM.md) was still fitting and before its scores
were read. The original declaration, four configurations and output directory
remain unchanged.

## Fixed comparison

| Configuration | Leaves | Maximum depth | Learning rate | Trees |
| --- | ---: | ---: | ---: | ---: |
| Existing context_regularized comparator | 31 | 7 | .02 | 640 |
| context_63_slow | 63 | 8 | .01 | 1280 |
| context_127_slow | 127 | 10 | .01 | 1280 |

The product of learning rate and tree count stays 12.8. This is a rough matched
boosting budget, not a guarantee of equal model capacity or predictions. More
leaves and depth allow additional interactions among the 625 temporal features;
smaller updates and twice as many trees may fit those interactions more
smoothly. This tests capacity and shrinkage together, so an observed difference
cannot be attributed to the leaf count alone.

All other parameters match `context_regularized`: minimum 120 rows per leaf,
minimum Hessian .01, L1 .2, L2 10, feature fraction .7, row fraction .8 with
bagging every iteration, and 127 bins. Every fit uses seed 2026 and one CPU
thread. No early stopping or adaptive arm is added.

Features, 2.5-Hz training rows, raw clean/player-slot-dropout views, equal
provider/match/view weights, native unknown targets and whole-provider
exclusions are identical. The extension reuses checksum-verified training
feature caches; no features or normalization are fitted on the excluded
provider. Tree-seed variability remains unmeasured.

The same two fixed GRU control seeds provide conditional H/A probabilities.
Each LightGBM D head is evaluated as a complete D replacement and a 50/50 D
probability blend, followed by the same cost-3 decoder. The declared budget is
six new fits and 120 new paired evaluation rows. All six checkpoints are sealed
before extension scoring. The original regularized comparator and GRU control
rows are reused without changing their scores.

Official spell F1 remains primary, macro F1 is the exact-tie breaker, and the
70/30 summary is supplementary. The original per-seed clean/stress guardrails
apply. Results will show both GRU seeds, their mean, every provider, matched
changes against the 31-leaf comparator, exact spell diagnostics and measured
training time. Increasing trees and leaves can substantially increase runtime;
any accuracy gain must be assessed alongside that cost.

## Reproduction and artifacts

The frozen declaration is
`data/experiments/capacity_lgbm_v1/predeclared.json`. Input and checkpoint
requirements are the same as the original study. The new runner never edits
the original `CONFIGS` or declaration.

```sh
python -m experiments.capacity_lgbm_experiment freeze
python -m experiments.capacity_lgbm_experiment fit --source sportec --config context_63_slow
# Complete both configurations for all three providers, then seal all six.
python -m experiments.capacity_lgbm_experiment seal
python -m experiments.capacity_lgbm_experiment evaluate
python -m experiments.capacity_lgbm_experiment report
```

The configuration contract was checked before fitting: only learning rate,
tree count, leaf count and maximum depth differ from the original regularized
arm. The original frozen-source check still passes. Models and frame outputs
remain private; only aggregate evidence will enter the public report. No
organizer score or model adoption is part of this extension.
