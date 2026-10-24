# Testing the frozen-ball shortcut

`refine_domain.py` tests a separate dead/alive classifier that cannot inspect ball
coordinates, observation flags, source possession/status fields, or target labels
at inference. Its 67 features summarize collective player motion and pitch shape:
speed quantiles and spread, stationary/running fractions, velocity coherence,
team-centroid separation, touchline occupancy, and centred temporal means.
The features are symmetric under team-name swaps and player-slot permutations.
All derivatives and windows reset at period and timestamp breaks.

The player-only head uses histogram gradient boosting. Training uses the original
three Sportec matches, sampled every sixth frame, plus a deterministic copy with
25% of player slots hidden within each period. Counts are normalized by observed
players; bench-slot padding cannot directly reveal the provider. The H/A
conditional distribution comes from the original frozen gap-augmented model.

Two model capacities were declared before validation: 31 leaves/160 iterations
and 63 leaves/260 iterations. The D probability is an arithmetic blend of the
player-only head and the original model's D probability, with original-model
weights 0, 0.25, 0.5, 0.75, and 1. Each is decoded with switch costs 1, 3, or 8.
Clean J03WOY validation selects the highest spell F1 among macro F1 at least 0.90,
otherwise the highest macro F1. The fixed 41% missing-ball stress is supplementary.
The pure player-only D variant is reported independently even if it does not win.

No test or competition target data enters fitting, parameter selection, or feature
design. J03WPY, J03WR9 and J03WQQ are not opened by this experiment.

## Validation result: a plateau for this refinement

| Frozen method | Clean macro F1 | Clean spell F1 | Masked macro F1 | Masked spell F1 |
| --- | ---: | ---: | ---: | ---: |
| Original gap-augmented model, cost 3 | 0.9188 | 0.6108 | 0.8983 | 0.5603 |
| Selected 75% original / 25% player D blend | 0.9189 | 0.6120 | 0.8980 | 0.5636 |
| Player-only D head with original conditional H/A | 0.8597 | 0.5163 | 0.8459 | 0.4827 |

The selected blend uses 63 leaves, 260 iterations and decoder cost 3. Its clean
spell gain is **0.00124**, with **0.00016** macro F1 gain; the masked spell gain is
**0.00329**, with a **0.00025** macro F1 decrease. Those gains do not justify a
claim of meaningful progress or a reliable advantage on one validation match.
The declared two capacities and 30 blend/decoder configurations are complete;
there was no additional search after examining these results.

The pure player-only D diagnostic selected 31 leaves, 160 iterations and cost 3.
Its D F1 is **0.9265** clean and **0.9257** masked, versus **0.9993 / 0.9893** for
the selected blend. The D-head probabilities themselves do not change when ball
observations are hidden; the final decoder can still change D decisions because
its H/A evidence comes from the ball-dependent base model.

This establishes an empirical plateau for this particular D-head refinement,
not a ceiling for possession estimation. The large loss when removing ball
information shows that the clean benchmark's almost-perfect D classification is
not supported by equally strong independent player-motion evidence.

The frozen selected and diagnostic models are ready for the parent's fresh
held-out comparison. Neither has yet read or scored that held-out match here.

## Source artifact check

On training match J03WMX, there were **zero** adjacent within-label frame pairs
where every visible player had exactly unchanged coordinates, for each H/A/D
class. Individual player pairs were exactly stationary 9.75% of the time during
D, versus 1.17% during H and 0.86% during A. This differs from the ball's 100%
within-D coordinate freeze. The player-only head therefore removes the direct
ball-coordinate shortcut, although all results still share the same provider
and player tracking may contain other provider-specific structure.

## Reproduction and interface

```sh
python experiments/refine_domain.py
python -m unittest experiments.tests.test_domain_features -v
```

Artifacts live under ignored `data/experiments/refine_domain/`. The experiment
refuses to overwrite a frozen run. `model.joblib` stores the validation-selected
blend; `player_only_model.joblib` stores the separately selected zero-ball-weight
diagnostic. `validation.json` contains all predeclared candidates, and
`frozen_config.json` records source/model hashes and configurations.

```python
import joblib
from experiments.refine_domain import predict
model = joblib.load("data/experiments/refine_domain/model.joblib")
predicted_labels = predict(tracking_inputs, model)
```

The D head alone is independent of the ball. The complete H/A/D model still uses
ball geometry for team attribution through its frozen base model. A player-only
D classifier is a diagnostic of transferable evidence, not proof of reliable
dead-ball detection on the competition feed.

Three synthetic tests verify ball/target isolation, team/slot symmetry and period
window isolation. Vectorized feature quantiles were checked against NumPy's
linear NaN-quantiles to numerical precision. Code and model SHA-256 hashes are
recorded before held-out access.
