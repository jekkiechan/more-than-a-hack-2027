# One fixed posterior ensemble

The final equal-weight ensemble reached **0.93972 frame macro-F1 / 0.70922 mean spell F1** on the existing validation match `J03WOY`. Its spell F1 is 0.01011 above the best individually selected member. This is a modest validation gain, not evidence of a general accuracy ceiling or a reliable improvement on unseen matches.

Exactly one candidate was evaluated. It averages the frozen spell-weighted **63-leaf / 260-iteration, beta=1** model with the frozen uniform-weight **127-leaf / 400-iteration** model. Both use the same physical features. Each member's H/A/D posterior is cast to float32, the two are averaged equally in float32, and the already fixed **team-switch cost 3 / dead-alive cost 8** decoder is applied. There was no new fitting, weight search, or decoder search.

| Model | Decoder | Clean macro-F1 | Clean spell F1 | Masked macro-F1 | Masked spell F1 |
|---|---|---:|---:|---:|---:|
| Spell-weighted 63/260 | Uniform cost 3 | 0.93637 | 0.69911 | 0.91393 | 0.62935 |
| Uniform-weight 127/400 | Fixed costs 3/8 | 0.93891 | 0.68881 | 0.91443 | 0.61007 |
| **Equal-weight ensemble** | **Fixed costs 3/8** | **0.93972** | **0.70922** | **0.91649** | **0.63036** |

The comparison uses each member's previously selected decoder. Because the weighted member used uniform cost 3, the ensemble's entire increment cannot be attributed to posterior averaging alone. No further candidate was added to isolate that difference. The ensemble predicts 412 possessions versus 456 in the clean validation reference.

The independent masked-observation validation condition is supplementary and did not affect any ensemble setting. Sportec's original source freezes ball coordinates during dead play, so high dead-ball performance can depend on that provider-specific shortcut. Both the small ensemble gain and earlier model choices were assessed on the same repeatedly used validation match; an untouched holdout is needed before claiming generalization. This experiment did not access any held-out match or competition ground truth. Its direction follows earlier diagnostic work, so the original test match is not a fresh evaluation for the refinements.

`data/experiments/refine_ensemble/` contains `model.joblib` with both unchanged model dictionaries, component hashes, frozen configuration, validation scores, and predictions. The model and configuration were written before ensemble validation scoring. `config.json` and `frozen_config.json` contain the same frozen manifest.

```python
import joblib
from experiments import refine_ensemble

model = joblib.load("data/experiments/refine_ensemble/model.joblib")
predictions = refine_ensemble.predict(tracking_data, model)
```

`predict` extracts physical features once for both members. For an existing shared feature cache, use `predict_from_features(matrix, names, data, model)`. The cache must use the frozen `refine_features.features` schema; only tracking metadata are needed from `data` for decoding.

Verification regenerated every clean and masked validation prediction from the shared feature cache and both frozen classifiers; both matched archived-posterior decoding exactly. Component/model/source hashes passed. Float32 averaging also passed order-invariance, normalization, and identical-member checks. Details are saved in `verification.json` beside the frozen model.

Reproduce the single candidate in a fresh directory:

```bash
python -m experiments.refine_ensemble --out "$PWD/data/experiments/refine_ensemble_reproduction"
```
