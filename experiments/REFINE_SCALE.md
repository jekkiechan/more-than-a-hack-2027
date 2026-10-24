# Final capacity check on cached physical features

The 63-leaf / 260-iteration model improved clean validation spell F1 by 0.0152
over 31 leaves / 160 iterations. This single extra scaling check tests whether
further capacity still helps before claiming that model-size gains have flattened.
It does not change the frozen feature implementation or previous model files.

## Predeclared experiment

- One multiclass histogram gradient-boosting model: **127 leaves, 400 iterations**,
  learning rate 0.08, minimum leaf size 35, L2 regularization 1, no early stopping,
  random state 17, and one CPU thread.
- Uniform training-frame weights, identical to the physical-feature baseline.
- Reuse the weighted experiment's completed cache of 283,164 rows and 137 features:
  `data/experiments/refine_weighted/training_features.joblib`. It contains the
  existing clean/masked/team-swapped features; the weighting experiment does not
  alter the cached feature values. No training-feature re-extraction is allowed.
- Reuse the same owner's clean/masked J03WOY feature matrices, with the same
  validation mask seed 2026.
- Four configurations only: uniform decoder penalties **1, 3, 8**, and the
  already selected separate team/dead penalties **3/8**. No adaptive decoder or
  further grid.
- Select highest clean validation spell F1 among candidates with clean macro F1
  at least 0.90; otherwise select highest clean macro F1. Masked scores remain
  supplementary.

Only J03WMX, J03WN1, J03WOH training inputs and J03WOY validation can be loaded.
No held-out test or competition ground-truth path is constructed. The script
verifies cache dimensions, training match identities and hashes, and the frozen
feature implementation's source hashes before training. The cache owner must
signal that every cache file is completely written before this run starts.

## Run and API

```bash
# Register the fixed experiment before the shared cache is ready.
python experiments/refine_scale.py --declare

# Run only after the cache owner confirms completion.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/refine_scale.py
```

`--check` verifies equivalence between uniform and equal team/dead costs, and
target independence of decoding. There is no feature-extraction fallback and no
test option. Already frozen outputs cannot be overwritten.

The saved estimator uses the standard `refine_features` model dictionary and
137-feature API. Use `experiments.refine_scale.predict(data, model)` to apply the
selected decoder; it dispatches between uniform and separate costs. Probability
outputs and decoding explicitly use float32, matching the archived posteriors.
`experiments.refine_scale.predict_probabilities(data, model)` exposes H/A/D scores.
Inference allowlists tracking fields and excludes reference targets.

## Results

The completed run took **549.3 seconds** on one CPU thread, including cached data
loading, training, and validation scoring. The selected configuration is
127 leaves / 400 iterations with **team cost 3 and dead cost 8**.

| Decoder | Clean macro F1 | Clean spell F1 | Masked macro F1 | Masked spell F1 |
| --- | ---: | ---: | ---: | ---: |
| Uniform 1 | 0.937957 | 0.635461 | 0.912854 | 0.543631 |
| Uniform 3 | 0.938765 | 0.686359 | 0.914329 | 0.608676 |
| Uniform 8 | 0.936684 | 0.684058 | 0.916079 | 0.631447 |
| **Team 3 / dead 8** | **0.938911** | **0.688814** | **0.914428** | **0.610069** |

The clean objective selects team/dead 3/8; uniform 8 has stronger masked results
but weaker clean spell F1. Choosing from the masked score instead would change
the predeclared selection criterion.

Holding the decoder at uniform 3, scaling from 63 leaves / 260 iterations improves
clean spell F1 from 0.671795 to 0.686359: **+0.014564**. With the same fixed 3/8
decoder, the corresponding improvement is 0.673046 to 0.688814: **+0.015768**.
These gains remain meaningful on this validation match, so this experiment does
**not** demonstrate that the capacity curve has reached a ceiling.

The separate [training-weighting experiment](REFINE_WEIGHTED.md), using the
smaller 63/260 backbone, subsequently reached clean spell F1 **0.699111** with
macro F1 0.936371. It outperforms this larger uniform-weight model under the shared
selection objective, suggesting the short-possession training objective is the
stronger current choice. This comparison is still validation selection, not a
fresh generalization measurement.

Frozen artifacts under ignored `data/experiments/refine_scale/`:

- `model.joblib`, SHA-256
  `80600321efce1c5c21ab9e773e29d57161ce0d7442ff0456c167ae6ad2d01965`.
- `frozen_config.json`: selected decoder, training/cache/source/model hashes,
  runtime, and `test_evaluated: false`.
- `validation.json`: all four clean/masked candidate scores and the selected one.
- `J03WOY_clean_probabilities.npz` and `J03WOY_masked_probabilities.npz`: H/A/D
  float32 probabilities and timing metadata, without reference labels.
- `predeclared.json`: the specification saved before cache loading and training.

The shared cache's additional provenance record is
`data/experiments/refine_weighted/cache_manifest.json`. Existing frozen artifacts
and cached feature values were never modified by this experiment.

This remains model selection on a repeatedly used validation match. Any observed
gain is not independent evidence of generalization, and absence of a gain would
describe this bounded size comparison rather than an absolute performance ceiling.
