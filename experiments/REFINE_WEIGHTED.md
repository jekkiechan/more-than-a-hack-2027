# Training weights for short possession spells

This bounded experiment tests whether uniform frame loss underemphasizes short possession spells. A brief possession contributes few training rows but still counts as a complete spell in the official spell metric. The model keeps the existing physical features and decoder; only the supervised training weights change. This is an approximation to the evaluation objective, not direct optimization of spell F1.

## Frozen experimental design

`refine_weighted.py` writes `data/experiments/refine_weighted/predeclared.json` before feature extraction or validation scoring. The round contains exactly two weighting strengths, beta 0.5 and 1.0, each trained with the same 63-leaf, 260-iteration HistGradientBoostingClassifier. Each model is evaluated with switch costs 1, 3, and 8: six validation candidates total. There is no capacity sweep in this experiment.

For a home/away training run lasting `duration_s`, each frame receives:

```text
raw_weight = clip((5 / duration_s) ** beta, 0.25, 4)
```

Dead-ball frames receive raw weight 1. Runs split at label changes, half boundaries, and discontinuous timestamps. Durations are calculated from the full-resolution training labels before subsampling. After augmentation, all weights are divided by their global mean so the training sample-weight mean is 1. Consequently the final dead-ball weight is one divided by that normalizer.

The training data are J03WMX, J03WN1, and J03WOH only. J03WOY is the sole validation and selection match. No path to J03WQQ, J03WPY, or J03WR9 is constructed or opened. The separate fresh-test protocol remains in `data/experiments/refinement_protocol.json`; this script never evaluates it.

Each training match contributes clean and synthetic missing-ball views, each with a team-swapped copy. The missing-ball process is the existing label-independent 41% block mask. Feature rows are sampled every six frames from the 25 Hz data, giving approximately 4.17 Hz and 283,164 rows after all four views. The same run weight follows a sampled frame into every view. The script verifies the complete augmented label order against the cached feature order before fitting.

All 137 features come unchanged from `experiments.refine_features.features`. HGB parameters also match that backbone: learning rate 0.08, minimum leaf size 35, L2 regularization 1, no early stopping, random seed 17, and one CPU thread. The classifier is symmetrized over team-swapped predictions. Decoding uses the existing period/time-gap-safe decoder.

Selection maximizes clean validation spell F1 among candidates with macro F1 at least 0.90; if none qualify, it selects the highest clean macro F1. Masked validation is supplementary and never enters selection. Each beta is fitted once; there is no further weight search.

## Leakage boundary and artifacts

Training labels are used to form supervised targets and weights, which is permitted. They never become features, prediction-time weights, episode boundaries, or additional inference inputs. The selected model is directly compatible with `experiments.refine_features.predict(data, model)`, whose input allowlist discards labels and ownership/status fields.

Artifacts are under `data/experiments/refine_weighted/`:

- `model.joblib` and `frozen_config.json`: selected model, decoder cost, training configuration, file/source hashes, and no-test-evaluation declaration.
- `validation.json`: all six candidates and the selected result using the unchanged official metric wrapper.
- `J03WOY_{clean,masked}_probabilities.npz`: selected validation probabilities and timing metadata, without target labels.
- `weight_summaries.json`: per-match run counts and weight distributions.
- `training_features.joblib` and `J03WOY_{clean,masked}_features.npy`: reusable local caches, excluded from git with the rest of `data/`.
- `cache_manifest.json`: data, source, cache-file, and training-array hashes plus row ordering and mask seeds. This provenance record was added **after extraction**, while fitting was active; its timestamp makes that explicit. The parent evaluation gate separately regenerates validation predictions from raw inputs before comparing final candidates.

```bash
python experiments/refine_weighted.py --check
python experiments/refine_weighted.py
```

The completed experiment refuses to overwrite a frozen output. Checks verify run durations and clipping, unchanged raw D/X weights, and separate runs across half or timestamp boundaries. Feature/label alignment checks run before each fit.

## Validation results and frozen selection

The selected model uses **beta 1.0 and switch cost 3**. All six candidates met the clean macro-F1 threshold, so spell F1 determined the choice.

| Beta | Switch cost | Clean macro F1 | Clean spell F1 | Masked macro F1 | Masked spell F1 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1 | 0.93746 | 0.65592 | 0.91427 | 0.56653 |
| 0.5 | 3 | 0.93722 | 0.69083 | 0.91531 | 0.62574 |
| 0.5 | 8 | 0.93197 | 0.65300 | 0.91383 | 0.60261 |
| 1.0 | 1 | 0.93424 | 0.63724 | 0.91184 | 0.55242 |
| **1.0** | **3** | **0.93637** | **0.69911** | **0.91393** | **0.62935** |
| 1.0 | 8 | 0.93403 | 0.66928 | 0.91272 | 0.61080 |

The selected unweighted physical backbone, with the same 63/260 capacity and switch cost 3, achieved clean macro F1 0.93646 and spell F1 0.67179. Duration weighting improves validation spell F1 by **0.02732 absolute** with almost unchanged macro F1. This supports the training-loss mismatch hypothesis on this validation match. It does not reach 0.90 spell F1.

Selected clean accuracy is 0.94882 and live-team accuracy is 0.90466. It predicts 444 possession spells against 456 reference spells; under masked validation these counts are 423 and 456. The chosen beta's global weight divisor is 0.86658, giving mean normalized H/A weights 0.86415 each and D weight 1.15396. These class averages reflect the mixture of upweighted short runs and downweighted long runs; no independent class weighting was added.

Model SHA-256: `738d371f5fbc1a2059c7494a7a9eee9784de10d505e00593494d82c3b0efb5db`.

Frozen configuration SHA-256: `12d90482e010800153ca88d0d0342843b2cce854374976612addd661dcf3122b`.

Reload checks verified the model/source hashes, 260 fitted iterations, 137 features, normalized probabilities, and agreement between reloaded-model probabilities and the saved clean/masked validation probabilities on 1,000 frames each. Maximum float32 serialization differences were below 3e-8. These checks use cached validation features; the separate parent gate regenerates validation features from raw inputs before the fresh test.

## Interpretation

The clean validation result can indicate whether this particular training-loss adjustment helps the spell objective. It does not establish fresh held-out performance. Any improvement must still be tested once on J03WQQ by the parent evaluation gate after all candidates are frozen.

The open Sportec source freezes ball XY during dead-ball periods. The physical features can exploit that upstream artifact. Synthetic ball masking probes one missingness pattern; it does not reproduce all broadcast occlusions, camera cuts, or player-identity errors. Neither a validation gain nor a later same-source test gain removes those limitations.
