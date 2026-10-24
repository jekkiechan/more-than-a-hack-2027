# Training with synthetic ball-observation gaps

This bounded experiment tests whether missing-data augmentation improves the
existing multiclass gradient-boosting model's robustness without sacrificing its
clean validation performance. It uses the fixed Sportec split and never opens a
test NPZ. Synthetic gaps are a supplementary stress measurement; they are not a
simulation of real broadcast occlusion and do not determine model selection.

## Reproduce

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/robust.py \
  --baseline data/experiments/supervised/model.joblib
```

The output directory defaults to `data/experiments/robust/`. The script refuses
to overwrite a frozen result; supply a fresh `--out` for a separate run. The
optional baseline is loaded before training and its SHA-256 is recorded, so the
comparison remains identifiable if another process replaces the source file.

## Fixed protocol

- Training matches: J03WMX, J03WN1, J03WOH; validation: J03WOY.
- Features, team-swap augmentation, probability symmetrization, and temporal
  decoding reuse `experiments/supervised.py`. Inference receives only an explicit
  allowlist of tracking/period/pitch/observation arrays, excluding reference labels
  and source ownership/state fields.
- Each training match contributes a clean view and one view with ball coordinates
  removed using `stress.mask_ball`. Expected masking rate is 41%, mean gap length
  one second, and seeds are 2026, 2027, 2028 in training-split order. Neither labels
  nor ball status controls masking. The original data is unchanged.
- Every view also receives home/away swap augmentation, yielding four times the
  sampled training rows. Sampling matches the existing baseline: requested 4 Hz,
  integer stride `round(source_fps / 4)` (25 Hz data therefore becomes 4.167 Hz).
- One multiclass `HistGradientBoostingClassifier`: 31 leaves, 160 iterations,
  learning rate 0.08, minimum 35 samples per leaf, L2 regularization 1,
  `early_stopping=False`, random state 17, one CPU thread.
- Decode penalties: 0, 3, 8, 16, 32 in the shared decoder's 10 Hz units. Select the
  highest clean validation spell F1 among candidates with clean macro F1 at least
  0.90; if none qualify, select the highest clean macro F1.
- All five candidates are also measured on validation masked with seed 2026.
  Those measurements are not used for selection.

Metrics use the unchanged competition evaluator through the shared wrapper.
Frame macro F1 and possession-spell F1 are reported separately. The model,
configuration, input/source hashes, and full compact validation metrics are saved
before any optional baseline comparison. There is no test-evaluation option.

## Results

The run completed in 161.7 seconds on one CPU thread with 283,164 augmented
training rows and 65 features. Both the original clean-training model and the
selected augmented model use switch cost 3. The generated validation mask removed
40.321% of ball observations; the clean validation data has none missing.

| Training | Validation input | Frame macro F1 | Possession-spell F1 |
| --- | --- | ---: | ---: |
| Clean only | Clean | 0.919051 | 0.602187 |
| Clean + masked | Clean | 0.918769 | 0.610784 |
| Clean only | Masked | 0.744789 | 0.315578 |
| Clean + masked | Masked | 0.898253 | 0.560298 |

Augmentation improves masked macro F1 by 0.153463 and masked spell F1 by
0.244720. Clean macro F1 decreases by 0.000282 while clean spell F1 increases by
0.008597. These are validation differences on one match, not confidence intervals
or held-out test claims. Under the synthetic gaps the augmented model predicts
350 possession spells versus 937 for the baseline; the reference contains 456.

All predeclared decoding candidates:

| Switch cost | Clean macro F1 | Clean spell F1 | Masked macro F1 | Masked spell F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.915311 | 0.315247 | 0.893807 | 0.285581 |
| **3** | **0.918769** | **0.610784** | **0.898253** | **0.560298** |
| 8 | 0.917314 | 0.581915 | 0.899229 | 0.548248 |
| 16 | 0.911266 | 0.533903 | 0.894324 | 0.500575 |
| 32 | 0.901959 | 0.471364 | 0.886418 | 0.441463 |

Frozen artifacts, all under ignored `data/experiments/robust/`:

- `model.joblib`: selected estimator and decoder configuration, SHA-256
  `2df96983affc51c52fe50b9bd9a77b238f730933f8051ebda4918be4f664612c`.
- `frozen_config.json`: training/validation provenance, source hashes, selection
  protocol, model hash, and `test_evaluated: false`.
- `validation.json`: all five clean/masked candidates and selected metrics.
- `baseline_validation.json`: original frozen-model comparison on identical data
  and mask, baseline SHA-256
  `a8501635ca862868de65ae85ff5c4a3e56d3deffc7de9943619e19b1b4a1c0ac`.

The test match was not opened by this experiment. Subsequent independent testing
of the frozen model is controlled by the parent benchmark runner.

## Verification and limits

The script compiled successfully. A synthetic check confirmed masking is
identical when possession labels and owner fields are replaced, these fields
cannot enter the inference dictionary, original coordinates are unchanged, and
masked observations have their observation flag cleared.

This is one fixed split and one random mask per match. Missingness is generated
independently of labels; real ball-tracking failures can concentrate around
possession changes and difficult camera conditions. Results cannot establish the
same robustness on the competition match, and validation remains a model-selection
set rather than independent final evidence.
