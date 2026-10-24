# Bounded decoder refinement

Two rounds found a small validation improvement, **spell F1 0.6108 → 0.6125**, while preserving frame macro-F1 above 0.90. They did **not** recover additional short possessions at IoU 0.5. This is evidence of diminishing returns from these decoder changes with the existing frame classifier, not a general accuracy ceiling.

The frozen robust classifier was unchanged. Only the original three training matches (`J03WMX`, `J03WN1`, `J03WOH`) and validation match `J03WOY` were loaded. No held-out match or competition ground truth was accessed by this experiment. The configuration was frozen before supplementary missingness evaluation and before any parent-run held-out scoring.

The direction of this refinement was motivated by the earlier post-hoc diagnosis of short-possession errors on the original test match. That earlier match is therefore not a fresh independent evaluation for these refinements; any new generalization claim needs the parent's separate untouched holdout.

The selection rule stayed fixed: maximize clean-validation spell F1 among candidates with frame macro-F1 ≥0.90; otherwise maximize frame macro-F1. The budget was 16 configurations including the baseline, with two rounds and no further adaptive searches.

| Decoder | Validation macro-F1 | Validation mean spell F1 |
|---|---:|---:|
| Frozen baseline: all transition costs 3 | 0.91877 | 0.61078 |
| No team-switch smoothing, dead cost 8 | 0.91557 | 0.31968 |
| Team cost 1, dead cost 8 | 0.91717 | 0.58787 |
| **Selected: team cost 3, dead cost 8** | **0.91890** | **0.61252** |
| Team cost 8, dead cost 8 | 0.91731 | 0.58191 |
| Best learned boundary adaptation | 0.91795 | 0.59623 |

Round 1 separated H↔A switch penalties from transitions into/out of D. The 12 combinations used team costs `{0,1,3,8}` and dead costs `{1,3,8}`. Costs use the existing 10-Hz convention and scale with native sampling rate. Removing or sharply reducing team smoothing introduced enough extra switches to reduce spell F1. Increasing dead-transition smoothing gave the small selected gain.

Round 2 fitted one gradient-boosted boundary classifier on the three training matches only. Its target was proximity within 0.16 seconds to a true H/A transition. Features combined symmetric geometric contact evidence with changes in the frozen classifier's team posterior over 0.12–1 second. The head was trained with balanced weights; its scores are boundary evidence, not calibrated probabilities. Four configurations lowered the H/A penalty near predicted boundaries, using base costs `{3,8}`, powers `{1,2}`, and the selected dead cost. All four worsened spell F1 (0.5323–0.5962). The selected configuration does **not** use this head.

| True validation possession duration | True spells | Baseline matched at IoU ≥0.5 | Selected matched |
|---|---:|---:|---:|
| <1 second | 41 | 12 | 12 |
| 1–<3 seconds | 156 | 47 | 47 |
| 3–<10 seconds | 170 | 109 | 109 |
| ≥10 seconds | 89 | 82 | 82 |

Predicted spell count changed from 360 to 359 versus 456 reference spells. At IoU 0.3, 0.4 and 0.5, matched counts were unchanged; at each of IoU 0.6 and 0.7, one additional spell matched. The +0.00173 mean spell F1 gain is small and was selected on one repeatedly used validation match; it is not evidence of a statistically reliable improvement.

After freezing, the fixed independent missingness process hid 40.32% of validation ball observations. Selected macro-F1 was **0.89822**, spell F1 **0.56259**, versus baseline **0.89825 / 0.56030**. This supplementary stress result did not influence selection. Sportec's original source freezes ball XY during dead play, so gains from D-transition decoding may depend on a provider-specific artifact.

The boundary head could not rescue the existing posterior through transition costs alone. Stronger trajectory representations or independently evaluated event/interaction models remain hypotheses for future work; this bounded experiment does not establish their potential performance.

Reproduce in a fresh output directory:

```bash
python -m experiments.refine_decode --out "$PWD/data/experiments/refine_decode_reproduction"
```

Frozen inference API:

```python
import json
import joblib
from experiments import refine_decode

model = joblib.load("data/experiments/robust/model.joblib")
config = json.load(open("data/experiments/refine_decode/frozen_config.json"))
predictions = refine_decode.predict(tracking_data, model, config)
```

Artifacts remain under ignored `data/experiments/refine_decode/`: frozen configuration and hashes, all validation trials, boundary head, supplementary stress scores, and duration diagnostics. Existing frozen files were not modified. Checks confirmed equivalence to the original decoder when costs match, exact team-swap symmetry, period-safe context shifts, and label-independent inference.

## One fixed transfer to the physical-feature model

After the physical-feature experiment independently froze its selected model, the already selected team/dead costs **3/8** were applied once to its validation posteriors. There was no further grid or parameter adjustment. The composition was frozen before its validation scoring.

| Validation condition | Physical features, uniform cost 3 | Physical features, fixed costs 3/8 |
|---|---:|---:|
| Clean macro-F1 | 0.93646 | 0.93652 |
| Clean spell F1 | 0.67179 | 0.67305 |
| Masked macro-F1 | 0.91306 | 0.91341 |
| Masked spell F1 | 0.60640 | 0.60814 |

The decoder increment remains small (+0.00125 clean spell F1); most of the improvement over the original robust model comes from the physical-feature model. These remain validation results. The parent must select the final method before any new untouched holdout is scored.

The wrapper is `experiments.refine_combined.predict(data, physical_model, frozen_config)`. It explicitly uses the same float32 posterior precision as the exported validation artifacts. Frozen metadata and aggregate results are under `data/experiments/refine_combined/`; the wrapper verifies the component model and source hashes, accepts only the fixed costs, and never loads a held-out match.
