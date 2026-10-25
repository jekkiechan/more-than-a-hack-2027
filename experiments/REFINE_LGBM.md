# LightGBM versus the weighted physical-feature HGB

This bounded experiment asks whether swapping the gradient-boosting library helps.
It keeps the short-possession-weighted physical-feature model from
[REFINE_WEIGHTED.md](REFINE_WEIGHTED.md) unchanged apart from the classifier.
The answer is a **validation-only** comparison on one Sportec match. It does not
claim test performance, competition performance, or broadcast transfer.

## Setup

`experiments/refine_lgbm.py` (committed in `aa5f5e5` before any run) fixes six
configurations and three switch costs in advance: 18 validation candidates.

- **Data.** Training uses Sportec `J03WMX`, `J03WN1`, and `J03WOH`. Validation uses
  `J03WOY` only. Only these four matches were downloaded and converted
  (`audit_dataset.py --matches ...`). No path to `J03WPY`, `J03WR9`, or `J03WQQ`
  was constructed or opened. The converted frame counts match
  [BENCHMARK_AUDIT.md](BENCHMARK_AUDIT.md).
- **Inputs, weighting, and decoding.** Every configuration uses the same 137
  `refine_features` physical features. Training adds a clean view and a 41%
  block-masked view, each with a team swap, sampled at about 4.17 Hz (283,164
  rows). All models use the same beta-1.0 short-spell training weights, the same
  team-swap symmetrization, and the same period/time-gap-safe Viterbi decoder.
  Masked validation uses seed 2026.
- **Baseline.** `hgb_baseline` is refitted in the same run: HistGradientBoosting,
  63 leaves, 260 iterations, learning rate 0.08, and no early stopping.
- **LightGBM configurations.** There are five. `lgbm_default` uses library
  defaults. Four "Kaggle-style" settings add lower learning rates with more trees,
  row and column subsampling, L2 regularization, and larger leaves; their exact
  parameters are in `CONFIGS`. Every LightGBM model uses `n_jobs=1`, seed 17, and
  no early stopping.
- **Selection rule (`robust.SELECTION`).** Choose the highest clean spell F1 among
  candidates with clean macro F1 ≥ 0.90. Masked results are supplementary.
- **Code changes.** None. The script ran unmodified with the versions below.

**Environment.** The run used Python 3.13.17 on a CPU-only machine with 4 cores
and 15 GB RAM. Package versions were lightgbm 4.7.0, scikit-learn 1.8.0, numpy
2.3.5, scipy 1.17.0, pandas 2.2.3, joblib 1.5.3, and threadpoolctl 3.6.0. The
pinned `requirements.txt` versions installed unchanged on 3.13; lightgbm is
unpinned there.

**Reproduction check.** The refitted `hgb_baseline` reproduces the beta-1.0 rows
of REFINE_WEIGHTED.md to every printed digit, for example 0.93637 / 0.69911 clean
and 0.91393 / 0.62935 masked at cost 3.

## Results (validation J03WOY)

Aggregate rows are in [results/refine_lgbm_validation.json](results/refine_lgbm_validation.json).
Bold marks each configuration's best clean spell F1.

| Config | Switch cost | Fit s | Clean macro F1 | Clean spell F1 | Masked macro F1 | Masked spell F1 |
|---|---:|---:|---:|---:|---:|---:|
| hgb_baseline | 1 | 168 | 0.93424 | 0.63724 | 0.91184 | 0.55242 |
| hgb_baseline | 3 | 168 | 0.93637 | **0.69911** | 0.91393 | 0.62935 |
| hgb_baseline | 8 | 168 | 0.93403 | 0.66928 | 0.91272 | 0.61080 |
| lgbm_default | 1 | 59 | 0.92715 | 0.62380 | 0.90319 | 0.55015 |
| lgbm_default | 3 | 59 | 0.93161 | **0.68479** | 0.90876 | 0.60843 |
| lgbm_default | 8 | 59 | 0.92971 | 0.65335 | 0.90639 | 0.57984 |
| lgbm_kaggle_a | 1 | 510 | 0.93720 | 0.66667 | 0.91538 | 0.57640 |
| lgbm_kaggle_a | 3 | 510 | 0.93930 | **0.71438** | 0.92007 | 0.65212 |
| lgbm_kaggle_a | 8 | 510 | 0.93407 | 0.66568 | 0.91783 | 0.61622 |
| lgbm_kaggle_b | 1 | 746 | 0.93875 | 0.66024 | 0.91773 | 0.57733 |
| **lgbm_kaggle_b** | **3** | 746 | 0.94057 | **0.71614** | 0.91919 | 0.64225 |
| lgbm_kaggle_b | 8 | 746 | 0.93725 | 0.68986 | 0.92111 | 0.64405 |
| lgbm_kaggle_c | 1 | 380 | 0.93570 | 0.66201 | 0.91367 | 0.58087 |
| lgbm_kaggle_c | 3 | 380 | 0.93730 | **0.71195** | 0.91847 | 0.64799 |
| lgbm_kaggle_c | 8 | 380 | 0.93569 | 0.67631 | 0.91604 | 0.61686 |
| lgbm_kaggle_d | 1 | 313 | 0.93872 | 0.65274 | 0.91724 | 0.57364 |
| lgbm_kaggle_d | 3 | 313 | 0.94049 | **0.71285** | 0.91883 | 0.63026 |
| lgbm_kaggle_d | 8 | 313 | 0.93769 | 0.68786 | 0.91946 | 0.63367 |

The table below compares each configuration with `hgb_baseline` at switch cost 3,
which is best for every configuration. Positive values favour LightGBM.

| Config | Δ clean macro | Δ clean spell | Δ masked macro | Δ masked spell |
|---|---:|---:|---:|---:|
| lgbm_default | −0.00476 | −0.01432 | −0.00517 | −0.02092 |
| lgbm_kaggle_a | +0.00293 | +0.01527 | +0.00614 | +0.02277 |
| lgbm_kaggle_b | +0.00420 | +0.01703 | +0.00526 | +0.01290 |
| lgbm_kaggle_c | +0.00093 | +0.01284 | +0.00454 | +0.01864 |
| lgbm_kaggle_d | +0.00412 | +0.00374 | +0.00490 | +0.00091 |

**Selection.** The predeclared rule picks **`lgbm_kaggle_b`, switch cost 3**:
clean macro F1 0.94057 and spell F1 0.71614.

**Runtime.** Fitting and scoring took about 30 minutes of wall time:

- `hgb_baseline` ran alone first. Its 545 s include about 6 minutes of
  one-time feature extraction.
- The five LightGBM configurations then ran as four parallel processes, with
  `lgbm_default` and `lgbm_kaggle_c` in one process. They took 1,273 s, limited by
  `kaggle_b`.

`fit_seconds` measures fitting only. For LightGBM it was recorded while four
single-threaded fits shared four cores, at about 74% CPU each. HGB fitted without
that contention. Fit times are therefore only roughly comparable.

## Interpretation

**Does LightGBM beat the baseline?** Yes for the four tuned configurations, on
this one validation match. Each beats `hgb_baseline` on both clean and masked
spell F1 at their common best cost, 3. The selected `kaggle_b` gains **+0.0170
clean** and **+0.0129 masked** spell F1, with +0.004 to +0.005 macro F1.
`kaggle_a` has the largest masked gain, +0.0228. `kaggle_d`'s masked gain of
+0.0009 is effectively a tie. Out-of-the-box `lgbm_default` is worse than HGB in
every condition. The gain comes from the settings (more, slower trees, subsampling
and regularization), not from the library itself.

**Is the margin meaningful?** It is a consistent direction rather than a
demonstrated improvement.

- *For:* four distinct configurations agree in sign at cost 3, and at cost 1 all
  four beat HGB on both conditions. Macro F1 also moves up slightly. This does
  not look like a single lucky seed.
- *Against:* these 15 LightGBM candidates share one baseline, one validation
  match, and about 456 reference spells. With about 900 predicted plus reference
  spells, a gain of about 0.017 is roughly 8 additional matched spells at each
  IoU threshold. For context,
  [REFINEMENT.md](REFINEMENT.md) reports a 5th–95th percentile spell-F1 interval
  about 0.064 wide on the comparable `J03WQQ` match. The gap here is a quarter of
  that width. A paired comparison would be tighter, but none was computed.
- *Limits:* seed variance from LightGBM row and column subsampling was not
  measured. Picking the maximum of 15 candidates is optimistically biased. At
  cost 8 the ranking is mixed: `kaggle_a` has 0.66568 clean spell F1 versus HGB's
  0.66928.
- *Earlier validation history:* the validation match has already guided earlier
  rounds of selection. The weighting and ensemble validation gains of +0.027 and
  +0.010 shrank or reordered on `J03WQQ` (see REFINEMENT.md).

`lgbm_kaggle_b` at cost 3 also exceeds the frozen final ensemble's validation
scores of 0.70922 clean and 0.63036 masked spell F1. Those are validation numbers
only. They do not reopen the frozen protocol, and no held-out or competition data
were scored in this experiment.

**Next step if pursued.** Freeze `lgbm_kaggle_b` as one predeclared candidate.
Then score it once, under a new protocol, on an untouched labelled match. That
match cannot be `J03WQQ`, `J03WPY`, or `J03WR9` without the caveats in
REFINEMENT.md, and is preferably an independent broadcast-derived match. Sportec's
frozen dead-ball coordinates still make D unusually easy here, so none of these
numbers transfers directly to the organizer's tracking.
