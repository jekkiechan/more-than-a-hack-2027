# Sportec extra held-out evaluation (`sportec_extra_eval_v1`)

## Question
Do the adopted (`adop`), candidate (`cand`, original_full), raw-feed (`raw`) and v2-trained (`v2train`) arms keep their ordering on two further Sportec matches, J03WPY and J03WQQ, that were not used in training, and does 5 Hz phase test-time augmentation (TTA) help?

## Method
- Every arm's `leave_out_sportec` fold models (two seeds, 2026/2027) exclude all Sportec data, so no retraining was needed. Fixed f062 decoder; official metrics via `audit.score`.
- J03WPY: 146,211 frames at 25 Hz (97.5 min), 146,211 graded. J03WQQ: 142,345 frames (94.9 min), 142,345 graded (no X frames in either match).
- Views per match: clean (1); v1 state-blind feed-like via `feed_robustness_v1.raw_view` (2 realizations); `feed_like_v2` challenge-calibrated corruption (4 realizations). Seeds: `int(sha256(f"sportec_extra_eval_v1/{view}/{rep}/{match}")[:8], 16)`. Development labels only place v2 defects and score cells.
- Modes: `phase0` (archived sampling phase) and `phase_tta` (mean of the five 5 Hz phases). Numbers are per-match means over realizations.
- J03WOY (already used) comes from existing results: clean/v1 `cand`/`adop` from `feed_benchmark_audit_v1.json` (tta), v2 from `feed_like_v2_benchmark` scores J03WOY__{1..4}. Missing J03WOY clean/v1 `raw` and `v2train` (both modes) were computed with the same code on the cached `seed_gap_context_v1` inputs; the same recomputation reproduced the existing `cand`/`adop` J03WOY numbers exactly (max abs diff 0). J03WOY v1 views are the older cached `feed_like_corrupt_1/2` realizations, not the fresh `raw_view` draws used for the new matches.

## Sanity checks
- `ms.view` + `build_record` on J03WOY.npz reproduces the cached record `x` array exactly (shape [28508, 204], max abs diff 0.0), indices and labels also equal.
- `cand` phase0 on cached J03WOY: clean spell F1 0.6904 (expected 0.6904), feed mean 0.5174 (expected 0.5174); differences 0.0e+00 / 0.0e+00.

## Spell F1: arms x views
Cells: mean over the two new matches (J03WPY, J03WQQ) / 3-Sportec mean incl. J03WOY.

| arm | mode | clean (1) | v1 feed-like (2) | feed_like_v2 (4) |
|---|---|---|---|---|
| adop | phase0 | 0.649 / 0.656 | 0.506 / 0.502 | 0.576 / 0.574 |
| adop | phase_tta | 0.654 / 0.657 | 0.499 / 0.498 | 0.581 / 0.577 |
| cand | phase0 | 0.689 / 0.689 | 0.528 / 0.524 | 0.590 / 0.589 |
| cand | phase_tta | 0.692 / 0.688 | 0.532 / 0.526 | 0.600 / 0.595 |
| raw | phase0 | 0.657 / 0.657 | 0.519 / 0.513 | 0.579 / 0.576 |
| raw | phase_tta | 0.664 / 0.660 | 0.517 / 0.511 | 0.589 / 0.581 |
| v2train | phase0 | 0.653 / 0.655 | 0.509 / 0.503 | 0.581 / 0.578 |
| v2train | phase_tta | 0.664 / 0.660 | 0.503 / 0.501 | 0.595 / 0.585 |

## Macro F1: arms x views
Cells: mean over the two new matches (J03WPY, J03WQQ) / 3-Sportec mean incl. J03WOY.

| arm | mode | clean (1) | v1 feed-like (2) | feed_like_v2 (4) |
|---|---|---|---|---|
| adop | phase0 | 0.895 / 0.892 | 0.838 / 0.833 | 0.871 / 0.865 |
| adop | phase_tta | 0.897 / 0.893 | 0.840 / 0.835 | 0.873 / 0.867 |
| cand | phase0 | 0.899 / 0.895 | 0.840 / 0.836 | 0.868 / 0.863 |
| cand | phase_tta | 0.901 / 0.897 | 0.842 / 0.837 | 0.870 / 0.865 |
| raw | phase0 | 0.897 / 0.892 | 0.844 / 0.839 | 0.868 / 0.862 |
| raw | phase_tta | 0.899 / 0.894 | 0.844 / 0.839 | 0.870 / 0.863 |
| v2train | phase0 | 0.896 / 0.892 | 0.837 / 0.833 | 0.870 / 0.865 |
| v2train | phase_tta | 0.899 / 0.895 | 0.838 / 0.834 | 0.871 / 0.866 |

## Per-match spell F1 (phase0 / phase_tta)

| arm | view | J03WPY | J03WQQ | J03WOY (existing) |
|---|---|---|---|---|
| adop | clean (1) | 0.616 / 0.625 | 0.683 / 0.684 | 0.671 / 0.662 |
| adop | v1 feed-like (2) | 0.485 / 0.478 | 0.526 / 0.519 | 0.494 / 0.498 |
| adop | feed_like_v2 (4) | 0.540 / 0.553 | 0.612 / 0.609 | 0.571 / 0.568 |
| cand | clean (1) | 0.667 / 0.669 | 0.710 / 0.716 | 0.690 / 0.679 |
| cand | v1 feed-like (2) | 0.498 / 0.507 | 0.557 / 0.557 | 0.517 / 0.514 |
| cand | feed_like_v2 (4) | 0.559 / 0.576 | 0.622 / 0.624 | 0.586 / 0.586 |
| raw | clean (1) | 0.623 / 0.633 | 0.691 / 0.694 | 0.657 / 0.652 |
| raw | v1 feed-like (2) | 0.500 / 0.487 | 0.539 / 0.548 | 0.500 / 0.498 |
| raw | feed_like_v2 (4) | 0.548 / 0.562 | 0.609 / 0.617 | 0.571 / 0.564 |
| v2train | clean (1) | 0.618 / 0.636 | 0.687 / 0.692 | 0.659 / 0.650 |
| v2train | v1 feed-like (2) | 0.486 / 0.475 | 0.532 / 0.532 | 0.491 / 0.497 |
| v2train | feed_like_v2 (4) | 0.548 / 0.568 | 0.615 / 0.622 | 0.572 / 0.565 |

## Paired cand - adop (spell F1 / macro F1)

| mode | view | J03WPY | J03WQQ | J03WOY | mean 2 new | mean 3 |
|---|---|---|---|---|---|---|
| phase0 | clean (1) | +0.052 / +0.004 | +0.027 / +0.004 | +0.019 / +0.000 | +0.039 / +0.004 | +0.033 / +0.003 |
| phase0 | v1 feed-like (2) | +0.013 / -0.000 | +0.031 / +0.005 | +0.023 / +0.003 | +0.022 / +0.002 | +0.022 / +0.003 |
| phase0 | feed_like_v2 (4) | +0.020 / -0.003 | +0.009 / -0.003 | +0.015 / -0.002 | +0.015 / -0.003 | +0.015 / -0.003 |
| phase_tta | clean (1) | +0.044 / +0.004 | +0.032 / +0.005 | +0.017 / +0.002 | +0.038 / +0.004 | +0.031 / +0.004 |
| phase_tta | v1 feed-like (2) | +0.029 / -0.001 | +0.038 / +0.005 | +0.017 / +0.003 | +0.033 / +0.002 | +0.028 / +0.003 |
| phase_tta | feed_like_v2 (4) | +0.023 / -0.002 | +0.015 / -0.003 | +0.017 / -0.001 | +0.019 / -0.002 | +0.019 / -0.002 |

## Phase-TTA delta (phase_tta - phase0), spell F1

| arm | view | J03WPY | J03WQQ | J03WOY | mean 2 new | mean 3 |
|---|---|---|---|---|---|---|
| adop | clean (1) | +0.009 | +0.001 | -0.009 | +0.005 | +0.000 |
| adop | v1 feed-like (2) | -0.007 | -0.007 | +0.004 | -0.007 | -0.004 |
| adop | feed_like_v2 (4) | +0.014 | -0.004 | -0.003 | +0.005 | +0.002 |
| cand | clean (1) | +0.001 | +0.006 | -0.011 | +0.004 | -0.001 |
| cand | v1 feed-like (2) | +0.009 | -0.000 | -0.003 | +0.004 | +0.002 |
| cand | feed_like_v2 (4) | +0.017 | +0.002 | -0.000 | +0.010 | +0.006 |
| raw | clean (1) | +0.010 | +0.003 | -0.005 | +0.006 | +0.003 |
| raw | v1 feed-like (2) | -0.013 | +0.009 | -0.003 | -0.002 | -0.002 |
| raw | feed_like_v2 (4) | +0.014 | +0.008 | -0.007 | +0.011 | +0.005 |
| v2train | clean (1) | +0.018 | +0.005 | -0.008 | +0.011 | +0.005 |
| v2train | v1 feed-like (2) | -0.012 | -0.000 | +0.006 | -0.006 | -0.002 |
| v2train | feed_like_v2 (4) | +0.020 | +0.007 | -0.007 | +0.014 | +0.007 |

## Reading

- `cand` has higher spell F1 than `adop` in all 18 match x view x mode comparisons (3-match mean +0.015 to +0.033), and the ordering holds on each new match separately. Macro F1 differences are within 0.006 in either direction, so the gain is in spell structure, not frame-level accuracy.
- `cand` is also highest of the four arms in every view and mode on the 3-match mean. `v2train` is below `cand` in all 18 comparisons (3-match mean -0.010 to -0.034) and sits close to `adop`/`raw` (within about 0.01), with a small edge on feed_like_v2 only.
- Phase TTA effects are small (mostly within +/-0.01 spell F1) and not consistently signed: slightly negative on J03WOY clean for all arms, mostly positive on the new matches' clean and v2 views, mixed on v1. There is no evidence of a reliable benefit or harm.

## Caveats
- These matches were spent confirmation matches: they are exposed (scored here, and Sportec fold models excluded them from training) but not trained on. This is exposed held-out evidence, not fresh sealed evidence.
- n is small: two new matches, three with J03WOY; realizations within a match are correlated draws of one match, so no confidence intervals are given and differences of ~0.01 are not distinguishable from noise.
- Sportec has complete ball tracking with no native missing data, so all feed-like behaviour is simulated (v1 or v2); v1 for the new matches is a different generator draw from the cached J03WOY v1 views.
- Provider-specific shortcuts: all arms were selected on earlier evidence involving J03WOY; no decoder, seed or arm selection was done here. No organizer labels or challenge inputs were read.
- Aggregate numbers only: `experiments/results/sportec_extra_eval_v1.json`; per-cell JSON in ignored `data/experiments/sportec_extra_eval_v1/`. Code: `experiments/sportec_extra_eval_v1.py`.
