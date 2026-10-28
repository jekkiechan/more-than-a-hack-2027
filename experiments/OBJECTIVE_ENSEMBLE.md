# Combined-objective seed ensemble comparison

Status: comparison completed; pooled deployment preparation underway.
Date: 2026-10-28. Branch: `possession/combined-objective-ensemble`.

The fixed combined-loss probability ensemble passes the declared practical screen against the adopted original-loss ensemble. It improves clean and feed-like spell F1, with a small SkillCorner spell regression. The user requested comparison and consideration of a baseline bump; deployment fits and package verification are separate from this completed comparison.

## Matched results

| View | Current spell | Combined spell | Delta | Current macro | Combined macro | Auxiliary 70/30, current → combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.626531 | 0.633205 | +0.006674 | 0.854559 | 0.854226 | 0.694940 → 0.699511 |
| feed_like | 0.487419 | 0.498830 | +0.011410 | 0.811127 | 0.811387 | 0.584532 → 0.592597 |

| Provider | Clean spell delta | Feed-like spell delta | Clean macro delta | Clean D-F1 delta |
| --- | ---: | ---: | ---: | ---: |
| sportec | +0.002072 | +0.010434 | -0.001045 | -0.003501 |
| metrica | +0.025024 | +0.030917 | -0.000120 | -0.005533 |
| skillcorner | -0.007075 | -0.007120 | +0.000165 | +0.002373 |

The largest gain is on Metrica. SkillCorner loses .007075 clean and .007120 feed-like spell F1. Clean D recall declines .026597 on Metrica, while its D precision rises .018437; its D-F1 change is -.005533. Equal-provider D-F1 declines .002220 clean and .001915 feed-like. These tradeoffs remain visible despite passing the provider D-F1 and macro guards.

Short-spell recall at IoU .5 rises .035422 clean and .015908 feed-like. The equal-provider official F1 improves at every IoU threshold .3–.7; the clean IoU .7 gain is only .001118. Sportec clean IoU .7 falls .020701, and SkillCorner loses at all five clean thresholds. Full D precision/recall, thresholds, spell counts, fragmentation, boundary censoring and complete-boundary timing are retained in the [per-cell rows](results/objective_ensemble_rows.json).

## Frozen comparison

The two arms average native H/A/D probabilities 50/50 before the unchanged Viterbi decoder (cost 3). They use the same five development matches, native grading/masks and time boundaries: Sportec J03WOY, Metrica Game 3, and SkillCorner 1996435/1996436/2006229. All learned components of each fold exclude the evaluated provider. There are 15 matched cells (five clean and ten feed-like), producing 30 recipe rows. The current ensemble reproduces all 15 saved probability/prediction digests and scores. The candidate uses the already-trained combined-loss seeds 2026/2027, with no fitting, weight search, decoder search or feature rebuild for this comparison.

The new practical screen was frozen before scoring: positive equal-provider clean and feed-like spell deltas; each provider clean spell, macro and D-F1 delta at least -.02. All five checks pass. These criteria were chosen after inspecting the earlier objective study. They do not retroactively change that study’s failed +.01 both-seed screen. This result is development evidence on previously exposed matches, with synthetic feed-like stress; it is not independent confirmation. Separate seed results remain in [the original study PR #25](https://github.com/jekkiechan/more-than-a-hack-2027/pull/25).

Spell F1 ranks candidates; macro F1 breaks exact ties. The 70/30 summary is auxiliary only. Replicates are averaged within match, matches within provider, then providers equally. No organizer labels were read or scored, and all historical scoring gates remain closed.

## Evidence and reproduction

- [Frozen declaration](results/objective_ensemble_declaration.json): `686a10993b74234e0bb02134cd4ec79f781c87a140dcf88cfa31cdbc8f4a0c6e`.
- [Aggregate report](results/objective_ensemble.json): `cf267890905ece05aebe366c86c9ea635c07fa5267831f0658ac8932863f51d4`.
- [Per-cell rows](results/objective_ensemble_rows.json): 30 rows; all 15 baseline cases reproduce.
- Eight new synthetic contracts pass; the full suite passed 181 tests before deployment additions.

The [private archive receipt](results/objective_ensemble_artifacts.json) records 387 files / 639,385,391 bytes uploaded to the private cache at `ae8e6ed`. A fresh remote clone restored every file and passed the frozen input/source check. The [independent verifier](results/objective_ensemble_verification.json) reproduced all 15 candidate blends/decodes, all 15 baseline prediction digests and saved score rows, plus all 192 aggregate checks exactly. A source snapshot preserves the baseline manifest/policy used by this comparison even if the working package is later promoted. Reproduce using that snapshot and the versioned private cache rather than the subsequently active manifest.
