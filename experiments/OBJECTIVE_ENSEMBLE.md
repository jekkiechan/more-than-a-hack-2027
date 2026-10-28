# Combined-objective seed ensemble comparison

Status: comparison and pooled deployment completed; working baseline adopted before the separate public diagnostic.
Date: 2026-10-28. Branch: `possession/combined-objective-ensemble`.

The fixed combined-loss probability ensemble passes the declared practical screen against the adopted original-loss ensemble. It improves clean and feed-like spell F1, with a small SkillCorner spell regression. The user requested comparison and consideration of a baseline bump. The combined ensemble was adopted after two pooled deployment fits and label-free package verification, before the separately requested public check.

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

## Objective change

The combined training objective retains the factorized H/A/D GRU and its original D/live loss. It reweights conditional H/A cross-entropy using complete native H/A spell durations: raw `sqrt(5 seconds / duration)` weights clipped to [.5, 2], then normalized within eligible live rows of each training record. Unknown, D and boundary-censored targets stay neutral. A fixed .1 auxiliary term penalizes adjacent dead/live and conditional H/A change errors, balancing stable and changed pairs. Full-native interval masks prevent bridging unknown targets or time breaks. There is no extra prediction head and no inference-time label dependence. These surrogates improve the matched official metric here; they do not optimize interval matching exactly.

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


## Pooled deployment and adoption

Both new members completed the frozen eight-epoch budget on the same 13 TRAIN matches: 52 records, 62,643 parameters, 30,560 windows per epoch, 3,824 updates. Normalization and schema exactly match the restored original-loss controls. The [deployment declaration](results/combined_objective_deployment_declaration.json), [fit summary](results/combined_objective_deployment.json), and [promotion receipt](results/baseline_combined_objective_promotion.json) preserve that scope. The preceding ensemble and new production entrypoint each reproduce their sealed CSV byte-for-byte. The [entrypoint receipt](results/combined_objective_entrypoint_verification.json) records the 199,319-row replay.

## Separate requested public diagnostic

The package and its adoption were sealed before this check. Following the user’s request for a public score, a distinct exclusive gate scored exactly the two sealed CSVs once with the unchanged official evaluator and 1,000 bootstrap draws. Existing gates were preserved. The published window has 7,500 graded frames and 17 reference possessions; the other full-match labels remain unknown.

| Public five-minute diagnostic | Prior original-loss ensemble | Combined-loss ensemble | Delta |
| --- | ---: | ---: | ---: |
| Spell F1 | 0.557576 | 0.557576 | +0.000000 |
| Macro F1 | 0.862375 | 0.854088 | -0.008287 |
| D precision | 0.896291 | 0.893175 | -0.003116 |
| D recall | 0.805990 | 0.819605 | +0.013615 |
| D F1 | 0.848746 | 0.854810 | +0.006065 |
| Auxiliary 70/30 | 0.649015 | 0.646529 | -0.002486 |

Both predict 16 possessions against 17 references. The primary spell score ties exactly; macro F1 falls 0.008287, so the preceding ensemble leads on the official exact-tie secondary metric for this diagnostic. D recall/F1 improve. At IoU .5, matched spells fall from 11 to 10; at .6 they rise from 7 to 9; at .7 they fall from 6 to 5. These threshold changes average to the same headline spell F1.

Official separate 90% spell intervals are [0.3879, 0.7158] for the preceding ensemble and [0.3879, 0.7273] for combined. They are not paired difference intervals and do not establish full-match performance. The check does not confirm a public-window improvement. No model, threshold, seed, decoder, policy or adoption change follows it.

[Public declaration](results/combined_objective_public_check_declaration.json) · [Public results](results/combined_objective_public_check.json). Both the exclusive marker and completed report keep this gate closed.


## Delivery verification

All **198 synthetic/contract tests pass**. The [fit archive](results/combined_objective_deployment_artifacts.json) records 204 private files; the [baseline archive](results/baseline_combined_objective_artifacts.json) records 179 package/source/output/gate files and the merged default cache. A fresh remote clone restored all three model-package files and 169 source files, then the restored source’s normal prediction entrypoint reproduced the full candidate CSV exactly. See the [cold replay receipt](results/combined_objective_remote_verification.json). No public reference was loaded during that restore/replay. The active manifest and policy still match the pre-public seals.

Updated Pages: [progress and charts](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9), [metric/optimization explainer](https://chatgpt.com/space/page_4198931ce5288191bc4f646773074e10), and [historical D-head study](https://chatgpt.com/space/page_c9f856e9f2b88191b9af86da8c58a059). The next evidence priority is independent broadcast confirmation; no further public tuning or repeat scoring is proposed.
