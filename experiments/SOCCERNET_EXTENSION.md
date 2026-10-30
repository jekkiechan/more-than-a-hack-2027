# SoccerNet sparse H/A warm-start extension

Status: stage 1 completed as a negative primary result; larger H/A capacity and data-scaling continuation is running. Retain the adopted baseline for now. The selected GAR50 arm passes provider/feed-like guardrails but fails the predeclared improvement screen. Defer the public diagnostic until final external model selection; it has not run or produced predictions.

Date: 30 October 2026. Study: `soccernet_extension_v1`.

Baseline: `joint_gru_combined_ensemble_v1`, manifest v5 and decision policy v5. Manifest SHA-256 `c0eeeb8a11c59f2821ca306d7d7ab27052f98ccfd8ad720256d8fa3eae56baf0`; policy SHA-256 `e5cc29d00b6608b625f3ba539ea9632565334190264f8af498c2fbb35695fcb3`.

Declaration SHA-256: `15e80fe3e7e9499141475f743a2f913edf229cfcd46340ccbcf77e96a53d07ca`. Checkpoint seal SHA-256: `ae8f755ea74a53184dcad2a5f3310564e32c31f50f31cfc10bea90eb8d059cc5`. Source snapshot commit: `6e97f496a4ab2d01cccf76a5fdc795562c56bc58`. Frozen source/config/model seals were preserved through evaluation; only separately sealed runtime drivers handled three numerical/receipt corrections described below.

## Question and decision

The study asked whether sparse SoccerNet-GAR event-team supervision improves the adopted combined-objective model after a matched H/A-only warm-start. It does not. The selected GAR50 ensemble scores 0.631957 clean and 0.495121 feed-like spell F1, against the frozen baseline’s 0.633205 and 0.498830. GAR50 is +0.002703 clean versus the matched old-data-only fine-tune, but it is -0.001248 versus baseline clean and -0.003709 feed-like. The declared clean improvement requirement was baseline +0.01 and strictly above old-only; it fails. Keep `joint_gru_combined_ensemble_v1`; no promotion or automatic adoption.

GAR50 was the highest clean-spell GAR arm and passed all seven external guardrails: clean D-F1 and macro floors for each provider, plus the feed-like spell floor. GAR25 passed those same guards, but ranked below GAR50 on the frozen clean ensemble criterion. The selection does not turn a failed improvement screen into a gain.

## Protocol and training

The primary development protocol is the same five previously evaluated matches, with a full-provider exclusion for every fit: Sportec J03WOY, Metrica `metrica_3`, and SkillCorner `skillcorner_had_1996435`, `skillcorner_had_1996436`, `skillcorner_had_2006229`. Each match has one clean view and two synthetic `feed_like` replicates (`corrupt_1`, `corrupt_2`), yielding 15 matched cells. Aggregates average across matches/replicates within provider, then weight Sportec, Metrica and SkillCorner equally. `feed_like` is synthetic stress, not an independent broadcast test.

The baseline arm reuses the original fold checkpoint. Each fine-tune begins from that fold’s combined-objective checkpoint and leaves its feature schema, median and scale unchanged. The clone trains conditional H/A only for 256 AdamW updates (learning rate 1e-4, weight decay 1e-4, batch 64, gradient clip 1). The original checkpoint supplies P(D) bitwise; the fine-tuned clone supplies conditional H/A. Decoding the changed H/A probabilities can still change D precision and recall. The three training arms are a matched 0% GAR / old-data-only control, GAR25, and GAR50. Old windows are source/match balanced; GAR windows are game balanced; group losses implement the declared mixture. These are warm-start H/A adaptations, not full combined-loss retrains.

There were 18 full-provider fits: 3 provider-exclusion folds × 2 seeds (2026/2027) × 3 arms. Two pooled GAR50 fits for deployment were also completed after selection; their existence does not imply adoption. The saved seed-specific diagnostics and the actual probability ensembles are distinguished below.

The frozen data preparation used official SoccerNet-GAR TRAIN only: 45 games and 62,159 metadata clips. Before opening TAR members, the fixed SHA-256 ranking selected up to 64 clips/game from PASS, CROSS and SHOT actions: 2,880 selected, 2,877 prepared, and three rejected for conflicting center-team labels. Each record remains a separate 144-native-frame clip, sampled to 24 rows; only its center carries a home/away event-team silver label. All other targets are unknown. It adds no dense D or continuous possession labels. Features use visible-only raw coordinates and input tracking fields; event IDs/types, player/team identities and sidecar metadata are excluded from model features. Clip/period breaks prevent temporal concatenation.

Independent preparation checks recomputed 16 raw clips exactly, matched all selected IDs to TRAIN metadata, and verified the prepared 204-feature schema and unseen-fold checkpoint contracts. GAR valid and test sets were not opened for training, model selection, or this evaluation; organizer labels were not read. No full combined-loss-from-scratch study was run.

## Matched results

Each row below is the equal-provider **actual probability ensemble** (the mean of seed 2026 and 2027 native probabilities, decoded once), not the arithmetic mean of the two seed scores. Spell F1 is the primary official metric: greedy one-to-one H/A matching averaged over IoU .3/.4/.5/.6/.7. Macro F1 is the tie-break; 70/30 is auxiliary only.

| Recipe | View | Spell F1 | Macro F1 | 70/30 | D-F1 | D precision | D recall | Short-spell recall | Predicted spells | IoU .3 | .4 | .5 | .6 | .7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | clean | 0.633205 | 0.854226 | 0.699511 | 0.849125 | 0.905032 | 0.802897 | 0.384161 | 374.666667 | 0.723165 | 0.695803 | 0.646311 | 0.587839 | 0.512906 |
| baseline | feed_like | 0.498830 | 0.811387 | 0.592597 | 0.844609 | 0.895965 | 0.802025 | 0.203371 | 317.666667 | 0.610564 | 0.570382 | 0.503104 | 0.438757 | 0.371340 |
| old_only | clean | 0.629253 | 0.853474 | 0.696519 | 0.848425 | 0.907517 | 0.800026 | 0.373357 | 365.777778 | 0.718852 | 0.691094 | 0.647420 | 0.582318 | 0.506582 |
| old_only | feed_like | 0.495753 | 0.810965 | 0.590317 | 0.844738 | 0.899006 | 0.800013 | 0.199905 | 311.333333 | 0.606218 | 0.563195 | 0.498749 | 0.438760 | 0.371846 |
| gar25 | clean | 0.630438 | 0.853589 | 0.697383 | 0.848219 | 0.908267 | 0.799273 | 0.375273 | 365.222222 | 0.722065 | 0.696682 | 0.646962 | 0.582059 | 0.504422 |
| gar25 | feed_like | 0.494076 | 0.811339 | 0.589255 | 0.845134 | 0.900101 | 0.799983 | 0.202657 | 312.055556 | 0.605590 | 0.560870 | 0.498726 | 0.436545 | 0.368647 |
| gar50 | clean | 0.631957 | 0.853560 | 0.698438 | 0.848319 | 0.909011 | 0.798990 | 0.369960 | 366.222222 | 0.724978 | 0.697677 | 0.646410 | 0.585283 | 0.505436 |
| gar50 | feed_like | 0.495121 | 0.811327 | 0.589983 | 0.844769 | 0.900023 | 0.799377 | 0.201306 | 308.444444 | 0.603033 | 0.560403 | 0.500436 | 0.439382 | 0.372350 |

The provider summary below reports spell F1, macro F1, D precision/recall/F1 and short-spell recall. Values are equal means within each provider over that provider’s selection matches and replicates.

| Recipe | Provider | Clean spell | Clean macro | Clean D-F1 (P/R) | Feed-like spell | Feed-like macro | Feed-like D-F1 (P/R) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | sportec | 0.642981 | 0.872372 | 0.911389 (0.929460 / 0.894008) | 0.473066 | 0.818100 | 0.905513 (0.918465 / 0.892921) |
| baseline | metrica | 0.797222 | 0.916912 | 0.889095 (0.930856 / 0.850920) | 0.602041 | 0.867830 | 0.889216 (0.927756 / 0.853751) |
| baseline | skillcorner | 0.459411 | 0.773392 | 0.746889 (0.854779 / 0.663764) | 0.421381 | 0.748233 | 0.739098 (0.841675 / 0.659403) |
| old_only | sportec | 0.640898 | 0.871203 | 0.909889 (0.930415 / 0.890249) | 0.462777 | 0.815633 | 0.905014 (0.919707 / 0.890785) |
| old_only | metrica | 0.788186 | 0.916865 | 0.889908 (0.932774 / 0.850809) | 0.599674 | 0.867101 | 0.888480 (0.928863 / 0.851464) |
| old_only | skillcorner | 0.458677 | 0.772353 | 0.745480 (0.859363 / 0.659020) | 0.424809 | 0.750162 | 0.740719 (0.848449 / 0.657790) |
| gar25 | sportec | 0.641604 | 0.869972 | 0.909795 (0.930416 / 0.890068) | 0.458586 | 0.815808 | 0.905331 (0.919927 / 0.891192) |
| gar25 | metrica | 0.787060 | 0.917204 | 0.890120 (0.933186 / 0.850854) | 0.601068 | 0.867607 | 0.889124 (0.929806 / 0.851853) |
| gar25 | skillcorner | 0.462649 | 0.773593 | 0.744743 (0.861198 / 0.656897) | 0.422573 | 0.750603 | 0.740947 (0.850569 / 0.656902) |
| gar50 | sportec | 0.641500 | 0.869684 | 0.909797 (0.930568 / 0.889932) | 0.455325 | 0.815200 | 0.904836 (0.919919 / 0.890242) |
| gar50 | metrica | 0.791585 | 0.917630 | 0.890272 (0.933440 / 0.850920) | 0.607509 | 0.868058 | 0.888996 (0.929779 / 0.851642) |
| gar50 | skillcorner | 0.462786 | 0.773365 | 0.744888 (0.863025 / 0.656117) | 0.422529 | 0.750723 | 0.740474 (0.850371 / 0.656246) |


| Recipe | Provider | View | IoU .3 | .4 | .5 | .6 | .7 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | sportec | clean | 0.750311 | 0.720497 | 0.660870 | 0.591304 | 0.491925 |
| baseline | sportec | feed_like | 0.607879 | 0.559012 | 0.480385 | 0.403123 | 0.314933 |
| baseline | metrica | clean | 0.866667 | 0.847222 | 0.816667 | 0.766667 | 0.688889 |
| baseline | metrica | feed_like | 0.708140 | 0.676215 | 0.607833 | 0.540943 | 0.477073 |
| baseline | skillcorner | clean | 0.552519 | 0.519691 | 0.461397 | 0.405547 | 0.357903 |
| baseline | skillcorner | feed_like | 0.515673 | 0.475919 | 0.421095 | 0.372206 | 0.322015 |
| old_only | sportec | clean | 0.745636 | 0.713217 | 0.665835 | 0.593516 | 0.486284 |
| old_only | sportec | feed_like | 0.594161 | 0.543446 | 0.466877 | 0.394404 | 0.315000 |
| old_only | metrica | clean | 0.860759 | 0.846695 | 0.815752 | 0.745429 | 0.672293 |
| old_only | metrica | feed_like | 0.704854 | 0.669675 | 0.603989 | 0.545864 | 0.473986 |
| old_only | skillcorner | clean | 0.550162 | 0.513369 | 0.460672 | 0.408009 | 0.361170 |
| old_only | skillcorner | feed_like | 0.519638 | 0.476463 | 0.425381 | 0.376012 | 0.326553 |
| gar25 | sportec | clean | 0.751880 | 0.721805 | 0.661654 | 0.588972 | 0.483709 |
| gar25 | sportec | feed_like | 0.596172 | 0.535728 | 0.462955 | 0.386162 | 0.311914 |
| gar25 | metrica | clean | 0.860759 | 0.843882 | 0.815752 | 0.745429 | 0.669480 |
| gar25 | metrica | feed_like | 0.704799 | 0.672768 | 0.608699 | 0.550726 | 0.468346 |
| gar25 | skillcorner | clean | 0.553556 | 0.524360 | 0.463481 | 0.411775 | 0.360076 |
| gar25 | skillcorner | feed_like | 0.515800 | 0.474113 | 0.424524 | 0.372748 | 0.325681 |
| gar50 | sportec | clean | 0.760000 | 0.722500 | 0.660000 | 0.585000 | 0.480000 |
| gar50 | sportec | feed_like | 0.586660 | 0.530078 | 0.461086 | 0.385349 | 0.313453 |
| gar50 | metrica | clean | 0.863955 | 0.847125 | 0.813464 | 0.760168 | 0.673212 |
| gar50 | metrica | feed_like | 0.706987 | 0.676376 | 0.616679 | 0.560071 | 0.477432 |
| gar50 | skillcorner | clean | 0.550978 | 0.523405 | 0.465767 | 0.410681 | 0.363096 |
| gar50 | skillcorner | feed_like | 0.515452 | 0.474757 | 0.423542 | 0.372727 | 0.326166 |
The five IoU scores are shown per provider below. GAR50 clean spell is 0.641500 / 0.791585 / 0.462786 and feed-like is 0.455325 / 0.607509 / 0.422529 for Sportec / Metrica / SkillCorner. Against baseline, GAR50 is -0.001481 / -0.005637 / +0.003375 clean and -0.017741 / +0.005468 / +0.001148 feed-like. The equal-provider mean can therefore conceal a larger Sportec stress decline.

Seed sensitivity is reported as independent members. GAR50 seed means do not imply its actual ensemble score; actual ensemble inference first averages probabilities, then decodes.

| Recipe | Seed 2026 clean/feed-like | Seed 2027 clean/feed-like | Arithmetic mean of seed scores clean/feed-like | Actual probability ensemble clean/feed-like |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.626425 / 0.487243 | 0.626554 / 0.491574 | 0.626489 / 0.489409 | 0.633205 / 0.498830 |
| old_only | 0.626460 / 0.486422 | 0.622128 / 0.490722 | 0.624294 / 0.488572 | 0.629253 / 0.495753 |
| gar25 | 0.625991 / 0.486119 | 0.624383 / 0.491096 | 0.625187 / 0.488608 | 0.630438 / 0.494076 |
| gar50 | 0.621529 / 0.484079 | 0.625780 / 0.490075 | 0.623655 / 0.487077 | 0.631957 / 0.495121 |

The public window has only 17 possessions. The five-match development cohort and the public point labels were previously exposed; these results are development evidence, not independent confirmation. This stage was not scored publicly; the later [capacity continuation](SOCCERNET_CAPACITY.md) owns the completed one-time public diagnostic.

## Corrections and verification record

Three runtime-only corrections were needed; none changed weights, features, targets, input probabilities, decoder, metrics, selection thresholds or completed scored cells:

1. A zero-GAR control’s NumPy identifiers raised `TypeError: unhashable type numpy.ndarray` while writing receipts, after its weights had been saved. Twelve GAR-arm receipts were recovered from unchanged saved checkpoints; six old-only receipts had completed normally. Only scalar source/match/clip identifiers were converted for the pooled runtime. The 18 fold checkpoints and declared update budgets remained unchanged.
2. The initial baseline replay guard used `atol=1e-7`, stricter than the historical `atol=1e-6` contract. Before any candidate scoring, an independent 30-cell replay found maximum absolute probability drift `8.344650268554688e-7`; all decoded baseline predictions were exactly equal to saved predictions. A sealed runtime amendment changed only that comparison tolerance to `1e-6`.
3. A float32 categorical sum assertion at `2e-7` rejected two-ULP rounding. Its runtime amendment changed only the assertion to `1e-6`; returned probability values were unchanged, and comparison outcomes were not inspected before correction. Previously completed cells were unchanged.

The declaration, frozen source hashes and checkpoint seal stayed intact. Runtime drivers and executed function bytes have separate receipts. The standard suite (308 experiment tests, 18 root tests) and exact file-backed fit contract passed. The independent data audit and all 180 saved metric-row replays passed; see the [verification receipt](results/soccernet_extension_verification.json).

## Public diagnostic and artifacts

The selected GAR50 arm’s two pooled fits are complete. Its public diagnostic was cancelled in favor of the requested [capacity/data-scaling continuation](SOCCERNET_CAPACITY.md); this stage never produced public predictions or opened its score gate. The continuation selected a new candidate externally, sealed its pooled weights, and completed one public diagnostic without subsequent tuning. The 17-possession public window remains a pipeline check, not independent efficacy evidence.

The aggregate and weights, prepared data, fit receipts, replay validation, and execution receipts are SHA-verified in the local `data/experiments/soccernet_extension_v1/` tree. No new private upload was made; GAR source artifacts are retained locally under the dataset terms and are not included in this repository. The aggregate report is `data/experiments/soccernet_extension_v1/report.json` (180 rows); frozen declaration, decision, preparation manifest, restore manifest, baseline replay and verification receipts are in the same local directory.

Promotion remains separate and was not performed. The capacity/data-scaling continuation is complete and also retains the combined-objective baseline under its frozen screen. A new efficacy claim requires genuinely independent broadcast matches with dense, blind possession labels; neither these sparse SoccerNet event-team anchors nor the previously exposed five-match/public checks provide that evidence.
