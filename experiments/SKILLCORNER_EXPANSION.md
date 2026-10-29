# SkillCorner training-data expansion

Study date: 29 October 2026. This is a bounded comparison of the adopted combined-objective GRU trained on the original thirteen matches versus the same recipe with six additional SkillCorner matches. The goal is more useful competition-relevant training data; a new provider is not required.

## Data and comparison

Nine additional A-League matches were acquired at the project's existing SkillCorner source pin `4340d274572876239c154c90bc507a9b3250a656`. Tracking objects match their pinned LFS SHA-256 values; match metadata and dynamic events match their Git objects. The unchanged adapter produces native 10 Hz tracking, observation masks and provider/event-derived H/A/D/X silver targets.

| Role | Additional match IDs |
| --- | --- |
| Training | 2006363, 2007448, 2007721, 2010085, 2011166, 2013725 |
| Development selection | 2015213, 2016236 |
| Reserved test | 2017461 |

These chronological roles were fixed before model evaluation. The [candidate manifest](eval_set/candidates/skillcorner_expansion_v1.json) contains nineteen training and seven selection matches. It preserves the three historical Sportec test exclusions. The reserved match is outside both fitting and selection.

The new six training matches provide 4.66 labeled hours after the existing minimum-twelve-detected-player grading rule. All nine provide 14.18 labeled hours before that rule and 6.81 afterward. Across the nine, live-ball missingness is 6.47% before the rule and 0.89% afterward. Broadcast cuts and this filtering therefore remain material limitations; more volume is not evidence of realistic tactical live-ball-gap coverage. The native arrays and flags remain intact for future declared preparation recipes.

The primary evaluation uses full provider holdouts on the same seven matches for both arms: Sportec J03WOY, Metrica Game 3, and five SkillCorner matches (the original three plus the new two). The original five-match subset is retained for historical parity. Adding SkillCorner training data cannot change the SkillCorner-held-out model, so that fold reuses its control and is reported as parity. A separate pooled comparison evaluates both training cohorts on the identical seven excluded selection matches; it measures the practical same-provider effect without conflating it with provider exclusion.

## Frozen controls

Both arms use seeds 2026/2027, the same 62,643-parameter hidden24 GRU with player-only D path, the combined duration/transition loss, five-second context and ten-second cores, original augmentation, source/match loss-mass formulas and cost-3 decoder. Native H/A/D probabilities are averaged across seeds before decoding. Separate-seed scores are reported as diagnostics, not averaged as an ensemble.

Historical training feature caches and checkpoints are restored and hash-verified. The original pooled normalization and 30,560-window count reproduced bitwise before fitting. New feature preparation reuses the shared content-addressed clean cache and existing raw gap/slot augmentation, with source indices 8–13. Native X labels, observation flags and half/time breaks are fixed before corruption. No event/state/ownership target field enters inference features.

| Candidate fold | Training matches | Windows drawn per epoch | Optimizer updates |
| --- | ---: | ---: | ---: |
| Hold out Sportec | 16 | 23,748 | 2,976 |
| Hold out Metrica | 17 | 25,976 | 3,248 |
| Pooled | 19 | 30,560 | 3,824 |

Each candidate uses eight epochs and batch size 64. Uniform sampling without replacement caps each epoch at its matched baseline window count. The baseline-size path uses the original DataLoader unchanged, with bitwise synthetic fit parity. Candidate normalization is recomputed using its training records only. File-backed normalized arrays preserve values and bound memory use. This tests additional data at fixed optimizer cost; it does not test an expanded-budget retraining recipe.

All six candidates are sealed before any evaluation. The [declaration](results/skillcorner_expansion_declaration.json) binds inputs, preparation/evaluation caches, sources, baseline checkpoints, budgets and criteria. The private working directory is `data/experiments/skillcorner_expansion_v1/`.

## Evaluation and decision rule

Use the unchanged official spell F1 averaged across IoU .3/.4/.5/.6/.7; macro F1 breaks exact ties. Replicates are averaged within match, matches within provider, then providers equally. The 70/30 summary is auxiliary only. Clean and the two fixed feed-like stress replicates are separate views.

Before fitting, adoption requires primary clean spell gain at least .01, primary feed-like spell regression no worse than .01, and each provider's clean macro/D-F1 regression no worse than .02. The pooled supplement additionally requires positive clean spell gain, feed-like loss no worse than .01, and the same provider frame-quality guards. All metrics, seed breakdowns, short-spell recall and fragmentation are retained, including failed guards. No extra arms, thresholds, seeds or decoder variants follow these results.

The user separately requested competition spell F1. The fixed pooled candidate is generated and format-validated without reference access, reproduced through the normal prediction entrypoint, and sealed after the external decision. One distinct exclusive gate is claimed before fetching/opening the public reference. The existing baseline score is reused. That already-exposed five-minute window has seventeen possessions; it cannot select the dataset/model or establish full-match gains. Existing gates are never reset.

## Verification and artifacts

Validation includes the existing synthetic/contract suite, new data/sampler contracts, independent score/cache/gate integrity contracts, exact file-backed synthetic-fit parity, and bundled submission format validation. Real checkpoints, normalization, optimizer budgets and saved probabilities require independent replay before delivery. Array/model/frame artifacts stay in private storage; only code, manifests and aggregate evidence belong in this public repository.

Final outcome, measured scores and remote artifact receipts will be recorded after the frozen comparison completes.
