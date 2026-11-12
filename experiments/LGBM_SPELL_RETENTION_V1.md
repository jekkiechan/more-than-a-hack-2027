# LightGBM spell-retention study

12 November 2026. **Partial success; no eligible arm and no promotion.**
Conservative D-end updates improve spell and D-F1 in all three development views,
but retain only 21.95% of the unrestricted refiner’s v2 D-F1 gain and reduce
historical-v1 short-spell recall by .001576. This supports the mechanism without
establishing retention of the large gain or any public spell guarantee.
Keep reconstructed Fable adopted.

## Question and fixed protocol

Can balanced training exposure, missingness ablation, or conservative D-boundary
updates retain the refiner’s dead-ball gain without sacrificing development spell
F1? The organizer window is not read or scored in this study. The prior public
result motivates investigation but supplies no thresholds, ranking or selection.

The frozen budget is 18 fits: three feature/weight recipes, three held-out
providers and two seeds (2026/2027). Each LightGBM uses the original 300-tree
parameters and stride-five sampling with X targets excluded. All 35 cells cover
five exposed development matches, clean inputs, two historical-v1 realizations
and four v2 realizations. Realizations are averaged within match, matches within
provider, then providers equally. Native grading populations and half/time
boundaries are preserved. The adopted control must replay within 1e-9 before
candidate scoring.

The three recipes are full features with original 1:2:4 view-family mass;
identical features with equal 1:1:1 mass; and the original missingness ablation
with block-local filled ball speed and signed boundary distance. All use the
same 50/50 P(D) blend, f062 decoder, snap rule and frontier H/A mixture. Each
inference ensemble averages the two seeds’ probabilities before decoding.

Three additional arms use only the full-model proposals: move both D-run
endpoints, starts only, or ends only. Each applies reciprocal one-to-one
D-run overlap matches, retains baseline H/A assignments, and rejects edits that
change ordered H/A spell topology or D-run count. Newly live endpoints need an
adjacent baseline team anchor. Predicted X regions and half/time breaks are hard
barriers; reference X is used only to exclude ungraded targets. This preserves sequence structure, not interval IoU or spell F1.

Eligibility requires the original spell, short-recall, macro and clean-provider
guards, plus no aggregate D-F1 decline in any view and a strict v2 D-F1 gain.
Eligible ensembles rank by v2 official spell F1, with macro only for exact ties.
Constituent seed effects are reported; seeds are not selected after scoring.
No post-result tweaks, GAR scoring, public scoring or adoption are authorized
by this declaration.

## Evidence limits

These are five heavily exposed development matches and synthetic views, not
new confirmation. Only the refiner’s learned component excludes the held-out
provider. Cached upstream probabilities use each row’s own provider fold;
this is not nested full-pipeline cross-fitting. The label/model-conditioned v2
generator can reward missingness shortcuts. The block-safe adapter differs
from the historical R2 implementation. No pooled deployment candidate or
public spell-retention claim is established by this study.

## Matched results

All values are equal-provider development means. Clean/v1/v2 are distinct input
conditions; these are not public-window scores. Full and balanced recipes improve
aggregate spell and short recall in every view, but retain the v1 D/macro failure.
Removing missingness cues improves spell, macro and D-F1 everywhere while losing
short recall. No ensemble passes every predeclared guard.

| Arm | Spell clean | Spell v1 | Spell v2 | D-F1 clean | D-F1 v1 | D-F1 v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adopted control | 0.681508 | 0.534731 | 0.598690 | 0.875558 | 0.861220 | 0.869770 |
| Unrestricted full | 0.693127 | 0.538825 | 0.613612 | 0.897420 | 0.850613 | 0.896046 |
| Equal view mass | 0.693233 | 0.539932 | 0.612666 | 0.898414 | 0.852284 | 0.895557 |
| Missingness ablation | 0.684915 | 0.539115 | 0.603606 | 0.892803 | 0.864962 | 0.881698 |
| Preserve topology: both ends | 0.684328 | 0.535612 | 0.600275 | 0.879902 | 0.861812 | 0.874446 |
| Preserve topology: D starts | 0.681844 | 0.534985 | 0.598368 | 0.876090 | 0.859810 | 0.873827 |
| Preserve topology: D ends | 0.685998 | 0.535534 | 0.600820 | 0.883427 | 0.861674 | 0.875536 |

The unrestricted block-safe ensemble reaches v2 spell **.613612** (+.014923) and
D-F1 **.896046** (+.026276). Equal view mass reduces the v1 macro loss from
−.004346 to −.003598 and the v1 D-F1 loss from −.010607 to −.008936, but does not
remove either failure. The full arm’s v1 D precision falls .043781 while recall
rises .025378, consistent with excess false D assignments in that view.
The no-missingness arm retains 45.40% of the v2 D-F1 gain
and 32.94% of the spell gain; its clean/v2 short-recall losses are .007564/.007639.
These are ablations of this complete recipe, not isolated proof of causation.

## Conservative boundary result

D-end updates are the most promising conservative arm, not an eligible winner.
They improve v2 spell **.598690 → .600820**, D-F1 **.869770 → .875536**, and macro
**.854915 → .857818**. Clean and v1 spell/D-F1 also improve. They retain 21.95%
of the unrestricted v2 D-F1 gain and 14.28% of its spell gain. Historical-v1
short recall falls **.230361 → .228785** (−.001576). Its 2026 constituent also
fails v2 short recall; the ensemble does not make that seed weakness disappear.

| D-end arm metric | Clean | Historical v1 | V2 |
| --- | ---: | ---: | ---: |
| Spell F1 | 0.685998 (+0.004490) | 0.535534 (+0.000804) | 0.600820 (+0.002130) |
| Macro F1 | 0.880588 (+0.004384) | 0.828154 (+0.000844) | 0.857818 (+0.002903) |
| 70/30 auxiliary | 0.744375 (+0.004458) | 0.623320 (+0.000816) | 0.677919 (+0.002362) |
| D precision | 0.907549 (+0.008455) | 0.843139 (-0.005044) | 0.883613 (+0.001897) |
| D recall | 0.866305 (+0.006830) | 0.883572 (+0.005973) | 0.871566 (+0.009088) |
| Short recall (<3s, IoU .5) | 0.453184 (+0.000951) | 0.228785 (-0.001576) | 0.307893 (+0.000349) |
| Predicted spell count | 358.444444 (+0.000000) | 304.388889 (-0.888889) | 336.138889 (-0.833333) |
| Reference spell count | 404.333333 (+0.000000) | 404.333333 (+0.000000) | 404.333333 (+0.000000) |

Counts above are means under the declared weighting, not pooled integers. The
adapter preserves the full prediction’s H/A sequence and D-run count per input
block. Official graded spell counts can still change when shifted boundaries
interact with excluded reference-X margins. Preserving topology cannot guarantee
interval IoU or short-spell recall.

The two-ended arm is a useful alternative: v2 spell +.001586 and D-F1 +.004676,
with only historical-v1 short recall failing (−.001323). Its v1 short loss is
localized to Metrica (provider delta −.003968); SkillCorner and Sportec are flat.
D-start-only edits lose v2 spell and v1 D-F1, so the conservative signal is
stronger at D ends. No additional arm or duration threshold was fitted afterward.

## Provider and seed checks

| D-end arm: v2 provider | Spell delta | D-F1 delta |
| --- | ---: | ---: |
| Metrica | +0.002482 | +0.005203 |
| SkillCorner | +0.001664 | +0.011279 |
| Sportec | +0.002244 | +0.000817 |

The D-end arm’s v1 short-recall provider deltas are Metrica −.003968,
SkillCorner −.003297 and Sportec +.002538. Its v2 SkillCorner short recall also
falls .002760, despite the positive aggregate. Provider heterogeneity remains.

| Arm | Seed 2026 v2 spell | Seed 2027 v2 spell | Probability ensemble v2 spell | Ensemble failed guards |
| --- | ---: | ---: | ---: | --- |
| Unrestricted full | 0.614408 | 0.613890 | 0.613612 | feed_like_v1/macro_f1, feed_like_v1/d_f1_retention |
| Equal view mass | 0.612283 | 0.612839 | 0.612666 | feed_like_v1/macro_f1, feed_like_v1/d_f1_retention |
| Missingness ablation | 0.603472 | 0.603444 | 0.603606 | clean/short_spell_recall_iou_0_5_lt3s, feed_like_v2/short_spell_recall_iou_0_5_lt3s |
| Preserve topology: both ends | 0.600440 | 0.600249 | 0.600275 | feed_like_v1/short_spell_recall_iou_0_5_lt3s |
| Preserve topology: D starts | 0.598398 | 0.598323 | 0.598368 | feed_like_v1/short_spell_recall_iou_0_5_lt3s, feed_like_v2/spell_f1, feed_like_v1/d_f1_retention |
| Preserve topology: D ends | 0.600637 | 0.600728 | 0.600820 | feed_like_v1/short_spell_recall_iou_0_5_lt3s |

Both full-feature seeds fail the same v1 D/macro checks; both two-ended seeds
fail v1 short recall. Two seeds on the same exposed matches are a sensitivity
check, not statistical significance or independent confirmation. The full
probability ensemble scores below both constituent spell scores; ensembling is
not automatically an improvement.

Across all 35 views, D-end editing considers 3,030 reciprocal overlap proposals,
accepts 1,569, and falls back on 34 of 112 input blocks. It changes 44,527 native
frame predictions. These are repeated synthetic realizations, not independent
events. Detailed indices remain private; the public artifact contains counts.

## Decision and next evidence

The intuition that the refiner contains useful information is supported. A
conservative deployment of that information can improve development spell and
D-F1 together. The stronger claim—keep most of the large D gain without any
regression—is not supported by these six fixed strategies. Every arm still
fails at least one guard; no candidate is selected for confirmation or adopted.
The original pooled public diagnostic is unchanged and its gate remains closed.

Next, localize which development short spells are harmed by D-end movement and
measure missingness transfer with fixed label-independent masks. Any proposed
short-spell protection requires a separate bounded declaration; do not search
public thresholds or reselect seeds. A candidate that clears development checks
still needs exact-package transfer evaluation and ultimately independently
annotated continuous matches.

## Validation and delivery

The control matches all declared aggregate metrics within 1e-9 across 35 cells.
All 18 fits and 210 provider-fold probability files are sealed before evaluation.
Thirteen focused tests pass; the complete experiment suite runs 1,000 tests
(999 pass, one skip), all 26 root tests pass, and the exact file-backed training
contract passes. An independent replay verifies all 630 saved candidate rows, 35 baseline rows,
equal-provider aggregates, 322 source hashes, all 18 models, 210 probability files
and unchanged grading counts with no discrepancy.
Native ball-observation masks match finite ball coordinates
across all 15 raw input files (1,390,404 rows).

A pre-evaluation [output-policy correction](results/lgbm_spell_retention_v1_output_policy.json)
redirects the frozen runner’s detailed diagnostics to private storage and
publishes only counts. Sources, arms, fits, metric computation and selection
rules are unchanged. Use `scripts/run_lgbm_retention_evaluation.py` for the
one-shot evaluation path; completed stages must not be rerun or overwritten.

## Evidence

- [Declaration](results/lgbm_spell_retention_v1_declaration.json)
- [Runner](lgbm_spell_retention_v1.py)
- [Feature recipes](lgbm_retention_features.py)
- [Topology adapter](lgbm_retention_topology.py)
- [Earlier public diagnostic](LGBM_PUBLIC_V1.md)
- [All scalar results and guards](results/lgbm_spell_retention_v1.json)
- [Matched control](results/lgbm_spell_retention_v1_control.json)
- [Training seal](results/lgbm_spell_retention_v1_training.json)
- [Test receipt](results/lgbm_spell_retention_v1_checks.json)
- [Independent verifier](../scripts/verify_lgbm_retention.py)
- [Independent verification receipt](results/lgbm_spell_retention_v1_verification.json)
- [Private archive delivery and fresh-restore verification](results/lgbm_spell_retention_v1_private_delivery.json)

The private archive contains prepared cells, fitted models, fold probabilities,
saved predictions, detailed diagnostics and a source snapshot. The delivery
receipt records exact remote availability. The existing restricted
[research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/experiments.html#lgbm-retention)
presents the matched results; repository evidence is authoritative.
