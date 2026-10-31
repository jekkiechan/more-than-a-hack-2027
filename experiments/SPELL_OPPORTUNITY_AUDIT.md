# Combined-objective spell opportunity audit

The most broadly supported spell-F1 opportunity is recovering brief H/A spells. In the saved combined-objective ensemble, under-three-second spell recall is **.384 on clean inputs** and **.203 on feed-like stress** at IoU .5. These spells are about 40% of reference spells, yet account for 61% of unmatched references in the clean cells and 56% in the feed-like cells. At pooled cell counts, longer-spell recall is about .72 clean and .59 under stress. This pattern is consistent across the five fixed development matches, though it does not establish a causal mechanism.

## Evidence and method

This is a post-hoc audit of the actual combined-objective two-seed probability ensemble on the five historically exposed full-provider-holdout matches: Sportec J03WOY, Metrica metrica_3, and three SkillCorner matches. It covers one clean cell per match and two saved feed-like replicates per match. There was no fitting or model inference. The active decoder was not changed. The audit re-scored the saved decoded predictions with the unchanged official metric and verified both candidate score rows and the saved matched-control decode against their receipts across all 15 cells.

The reports use the official global greedy H/A spell assignment at IoU .3/.4/.5/.6/.7. Stress replicates are averaged within match, matches within provider, and providers equally for headline summaries. Frame and miss-geometry counts are descriptive; repeated frames and synthetic replicates are not independent observations. No oracle repair was applied.

| View | Spell F1 | Macro F1 | Under-3s recall at IoU .5 | D precision / recall |
| --- | ---: | ---: | ---: | ---: |
| Clean | .6332 | .8542 | .3842 | .9050 / .8029 |
| Feed-like | .4988 | .8114 | .2034 | .8960 / .8020 |

Among unmatched references, the largest within-span error category is wrong-team output: an equal-provider mean of about 47% in clean cells and 53% in feed-like cells. False-D output is the dominant category for about 25% and 20%, respectively; the remaining 28% and 27% have same-team overlap or boundary/assignment patterns. These labels describe the largest frame-error count inside each unmatched span. They can hide other errors in the same spell and are not mutually exclusive causes.

Across graded live frames, wrong-team errors are more common than false-D errors: 8.17% vs 4.09% clean, and 14.20% vs 4.59% feed-like. The model also misses about 19.7% of D frames in both views. That D recall loss matters, but false-D on live frames is not the main frame error in these cells.

## Other spell failure modes

| Diagnostic | Clean | Feed-like |
| --- | ---: | ---: |
| Reference spells overlapped by multiple same-team predictions | 65 / 1,963 (3.3%) | 103 / 3,926 (2.6%) |
| Predictions overlapping multiple same-team references | 157 / 1,998 (7.9%) | 408 / 3,490 (11.7%) |
| Reference spells touching a graded-run edge | 789 / 1,963 (40.2%) | 1,578 / 3,926 (40.2%) |

Merge-like overlaps are more common than fragmentation. They are still much less frequent than missed short spells. The larger feed-like merge count is a stress diagnostic; it does not show that merges drive the clean score.

Graded-edge censoring is concentrated in SkillCorner: 784 of its 1,125 clean reference spells (69.7%) touch a graded-run edge, and 61% of those spells match at IoU .5. These spans stay in official scoring, but a boundary cut by an X edge cannot support an ordinary timing interpretation. For complete, matched spells, the mean of provider-level median start/end errors is −.15s / +.07s clean and −.27s / +.19s feed-like. SkillCorner contributes the largest extension pattern: its cell-median start/end errors are about −.38s / +.28s clean and −.72s / +.59s feed-like. Here signed error is predicted boundary time minus reference time.

## Gap and contact signal

The long-gap result points toward team attribution when reliable contact evidence is old, but its native support is limited.

| View / stratum | Graded live-frame support | Wrong-team rate | False-D rate |
| --- | ---: | ---: | ---: |
| Clean, native ball gaps ≥10s | 108 frames, all SkillCorner | 5.6% (6 frames) | 75.9% (82 frames) |
| Feed-like, ball gaps ≥10s | 163,799 frames across 10 saved cells | 23.1% equal-provider mean | 5.1% equal-provider mean |
| Feed-like, nearest reliable contact 10–20s ago | 36,340 pooled frames across 10 saved cells | 32.3% equal-provider mean | 7.0% equal-provider mean |

Within feed-like gaps, wrong-team rates rise from 19.4% at 10–20s to about 24–25% beyond 20s; false-D rates stay around 4–7%. The contact measure is a tracking-derived geometry proxy, not possession truth. The synthetic feed-like cells give strong evidence for a stress mechanism, while the native clean set has only 108 graded live frames in long gaps and reverses the error mix. Treat contact-aware long-gap attribution as a second, stress-focused hypothesis that needs independent broadcast evidence.

## Recommended next mechanism

Prioritize brief-spell recovery in a model-side temporal experiment. The broadest opportunity is in missing H/A events under three seconds, while split errors are rare. A useful bounded comparison would test whether overlapping temporal context or a duration-sensitive training signal preserves short spells, with the saved baseline and decoder held fixed. Judge it on the same external cells by headline spell F1, short-spell recall, and provider clean D/macro guards. The ongoing evidence is descriptive development evidence; it is not an expected gain or confirmation result. A separate contact-age intervention for long-gap H/A attribution is promising for synthetic stress but should not be selected without better native support.

## Provenance and limits

The complete aggregate output is `results/spell_opportunity_audit.json`; the corrected pre-run receipt is `results/spell_opportunity_audit_declaration_v2.json`. The receipt binds the active baseline and policy, the external NPZs, matched input snapshots, saved outputs, scorer, and diagnostic sources. No organizer files, labels, scores, or scoring gates were read or used. Metrica and SkillCorner references are event-derived silver labels, every development match has prior exposure, and feed-like inputs are synthetic. There is no independent confirmation set.

The audit was run with `/workspace/venv/bin/python -m experiments.spell_opportunity_audit --freeze` followed by `--run` after restoring the private cache. Both commands refuse to replace an existing receipt or result. In a clean worktree with the same verified cache restored and no prior audit artifacts, those commands recreate the sealed aggregate. The focused contracts run with `/workspace/venv/bin/python -m unittest experiments.tests.test_spell_opportunity_audit -v`.

The first implementation attempt stopped while building its first row because it looked for a reference-spell count under a per-IoU object. It wrote no result file and no metric values from that attempt were used. The generated receipt from that attempt was removed before its exact hash was retained. The corrected source and inputs were frozen under declaration v2 before the completed run; that correction is recorded in both v2 artifacts.
