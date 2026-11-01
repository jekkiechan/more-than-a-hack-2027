# Provider-excluded stacking of complementary possession models

Study `smart_stacking_v1`, 1 November 2026. Base commit `cef9490`;
branch `research/smart-stacking-v1`. This is a bounded development experiment,
with no organizer reference access or model promotion.

## Question and controls

Can a learned stack combine models that individually approach the target but
do not establish a robust overall improvement? Four restored model families
contribute two matching seeds, 2026/2027: adopted calibrated dense, raw-feed
augmentation, within-record short sampling, and original-possession diversity.
The latest full-model research leader reaches .662590/.528180 clean/feed-like
official spell F1; the adopted baseline reaches .658625/.518527.

The later fixed public diagnostic for the research leader is lower than the
adopted model. It motivates caution but supplies no stack-training labels,
threshold selection, seed selection or new scoring. Historical public gates
remain closed. See [PR #66](https://github.com/jekkiechan/more-than-a-hack-2027/pull/66).

Eleven fixed arms compare the adopted model, raw-feed model, short-sampled H/A
with raw-feed D, original-diversity model, uniform four-family probability
average, two single-model calibration controls, a static learned stack, a
temporal H/A stack retaining raw-feed D, a temporal joint stack, and a fixed
50/50 temporal-H/A/adopted blend. No base model is retrained.

## Leakage prevention and additional supervision

The five exposed development matches remain Metrica 3, Sportec J03WOY, and
SkillCorner 1996435/1996436/2006229. All learned base H/A/D components,
teachers and normalizers exclude the evaluated provider. For each outer
provider, regenerate level-one predictions on the OTHER providers' development
matches using that SAME outer fold's checkpoints. Those development matches
are disjoint from every base TRAIN match. Fit the meta-model only on those
other-provider labels, then evaluate the excluded provider.

Reusing the usual other-provider OOF arrays would be invalid here: those base
checkpoints had seen the outer provider's TRAIN matches. This study regenerates
the required arrays instead. The single-model calibration controls receive
exactly the same extra development-label supervision and training weights as
the multi-model stacks. The comparison with an uncalibrated base model alone
cannot attribute gains solely to model combination.

This uses more labeled development matches than the original base training
recipe and cannot claim new independent evidence. All five matches have prior
researcher exposure. Metrica/SkillCorner references include event-derived silver
labels. No GAR VALID/TEST or organizer reference labels are used.

## Fixed meta-model and decoder

Fit separate conditional H/A and dead-ball logistic models. Static features
are the four experts' log odds. The temporal H/A model adds each expert's
current log odds minus its centered one-second local mean, separately within
each inference time block. H/A has no intercept, preserving team symmetry.
D receives only player-only D log odds plus an intercept, without H/A,
ball-visibility, provider identity or target-derived inputs.

Use L2 penalty .02 around equal expert weights, zero temporal terms and zero
D intercept. L-BFGS-B has a fixed 200-iteration ceiling and no parameter search.
Deterministic 5 Hz native-row sampling and equal provider/match/clean-feed
weights control exposure. TRAIN-only conditional H/A weights emphasize short
possessions by `clip(sqrt(3/duration_seconds), 1, 4)`; D uses ordinary within-cell
weights. Preserve X targets and all native grading/time boundaries.

Three outer providers × two seeds × four meta-model types yield 24 models,
48 binary fits. Each component seed has its own stack fit. Average the two
resulting native H/A/D probability arrays before the unchanged temperature-4,
cost-3 decoder. Individual seed scores are descriptive; their metric average
is not the ensemble score.

Retaining a D component means preserving its probabilities. H/A emissions can
still change decoded D labels through joint segmentation.

## Ranking, robustness and stopping

Score 11 arms × 15 match/view cells × 3 members = 495 original cells with the
official scorer. Average corruption replicas within match, matches within
provider, then providers equally. Rank unrounded feed-like spell F1, macro
F1 only for exact ties; report the fixed 70/30 summary separately.

Guards against the adopted baseline are clean aggregate spell within .005,
each provider's clean macro/D-F1 within .02, and each provider's feed-like
spell within .01. Report the historical +.01 clean-gain screen separately.
A learned winner must also beat original diversity, uniform averaging, and
both single-model calibration controls. The .55 feed target is a stretch.

Ten new raw-feed corruption inputs, with fixed seeds 862401/862402 hashed by
match, are prepared before candidate scoring. Only a qualifying learned winner
may receive the declared fixed comparison against adopted/original/uniform on
those inputs, without reselection. Such a check is synthetic-corruption
sensitivity on the same matches, not independent-match confirmation. The
declaration also fixes a one-expert-at-a-time fallback diagnostic for that same
winner. No post-result arms, rates, temperatures, context lengths or seeds.

## Preserved runtime correction

The first inference attempt stopped before producing any level-one array or
candidate score. Ensemble NPZ metadata provides the single evaluation input
through `evaluation_input_hashes_json`, whereas the frozen runner looked for
the member-only `evaluation_input_path` field. All three failure logs and the
original runner remain intact. A separately sealed adapter verifies and loads
the existing hash-bound path, and memoizes unchanged source cells to reduce
repeated I/O. It changes no numerical inputs, targets, models or parameters.

Evidence: [frozen declaration](results/smart_stacking_v1_declaration.json),
[metadata amendment](results/smart_stacking_v1_runtime_amendment.json).

The first scoring attempt also stopped before any completed score or output:
NPZ scalar `fps` was still a zero-dimensional array, which the official scorer
cannot round. A second sealed adapter converts that metadata to the identical
Python float and accepts only the already-existing identical scoring lock.
The original failure log, lock and all 24 fitted models are preserved. No
numerical value, inference, model, arm or scoring population changes.
See the [scalar amendment](results/smart_stacking_v1_scalar_amendment.json).

## Reproduction and artifacts

Restore only the exact artifacts bound by the declaration, including the six
provider-excluded teachers and four model families. The local restore receipt
records 230 verified dependencies from the existing private research archives;
no base refits substitute for a missing checkpoint. Keep source overlays from
older archives isolated. Use the already-completed outputs instead of repeating
a frozen command:

```sh
.venv/bin/python -m experiments.smart_stacking_v1 declare
.venv/bin/python -m experiments.smart_stacking_v1 prepare_fresh
# The initial metadata failure is preserved; the sealed adapter is required.
.venv/bin/python -m experiments.smart_stacking_runtime_v1 infer --provider Metrica
# Also infer SkillCorner and Sportec, then:
.venv/bin/python -m experiments.smart_stacking_runtime_v1 fit
.venv/bin/python -m experiments.smart_stacking_scalar_runtime_v1 evaluate
.venv/bin/python scripts/verify_smart_stacking_v1.py
.venv/bin/python -m experiments.smart_stacking_runtime_v1 fresh
```

Native tracking, predictions and models stay in ignored
`data/experiments/smart_stacking_v1/`. Publication of aggregate research results
does not establish remote availability of new reconstructive artifacts.

## Completed results

Decision: **retain the adopted baseline; no learned stack beats the strongest single recipe under the declared rule.** All 24 meta-models and 495 original score cells completed. No extra settings follow this result.

| Fixed arm | Clean spell | Feed-like spell | Feed macro | Auxiliary 70/30 | Feed-first guards |
| --- | ---: | ---: | ---: | ---: | --- |
| Adopted calibrated dense | 0.658625 | 0.518527 | 0.823343 | 0.609972 | Pass |
| Raw-feed model | 0.654740 | 0.525240 | 0.826937 | 0.615749 | Pass |
| Short-sampled H/A + raw-feed D | 0.657215 | 0.527396 | 0.824460 | 0.616515 | Pass |
| Original-diversity model | 0.662590 | 0.528180 | 0.825370 | 0.617337 | Pass |
| Equal four-family probability average | 0.659675 | 0.523795 | 0.826276 | 0.614539 | Pass |
| Single adopted-model calibration | 0.641212 | 0.506832 | 0.811975 | 0.598375 | Fail |
| Single original-model temporal calibration | 0.642108 | 0.514128 | 0.814233 | 0.604159 | Fail |
| Learned static joint stack | 0.639221 | 0.513720 | 0.816411 | 0.604527 | Fail |
| Learned temporal H/A, raw-feed D | 0.661814 | 0.525872 | 0.824854 | 0.615566 | Pass |
| Learned temporal joint stack | 0.638640 | 0.513886 | 0.816465 | 0.604660 | Fail |
| Fixed half temporal H/A / adopted blend | 0.658817 | 0.519920 | 0.824383 | 0.611259 | Pass |

The best guard-passing learned arm is the **temporal H/A stack retaining raw-feed D**. It gains **+.007344** feed spell and **+.003190** clean spell against the adopted package, but loses **.002308** feed spell against the original-diversity recipe. Its four-family stack also trails the simpler short-H/A/raw-feed-D cross (.527396). Uniform averaging reaches only .523795. No learned arm qualifies as the study winner, and none passes the historical +.01 clean-improvement screen.

### Provider and seed spread

| Arm | Metrica feed spell | SkillCorner feed spell | Sportec feed spell |
| --- | ---: | ---: | ---: |
| Adopted calibrated dense | 0.606749 | 0.454592 | 0.494242 |
| Raw-feed model | 0.617628 | 0.457600 | 0.500494 |
| Short-sampled H/A + raw-feed D | 0.620941 | 0.456672 | 0.504574 |
| Original-diversity model | 0.616220 | 0.450872 | 0.517448 |
| Equal four-family probability average | 0.614353 | 0.456187 | 0.500844 |
| Learned temporal H/A, raw-feed D | 0.622393 | 0.454392 | 0.500830 |
| Learned temporal joint stack | 0.596567 | 0.453599 | 0.491492 |

Relative to original diversity, the best learned stack gains .006173 on Metrica and .003521 on SkillCorner, but loses .016618 on Sportec. Its feed-first guards compare with the adopted package; passing them does not mean matching the strongest recipe in every provider. Even against adopted, SkillCorner declines slightly (.454592 → .454392).

| Member | Temporal H/A clean | Temporal H/A feed | Adopted feed | Original-diversity feed |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 0.654685 | 0.515526 | 0.509778 | 0.516486 |
| 2027 | 0.654536 | 0.519563 | 0.506091 | 0.518873 |
| ensemble | 0.661814 | 0.525872 | 0.518527 | 0.528180 |

The ensemble averages native probabilities from the two separately fitted seed pipelines. Component-score spread does not estimate independent-match uncertainty. A paired provider-stratified whole-match bootstrap gives a conditional best-stack-minus-original difference of −.002308, interval [−.003519, −.000911]. Metrica and Sportec each contain one match, so they contribute no resampling uncertainty. This is not a population-level or unseen-provider significance claim.

### Failed guards and the D-path effect

All four control recipes, uniform averaging, temporal H/A stacking and the fixed safe blend pass the feed-first guards. Both single-model calibrators, the static joint stack and the temporal joint stack fail the aggregate clean-spell floor. All four also fail the Metrica feed-spell floor; the single adopted-model calibrator additionally fails Sportec feed spell. No provider clean macro/D-F1 guard fails. These failures remain part of the result.

The temporal H/A and joint arms use the same fitted H/A coefficients. Changing only from retained raw-feed D to fitted D reduces clean/feed spell from **.661814/.525872** to **.638640/.513886**. Thus the joint regression is directly attributable to the D combination within this ablation. Both single-model joint calibrators also regress. Conditional frame likelihood with short-spell weighting did not translate into better overall interval recovery. This does not establish that every stacking architecture or spell-oriented objective must fail.

### D and short-possession diagnostics

| Arm | Feed D precision | Feed D recall | Feed D-F1 | Short-spell recall | Predicted spells | Reference spells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adopted calibrated dense | 0.843412 | 0.877506 | 0.859048 | 0.198820 | 282.333 | 404.333 |
| Original-diversity model | 0.842902 | 0.880688 | 0.860203 | 0.220039 | 305.556 | 404.333 |
| Learned temporal H/A, raw-feed D | 0.848824 | 0.869208 | 0.857084 | 0.205198 | 294.389 | 404.333 |
| Learned temporal joint stack | 0.786782 | 0.913336 | 0.844928 | 0.170654 | 270.833 | 404.333 |

Counts are equal-provider means. Short recall uses globally one-to-one matched reference H/A spells shorter than three seconds at IoU .5. The best learned stack raises short recall only .198820 → .205198 against adopted, below original diversity’s .220039.

### Robustness follow-up and limits

The ten new corruption inputs were prepared and hash-bound before candidate scoring. The declared follow-up and expert-fallback diagnostic are **skipped**, because no eligible learned arm beats the strongest single-model, uniform and calibration controls. No new corruption scores are claimed and no public diagnostic follows. Original clean/feed-like results are exposed development evidence, not independent confirmation or proof of full-match organizer robustness.

The .55 original feed-like target remains unmet. Retain the adopted package and keep original diversity as the numerical research leader. A future stacking study should first establish complementary errors from genuinely different model families and obtain stronger match-level evaluation; this result does not justify another weight, context or decoder sweep on these five matches.

Evidence: [aggregate and all rows](results/smart_stacking_v1_aggregate.json), [model seal](results/smart_stacking_v1_model_seal.json), [follow-up skip](results/smart_stacking_v1_fresh_skipped.json).


## Verification and delivery

The separate verifier reconstructs all **495** native probability outputs with
independent temporal-feature and logistic algebra, then independently decodes
and replays the official spell/macro/D metrics. It checks all reference/grading
counts, all meta-model provider exclusions and equal-provider aggregation.
Maximum probability replay error is recorded in the
[verification receipt](results/smart_stacking_v1_verification.json); decoded
labels match exactly. All 120 restored target-model arrays pass the declared
2e-6 runtime tolerance (maximum 1.461e-6); adopted decoded labels match exactly.
Three complete control summaries reproduce their archived scores to 1e-12.

After integrating current main through PR #66, **564 experiment tests** pass,
plus 26 root tests, exact file-backed fit parity, documentation links and the
bundled 199,319-row submission-format check. No new pooled deployment fit or
organizer prediction/score is part of this study.

The [artifact receipt](results/smart_stacking_v1_artifacts.json) records
**582 locally verified native/metadata files / 2,286,245,722 bytes**, separate
from 230 hash-verified inherited dependencies. New reconstructive artifacts
are local in ignored storage; new remote upload and independent remote restore
are not claimed. Source and aggregate evidence are committed locally. Automatic approval review
rejected the GitHub push/PR because the request authorized testing but not
external publication. GitHub and research-site publication are paused pending
explicit user approval. The locally checked site preserves restricted access;
its current deployed version remains 20. The
[delivery snapshot](results/smart_stacking_v1_delivery.json) records the actual
state and clarifies that the earlier artifact receipt names an intended PR
destination, not an observed publication. The report and complete publication
draft are ready for review; no new remote artifact upload is claimed.
