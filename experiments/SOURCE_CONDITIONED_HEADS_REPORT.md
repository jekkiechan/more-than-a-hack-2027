# Source-conditioned output head comparison

## Decision

Retain the shared-head C baseline. Neither source-specific head variant passed the declared practical guard: both lost more than 0.01 clean spell F1, and both lost more than 0.01 feed-like spell F1 against C. The provider-specific clean macro-F1 and dead-class F1 guards passed for both challengers, but they do not override the spell guard. No model is promoted, and this development-only comparison does not establish independent broadcast performance.

| Arm | Clean spell F1 | Feed-like spell F1 | Clean macro-F1 | Clean D-F1 | Δ clean spell vs C | Feed guard | Clean +0.01 screen |
|---|---:|---:|---:|---:|---:|---|---|
| C | 0.647215 | 0.505811 | 0.857677 | 0.850177 | — | baseline | baseline |
| H | 0.634885 | 0.494700 | 0.854828 | 0.852542 | −0.012329 | fail | fail |
| HD | 0.635602 | 0.495254 | 0.856524 | 0.856842 | −0.011613 | fail | fail |

## Scope and method

The frozen finite matrix was C/H/HD × three leave-one-provider-out folds (`leave_out_sportec`, `leave_out_metrica`, `leave_out_skillcorner`) × seeds 2026/2027: 18 fits, each at 2,048 updates. There were 90 per-seed prediction cells and 45 two-seed ensemble cells. Ensembles average native probabilities before the unchanged cost-3 decoder and official scorer. The 70/30 value is the declared auxiliary combination of spell and macro F1; the guard/ranking decision uses the separately reported spell and provider macro/dead metrics.

C is the frozen D2 shared-projection recipe. H separates the target and GAR H/A output projections while sharing the D projection. HD also separates the target and GAR D projections. The target projections are used for every inference provider because an unseen organizer provider cannot be identified as SoccerNet GAR. GAR silver labels train auxiliary GAR projections only. H adds 386 projection parameters and HD adds 435; the comparison therefore has a capacity confound. It does not identify a causal benefit of source conditioning independent of extra parameters.

The recurrent feature paths, D2 sampling order, per-row source routing, input dropout, separate H/A and D optimizers, loss terms and weights, gradient clipping, update budget, fold-frozen normalizers, five-second context, ten-second cores, period/time-break handling, and direct/team-swapped inference were preserved by the frozen runner. The H-only arm provides a check that its shared D path stays near C: across all six fold/seed pairs, maximum D recurrent-state difference was `1.79e-7`, D projection weight difference `8.94e-8`, bias difference `1.49e-8`, and target-route native dead-probability difference `1.31e-6` on clean selection inputs. The inference replay tolerance was `2e-6`.

The original trained D2 fold weights were unavailable after documented artifact checks. C was therefore explicitly reconstructed from the exact fold teachers, prepared target caches, verified GAR train records, and frozen D2 recipe; it is not represented as restored historical D2 trained weights. The reconstructed C ensemble replayed the published D2 rounded clean/feed-like spell values: `0.6472145591` vs `0.6472` and `0.5058107894` vs `0.5058`, both within the declared four-decimal rounding tolerance of `0.000051`. The raw SoccerNet train source pins and prepared 45-game dense manifest were hash-verified; the manifest SHA-256 is `d2639fa1a47b38bff1b293e8b2592fb1d7392c09700a2c9873c1a90a72397b51`.

The L calibration arm was excluded before candidate scoring. Its training-only event schema evidence was Sportec-specific and did not establish a transferable rule for GAR, Metrica, and SkillCorner; the leave-out-Sportec fold also had no eligible Sportec calibration training matches. This excludes it from this matrix, not from future source-specific calibration research.

## Ensemble metrics

Equal-provider means combine the three provider aggregates, not individual frame counts. Values below are equal-provider means; the scorer aggregates each source's selection cells first (five clean cells or ten feed-like cells overall). `P/R` are D-class precision/recall; `short` is short-spell recall.

| Arm | View | Spell F1 | 70/30 | Macro-F1 | D-F1 | D P/R | Short recall | GT spells | Predicted spells |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | clean | 0.6472 | 0.7104 | 0.8577 | 0.8502 | 0.9185 / 0.7969 | 0.4025 | 404.3 | 360.7 |
| C | feed-like | 0.5058 | 0.5987 | 0.8155 | 0.8477 | 0.9108 / 0.7979 | 0.2171 | 404.3 | 318.4 |
| H | clean | 0.6349 | 0.7009 | 0.8548 | 0.8525 | 0.9177 / 0.8009 | 0.3967 | 404.3 | 379.7 |
| H | feed-like | 0.4947 | 0.5894 | 0.8105 | 0.8488 | 0.9098 / 0.7998 | 0.2094 | 404.3 | 328.1 |
| HD | clean | 0.6356 | 0.7019 | 0.8565 | 0.8568 | 0.9107 / 0.8117 | 0.3967 | 404.3 | 376.3 |
| HD | feed-like | 0.4953 | 0.5902 | 0.8119 | 0.8518 | 0.9024 / 0.8092 | 0.2070 | 404.3 | 326.2 |

### Provider metrics

Each cell gives spell F1 / macro-F1 / D-F1. These are full-provider aggregates over the full 15-cell set: three Sportec, three Metrica, and nine SkillCorner cells across clean and feed-like views. The clean ensemble contains one cell per provider; feed-like contains two per match.

| Arm | View | Metrica | SkillCorner | Sportec |
|---|---|---:|---:|---:|
| C | clean | 0.8050 / 0.9254 / 0.9020 | 0.4869 / 0.7819 / 0.7486 | 0.6498 / 0.8658 / 0.9000 |
| C | feed-like | 0.6038 / 0.8749 / 0.8953 | 0.4411 / 0.7619 / 0.7508 | 0.4726 / 0.8098 / 0.8970 |
| H | clean | 0.7978 / 0.9246 / 0.9014 | 0.4558 / 0.7735 / 0.7562 | 0.6510 / 0.8664 / 0.9000 |
| H | feed-like | 0.5931 / 0.8718 / 0.8949 | 0.4203 / 0.7486 / 0.7542 | 0.4707 / 0.8111 / 0.8973 |
| HD | clean | 0.7995 / 0.9251 / 0.9023 | 0.4572 / 0.7780 / 0.7679 | 0.6502 / 0.8665 / 0.9004 |
| HD | feed-like | 0.5919 / 0.8717 / 0.8948 | 0.4225 / 0.7527 / 0.7629 | 0.4714 / 0.8112 / 0.8977 |

The declared practical guard requires feed-like spell F1 to be within 0.01 of C and requires each provider's clean macro-F1 and D-F1 to be within the declared tolerance of C. Both H and HD passed all six provider-specific clean macro/D checks, but failed feed-like spell F1 and the separate +0.01 clean spell screen. In particular, all three arms showed similar feed-like degradation, while neither new head recovered the clean spell loss.

### Spell F1 at each overlap threshold

| Arm | View | IoU .3 | IoU .4 | IoU .5 | IoU .6 | IoU .7 |
|---|---|---:|---:|---:|---:|---:|
| C | clean | 0.7369 | 0.7068 | 0.6624 | 0.5979 | 0.5322 |
| C | feed-like | 0.6182 | 0.5719 | 0.5099 | 0.4449 | 0.3842 |
| H | clean | 0.7289 | 0.6971 | 0.6442 | 0.5851 | 0.5192 |
| H | feed-like | 0.6060 | 0.5616 | 0.4984 | 0.4346 | 0.3729 |
| HD | clean | 0.7304 | 0.6975 | 0.6463 | 0.5866 | 0.5172 |
| HD | feed-like | 0.6065 | 0.5625 | 0.4989 | 0.4351 | 0.3733 |

### Seed spread

These single-seed values are descriptive and are not the ensemble ranking target. Across seeds, clean spell F1 spans 0.6355–0.6372 for C, 0.6271–0.6325 for H, and 0.6286–0.6306 for HD. Feed-like spell F1 spans 0.4958–0.5018 for C, 0.4901–0.4966 for H, and 0.4902–0.4958 for HD.

| Arm | Seed | Clean spell / macro / D-F1 / short | Feed-like spell / macro / D-F1 / short | 70/30 clean / feed |
|---|---:|---:|---:|---:|
| C | 2026 | 0.6355 / 0.8506 / 0.8364 / 0.3832 | 0.5018 / 0.8091 / 0.8360 / 0.1987 | 0.7001 / 0.5940 |
| C | 2027 | 0.6372 / 0.8567 / 0.8511 / 0.3885 | 0.4958 / 0.8097 / 0.8464 / 0.2162 | 0.7031 / 0.5899 |
| H | 2026 | 0.6271 / 0.8483 / 0.8391 / 0.3849 | 0.4966 / 0.8057 / 0.8373 / 0.2182 | 0.6935 / 0.5894 |
| H | 2027 | 0.6325 / 0.8541 / 0.8537 / 0.3997 | 0.4901 / 0.8049 / 0.8478 / 0.2223 | 0.6990 / 0.5845 |
| HD | 2026 | 0.6286 / 0.8512 / 0.8459 / 0.3815 | 0.4958 / 0.8078 / 0.8424 / 0.2157 | 0.6954 / 0.5894 |
| HD | 2027 | 0.6306 / 0.8547 / 0.8555 / 0.3977 | 0.4902 / 0.8051 / 0.8483 / 0.2210 | 0.6979 / 0.5847 |

### Short spells, fragmentation, and boundaries

Counts below sum the cells within each view, while boundary timing pools complete spell boundary matches at IoU 0.5. Start/end errors are prediction time minus reference time in seconds, so negative start values mean predictions tend to start early. The source-conditioned experiment's frozen aggregate omitted the ensemble overlap geometry fields; the accompanying diagnostics supplement recomputes these counts from the sealed two-seed native probabilities and declared caches. It matches every aggregate scorer metric and binds the score/model seals. Do not interpret the aggregate's default zero merge count as measured zero.

| Arm | View | Fragmented reference spells | Predictions overlapping multiple same-team refs | Short matched / short refs | Short recall (equal-provider mean) | Start error mean; per-cell p90 abs range | End error mean; per-cell p90 abs range |
|---|---|---:|---:|---:|---:|---:|---:|
| C | clean | 41 | 161 | 274 / 783 | 0.4025 | −0.492 s; 1.34–5.40 s | +0.235 s; 1.64–5.50 s |
| C | feed-like | 76 | 403 | 355 / 1,566 | 0.2171 | −0.566 s; 2.89–5.26 s | +0.481 s; 2.53–7.62 s |
| H | clean | 68 | 152 | 271 / 783 | 0.3967 | −0.351 s; 1.32–4.84 s | +0.176 s; 1.64–3.91 s |
| H | feed-like | 106 | 400 | 343 / 1,566 | 0.2094 | −0.545 s; 2.90–7.96 s | +0.410 s; 1.90–5.54 s |
| HD | clean | 69 | 152 | 271 / 783 | 0.3967 | −0.360 s; 1.38–4.84 s | +0.168 s; 1.62–3.91 s |
| HD | feed-like | 108 | 398 | 339 / 1,566 | 0.2070 | −0.557 s; 2.84–7.96 s | +0.390 s; 1.75–5.54 s |

The short-spell count uses the scorer's short-reference definition; its raw matched/reference total is descriptive, while the recall column is the official equal-provider mean. Boundary timing excludes matches touching graded-run edges. The summed excluded-match counts were 485/895 (C clean/feed), 477/870 (H), and 473/868 (HD); all per-cell errors and exclusions are in the diagnostics JSON. Fragmentation and overlap counts are diagnostic totals, not the selection criterion.

## Reproducibility, checks, and artifacts

The experiment source was frozen at commit `509242a8abef47b86ce40cfeaac8b9db10e54d02`, with declaration SHA-256 `919946e3753ab9172e5ca8558cfe320f6e0c5c5758ba1ad338e03749dadcf26e`. Fit/model/score receipts and prediction archives are sealed under the ignored `data/experiments/source_conditioned_heads_v1/` data directory and are not included in the repository. The declared execution runner verifies input hashes, fold/model/score identities, exact cell keys, and prediction archive hashes before scoring or aggregation.

The environment restarted during candidate fitting. The old idle queue driver was terminated after checking that no fit workers remained. Completed receipts and logs were retained; there were no incomplete temporary artifacts. The unchanged queue resumed with three workers and skipped completed fit receipts. All final 18 models completed 2,048 updates; no completed model was refit. This recovery is recorded in `data/experiments/source_conditioned_heads_v1/scheduler_resume.json`.

Verification passed: C initial-parameter inference replay on six actual inputs (maximum error below `2e-6`), reconstructed C aggregate gate, synthetic shared-C training parity, target-route independence, source-gradient routing, D player-only input behavior, team-swap/time-break behavior and state serialization tests (13 focused tests), all 18 model/fit identities, exact score/archive key checks, and independent recomputation of all 90 member rows and all 45 ensemble cells. The independent ensemble values agree with the aggregate to `1e-12`. The shared-D comparison is recorded separately in `data/experiments/source_conditioned_heads_v1/h_shared_d_vs_c.json`.

Primary machine-readable results are in `experiments/results/source_conditioned_heads_scores.json`; the corrected ensemble geometry supplement is `experiments/results/source_conditioned_heads_diagnostics.json`. The immutable protocol and finite arm definitions are documented in `experiments/SOURCE_CONDITIONED_HEADS.md`. No organizer labels were accessed. Future action, if any, should be a separately declared study that controls for added head capacity or evaluates genuinely unseen broadcast feeds; this result alone does not justify changing the deployed baseline.

The [independent verification receipt](results/source_conditioned_heads_independent_verification.json) binds the final public aggregate and diagnostics supplement, verifies the 30 source hashes and all 135 member/ensemble cells, and records the failed candidate screens.
