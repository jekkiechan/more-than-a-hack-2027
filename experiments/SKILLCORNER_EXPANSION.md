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

Validation includes the existing synthetic/contract suite, new data/sampler contracts, independent score/cache/gate integrity contracts, exact file-backed synthetic-fit parity, and bundled submission format validation. Real checkpoints, normalization, optimizer budgets and saved probabilities passed independent replay; the [fit](results/skillcorner_expansion_fit_verification.json) and [external](results/skillcorner_expansion_external_verification.json) receipts record the checks. Array/model/frame artifacts stay in private storage; only code, manifests and aggregate evidence belong in this public repository.

## Outcome

**Retain the active thirteen-match training manifest and `joint_gru_combined_ensemble_v1`.** The six-match expansion fails both frozen clean-spell screens. It remains an experimental corpus, with no active training or model promotion. The reserved match 2017461 remains unscored.

These are matched seven-match comparisons; their control scores differ from the historical five-match aggregate because the evaluation cohort has expanded. Provider holdout excludes every learned component. The pooled supplement excludes the same seven evaluation matches from training but uses all three providers.

| Protocol / view | Original 13 spell F1 | Expanded 19 spell F1 | Delta | Control / expanded macro F1 |
| --- | ---: | ---: | ---: | ---: |
| primary / clean | 0.641468 | 0.643523 | +0.002055 | 0.850920 / 0.851632 |
| primary / feed_like | 0.510206 | 0.502104 | -0.008102 | 0.809704 / 0.807428 |
| pooled / clean | 0.696722 | 0.686765 | -0.009958 | 0.877504 / 0.878381 |
| pooled / feed_like | 0.554822 | 0.546531 | -0.008290 | 0.836023 / 0.838120 |

Primary clean gain is below +.01; pooled clean gain is negative. Both feed-like spell guards and all per-provider clean macro/D-F1 guards pass. Passing frame-quality guards does not compensate for failed spell-F1 criteria. No criteria changed after seeing the results.

### Provider and seed spread

| Protocol | Provider | Control / expanded clean spell | Control / expanded feed-like spell | Expanded clean D precision / recall |
| --- | --- | ---: | ---: | ---: |
| primary | sportec | 0.642981 / 0.655319 | 0.473066 / 0.463942 | 0.926117 / 0.891351 |
| primary | metrica | 0.797222 / 0.791049 | 0.602041 / 0.586858 | 0.913610 / 0.879737 |
| primary | skillcorner | 0.484201 / 0.484201 | 0.455512 / 0.455512 | 0.865191 / 0.665868 |
| pooled | sportec | 0.715764 / 0.697199 | 0.521111 / 0.518676 | 0.922278 / 0.933721 |
| pooled | metrica | 0.833741 / 0.821167 | 0.640623 / 0.620133 | 0.902262 / 0.906068 |
| pooled | skillcorner | 0.540661 / 0.541929 | 0.502731 / 0.500786 | 0.837408 / 0.836004 |

SkillCorner primary parity is required by the design: those models exclude all SkillCorner training matches. In the pooled supplement, SkillCorner clean spell gains only .001267; Sportec and Metrica regress by .018566 and .012575. More same-provider volume does not establish general transfer.

| Protocol | Member | Clean spell delta | Feed-like spell delta |
| --- | --- | ---: | ---: |
| primary | 2026 | -0.003666 | -0.005498 |
| primary | 2027 | +0.003526 | -0.005124 |
| primary | ensemble | +0.002055 | -0.008102 |
| pooled | 2026 | -0.001470 | -0.000469 |
| pooled | 2027 | -0.008177 | -0.009713 |
| pooled | ensemble | -0.009958 | -0.008290 |

Under-three-second recall declines by .001168/.008198 in primary clean/feed-like and .014785/.014857 in pooled clean/feed-like. Pooled D recall increases by .028718/.032676 while D precision declines by .020632/.025423. Those changes are descriptive and do not prove why overall spell recovery falls.

All five IoU scores, short-reference counts, predicted/reference spell counts, fragmentation and each member/ensemble cell are retained in the [252 aggregate rows](results/skillcorner_expansion_rows.json). The [decision and summaries](results/skillcorner_expansion.json) preserve every failed guard; the [data audit](results/skillcorner_expansion_data_audit.json) records masks, event-derived target quality and source provenance.

### Interpretation and next data work

This fixed-update study does not establish that SkillCorner data are inherently harmful, or test a larger optimization budget. Its result is that automatically adding this batch under the current recipe does not earn adoption. The labels remain provider/event silver; interruption-derived D, broadcast visibility filtering, X margins and weak native long-gap coverage remain limitations. The reserved match has source-quality metadata audits but no model scores; it is not evaluated to rescue a failed development result.

The next useful data task is a blind dense-label pilot on a match with competition-like observation gaps, with independent adjudication of possession boundaries. Additional event-silver volume or model-generated labels alone would not resolve that representativeness gap.


## Requested competition diagnostic

The fixed expanded pooled model was scored once after sealing the failed external adoption decision and its full-match predictions. The normal baseline prediction entrypoint produced two byte-identical, format-valid 199,319-row CSVs; their X masks match invalid input periods. An exclusive marker was claimed before the reference was fetched. The existing baseline result was reused; no historical gate was reset.

| Public five-minute diagnostic | Active baseline | Expanded candidate | Delta |
| --- | ---: | ---: | ---: |
| Mean spell F1, IoU .3–.7 | .557576 | .548571 | -.009004 |
| Macro F1 | .854088 | .874121 | +.020033 |
| D-F1 | .854810 | .890556 | +.035746 |
| Predicted / reference H/A spells | 16 / 17 | 18 / 17 | +2 predicted |

Candidate F1 at IoUs .3/.4/.5/.6/.7 is .628571/.628571/.571429/.571429/.342857. At IoU .5 both models recover ten true spells; the candidate has eight spurious spells versus six. Higher frame accuracy therefore does not imply better possession intervals. Candidate spell bootstrap 5th–95th percentiles are [.388571, .708571]; these are descriptive single-model intervals, not paired significance.

Only 7,500 frames and seventeen reference possessions are graded. This already-exposed diagnostic neither establishes full-match performance nor selects a model. Its lower spell point estimate does not change the external decision. No retraining, seed changes, decoder tuning or further candidate scoring followed. See the [saved diagnostic](results/skillcorner_expansion_public_check.json) and [execution receipt](results/skillcorner_expansion_public_execution.json).

## Reproduction and operational verification

The executable comparison was frozen at code commit `b448bb7880997946dce9ee3fc7403b37beb737aa`, before later main changes were integrated in a separate delivery checkout. Its declaration SHA-256 is `8bf2bc36a85c28fe9e6725c90905ab57b1a9142c98af54fcaa5e9f4e7a76963f`. Replays must use that checkout or the archived source snapshot: the seal intentionally checks the complete experiments/evaluation Python-file inventory, so later unrelated modules do not belong in its execution environment. Restore the saved declaration and artifacts; do not freeze a new study or reopen a completed public gate.

All six fit receipts and checkpoint hashes pass. The [independent fit verifier](results/skillcorner_expansion_fit_verification.json) checks all declared input/cache/control/source hashes, model capacity and optimizer updates, and recomputes training-only normalization once per fold with bitwise agreement for both seeds. It verifies source-balanced and within-source match-balanced loss formulas. Initial standalone verifier schema assumptions were corrected and audited; data, frozen runtime sources, models and thresholds were unchanged.

The integrated delivery checkout passes 258 experiment contracts plus seven independent integrity/gate contracts, exact file-backed synthetic-fit parity, and the bundled 199,319-row format check. The real expanded competition CSV is also valid and reproduces byte-for-byte.

The frozen public-check module has weaker generic re-entry handling: its existing-result path precedes seal validation, and its seal does not separately compare complete declaration bytes. This execution used a fresh gate, preflighted all result destinations and the exact declaration hash, and separately hash-bound the resulting private/public files. No existing-result or repeat-score path was used. The archived preflight and execution receipts preserve those additional checks.

The [independent external verifier](results/skillcorner_expansion_external_verification.json) passes all 42 paired cells / 252 rows, including exact candidate probability replay for 84 seed/cell comparisons (maximum absolute difference zero), source/match identities, hashes, probability means, official metric/decoder replay, all fifteen historical control rows, provider/equal-provider summaries and the adoption decision. The [verifier correction audit](results/skillcorner_expansion_verifier_corrections.json) preserves the standalone schema/inventory fixes.

### Long missing-ball behavior

The independent, non-gating diagnostic resets missing-ball runs at period/time breaks. The table pools reference-H/A frames within the two feed-like replicates whose whole missing-ball run lasts at least ten seconds; these are frame-weighted descriptive rates, not equal-provider adoption scores. Both protocols use the same 189,954 reference-H/A frames.

| Feed-like protocol | Wrong team, control → candidate | False dead-ball, control → candidate |
| --- | ---: | ---: |
| Full provider holdout | 23.23% → 24.05% | 4.75% → 5.48% |
| Pooled | 20.89% → 20.08% | 4.97% → 7.21% |

Neither arm emits X on this support. Pooled team identity improves locally while false D increases, and overall spell F1 still falls. Native clean long gaps have only 230 graded live frames across the expanded seven-match cohort; those limited observations cannot confirm competition-like gap robustness.

## Artifact availability

The experimental nine-match corpus, six checkpoints, cached features, predictions, source snapshot and verifier evidence are frozen locally in private-cache commit `5dd0a23fd734fa79a99246cd792e251fe5f20b78` on `archive/skillcorner-expansion-v1`. The study manifest lists 374 files / 2,305,049,329 bytes; a separate manifest identifies 67 existing base dependencies. No organizer reference CSV or new organizer inputs are included.

Automatic approval review rejected pushing this payload to `jekkiechan/mtah-data`, requiring explicit destination-specific authorization for the private tracking data and experiment artifacts. No alternate upload route was attempted. Remote availability and a fresh remote restore are **not** claimed. The active dataset and model remain unchanged.

The [artifact receipt](results/skillcorner_expansion_artifacts.json) records successful local restoration and SHA-256 verification of all 441 unique files / 4,281,194,341 bytes across the new archive and its base dependencies. The corrected unpublished commit contains 105 proper LFS pointers; the initial missing-clean-filter staging issue changed no payload checksums. This is a local restore, not a remote-clone or download check.

From the local archive checkout, reproduce the verified restore with the existing conflict-safe helper:

```sh
python scripts/competition_cache.py restore --manifest skillcorner_expansion_v1_cache_manifest.json --data-dir /path/to/code/data
python scripts/competition_cache.py restore --manifest skillcorner_expansion_v1_base_dependencies_manifest.json --data-dir /path/to/code/data
```

Use the frozen execution checkout or archived snapshot with those restored files. Saved competition diagnostic results are final; do not rerun its scorer. Public receipt JSON retains the original executor paths as provenance; these contain no tracking coordinates or credentials.
