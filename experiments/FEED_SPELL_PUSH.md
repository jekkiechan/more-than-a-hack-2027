# Feed-first spell F1: probability calibration and pooling

Completed 1 November 2026 against fork main `f6a2803`. **Keep f062 / g001 as the strongest feed-like development candidate.** Across five bounded rounds, it raises feed-like spell F1 from **0.505811 to 0.518527** (+0.012717), and clean spell F1 from **0.647215 to 0.658625** (+0.011410). It also beats the previous feed-like leader, 0.511326, by +0.007202. Both component seeds and all three providers improve spell F1 in both views. The adopted deployment manifest is unchanged; an opt-in, reference-free CSV generator is provided.

## What changed

Average the same two dense-model native probability arrays, then replace D probability d with `sigmoid(logit(d) / 4)`. Preserve conditional H/A probabilities and use H/A cost 3 and team/dead cost 3. The transform softens D probabilities toward 0.5; it does not establish probability calibration in a statistical sense. No network weights, normalization, input features, or training examples change.

The user explicitly requested maximizing feed-like spell F1, assuming this synthetic view represents the competition feed. This changes the study's selection objective from earlier clean-first searches. Rank by unrounded feed-like official spell F1, then feed-like macro for exact ties. Report clean results and the historical clean improvement screen separately. Keep a clean aggregate spell floor of adopted −0.005, each provider's clean D/macro within 0.02, and every provider's feed spell within 0.01. All guards pass; the historical +0.01 clean screen now also passes. Neither passing it nor generating a candidate CSV automatically promotes a package.

| Pipeline | Clean spell | Feed-like spell | Clean macro | Feed-like macro |
| --- | ---: | ---: | ---: | ---: |
| Adopted cost-3 decoder | 0.647215 | 0.505811 | 0.857677 | 0.815535 |
| Previous clean-selected r069 | 0.652958 | 0.511065 | 0.860730 | 0.820064 |
| Previous feed-like leader r073 | 0.652348 | 0.511326 | 0.859630 | 0.818536 |
| Feed-first winner f062 / g001 | 0.658625 | 0.518527 | 0.871419 | 0.823343 |

## Two bounded rounds

The first declaration fixes **194 configurations**: two historical controls, 90 joint H/A–D calibration settings, 12 D-bias settings, 18 predecode filters, 18 generalized-mean seed pools, 18 disagreement-dependent transforms, 18 conditional-H/A-entropy transforms, and 18 calibrated non-Viterbi anchor segmenters. It completes **2,910 ensemble cells** and **90 component-seed cells** for three finalists.

| Broad-round family | Feed-selected arm | Feed-like spell | Same arm clean spell | Guards pass |
| --- | --- | ---: | ---: | --- |
| reference | f001 | 0.511326 | 0.652348 | True |
| calibration | f062 | 0.518527 | 0.658625 | True |
| dead_bias | f099 | 0.515343 | 0.654473 | True |
| filter | f117 | 0.514112 | 0.651797 | True |
| pool | f138 | 0.517444 | 0.656412 | True |
| disagreement | f143 | 0.516842 | 0.658664 | True |
| entropy | f171 | 0.517322 | 0.661901 | True |
| anchors | f177 | 0.511171 | 0.649052 | True |


The second declaration is explicitly adaptive to the first result. Its **92 configurations** comprise two controls, 72 local temperature/bias/switch-cost settings, and 18 settings that calibrate each model before averaging. It completes **1,380 ensemble cells** and **90 component-seed cells**. The broad-round winner replays exactly as g001 and remains first at 0.518527 feed-like. The best member-before-pooling arm, g084, scores 0.518284 feed-like / 0.659998 clean. The local refinement adds no feed-like improvement, so this round ends here. The 286 configurations include repeated controls and equivalent settings; they are not 286 independent methods or independent experiments.

Generalized-mean pooling and uncertainty-aware transforms are useful comparisons, but neither beats simple arithmetic pooling followed by fixed D-logit softening. The calibrated anchor variants are negative here; this does not erase the earlier uncalibrated anchor result. Raw-coordinate preprocessing and supervised stacking were not evaluated in these rounds. A valid learned stack requires nested base-model predictions excluding the outer provider throughout; fitting a meta learner on the other providers' existing folds would allow outer-provider exposure through those base models. No such shortcut is used.

## Provider and seed results

| Provider | Clean spell delta | Feed-like spell delta | Clean D-F1 delta | Clean macro delta |
| --- | ---: | ---: | ---: | ---: |
| Metrica | +0.003650 | +0.002982 | +0.010322 | +0.005611 |
| Sportec | +0.021250 | +0.021665 | +0.025111 | +0.020724 |
| SkillCorner | +0.009330 | +0.013503 | +0.032294 | +0.014890 |

| Member | Adopted clean | Winner clean | Adopted feed-like | Winner feed-like |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 0.635521 | 0.643860 | 0.501847 | 0.509778 |
| 2027 | 0.637245 | 0.647564 | 0.495759 | 0.506091 |
| ensemble | 0.647215 | 0.658625 | 0.505811 | 0.518527 |

Metrica feed-like macro and D-F1 decline by 0.003924 and 0.007630, while its feed spell improves 0.002982. This is a measured tradeoff, not a universal gain in every metric. The declared clean provider guards pass without exceptions.

| View | D precision | D recall | D F1 | Predicted spells, adopted → winner | Short-spell recall, adopted → winner |
| --- | ---: | ---: | ---: | --- | --- |
| clean | 0.885894 | 0.863596 | 0.872752 | 360.667 → 327.889 | 0.402475 → 0.362061 |
| feed_like | 0.843412 | 0.877506 | 0.859048 | 318.444 → 282.333 | 0.217097 → 0.198820 |

These counts are realization → match → equal-provider means, not raw summed counts. Short-spell recall is complete-spell recall below 3 seconds at IoU 0.5. Reference spell count is unchanged at 404.333 per provider-weighted match mean; every candidate keeps its original graded frames and reference spells. Full diagnostic values are in the finalist rows and aggregate.

## Selection robustness and evidence limits

Select using only two providers' feed scores and guards, then apply the selected setting to the omitted provider:

- feed_spell_push_v1: 0.651934 clean / 0.513611 feed-like; feed delta +0.007800 versus adopted.
- feed_spell_refine_v2: 0.654099 clean / 0.510673 feed-like; feed delta +0.004862 versus adopted.

Both checks are positive versus adopted, but the full-cohort optimum remains more optimistic. The base models exclude the evaluation provider from both heads and normalization. The postprocessing search globally tunes on five repeatedly exposed development matches: Metrica 3, Sportec J03WOY, and SkillCorner 1996435/1996436/2006229. The second round reuses that same information. Even the two-provider selection check has historical research exposure and is not fresh confirmation.

The input probabilities are exact hash-verified restores of the reconstructed D2 controls, not recovered original D2 training bytes. The [reconstruction report](DENSE_TARGETED_FOLLOWUPS.md) preserves the per-game serialization discrepancy. Clean and the two frozen synthetic feed-like corruptions use identical native timestamps, masks, half/time breaks, grading population and official evaluator. Inference accepts only probabilities, time breaks and fps; truth labels enter scoring afterward. External selection uses no truth-derived inference mask, provider-specific rule, or organizer reference scoring.

The [feed defect comparison](STRESS_COMPONENT_ABLATION.md) motivates feed-like over composite stress. It does not establish synthetic-to-real equivalence. Under the user's proxy assumption this is a useful benchmark improvement; full-match competition F1 and performance on independent dense matches remain unknown. Earlier public diagnostics belong to their frozen candidates and do not score this new rule.

## Replay, incident and checks

Restore verifies all **51** probability/checkpoint files by SHA-256. The broad runner reproduces 45 adopted paths and their full official scores, verifies native member alignment and exact probability means, then reproduces all 15 prior feed-leader cells before scoring. The refinement replays all 15 broad-winner paths. Every finalist is checked through the full official wrapper. Separate scalar implementations replay all 286 arm summaries, feed rankings, guards, raw grading populations and two-provider selection. No arm or threshold was changed after candidate scoring within either declaration.

The first refinement execution stops before candidate scoring because the shared aggregate omits `graded_frames`, which the replay incorrectly tried to read. The failed lock, declaration and runner are preserved. An explicit [execution amendment](results/feed_spell_refine_v2_execution_amendment.json) removes only that unsupported aggregate lookup in memory, verifies graded-frame/reference counts on all 15 raw paired control cells, and uses a new `corrected/` output directory. Per-candidate raw population checks, exact paths, all parameters, tolerances and score guards remain unchanged. The earlier broad scalar-readout implementation had the same unsupported lookup; it was corrected before producing its receipt, with no rescoring.

All **519 experiment tests**, 26 integrity/gate tests, the synthetic file-backed fit contract and the bundled submission-format check pass. Tests include team swaps, period isolation, unavailable classes, member-order invariance, input immutability, single-member calibration-order equivalence and reference replay. CSV generation and delivery evidence are recorded below when complete.

## Reproduction and artifact state

Frozen declarations and results must not be overwritten or rerun into their completed destinations:

```sh
.venv/bin/python -m unittest experiments.tests.test_feed_spell_push experiments.tests.test_feed_spell_refine -v
.venv/bin/python -m experiments.feed_spell_predict --round broad --models-dir ../mtah-data/models/joint_gru_sn_dense_ensemble_v1 --output data/experiments/feed_spell_new.csv
python evaluation/validate_submission.py data/experiments/feed_spell_new.csv
```

The broad execution is `experiments.feed_spell_push_run freeze/run`; the refinement declaration is `experiments.feed_spell_refine_run freeze`, with the reviewed correction in `experiments.feed_spell_refine_runtime`. Restore dependencies before any new reproduction: private `research/dense-targeted-archive-v1`, commit `64b6eaafef0f25bd3229aae2021efc9e13a8a1c9`, manifest `dense_targeted_followups_cache_manifest_v1.json`; the pooled deployment uses the versioned dense baseline manifest. The new archive records its own restore mappings and these dependencies rather than duplicating old raw inputs or weights.

Evidence: [broad declaration](results/feed_spell_push_v1_declaration.json), [broad aggregate](results/feed_spell_push_v1_aggregate.json), [broad scalar replay](results/feed_spell_push_v1_readout.json), [refinement declaration](results/feed_spell_refine_v2_declaration.json), [refinement aggregate](results/feed_spell_refine_v2_aggregate.json), [refinement scalar replay](results/feed_spell_refine_v2_readout.json), [restore receipt](results/feed_spell_push_v1_restore.json).

The next useful distinction is whether this fixed rule transfers to fresh dense matches or new tracking-defect realizations, followed by diverse base-model ensembling with valid nested folds. Further fine temperature tuning on these five matches has just failed to improve the feed objective. This is a candidate for independent confirmation, not a ceiling on attainable spell F1.

## Full-length candidate

The opt-in f062 generator completed feature-only inference from the exact four hash-verified pooled deployment checkpoints. Its **199,319 ordered rows pass the official format validator**. The full CSV is private at `data/experiments/feed_spell_push_v1_candidate.csv`; the [prediction receipt](results/feed_spell_push_v1_prediction_receipt.json) binds inputs, source, model, selection and output hashes. No organizer reference labels were loaded or scored.

## Three further iterations requested by the user

Before any new public scoring, the user requested several more improvement rounds and a public diagnostic. Three further frozen rounds reuse the exact native cohort and the same feed-first ranking/guards:

| Round | Fixed configurations | Mechanisms | Best clean / feed-like spell | Decision |
| --- | ---: | --- | --- | --- |
| 3 | 91 | Conditional-entropy D transforms, separate H/A and D logit smoothing, uncertainty-dependent edge costs | 0.658625 / 0.518527 | Retain original temperature-4 winner |
| 4 | 78 | Blend dense and archived combined-objective recipes: H/A-only, D-only, joint, arithmetic/geometric; seed-weight alternatives | 0.658625 / 0.518527 | Retain original temperature-4 winner |
| 5 | 67 | Consensus emissions, exact calibrated H/A/D minimum durations, capped H/A log odds | 0.658625 / 0.518527 | Retain original temperature-4 winner |

These 236 further configurations add 3,540 ensemble cells and 180 component-seed cells. Across all five rounds, **522 declared configurations** (including repeated controls) complete **7,830 ensemble cells and 360 component-seed cells**. None of the last three rounds improves the feed-first winner. Every round reproduces all 45 f062 paths before candidate scoring and separately replays its scalar hierarchy, guards, feed ranking and provider selection.

The recipe-diversity round restores **30 exact native probability arrays** from the pinned `objective_alignment_cache_manifest.json`. All learned components in those old folds exclude the evaluated provider. Their simple 50/50 combined-objective ensemble reproduces **0.633205 clean / 0.498830 feed-like** exactly on the current grading cohort before any blending. These old NPZs contain probabilities and paths but do not carry timestamps/masks; provenance is inherited from the sealed study and archived input sources, and is checked by native shape, hashes and the full historical score replay. This limit is explicit rather than asserting metadata that the files do not contain.

Further frozen evidence: [round 3 declaration](results/feed_spell_iteration_r3_declaration.json), [aggregate](results/feed_spell_iteration_r3_aggregate.json), [scalar replay](results/feed_spell_iteration_r3_readout.json); [round 4 declaration](results/feed_spell_iteration_r4_declaration.json), [aggregate](results/feed_spell_iteration_r4_aggregate.json), [scalar replay](results/feed_spell_iteration_r4_readout.json); [round 5 declaration](results/feed_spell_iteration_r5_declaration.json), [aggregate](results/feed_spell_iteration_r5_aggregate.json), [scalar replay](results/feed_spell_iteration_r5_readout.json).

## Fixed public diagnostic after external work

Only after all five external rounds and scalar replays finish, seal two predeclared predictions: the broad winner and the strongest guard-passing external winner across all completed rounds. Rank externally using feed-like spell, then macro; ties preserve the earlier candidate. Generate and validate full CSVs, verify the adopted deployment CSV byte for byte, and verify the broad candidate against its earlier sealed CSV. Claim a distinct exclusive public gate before reading the reference. Score duplicate CSV hashes once; reuse the saved adopted baseline. No tuning, repair, selection or promotion follows public access.

The broad and final externally selected winners are identical, so **1 unique CSV** is scored once. The gate is permanently closed. The reference and grading population are unchanged: 7,500 frames and 17 reference possessions.

| Fixed prediction | Public spell F1 | Public macro F1 | D-F1 |
| --- | ---: | ---: | ---: |
| Adopted (saved score) | 0.630303 | 0.851387 | 0.843424 |
| Temperature-4 winner | 0.630303 | 0.864181 | 0.873502 |

The public spell delta is **+0.000000**. This public outcome accompanies the positive external result; it does not change the pre-public selection. The diagnostic is a small exposed window, and its bootstrap interval is conditional on that window, not independent dense confirmation or a full-match performance estimate. The candidate's public 5th–95th percentile interval is `[0.46060606060606063, 0.7878787878787878]`. Separate intervals are not a paired significance test.

[Public aggregate](results/feed_spell_public_v1.json); the complete prediction/model/input/source seal and permanent gate remain private.

The public candidate improves macro F1 by +0.012794 and D-F1 by +0.030078. It predicts 16 public possessions against 17 references, with the same matched counts as the saved adopted score at every official IoU threshold (12/12/12/9/7). This tie is not a claim that the frame predictions are identical.

## Delivered code and private restore

Code, frozen results and the five-round report are merged in [public PR #58](https://github.com/jekkiechan/more-than-a-hack-2027/pull/58), with successful CI. Source/results snapshot: `68ca82884f83602e800e4f792448e6ad87b48c65`.

Private [mtah-data PR #13](https://github.com/jekkiechan/mtah-data/pull/13), commit `1dac5a00baa1328cb99c4a497758974fe2e780fe`, delivers `feed_spell_push_delivery_v1_cache_manifest.json`. A fresh remote clone and empty-destination original-path restore verified **all 1,176 entries**, 58,838,258 bytes, with no missing files, size differences or SHA-256 mismatches. No inference, fitting or rescoring occurred during verification. Organizer reference labels are excluded. The archive binds the original source/results snapshot; this later delivery receipt is separate to avoid circular hashes.

The first archive checkout was interrupted during slow per-file LFS materialization and resumed with explicit skip-smudge checkout plus batch download. Git reported an index-update warning during that recovery. Every final file was independently read and verified against the manifest, then copied and verified again in an empty destination; the repository-index warning does not substitute for a file-content check. [Remote verification receipt](results/feed_spell_delivery_v1_remote_verification.json).

The private PR's first automatic review misidentified its destination as an unverified public repository. Fresh native GitHub metadata confirmed the exact repository is private and the user has admin/push access. The same native PR action then succeeded with no visibility or audience change; no approval bypass was used.

## Published research site

The existing restricted [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/experiments.html#feed-first) is published as **version 15**, source `879cfee9aab4923ad1c55d64b3f36a9bc9f04513`. Experiments contains the five-round comparison, an interactive family selector, matched controls and the separate public diagnostic. The overview timeline and next steps reflect the fixed external winner. Desktop (1440×1100) and mobile (390×844) checks pass for both pages, all eight selector options, the existing overview chart, page overflow and browser errors. Access remains custom policy revision 2 with the same two external viewers.

The Sites helper is unavailable in this runtime, so the existing authenticated source Git/static-archive workflow was used. The first gzip-archive save stalled without a listed saved version. After reconciling state and canceling its caller, an uncompressed TAR of the same source saved and published successfully. No extra deployment or access change followed. [Publication and QA receipt](results/feed_spell_delivery_v1_site_publication.json).
