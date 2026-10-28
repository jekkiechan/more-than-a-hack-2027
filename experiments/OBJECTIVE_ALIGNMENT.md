# Possession-aware objective study

Status: completed external development comparison. Date: 2026-10-28.

**Decision: retain `joint_gru_slotdrop_v1`.** All three loss recipes pass the declared robustness and provider frame-quality guardrails in both seeds. No recipe reaches the predeclared clean spell-F1 gain of at least .01 in both seeds. The combined recipe improves clean and feed-like spell F1 in both seeds and is a promising confirmation candidate; its seed-2027 clean gain is .00895368, below that screen.

The comparison uses five previously exposed external matches and full provider holdouts for H/A, D and normalization. It is not independent confirmation or a full organizer-match result. The two seeds are separately evaluated models; no seed ensemble is evaluated here. The adopted package and organizer scoring gates are unchanged.

## Headline scores

Within each view, average stress replicates within match, matches within provider, then providers equally. Rank on unrounded official spell F1 across IoU .3/.4/.5/.6/.7, with macro F1 for exact ties. The fixed 70/30 summary is auxiliary.

### Native clean

| Loss | Seed | Spell F1 | Macro F1 | D F1 | D precision | D recall | 70/30 auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 2026 | 0.60876 | 0.84343 | 0.83378 | 0.89469 | 0.78638 | 0.67916 |
| Control | 2027 | 0.61760 | 0.85434 | 0.85425 | 0.88620 | 0.82771 | 0.68862 |
| Duration weights | 2026 | 0.62681 | 0.84284 | 0.83077 | 0.89891 | 0.77732 | 0.69162 |
| Duration weights | 2027 | 0.62188 | 0.85451 | 0.85488 | 0.89533 | 0.82101 | 0.69167 |
| Transition auxiliary | 2026 | 0.61864 | 0.84842 | 0.83958 | 0.88979 | 0.80090 | 0.68758 |
| Transition auxiliary | 2027 | 0.61537 | 0.85552 | 0.85775 | 0.89149 | 0.82802 | 0.68742 |
| Combined | 2026 | 0.62642 | 0.84608 | 0.83537 | 0.90148 | 0.78258 | 0.69232 |
| Combined | 2027 | 0.62655 | 0.85551 | 0.85533 | 0.89810 | 0.81897 | 0.69524 |

### Feed-like stress

| Loss | Seed | Spell F1 | Macro F1 | D F1 | D precision | D recall | 70/30 auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 2026 | 0.47424 | 0.80125 | 0.83143 | 0.89035 | 0.78558 | 0.57234 |
| Control | 2027 | 0.48530 | 0.81087 | 0.85004 | 0.88207 | 0.82380 | 0.58297 |
| Duration weights | 2026 | 0.48355 | 0.79976 | 0.82738 | 0.89490 | 0.77466 | 0.57841 |
| Duration weights | 2027 | 0.48775 | 0.80977 | 0.85043 | 0.89017 | 0.81723 | 0.58435 |
| Transition auxiliary | 2026 | 0.48021 | 0.80506 | 0.83602 | 0.88333 | 0.79987 | 0.57766 |
| Transition auxiliary | 2027 | 0.48785 | 0.81345 | 0.85374 | 0.88423 | 0.82695 | 0.58553 |
| Combined | 2026 | 0.48724 | 0.80407 | 0.83319 | 0.89625 | 0.78317 | 0.58229 |
| Combined | 2027 | 0.49157 | 0.80973 | 0.85002 | 0.88995 | 0.81606 | 0.58702 |

## Frozen screens and guardrails

For each seed: clean spell gain >= .01; feed-like spell delta >= -.01; every provider clean macro F1 and D-F1 delta >= -.02. A recipe must pass in both seeds. No threshold, objective weight, epoch, seed, or decoder setting changed after scoring.

| Loss | Seed | Clean spell delta | Feed-like spell delta | Worst provider clean D/macro delta | Clean screen | Guards |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Duration weights | 2026 | +0.01805 | +0.00931 | -0.00410 | pass | pass |
| Duration weights | 2027 | +0.00428 | +0.00244 | -0.00424 | fail | pass |
| Transition auxiliary | 2026 | +0.00988 | +0.00597 | +0.00159 | fail | pass |
| Transition auxiliary | 2027 | -0.00223 | +0.00255 | -0.00105 | fail | pass |
| Combined | 2026 | +0.01766 | +0.01301 | -0.00356 | pass | pass |
| Combined | 2027 | +0.00895 | +0.00627 | -0.00324 | fail | pass |

All six candidate/seed rows improve feed-like spell F1. Duration weighting and combined pass the positive clean screen only at seed 2026. Transition alone is seed-inconsistent on clean spell F1. The .00895368 combined gain at seed 2027 does not become a pass by rounding or averaging seeds.

## Where the gains and regressions occur

The combined recipe gains clean spell F1 on Sportec and Metrica, while SkillCorner declines. Its provider D/macro guardrails still pass; provider spell regressions are a separate diagnostic and must remain visible.

| Combined seed | Provider | Clean spell delta | Feed-like spell delta | Clean D-F1 delta | Clean macro delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2026 | sportec | +0.02318 | +0.01717 | +0.00685 | +0.00730 |
| 2026 | metrica | +0.03951 | +0.02868 | -0.00356 | +0.00210 |
| 2026 | skillcorner | -0.00971 | -0.00684 | +0.00148 | -0.00146 |
| 2027 | sportec | +0.02609 | +0.01896 | -0.00197 | +0.00263 |
| 2027 | metrica | +0.01350 | +0.00544 | +0.00366 | +0.00413 |
| 2027 | skillcorner | -0.01273 | -0.00558 | +0.00154 | -0.00324 |

Duration weighting raises clean short-spell recall by .0656/.0483 and feed-like recall by .0236/.0216 across the two seeds, while increasing predicted and fragmented spells. Transition alone has no consistent short-spell benefit. Combined retains a smaller recall gain while improving clean F1 at every official IoU threshold in both seeds.

| Combined seed | Clean short-recall delta | Feed short-recall delta | Clean F1@.7 delta | Feed F1@.7 delta | Clean turnover F1 delta | Feed turnover F1 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | +0.05399 | +0.02470 | +0.02159 | +0.01778 | +0.02421 | +0.00875 |
| 2027 | +0.03821 | +0.01510 | +0.01081 | -0.00006 | +0.01949 | +0.00740 |

Short means <3 seconds and recall uses official global one-to-one matching at IoU .5. Turnover F1 uses the saved requested .5-second timing diagnostic with native-frame rounding. These supporting diagnostics do not replace the frozen adoption screen.

Combined increases false possessions per minute on clean SkillCorner by about .43/.53, and its short-recall gain there is nearly flat. The overall recall improvement is concentrated in the more fully observed sources. SkillCorner has many spells touching graded-run edges, so short-spell and boundary findings include censored observations. Live/dead timing improvements are mixed.

## Matched design and losses

The user requested trials with Luna subagents at xhigh. Luna agents implemented the losses and runner and independently reviewed protocol, masks, schemas and the readout. The study restores six sealed slot-dropout controls and fits 18 new models: three loss variants x three whole-provider holdouts x seeds 2026/2027. The same 13 external TRAIN matches supply fold-specific training, with five development matches used for selection evidence. All learned H/A, D and normalization components exclude the evaluation provider.

The architecture is the unchanged factorized bidirectional GRU, hidden 24, with 62,643 trainable parameters and a strictly player-only D path. Inputs are 5 Hz, with 10-second output cores and five seconds of context on either side. Eight epochs, batch 64, AdamW .001/.0001, clipping 1, source/match weights, H/A symmetry, original clean/legacy plus 25% whole-window ball-gap replacement and player-slot dropout, train-only normalization, and Viterbi cost 3 remain fixed.

| Held-out provider | Training records | Windows/epoch | Optimizer updates/seed |
| --- | ---: | ---: | ---: |
| Sportec | 40 | 23,748 | 2,976 |
| Metrica | 44 | 25,976 | 3,248 |
| SkillCorner | 20 | 11,396 | 1,432 |

**Duration weighting:** decompose categorical NLL into D/live gate loss and conditional H/A loss; reweight only the latter. At native resolution, complete H/A runs receive raw `clip(sqrt(5 seconds / duration), .5, 2)`. Runs touching unknown labels or match/half/time-block edges stay neutral. Normalize eligible sampled live multipliers to mean one within each record; D, censored-live and total eligible H/A mass stay fixed. Effective normalized weights can exceed the raw caps. Unknown targets retain zero loss weight.

**Transition auxiliary:** add a fixed lambda=.1 cost-sensitive probability surrogate. One component predicts D/live change from player-only P(D); the other predicts H/A change conditional on both labels being live. Each component has a fixed half share, and stable/change groups each have a fixed half share; absent groups or components contribute zero. Require every native target between sampled endpoints to be known within one continuous block, assign each edge to its left output-core frame, and use right context at core seams. There is no added head. The adjacent-marginal surrogate is uncalibrated and does not implement greedy spell matching. The combined arm adds both mechanisms.

## Preparation and runtime provenance

The 253-file restored GRU archive contains the controls and evaluation data but omits training feature caches. This study regenerates 13 caches under a separately frozen preparation declaration, using the exact original feature/view/seed recipe. Historical directories and declarations remain unchanged. All six control medians, scales, feature schemas, parameter counts, windows and optimizer budgets match exactly before candidate fitting. The archived source inventory differs only in the three later-promoted working-baseline files.

Evaluation reuses 15 verified native/feed-like feature records from the prior seed/context archive. Feed-like retains ball gaps, coordinates, player-slot gaps and same-team ID swaps, with the fixed corruption seeds/calibration. Composite stress is not evaluated in this study.

The first three fits use in-memory normalization (SkillCorner duration seed 2026, Sportec duration seed 2026, SkillCorner combined seed 2027). A separate runtime addendum, introduced after those fits began and before comparative scoring, maps the unchanged normalizer outputs through read-only temporary `.npy` files for the other 15 fits. Exact full model-state/history/prediction parity is tested for all three losses, with non-neutral spell weights. Every adapted fit has a hash-bound sidecar verifying normalization, budgets, call count, restoration and cleanup. Concurrent scheduling affects wall-time metadata; it is not an efficiency benchmark.

The shared preparation API landed on main during training. Its recipe uses different augmentation seeds, so this frozen comparison preserves its original preparation for exact control parity, as the new guide requires. Future studies should use the [shared planner/cache API](preparation.py) and its [guide](../.agents/skills/possession-preparation/SKILL.md). Merging main preserves all bound study sources and the adopted baseline.

## Verification, artifacts and next step

All 170 integrated synthetic/contract tests pass, and the bundled 199,319-row submission passes the official format validator. All 18 candidates were sealed before the 120 evaluations (90 candidate cases plus 30 restored-control cases). Independent read-only verification passes: raw/cache/code/model seals, training-only labels and symmetry, budgets, all saved predictions and decoded outputs, official scores, every control parity cell, provider aggregation, all IoUs and screen decisions.

Models, training caches and frame outputs remain in ignored/private storage. Remote archive verification is pending; the artifact receipt will record the exact private revision, manifest and fresh-restore result. No organizer labels were read or scored and no baseline package changed.

The next useful test is independent labeled broadcast confirmation of the combined recipe, with explicit attention to SkillCorner false possessions. Preserve the both-seed screen; further loss or decoder tuning on these five exposed matches would need a new declared development comparison.

## Evidence

- [Frozen study declaration](results/objective_alignment_declaration.json).
- [Frozen preparation declaration](results/objective_alignment_preparation.json).
- [Exact training parity](results/objective_alignment_training_parity.json).
- [Runtime addendum](results/objective_alignment_runtime.json).
- [All 120 rows, per-provider scores and diagnostics](results/objective_alignment_development.json).
- [Independent verification](results/objective_alignment_verification.json).
