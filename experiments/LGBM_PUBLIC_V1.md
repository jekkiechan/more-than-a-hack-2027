# Pooled LightGBM public diagnostic

12 November 2026. **Completed; keep reconstructed Fable as the adopted package.**
The requested single public diagnostic improves frame macro F1 and dead-ball F1
but reduces the primary possession-spell metric. This small reused window is
not a model-selection set; no tuning, repeat scoring or promotion follows.

## Fixed candidate and provenance

The user requested public evaluation first, with Luna subagents investigating
other experiment ideas in parallel. The starting point is
[Fable's development study](V2_TARGET_PUSH_V1.md), merged in PR #93:
`R2_full_w0.5` improves clean/v1/v2 spell by .011733/.004456/.014769,
with short recall improving in all three views. Its v1 macro retention guard
fails (−.004977 against a .001 allowance), and v1 D-F1 declines .012091.
Those failures remain unchanged.

No pooled refiner checkpoint existed. This separately declared adaptation fits
one 300-tree LightGBM with the original seed 2026 and hyperparameters on all
35 exposed development cells: five matches, clean, two v1 and four v2 views.
Every fifth native frame is sampled and X targets excluded: **532,637 rows**.
The new adapter invokes the original feature formulas separately within each
half/time block. This corrects cross-boundary context without editing the frozen
original; it is not an exact reproduction of the original LOPO candidate.

Inference blends refiner and predecessor P(D) at 50/50, applies the frozen
temperature-4/cost-3 decoder, then `snap_m0.0` and the unchanged frontier H/A
mixture. Training features use restored provider-fold upstream probabilities;
challenge inference uses the exact pooled baseline package. The distribution
change and non-nested stacking lineage remain limitations. The adaptation has
not itself received a matched clean/v1/v2 development evaluation.

The restore verifies 15 raw development inputs, 35 native probability cells,
the v2 calibration, all 160 research-archive members and ten baseline model
roles. The reference-free baseline replay matches the sealed **199,319-row CSV
byte-for-byte**. Candidate source, model, inputs and CSV were sealed before the
distinct exclusive public gate was claimed. Saved baseline scores were reused.

## Public result

Same official evaluator, 7,500 graded frames, 17 reference possessions,
1,000 bootstrap draws. Spell F1 is the mean over IoU .3/.4/.5/.6/.7.

| Metric | Adopted Fable | Pooled refiner adaptation | Change |
| --- | ---: | ---: | ---: |
| Spell F1 (primary) | .623529 | .569697 | **−.053832** |
| Macro F1 | .864234 | .886569 | +.022334 |
| 70/30 auxiliary summary | .695741 | .664759 | −.030982 |
| D precision | .818182 | .966592 | +.148411 |
| D recall | .974132 | .886317 | −.087815 |
| D-F1 | .889372 | .924716 | +.035344 |
| Predicted possessions in graded window | 17 | 16 | −1 |

At IoU .5, matches fall from 12 to 11; at .6, from 10 to 7; at .7, from 7 to 6.
Thus better frame classification does not translate into better possession
interval matching here. The candidate's spell 5–95% bootstrap interval is
[.412121, .727273], versus [.458824, .776471] for the saved baseline. These are
separate intervals, not a paired significance test. Short-spell recall was not
included in this public score summary. No full-match accuracy claim is possible.

## Decision and Luna follow-ups

The development gain remains worth investigating; the v1 regression alone was
not grounds to abandon the idea. However, neither the exposed-development result
nor this mixed diagnostic establishes that benefits outweigh regressions in
deployment. Retain the current package pending independent transfer evidence.
The public decline is not an automatic veto and cannot drive a repair or weight
search. No model or threshold changed after seeing it.

Luna agents independently reviewed transfer, alternatives, runtime and scoring
contracts. Their experiment proposals were made before the public result:

1. **Label-independent missing-ball stress.** Freeze the control, full refiner
   and missingness-ablated comparator. Use four fixed mask placements with gap
   lengths/rates matched to feed defects but assigned without labels. Keep player
   tracking fixed, recompute ball features, preserve unknowns and time breaks,
   and report provider spell, macro, D precision/recall and short recall. This is
   the highest-value next experiment because missingness ablation removes about
   two thirds of the original v2 gain.
2. **Localize the historical v1 regression on development only.** Decompose
   saved paired outputs by provider/class, D false positives/negatives and run
   boundaries. Only if a consistent mechanism appears should one separately
   declared correction be tested. Avoid another blend sweep: .25 already fails
   v1 macro and v2 short recall.

These are proposals, not completed experiments. The public diagnostic took
priority as requested. Exact-package GAR TEST10 remains unrun for this refiner;
any later use must disclose repeated silver-label exposure. Independent
continuous-match annotation remains the stronger transfer check.

## Reproduction and evidence

The runner is [lgbm_public_v1.py](lgbm_public_v1.py), with the isolated adapter
[lgbm_public_runtime.py](lgbm_public_runtime.py). Its sequential stages are
`declare`, `prepare`, `fit`, `components`, `predict`, `public`; completed stages
and gates must not be rerun or overwritten. Source hashes bind this execution.
Private inputs, fitted weights, training matrices and frame outputs remain under
ignored `data/experiments/lgbm_public_v1/` and authorized private storage.

- [Declaration](results/lgbm_public_v1_declaration.json)
- [Restore verification](results/lgbm_public_v1_restore.json)
- [Single-fit seal](results/lgbm_public_v1_fit.json)
- [Prediction seal](results/lgbm_public_v1_prediction_seal.json)
- [Closed public gate](results/lgbm_public_v1_public_gate.json)
- [Official public aggregate](results/lgbm_public_v1_public.json)

Nine new synthetic runtime/gate tests pass: block isolation, label independence,
team symmetry, feature ordering, seal integrity and gate-before-reference access.
The full suite initially found three missing-PyArrow errors; installing the
repository-pinned dependency resolves that environment prerequisite. The repeated suite then exposed two inherited dangling test-created links;
only those broken setup links were removed. Final validation: **987 experiment
tests, 986 pass and one skip; all 26 root tests pass**, exact file-backed fit
parity, submission format and documentation checks pass. See the
[verification receipt](results/lgbm_public_v1_verification.json).


## Delivery

[PR #94](https://github.com/jekkiechan/more-than-a-hack-2027/pull/94) publishes
the implementation and aggregate evidence. The
[private delivery receipt](results/lgbm_public_v1_private_delivery.json) records
a fresh isolated remote restore of all 369 archive members (45.7 MB archive)
on private branch `research/lgbm-public-v1`; private main is unchanged.

The [research progress site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site)
is published as version 44 with its restricted audience unchanged. Overview,
timeline and Research journal distinguish this diagnostic from adopted Fable
and the original development study. Desktop/mobile layout, anchor navigation
and study search pass. [Publication receipt](results/lgbm_public_v1_site_publication.json).
