# Dense ensemble targeted follow-ups

Status: completed 31 October 2026. Started against fork main at `1b06133`.
All three fixed comparisons miss the clean improvement screen; retain the adopted baseline.

The user requested Luna subagents to execute an error audit, missing-ball team
memory, boundary decoding, silver-label weighting, and submission verification.
The adopted package remains `joint_gru_sn_dense_ensemble_v1`. No model promotion
or organizer reference scoring is included.

## Study coordination

| Workstream | Question | Comparison |
| --- | --- | --- |
| `dense_error_audit_v1` | Which dense-model errors lose possession spells? | Read-only external provider-holdout breakdown |
| `dense_gap_memory_v1` | Does carrying the last reliable predicted control through missing-ball intervals help? | Fixed decaying H/A prior versus unchanged emissions |
| `dense_boundary_v1` | Do probability-supported transitions recover possession boundaries? | Fixed confidence-dependent decoder versus cost-3 Viterbi |
| `dense_label_weight_v1` | Does downweighting ambiguous silver-label transitions improve transfer? | Matched current training control versus one weighted-label arm |
| `dense_submission_audit_v1` | Can the actual submission be reproduced and validated? | Fresh raw-input preparation and exact sealed CSV replay |

Each experiment owns its declaration and result. The fitting workstream supplies
shared reconstructed controls so the audit and inference-only candidates do not
repeat training. Controls must pass their predeclared replay check before
candidate scores are interpreted. The independent reviewer checks contracts,
whole-provider exclusions, aggregation, and preserved evidence.

## Evidence and artifact scope

The exact four adopted deployment weights are restored and hash-verified, but
these are pooled models, unsuitable substitutes for provider-holdout models.
Original dense research fold checkpoints and native probability arrays were not
available locally or in the private deployment manifest. Recovery therefore
restores the existing six provider-excluded teacher checkpoints, old-provider
training caches and evaluation inputs, and reconstructs dense controls from the
pinned SoccerNet TRAIN source under an explicit declaration.

The five external development matches were previously exposed. Synthetic
feed-like corruption is not independent broadcast validation. All inference-only
candidates use the same reconstructed controls, masks, temporal breaks, grading
population and official evaluator. The +0.01 clean screen and existing provider
and feed-like guards remain visible even if a smaller improvement is observed.

## Completed submission verification

A fresh process rebuilt tracking inputs from raw ATD and generated all 199,319
ordered rows. The official format validator passed. The output is byte-identical
to the sealed dense candidate; inference took 82.97 seconds on this CPU runtime.
No organizer reference labels were read. See the [audit report](dense_submission_audit_v1/README.md)
and [receipt](dense_submission_audit_v1/report.json) for hashes and contract limits.

## Reconstructed training inputs

All 45 TRAIN games were rebuilt: 11,249 records and 5,063,339 native frames.
The raw-source hashes, preparation source, schema and manifest metadata match
the earlier study, but all 45 serialized game files have different byte hashes.
The cause remains unresolved; exact feature-array identity cannot be established
without the original files. Substituting only the historical game-file hashes
into the new manifest reproduces the historical manifest digest exactly.

Independent QA comparison found 273 shared values with zero differences; all
per-game record, frame and period-block counts match. This supports a declared
same-source reconstruction, not a claim of byte-identical input restoration.
The failed byte check is preserved in the
[preparation receipt](results/dense_label_weight_v1_preparation_reconstruction.json).
The [independent check](results/dense_label_weight_v1_independent_reconstruction_verification.json)
and [aggregate QA](results/dense_label_weight_v1_preparation_qa_aggregate.json)
provide public evidence. The unchanged full QA includes frame-level examples and
is kept privately for exact restore; its frozen hash is recorded in these receipts.

## Matched results

The reconstructed control passed the unchanged ±0.000051 replay gate:
clean spell F1 **0.6472145590743718**, feed-like **0.5058107894247519**.
Both differ from the recorded expected values by less than 3 × 10⁻¹¹.
The independent reviewer verified six fits, all 45 native prediction archives,
provider exclusions, probability averaging and decoder replay.

| Ensemble | Clean spell F1 | Feed-like spell F1 | Clean delta | Feed-like delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Reconstructed control | 0.647215 | 0.505811 | — | — | Matched reference |
| Fixed gap memory | 0.647582 | 0.504678 | +0.000367 | −0.001133 | No upgrade; misses +0.01 clean screen |
| Confidence boundary decoder | 0.644276 | 0.502017 | −0.002939 | −0.003794 | No upgrade; both declared spell guards fail |
| Silver-label weighting | 0.647915 | 0.505227 | +0.000701 | −0.000583 | No upgrade; misses +0.01 clean screen |

The [gap-memory report](DENSE_GAP_MEMORY.md) includes both seed members and the
probability ensemble. The small clean gain is not consistent across seeds and
comes with worse ensemble stress performance. The [boundary report](DENSE_BOUNDARY_V1.md)
shows improved recall of spells shorter than three seconds (+0.01815 clean),
but lower overall spell F1. Its frozen no-feed-decline guard fails even though
the separate active-policy −0.01 tolerance passes. The original score receipt
is preserved alongside an explicit guard-wording correction.

The [silver-label weighting report](results/dense_label_weight_v1_report.md)
records all 12 completed fits (six controls and six candidates, each 2,048
updates). Downweighting eligible transition neighborhoods to 0.25 yields a small
clean gain and a small stress decline, with no material improvement in short-spell
recall. No additional weighting values were tried.

## What the error audit adds

The [current error audit](DENSE_ERROR_AUDIT_V1.md) identifies short-possession
team attribution and robustness to missing-ball corruption as measurable targets.
Equal-provider wrong-team rates over all graded H/A/D frames rise from 5.33%
clean to 9.18–9.23% under feed-like corruption. Within GT possessions lasting
at most one second, wrong-team rates are 36.71% clean and 43.42–45.17% stressed.
The proportion of predicted spells overlapping multiple same-team reference
spells rises from 8.40% to 13.62–13.71%.

These descriptive rates are not counterfactual F1 gains. The audit's ≤1-second
frame-error bin differs from the boundary study's <3-second spell-recall metric.
Provider rates, raw counts, unmatched boundaries and denominators remain in the
linked report. All evaluation uses previously exposed development matches and
synthetic stress; no new independent broadcast labels were acquired.

## Verification and delivery

The independent review checks fit budgets, provider exclusions, source/model
hashes, identical paired evaluation rows, probability ensembling and aggregate
readouts: [review receipt](results/dense_targeted_followups_independent_review_v1.json).
CI passes 458 experiment tests, 26 integrity/gate tests, documentation links and
the bundled submission-format check. Baseline manifests, decision policy and
closed historical scoring gates remain unchanged.

Code and aggregate evidence: [PR #49](https://github.com/jekkiechan/more-than-a-hack-2027/pull/49).
Exact derived inputs, model weights, native predictions and full evaluator
outputs are archived in private [mtah-data PR #8](https://github.com/jekkiechan/mtah-data/pull/8).
A fresh isolated remote download verified all 338 manifest entries. The restore
check recovered 241 original-path files (1,277,182,412 bytes), with zero missing
files or size/hash mismatches. See the [remote verification receipt](results/dense_targeted_followups_remote_verification.json)
for pinned commits and manifest hash. The code/results snapshot is `d7bbe9d`;
this delivery receipt is added afterward to avoid circular archive hashes.
The private full boundary output includes transition locations; its public
aggregate preserves metrics and binds the original digest. Its separate
execution-incident receipt discloses an accidental duplicate deterministic
invocation that refused to overwrite the first result; no tuning or selection
followed.

The [research site](https://more-than-a-hack-progress.buffhen.chatgpt.site/experiments.html#targeted-dense)
is published as version 10 with the existing restricted audience preserved.
Desktop and mobile layouts passed overflow, anchor and browser-error checks.
