# Dense ensemble targeted follow-ups

Status: running. Started 31 October 2026 against fork main at `1b06133`.

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

## Pending results

The three model comparisons and current error audit are not yet scored. Data
restoration, contract checks, matched control reconstruction, candidate execution
and independent verification must finish before a research conclusion is drawn.
