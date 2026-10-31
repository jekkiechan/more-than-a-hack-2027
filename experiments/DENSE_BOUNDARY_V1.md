# Dense boundary decoder v1

The fixed confidence-dependent decoder did not improve possession spell F1 over exact cost-3 Viterbi on the five-match external development cohort. Clean spell F1 fell by **0.00294**, and the mean of two feed-like views fell by **0.00379**. The candidate increased recall for ground-truth alive spells shorter than three seconds by **0.01815** clean and **0.01364** feed-like, but that gain did not offset the spell-F1 declines.

The only candidate arm used the frozen formula `3 × fps/10 × (1 − sqrt(p_old,left × p_new,right))` for each directed state change; same-state transitions cost zero. The paired control called the existing cost-3 decoder on the same frame probabilities. We scored all 15 provider, match and view cells with the official evaluator. Aggregation averages realizations within match, matches within provider, then providers equally.

| View | Cost-3 spell F1 | Candidate spell F1 | Delta | Cost-3 macro F1 | Candidate macro F1 |
|---|---:|---:|---:|---:|---:|
| Clean | 0.647215 | 0.644276 | −0.002939 | 0.857677 | 0.858390 |
| Feed-like mean | 0.505811 | 0.502017 | −0.003794 | 0.815535 | 0.814888 |

The frozen study guard required clean spell improvement and no feed-like decline; both failed. The active policy’s separate feed-like tolerance of −0.01 passed, its +0.01 clean screen failed, and each provider’s clean D-F1 stayed within −0.02. This is a no-go result; no promotion is authorized. The full cell scores and hashes are in [the sealed result](results/dense_boundary_v1.json), with the guardrail wording correction in [the readout receipt](results/dense_boundary_v1_readout_correction.json).

The paired error audit of the reconstructed control found wrong-team predictions on 36.71% of reference spells of one second or less in clean views, rising to 43.42–45.17% in feed-like views. Predicted-spell merge rates rose from 8.40% clean to 13.71% and 13.62% under the two corruptions. These categories are descriptive and overlap; they do not attribute counterfactual F1 loss to a particular mechanism. The boundary decoder's higher short-spell recall therefore does not show that it fixed the more frequent wrong-team or merge errors. See [the error audit](results/dense_error_audit_v1.json).

The probabilities came from a **reconstructed D2 control** in the provider-left-out folds. Original historical fold checkpoints were unavailable, so this does not establish exact-checkpoint replay or a new independent holdout result. The external matches were previously exposed development data. The [execution binding](results/dense_boundary_v1_execution_binding.json) records the reconstruction, 15-cell control replay, source hashes and the frozen decoder declaration. Six focused tests passed for weak evidence, confident short possessions, period breaks, unknown targets and saved period metadata.

An execution note records an accidental duplicate deterministic score read: the second run reached the write-once guard after scoring and did not overwrite the single sealed report. No inputs, code or selection choices changed between invocations; the second metrics were not separately saved. See the [incident receipt](results/dense_boundary_v1_execution_incident.json).
