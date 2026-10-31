# Dense gap memory v1

Status: completed; no-go on the predeclared clean screen. The pre-score scope
correction is recorded in
`experiments/results/dense_gap_memory_v1_amendment_pre_score.json`; aggregate
readout is `experiments/results/dense_gap_memory_v1_readout.json`.

## Declared question and method

The hypothesis is that a last reliable H/A control estimate, taken only from
observed-ball inference probabilities, can help the dense ensemble maintain
team identity through missing-ball gaps. The frozen, label-free rule records an
anchor when `max(P(H), P(A)) >= 0.80` and the H/A margin is at least `0.20`.
During a ball gap it adds a team-symmetric log-odds bias capped at `0.25`, with a
five-second half-life. It transfers probability only between H and A; D/X and
observed-ball rows are unchanged. A period or time break clears memory.

The comparison uses reconstructed current dense controls, each seed and their
equal-probability ensemble, for the same provider-held-out matches under clean
and feed-like views. The reconstruction is expressly labeled; it is not claimed
to be byte-identical to original holdout checkpoints. The parent-authorized
control rebuild must pass its replay and source/input audit before the frozen
candidate transform runs. There are no candidate model fits, tuning on
organizer labels, promotions, or changes to the completed dense study.

The producer audit confirms the original trained D2 fold binaries were
unavailable. It reconstructed controls from the exact raw TRAIN source, source
modules, frozen fold teachers, and declared D2 recipe. Byte identity with the
historical fold checkpoints cannot be established. The separate control replay
matched its saved clean/feed-like aggregate with maximum error about `2.5e-11`,
inside the predeclared `5.1e-5` tolerance. The provenance receipt hash is
`90ce3366a85b1377d6c136af90f335c37f1825be247c231f48264483ad07c7f2`.

The runner requires a complete first pass across all 15 match/view/replicate
cells (five matches, three views), covering 45 member rows across two seeds
and their ensemble. It compares each reconstructed seed and equal-probability
ensemble to the saved control predictions and metric rows before it starts a
second pass that applies the candidate. Each candidate score is paired with its
replayed control, and only aggregate cell JSON is written. A later invocation
verifies the sealed replay receipts and their inputs, then continues candidate
scoring without rescoring control metrics.

The pre-score readout screen is an equal-provider clean ensemble spell-F1 gain
of at least `0.01`, feed-like ensemble spell-F1 delta no worse than `-0.01`,
and per-provider clean macro-F1 and D-F1 deltas of at least `-0.02` for every
seed and the ensemble. It guides interpretation only and grants no promotion.

Run after the control artifacts pass their separate audit:

```bash
python -m experiments.dense_gap_memory_v1_run \
  --prediction-root data/experiments/dense_label_weight_v1/predictions \
  --member-scores experiments/results/dense_label_weight_v1_member_scores_control_D2.json \
  --ensemble-scores experiments/results/dense_label_weight_v1_ensemble_scores_control_D2.json \
  --provenance experiments/results/dense_label_weight_v1_control_prediction_audit.json
```

The dense model emits H/A/D probabilities. The helper accepts that three-class
shape directly (and also supports H/A/D/X for contract checks), preserving D
and any X column exactly. The caller must provide the evaluator's native period
IDs, including its time-break splits. The unchanged baseline decoder remains
responsible for mapping probabilities to predictions; baseline probability and
prediction replay is a precondition for candidate scoring.

For each saved control bundle the runner retains the original provider `period`
IDs and timestamps for official scoring. It separately derives
`supervised.model_periods(inputs)` for memory resets and decoding, so a native
period ID such as 1 is not mistaken for a model block ID or invalid row.

This is distinct from the earlier Smart Tagging study: it has no event tags or
conditional possession evidence, and only carries an inference-time model
control estimate from observed-ball rows. Smart Tagging's fixed public
diagnostic declined spell F1; its outcome does not supply efficacy evidence for
this provider-held-out dense-model test.

The PR #35 label-free follow-up also found no-go results for context-conditioned
gaps, possession-rate diagnostics, and target adaptation against the earlier
combined-objective package. It reported under-segmentation and that the model
already holds one team through many gaps. That weakens the broad premise that
gap handling itself is the main spell-F1 lever. This study asks the narrower,
unanswered question whether the dense model's own last reliable control
probability carries useful evidence on its current matched holdout; it does not
use those earlier unmatched scores as a control.

## Results and limits

The memory candidate misses the +0.01 clean ensemble spell-F1 screen. The
feed-like guard and every provider's clean macro-F1/D-F1 guard pass.

| Member | Clean spell F1 | Δ vs control | Feed-like spell F1 | Δ vs control |
|---|---:|---:|---:|---:|
| Seed 2026 | 0.636252 | +0.000730 | 0.502275 | +0.000428 |
| Seed 2027 | 0.636994 | −0.000251 | 0.493456 | −0.002303 |
| Equal-probability ensemble | 0.647582 | +0.000367 | 0.504678 | −0.001133 |

The ensemble control was 0.647215 clean and 0.505811 feed-like. The worst
per-provider clean macro-F1 or D-F1 delta was −0.000162 (Metrica D-F1), above
the declared −0.02 floor. Keep this as a no-go; the small clean change does not
justify a follow-up sweep or promotion.

The inference helper and runner's nine contract tests pass: invalid inputs are
rejected, the rule is team-symmetric, effects are limited to gaps, and memory
clears at period breaks or unreliable observed-ball frames. This evaluation is
still limited to five previously exposed development matches and reconstructed
controls; it is not fresh independent confirmation. No organizer labels were
read, and no candidate model was fitted or promoted.
