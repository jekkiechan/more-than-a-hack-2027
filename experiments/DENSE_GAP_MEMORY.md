# Dense gap memory v1

Status: mechanism and contract tests implemented; provider-holdout evaluation
awaits the declared reconstruction and replay of current dense control fits.
The pre-score scope correction is recorded in
`experiments/results/dense_gap_memory_v1_amendment_pre_score.json`.

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

No holdout scores have been computed. The current workspace initially had no
dense provider-held-out arrays or saved control probabilities. The private
cache manifests contain pooled baseline checkpoints and explicitly exclude raw
SoccerNet data and frame outputs. The parent has authorized a separately
declared reconstruction of the current dense controls; those controls need
parity validation before candidate scoring. Pooled deployment weights do not
serve as holdout evidence.

The inference helper and runner's nine contract tests pass: invalid inputs are rejected,
the rule is team-symmetric, effects are limited to gaps, and memory clears at
period breaks or unreliable observed-ball frames. These tests do not establish
holdout efficacy.
