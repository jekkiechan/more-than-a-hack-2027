# Subsecond possession team attribution

Status: implementation and verification in progress, 31 October 2026.
Study: `dense_short_team_v1`, based on fork main `09cbce3`.

## Question and comparison

Does increasing the training contribution of complete H/A possessions lasting
at most one second improve team attribution and overall possession spell F1?

The adopted dense model already uses capped square-root duration weights for
complete possessions. This study adds one fixed multiplier of two specifically
to complete subsecond H/A spans, on both original-provider and SoccerNet TRAIN
records. The multiplier acts only on conditional H/A cross-entropy. Each
window divides by the multiplier mass over known H/A frames; the existing
duration factors stay in the numerator. With the multiplier disabled, the
training path must reproduce D2 loss and model updates.

Spans touching a record boundary, time discontinuity or unknown target are
censored and receive no extra weight. Eligibility uses native labels and frame
rates before mapping to the sampled training grid. H and A follow identical
rules. Evaluation and inference receive no eligibility or duration targets.

The architecture, sampler, training budget, D loss, transition loss and cost-3
decoder stay fixed. Six candidate fits cover three full-provider exclusions
and seeds 2026/2027, each with 2,048 updates. The six reconstructed D2 controls
are restored from the previous targeted-study archive, without refitting.
Seeds are also averaged as native probabilities before decoding.

## Evaluation and stop rule

The paired cohort contains five previously exposed development matches and
three views per match: clean and two fixed synthetic feed-like corruptions.
All learned components and normalization exclude the evaluated provider.
Official spell F1 averages matching F1 across IoU thresholds 0.3–0.7. Macro F1
breaks exact ties; the 70/30 summary is descriptive.

The existing +0.01 clean screen, feed-like floor of control minus 0.01, and
provider clean macro/D-F1 floors of control minus 0.02 remain unchanged.
Diagnostics distinguish wrong-team frame errors within ≤1-second reference
spells from complete ≤1-second spell recall at IoU 0.5 and the established
<3-second recall metric. Better short recall alone is not an upgrade.

No weight, threshold or seed sweep follows a negative result under this study
ID. Another experiment requires a distinct hypothesis and declaration. Prior
decoder, contact, native-resolution and boundary-weighting failures are recorded
in the [experiment index](../docs/research/EXPERIMENTS.md#settled-directions).

## Evidence limits

This is development evidence on reused matches, with synthetic stress and
SoccerNet-derived silver labels. The controls are the verified reconstruction
described in [the targeted follow-ups](DENSE_TARGETED_FOLLOWUPS.md), not the
unavailable original research checkpoint bytes. No organizer labels are read,
no public scoring gate is reopened, and this study does not promote a model.

## Results and delivery

Pending the fixed comparison and verification. The adopted package remains
`joint_gru_sn_dense_ensemble_v1`.
