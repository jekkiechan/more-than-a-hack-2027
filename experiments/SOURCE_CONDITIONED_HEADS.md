# Source-conditioned output heads: bounded prototype

## Question

Can dense SoccerNet-GAR silver supervision update shared temporal features
without forcing its event-team labels and missingness pattern through the same
H/A/D output calibration used for the labeled target-domain matches?

This is a proposed experiment design, not a fitted result. The tensor prototype
in [`source_conditioned_heads.py`](source_conditioned_heads.py) shares a
bidirectional H/A encoder and a separate player-only D encoder across `target`
and `soccernet` supervision. The `H` configuration separates H/A projections
and keeps D shared; `HD` separates both. Target and GAR projections start from
cloned weights. H/A and D widths are independently configurable, allowing
baseline-aligned H=96/D=24. Training must explicitly route target records to `target`, and
GAR records to `soccernet`. There is not yet a full sequence inference wrapper.
A future runner must select `target` for every deployment input; provider
identity is not an inference feature. There is no organizer-provider-specific
head or lookup. The GAR head is auxiliary and must not be selected at inference.

The D projection receives features from its separate player-only recurrent
encoder, selecting only the established D feature indices. H/A receives the
full team feature vector. Each route produces
factorized categorical probabilities: D is sigmoid(logit), and H/A divides
the remaining live mass. Normalization, window/core slicing, time-break handling
and team-swap symmetrization are not implemented in this prototype; these are
required runner contracts before a full inference comparison.

## Proposed comparison

Follow the finite C/H/HD matrix in
[`SUPERVISION_ALIGNMENT_PLAN.md`](SUPERVISION_ALIGNMENT_PLAN.md): three
full-provider folds (`leave_out_sportec`, `leave_out_metrica`,
`leave_out_skillcorner`), two seeds and three arms (18 fits total), with no
pooled fits in this phase. C is the existing shared-projection D2 control; H
separates target/GAR H/A projections and keeps D shared; HD separates both H/A
and D projections. Use the fold-specific combined-objective teachers and
width-96/24 initialization specified in that plan. If exact artifacts cannot
be restored, stop and report the blocker instead of silently substituting
pooled or adopted checkpoints. Match its update count, GAR fraction, sampling,
normalization, augmentation, loss weights, seeds and inference recipe.

Every held-out provider must be excluded from target labels, normalization and
all learned target components. The 45 GAR TRAIN fixtures remain separate.
GAR loss updates its projection plus shared features; target loss updates the
target projection plus shared features. This isolates output semantics, not
all domain effects. At H=96/D=24, the extra H/A projection costs `4*96+2 =
386` parameters; the extra D projection costs `2*24+1 = 49`, so HD adds 435
projection parameters over C and H adds 386. This capacity difference is
explicit in the plan. Do not add parameter-matched or tied-head arms to this
finite matrix; attribution controls belong in a later declaration. Record
parameter counts, head initialization, optimizer updates, effective label
counts by source and per-source gradient contribution to shared encoders.

## Label and transfer limits

GAR PASS/CROSS/SHOT event-team labels are sparse silver supervision. The dense
target stream is derived from those events, so it is not independent dense
ground truth; unknown frames stay masked. Keep GAR H/A/D semantics, masks and
time breaks as prepared, and retain the distinction between target labels and
GAR pseudo labels in the runner and report. Provider-specific missingness is a
known shortcut risk, especially for D. The player-only pathway, label masks,
whole-match/provider isolation and label-free evaluation route are required.

There is no way to know that an unseen organizer provider is GAR or which
provider it is during inference. Therefore every deployment input selects the
single pooled `target` head, and its learned target normalization. A result on
the already inspected organizer diagnostic cannot validate transfer or trigger
head selection; the diagnostic remains closed. Full-match transfer evidence
requires independently labeled complete matches and cannot be inferred from
the current short public window.

Before fitting, add runner-level contracts for target-only inference,
source-specific gradient routing, unknown masks, temporal breaks, team swap
equivariance, exact initialized-control replay, cohort completeness and saved
baseline parity. Freeze inputs, checkpoint hashes, folds, sources, seeds,
budget, guards and readout. Score all official H/A/D measures and the current
policy's clean/feed-like/provider gates; report H/A spell IoUs, D recall and
seed spread separately. No fit, evaluation, artifact restore or organizer
scoring was performed as part of this prototype.

## Current implementation boundary

The synthetic module contract covers routing, cloned initialization,
target-route independence, state-dict restoration, unknown-target masking,
player-only D inputs and independent H/D width configuration. No baseline replay
or parity test has passed yet. Runner integration remains necessary: restoring
fold teachers and prepared records; applying the plan's normalization,
batching/update schedule and exact loss; exposing C/H/HD arms; implementing
window/core and time-break handling plus team-swap symmetrization; recording
source gradient/update statistics; replaying control parity; and running the
three-fold evaluation. The existing `sn_dense_train.py` frozen runner was not
edited.
