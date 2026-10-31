# Source-conditioned output head comparison

This experiment tests whether source-specific output projections improve
full-provider transfer while retaining shared temporal features. The finite
matrix is C/H/HD across `leave_out_sportec`, `leave_out_metrica`, and
`leave_out_skillcorner`, with seeds 2026 and 2027: 18 fits, no pooled fits.
The L calibration arm was excluded before candidate scoring because its
training-only schema evidence is specific to Sportec and does not establish
transfer to GAR, Metrica, or SkillCorner; the leave-out-Sportec fold has no
eligible Sportec calibration training matches.

C is the frozen shared-projection D2 control. H separates target and GAR H/A
projections while sharing D. HD separates both H/A and D projections. Each
head is cloned from the corresponding fold teacher initialization. H/A uses
the widened 96-unit capacity model and 0.1 input dropout; D keeps the separate
24-unit full-width GRU and the frozen player-only input mask. The extra
projections add 386 parameters in H and 435 in HD. This finite comparison has
an explicit capacity confound; parameter-matched attribution controls are
deferred to a separate study.

The reusable trainer preserves the D2 batch sampler, one-pass per-row source
routing, separate H/A and D optimizers, D2 losses and weights, clipping, and
update budget. Fit receipts record available known labels by source; H/HD
trainer logs also record sampled effective H/A and D frame counts plus
target/GAR gradient contributions to the shared recurrent encoders. GAR
silver labels are auxiliary training targets; inference always uses only the
target projections because an unseen organizer provider cannot be identified
as GAR. The GAR head is never selected at inference.

Inference applies the fold-frozen normalization, 5-second context and
10-second cores, period/time breaks, and direct plus team-swapped views. It
averages conditional H/A and D probabilities separately across the two views,
then combines them as in the frozen D2 predictor. The official scorer and
cost-3 decoder are unchanged.

## Execution and validation status

`source_conditioned_heads_execution.py` freezes source, model, prepared-data,
and cache hashes before fitting. Its compute-saving order is fixed: fit and
seal the six C controls, score them, and require the reconstructed C ensemble
to match the published D2 clean/feed-like spell scores to four-decimal
rounding precision before fitting H/HD. If that gate passes, fit the fixed 12
H/HD models, seal all 18 model hashes, then score H/HD and seal all score and
prediction receipts before aggregate recomputation. Per-seed score rows are
recomputed from sealed probability archives and cached labels. Initial C
inference parity must also pass at `2e-6` maximum absolute probability
difference on actual cached inputs. The trained C check is an explicit
reconstruction from exact fold teachers and recipe, not a claim that historical
trained fold weights were restored. Any mismatch stops before candidate fits
and scores.

Synthetic contracts cover exact teacher initialization and RNG preservation,
shared-control training parity against the frozen trainer, target-route
inference independence, source-specific gradients, player-only D behavior,
team-swap and time-break handling, and state restoration. Real-data fits and
scores are not complete until the execution receipt and report are present.
This experiment uses development matches already exposed in prior work and
does not use organizer labels or authorize promotion.
