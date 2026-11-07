# Spell-balanced continuation

Date: 7 November 2026. Status: implementation and preflight.
The adopted package remains `joint_gru_sn_dense_original_diversity_phase_ensemble_v1`.

## Question and decision rule

The user asked to continue the arm that improves spell F1 in every view, allows
small macro-F1 regression, and receives public spell-F1 validation at the end.
The starting arm is the previous **Control + adopted** conditional H/A blend:
clean/v1/v2 spell **0.668249 / 0.534925 / 0.585615**. It also improves short-spell
recall in all three views; its only prior quality failure is clean macro F1,
-0.000111. See the [previous study](V2_ITERATIVE_DATA.md).

The [new parent plan](results/v2_spell_balanced_v1_plan.json) requires spell F1
strictly above the adopted equal-provider value on clean, historical v1 and v2.
It allows up to **0.001 absolute macro F1** loss in each view, a conservative
interpretation of the user's allowance. Short-spell recall must retain its
adopted value; the adopted decoded D mask is fixed. Reference spell counts must
remain invariant. Eligible candidates rank by v2 spell F1, then macro F1 for an
exact tie. Exceeding **0.60 v2** remains the goal; an incremental eligible
candidate may still be selected below it.

This is an explicitly changed rule for a new study. The previous strict-gate
failures, frozen results and active adoption policy remain unchanged.

## Development protocol and data

The same five exposed development matches produce 35 cells: Sportec J03WOY,
Metrica Game 3 and SkillCorner 1996435, 1996436 and 2006229, each with clean
inputs, two fixed v1 realizations and four fixed v2 realizations. Average
realizations within match, matches within provider, then providers equally.
Every learned component and normalizer excludes the held-out provider.

Training reuses the original 13-match TRAIN pool and 45 GAR TRAIN games with
generated dense silver labels. The four existing v2 TRAIN views, native masks,
time/half boundaries and distinct-possession sampler are retained. This is
additional optimization of existing supervision, not newly acquired gold data.
The five development matches have prior selection exposure, and synthetic v2
retains its historical model-conditioned calibration limitation.

## Round one: nearby expert mixtures

Thirty-six fixed pipelines vary the continuation expert weight (0.25, 0.5,
0.75), calibrated expert weight (0, 0.125, 0.25), arithmetic versus log-probability
pooling, and H/A transition cost (1.5 or 3). The remaining probability weight
belongs to the adopted expert. All combinations act on conditional H/A
probabilities and preserve the adopted decoded D mask.

The adopted control and the previous half-continuation/half-adopted cost-3
anchor must reproduce all 24 aggregate metrics each within 1e-10 before any new
pipeline is scored. Caching the adopted decoded mask avoids repeatedly decoding
the same control; the new helper must exactly match the shared historical
implementation before the experiment is frozen.

All pipelines use the two-seed probability ensemble and native-phase averaging.
Individual-seed diagnostics are limited to the top two eligible recipes plus
the anchor, deduplicated to at most three. These diagnostics do not rerank the
ensemble selection.

## Round two: conservative continuation

Six provider-excluded fits continue the existing CE checkpoints for another
512 updates at learning rate 0.0001. They retain the existing duration-weighted
H/A cross-entropy and live-transition objective, batch size 32 and 50/50
original/GAR mixture. D weights and normalization are unchanged. Resetting the
fixed stage seeds repeats the declared first-stage draw schedule; it does not
create fresh training-window exposure. Stage 1024 means 1,024 continuation
updates after the adopted warm-start, not the model's lifetime update count.

Only the at-most-three round-one shortlist recipes are evaluated with these
new weights. There is no second blend grid. The final candidate is selected
across both rounds under the frozen development rule, with the original anchor
available as an eligible fallback.

## Final pooled package and public diagnostic

After development selection, reproduce the chosen continuation stage using two
pooled seeds, all eligible original TRAIN sources and GAR TRAIN only. The
selected expert weights, native-phase averaging, fixed adopted D mask and H/A
cost must match the development pipeline. At most four pooled fitting stages
are permitted: two initial 512-update continuations and, only if selected, two
further 512-update stages.

Generate the complete 199,319-row CSV without public-reference access, validate
its format and seal source, input, model and prediction hashes. Then atomically
claim a distinct new public gate and score exactly one frozen candidate. Reuse
the saved adopted public baseline: **0.611765 spell / 0.849473 macro**. No
public-driven tuning, repair, reselection or automatic promotion follows.

The public window has 7,500 graded frames and 17 reference possessions and has
been used before. This requested check is a reused pipeline diagnostic, not
independent confirmation or full-match accuracy. A comparison with the saved
adopted package cannot isolate the effect of an individual training change.
GAR VALID and TEST are not used in this study.

## Verification and delivery

The new experiment will preserve frozen declarations, input/model/prediction
hashes, matched control replay, every failed check and independent verification.
Research arrays and weights stay in ignored local storage. Aggregate evidence,
tests, a PR and the existing restricted research dashboard complete delivery.
