# Spell-balanced continuation

Date: 7 November 2026. Status: development complete; final validation in progress.
The adopted package remains `joint_gru_sn_dense_original_diversity_phase_ensemble_v1`.

## Development results

**The qualifying v2 leader reaches 0.591174; the >0.60 target remains unmet.**
The two rounds produce 39 pipeline comparisons: 10 of 36 initial mixtures and
2 of 3 longer-continuation mixtures meet the new incremental rule. The leader
uses 50% continuation, 25% calibrated and 25% adopted conditional H/A probabilities,
arithmetic pooling and H/A cost 3, after 1,024 continuation updates.

| Pipeline | Clean spell | V1 spell | V2 spell | V2 macro |
| --- | ---: | ---: | ---: | ---: |
| Adopted | 0.667556 | 0.531576 | 0.583571 | 0.849975 |
| Starting promising blend | 0.668249 | 0.534925 | 0.585615 | 0.850181 |
| 50/25/25 mean · 512 updates | 0.669461 | 0.537144 | 0.590454 | 0.851303 |
| 50/25/25 mean · 1,024 updates | 0.668533 | 0.533876 | 0.591174 | 0.851496 |

The longer continuation gains 0.000720 v2 spell over the matched 512-update
mixture, while losing 0.000928 clean and 0.003268 v1 spell. Against the original
promising arm, its clean/v1/v2 spell changes are +0.000284 / -0.001049 / +0.005559.
The shorter mixture improves all three spell scores relative to that starting
arm and all six protected aggregate metrics relative to adopted. The declared
ranking prefers the eligible higher-v2 continuation; it does not claim dominance
over the shorter mixture.

Compared with adopted, the longer mixture retains every spell and short-recall
aggregate. V1 macro loses 0.000104, within the 0.001 allowance. All D metrics
remain exactly equal to adopted.

| View | Δ spell | Δ macro | Δ short recall | D F1 | D precision | D recall | 70/30 auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | +0.000978 | +0.000355 | +0.002246 | 0.870926 | 0.884844 | 0.861920 | 0.729461 |
| V1 | +0.002301 | -0.000104 | +0.002500 | 0.860563 | 0.843006 | 0.881217 | 0.621738 |
| V2 | +0.007603 | +0.001522 | +0.003216 | 0.866138 | 0.873034 | 0.862980 | 0.669271 |

The longer original 50/50 anchor fails v1 spell retention and exceeds the
macro-loss allowance; it is not selected. Every qualifying mixture retains
H/A cost 3. Full [round-one](results/v2_spell_balanced_v1_round1.json) and
[round-two](results/v2_spell_balanced_v1_round2.json) results preserve all failures.
The previous numerical leader, 0.592638, remains higher but regressed short-spell
recall in all three views; this new result satisfies the declared retention rule.

### Provider spread and seed diagnostics

The aggregate gains do not hold for every provider. In particular, SkillCorner
v2 spell falls 0.002437 and short recall falls 0.001935; v1 spell falls for both
SkillCorner and Sportec. The rule protects equal-provider view means, not every
provider separately. These regressions limit a claim of general robustness.

| View / provider | Spell | Δ spell | Macro | Δ macro | Short recall | Δ short recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean / Metrica | 0.846768 | +0.000550 | 0.936363 | +0.000288 | 0.611111 | +0.000000 |
| Clean / SkillCorner | 0.479557 | +0.002589 | 0.790701 | +0.000689 | 0.284731 | +0.006738 |
| Clean / Sportec | 0.679275 | -0.000206 | 0.887816 | +0.000089 | 0.329949 | +0.000000 |
| V1 / Metrica | 0.645818 | +0.016535 | 0.882415 | +0.004183 | 0.297619 | +0.015873 |
| V1 / SkillCorner | 0.447586 | -0.003414 | 0.774029 | -0.000347 | 0.233692 | -0.003297 |
| V1 / Sportec | 0.508224 | -0.006219 | 0.823802 | -0.004149 | 0.137056 | -0.005076 |
| V2 / Metrica | 0.697573 | +0.016666 | 0.900839 | +0.002573 | 0.353175 | +0.003968 |
| V2 / SkillCorner | 0.481702 | -0.002437 | 0.797575 | -0.000103 | 0.275430 | -0.001935 |
| V2 / Sportec | 0.594248 | +0.008579 | 0.856075 | +0.002096 | 0.233503 | +0.007614 |

Seed diagnostics keep the pooled adopted D mask fixed and vary H/A seed
probabilities. They are not a second selection pass or a mean-of-scores ensemble.

| View | Seed 2026 spell | Seed 2027 spell | Probability ensemble spell |
| --- | ---: | ---: | ---: |
| Clean | 0.667475 | 0.666814 | 0.668533 |
| V1 | 0.529742 | 0.528382 | 0.533876 |
| V2 | 0.590608 | 0.586303 | 0.591174 |

Reference spell counts are unchanged in all 35 cells. Predicted/reference counts
and all provider/seed D diagnostics remain in the complete aggregate files.

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
