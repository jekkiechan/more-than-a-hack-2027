# Original_full with sampling-phase averaging

The user requested all metrics and Astra's promotion advice on 1 November 2026.
The completed readout combines the saved scorecards, 19 GAR match evaluations
and the missing original-provider guard/seed audit. All 470 cells pass official
scalar replay with zero error. No organizer labels are rescored, models fitted,
or adoption metadata changed.

**Final advice: provisionally promote phase averaging as an incremental upgrade
to the chosen original_full baseline.** All active guards pass, both component
seeds improve, and TEST supplies a small corroborating spell gain. The historical
+.01 clean screen fails and needs an explicit new adoption exception.
Calibrated dense plus phases remains the TEST/original-v2 spell point-estimate
leader; original_full plus phases is not an established overall winner.

## Configuration

Use the existing original_full pooled width-96 two-seed models (2026/2027), each
with its sealed teacher normalization. Compute native probabilities on all five
5 Hz sampling phases of the 25 Hz input, average uniformly across phases and
seeds, then apply the unchanged f062 D-logit temperature 4 and cost-3 decoder.
The current baseline manifest still has phase averaging disabled.

## Original development cohort

The five previously exposed matches have equal-provider weighting. V1 uses
two corruption replicas; v2 uses four. Full-pitch v2 excludes SkillCorner from
that column only. Scores across different views are not interchangeable.

| View | Spell F1 | Macro F1 | Auxiliary 70/30 | Spell delta vs phase0 |
| --- | ---: | ---: | ---: | ---: |
| Clean | .667555533 | .871271369 | .728670284 | +.004965085 |
| V1 feed-like | .531575511 | .826852823 | .620158705 | +.003395652 |
| V2 feed-like | .583571208 | .849974650 | .663492240 | +.005072998 |

V2 full-pitch spell F1 is **.633287662**. V2 realization scores are .584931989,
.585022617, .575725495 and .588604729. These are correlated realizations of the
same matches, not four independent test sets. Auxiliary 70/30 does not rank models.

| Provider | Clean spell | Clean macro | V1 spell | V1 macro | V2 spell |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metrica | .846217331 | .936074453 | .629282417 | .878232170 | .680906455 |
| SkillCorner | .476968748 | .790012326 | .451000401 | .774375113 | .484138299 |
| Sportec | .679480519 | .887727328 | .514443716 | .827951187 | .585668869 |

On clean tracking, the phase gain is +.029300551 for Metrica, −.003488887 for
SkillCorner and −.010916408 for Sportec. Aggregate improvement is not universal
provider improvement. The previous calibrated model with phases still leads
the original v2 benchmark: spell .588019876, macro .852533627, full-pitch spell
.636540030. Original_full with phases leads that model on clean and v1.

## Two further Sportec matches

J03WPY and J03WQQ are excluded from the fold checkpoints' training, but had
prior research exposure. All feed defects are simulated. These numbers use
the provider-excluded research models; they are not new scores of the pooled
competition weights.

| View | Spell F1 | Macro F1 | D-F1 |
| --- | ---: | ---: | ---: |
| Clean | .692215796 | .900908932 | .923412892 |
| V1 feed-like | .531962583 | .841865112 | .904038150 |
| V2 feed-like | .600109841 | .870162114 | .913418123 |

The [Sportec report](SPORTEC_EXTRA_EVAL_V1.md) and
[full saved scorecard](results/original_full_phase_average_scorecard_v1.json)
include per-match scores, spell counts, D metrics, phase deltas and the
three-Sportec mean including the already-used J03WOY match.

## Existing public pipeline diagnostic

Reuse the one-time saved result. There are 199,319 output rows, 7,500 graded
frames and 17 reference possessions. This diagnostic does not select models.

| Metric | Value |
| --- | ---: |
| Mean spell F1 over five IoUs (primary) | .611764706 |
| Macro F1 | .849472596 |
| D precision | .810035842 |
| D recall | .923076923 |
| D-F1 | .862869870 |
| Auxiliary 70/30 | .683077073 |
| Predicted / reference possessions | 17 / 17 |
| Spell bootstrap 5th–95th percentiles | .447058824–.764705882 |
| Macro bootstrap 5th–95th percentiles | .699601585–.885059110 |

| IoU | Spell F1 | Precision | Recall | Matched | Missed | Spurious |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .3 | .705882353 | .705882353 | .705882353 | 12 | 5 | 5 |
| .4 | .705882353 | .705882353 | .705882353 | 12 | 5 | 5 |
| .5 | .705882353 | .705882353 | .705882353 | 12 | 5 | 5 |
| .6 | .588235294 | .588235294 | .588235294 | 10 | 7 | 7 |
| .7 | .352941176 | .352941176 | .352941176 | 6 | 11 | 11 |

Against phase0, spell is unchanged, macro falls .002269326 and D-F1 rises
.000449487. The full candidate CSV hash is
`eb011268af14b4de77b4b1c823916631442ae2090efbb9526314007ddf27ceea`.
A fresh label-free replay reproduces this CSV and the adopted phase0 CSV
byte for byte; both pass the 199,319-row format check. The
[replay receipt](results/original_full_phase_average_replay_v1.json) records
input, model and manifest hashes. No reference labels were loaded or scored.
The [evaluation-adapter replay](results/original_full_phase_average_adapter_replay_v1.json)
also reproduces the same sealed CSV exactly. Changing the association of the
phase/seed means changes native probabilities by at most 3.33 × 10⁻¹⁶ and
does not change decoded output.

The completed guard audit below fills original clean/v1 D and component-seed metrics. Original-five v2 D/per-seed metrics were not published and remain unavailable; they are not inferred from other views.

## Initial Astra consultation

Initially, Astra recommended **waiting for the exact-package GAR TEST comparison
before changing the default**, while keeping this configuration as the leading upgrade
candidate. The aggregate gains are positive but small and exposed, the provider
signs are mixed, and the historical +.01 clean improvement screen fails. The
prior adoption exception does not automatically waive that screen for a new
package. The active policy still names v1; v2's proposed guardrail is not yet a
policy change.

The [bounded continuation](GAR_DEPLOYMENT_EVAL_V2.md) compares the exact current
and predecessor pooled packages with/without phases. TEST is separate from
previously selected-on VALID. A positive TEST stress result with passing guards
and gains spread across games would support provisional working-baseline
promotion; independently labeled continuous full-match performance would
remain unknown. No promotion occurs in this report.

Sources: [machine-readable scorecard](results/original_full_phase_average_scorecard_v1.json),
[saved public result](results/feed_like_v2_public_result.json),
[Astra review receipt](results/original_full_phase_average_astra_review_v1.json),
[v2 study](FEED_LIKE_V2.md), [original audit](FEED_BENCHMARK_AUDIT_V1.md).

## Completed original-provider guards and component seeds

The fixed audit replays both published phase0/phase-average spell and macro scorecards exactly on the same exposed matches. It adds metrics, not independent evidence. All active v1 guard checks pass: **True**. The historical +.01 clean screen still fails.

| Provider | Phase0 clean D-F1 | Phase-average clean D-F1 | Delta | Phase-average v1 D-F1 |
| --- | ---: | ---: | ---: | ---: |
| Metrica | 0.916277233 | 0.916100022 | -0.000177211 | 0.896339905 |
| SkillCorner | 0.774034601 | 0.774268999 | +0.000234399 | 0.779802472 |
| Sportec | 0.923668209 | 0.922409617 | -0.001258592 | 0.905545431 |

| Component seed | Mode | Clean spell / macro / D-F1 | V1 spell / macro / D-F1 |
| --- | --- | --- | --- |
| 2026 | phase0 | 0.650749532 / 0.863323217 / 0.858278703 | 0.516485934 / 0.819027500 / 0.850069938 |
| 2026 | phase_avg | 0.658831066 / 0.864153479 / 0.858126259 | 0.521102438 / 0.819880187 / 0.850390751 |
| 2027 | phase0 | 0.660130215 / 0.869476754 / 0.871238953 | 0.518873316 / 0.822280139 / 0.862031006 |
| 2027 | phase_avg | 0.664833686 / 0.869886200 / 0.870407649 | 0.523290779 / 0.823646042 / 0.862448749 |

The [guard audit](PHASE_AVERAGE_GUARD_V1.md) and [full expanded scorecard](results/original_full_phase_average_scorecard_v2.json) include every provider/seed metric and per-IoU metric available in the saved evidence.

## Completed exact-package GAR evaluation

These scores use exact pooled deployment checkpoints. Ten TEST games are the primary matched comparison; the nine previously selected-on VALID games are reported separately. GAR uses six native sampling phases, preserving 29.97 Hz tracking; the target feed uses five at 25 Hz. The sources are event-derived silver labels inside discontinuous clips.

| Split / view | Configuration | Spell F1 | Macro F1 | D-F1 | D precision | D recall | Short-spell recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TEST / clean | original_full phase0 | 0.698836485 | 0.775378751 | 0.574382295 | 0.495390817 | 0.687138676 | 0.267333602 |
| TEST / clean | original_full phase_avg | 0.701640993 | 0.779072746 | 0.583474252 | 0.504421324 | 0.695752995 | 0.271664713 |
| TEST / clean | calibrated phase0 | 0.701271380 | 0.772717198 | 0.563372845 | 0.508639833 | 0.634317051 | 0.249385933 |
| TEST / clean | calibrated phase_avg | 0.704845626 | 0.775564626 | 0.567702809 | 0.514070034 | 0.636228882 | 0.252166384 |
| TEST / v2 | original_full phase0 | 0.699133249 | 0.776583079 | 0.580782008 | 0.497378020 | 0.701380561 | 0.269267777 |
| TEST / v2 | original_full phase_avg | 0.701288850 | 0.780501635 | 0.589850580 | 0.505974742 | 0.710660991 | 0.268273619 |
| TEST / v2 | calibrated phase0 | 0.701215165 | 0.774048811 | 0.568310942 | 0.511702840 | 0.641910546 | 0.247647491 |
| TEST / v2 | calibrated phase_avg | 0.704117630 | 0.776682332 | 0.574051091 | 0.516833917 | 0.647987354 | 0.248629697 |
| VALID / clean | original_full phase0 | 0.698758393 | 0.781931434 | 0.591530712 | 0.512757215 | 0.708789309 | 0.270460637 |
| VALID / clean | original_full phase_avg | 0.700777135 | 0.781430606 | 0.589668997 | 0.508065301 | 0.712354343 | 0.264668315 |
| VALID / clean | calibrated phase0 | 0.695597659 | 0.776402764 | 0.575058011 | 0.518367305 | 0.654110773 | 0.247806800 |
| VALID / clean | calibrated phase_avg | 0.698876720 | 0.777646461 | 0.577096414 | 0.519820246 | 0.656906607 | 0.244995474 |
| VALID / v2 | original_full phase0 | 0.696525687 | 0.781900543 | 0.595808361 | 0.510070530 | 0.724304066 | 0.268138671 |
| VALID / v2 | original_full phase_avg | 0.697917200 | 0.781770142 | 0.594012642 | 0.506478883 | 0.726805192 | 0.268284802 |
| VALID / v2 | calibrated phase0 | 0.694025916 | 0.776018969 | 0.578147282 | 0.518808411 | 0.659878436 | 0.249300531 |
| VALID / v2 | calibrated phase_avg | 0.695390967 | 0.777075235 | 0.579750110 | 0.517219669 | 0.666243056 | 0.246003612 |

| Split | Games | Records | Graded frames | Graded hours |
| --- | ---: | ---: | ---: | ---: |
| TEST | 10 | 2,439 | 968,965 | 8.980879 |
| VALID | 9 | 2,191 | 851,015 | 7.887656 |

| TEST comparison | View | Spell delta | Paired 95% interval | Games with positive delta |
| --- | --- | ---: | --- | ---: |
| Original_full phases − same package phase0 | clean | +0.002804508 | [-0.002035991, +0.007242381] | 6/10 |
| Original_full phases − calibrated phases | clean | -0.003204633 | [-0.007988212, +0.001380503] | 4/10 |
| Original_full phases − same package phase0 | v2 | +0.002155601 | [-0.000348097, +0.004636954] | 8/10 |
| Original_full phases − calibrated phases | v2 | -0.002828780 | [-0.007269256, +0.002556923] | 3/10 |

All **380 saved cells** pass official scalar replay; maximum scalar difference is 0. Whole-game aggregation is independently checked. Reference/graded populations stay fixed across arms, phases and stress draws. Native predictions remain ignored; new native artifacts are locally verified, with no new remote-native archive claim.

The [GAR report](GAR_DEPLOYMENT_EVAL_V2.md) owns protocol and artifact details; [verification](results/gar_deployment_eval_v2_verification.json) supplies all per-IoU means. The expanded scorecard includes paired intervals, per-game values and possession counts.

## Final promotion recommendation

Astra recommends provisional adoption of uniform phase averaging as an incremental upgrade to the chosen original_full baseline. All seven active stress/provider guards pass, both component seeds improve clean/v1 spell, and GAR TEST corroborates a small spell gain with higher macro and D-F1. The primary spell interval still includes zero. Calibrated dense plus phases remains the numerical spell leader on the original v2 benchmark and GAR TEST; original_full plus phases is not an established overall winner. Preserve the failed +.01 original-clean screen and record an explicit new user-approved exception if adopted. No further experiment is required before the adoption decision; continuous full-match gold confirmation remains unknown.

The adopted manifest and policy remain unchanged. A promotion would require its own explicit working-baseline decision and a preserved exception to the failed historical clean screen. No new local tuning or organizer scoring follows from this readout.
