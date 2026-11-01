# Original_full with sampling-phase averaging

The user requested all saved metrics and Astra's promotion advice on
1 November 2026. This is a scorecard of existing results; it does not rescore
organizer labels, fit models or change the adopted package.

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
Original five-match phase-average D metrics, per-seed scores and per-IoU
breakdowns were not published in the saved aggregate phase scorecards; they
are unavailable here rather than inferred from other models or views.

## Astra consultation and next decision

Astra recommends **waiting for the exact-package GAR TEST comparison before
changing the default**, while keeping this configuration as the leading upgrade
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
