# Refinement, remaining headroom, and competition checks

Refinement improved the fresh external test from **0.9303 frame macro F1 /
0.6654 spell F1** to **0.9408 / 0.7059**. It did **not** reach 0.90 on the
competition data: the selected model scored **0.6487 / 0.4300** on the public
five minutes. The predeclared PathCRF hybrid scored **0.6795 / 0.5067**. The
full-match labels are private, so no full-match accuracy is claimed.

The final validation-selected model is an equal-weight ensemble of a
short-possession-weighted physical-feature classifier and a larger unweighted
classifier, followed by the already fixed team/dead transition costs 3/8.
Validation improved from **0.9188 frame macro F1 / 0.6108 spell F1** to
**0.9397 / 0.7092**. All decisions were frozen before the final external scores
and competition public-label check.

## Final external confirmation

J03WQQ contains 142,345 frames and 401 reference possessions. This is the new
confirmation match for locally trained models, with the provenance limits below.

| Frozen method | Clean frame macro F1 | Clean spell F1 | Masked frame macro F1 | Masked spell F1 |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 0.9303 | 0.6654 | 0.9123 | 0.6160 |
| **Preselected ensemble** | **0.9408** | **0.7059** | **0.9225** | **0.6371** |

The clean improvement is **+0.0105 frame macro F1 / +0.0405 spell F1**. The final
model has H/A/D F1 **0.9019 / 0.9218 / 0.9989**, accuracy **94.80%**, and home
possession-share error **0.71 percentage points**. Its official 1,000-resample
5th–95th percentile intervals are **[0.9308, 0.9489]** for frame macro F1 and
**[0.6732, 0.7372]** for spell F1, conditional on this match.

The larger unweighted model scored 0.6795 spell F1 here, below the smaller
physical-feature model's 0.6919. Its validation gain therefore did not establish
a general advantage. The ensemble's gain over the weighted member is only
0.0068 spell F1 on this match; do not treat that small gap as a reliable ranking.

[Post-hoc analysis](REFINE_ERRORS.md) confirms improved short-spell recall at
IoU 0.5: **1–3 second possessions rise from 38.1% to 52.5%**, while sub-second
possessions rise from 13.5% to 21.6%. At least ten-second possessions remain
around 95% recall. This is useful progress with substantial short-spell errors
remaining. No model was changed after this diagnosis.

## What improved

| Method | Validation frame macro F1 | Validation spell F1 | Masked frame macro F1 | Masked spell F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous gap-augmented baseline | 0.9188 | 0.6108 | 0.8983 | 0.5603 |
| Separate team/dead transition costs | 0.9189 | 0.6125 | 0.8982 | 0.5626 |
| Player-motion dead-ball blend | 0.9189 | 0.6120 | 0.8980 | 0.5636 |
| Physical features, 63 leaves / 260 iterations | 0.9365 | 0.6718 | 0.9131 | 0.6064 |
| Physical features + fixed separate costs | 0.9365 | 0.6730 | 0.9134 | 0.6081 |
| Short-possession-weighted physical model | 0.9364 | 0.6991 | 0.9139 | 0.6294 |
| Larger physical model, 127 leaves / 400 iterations | 0.9389 | 0.6888 | 0.9144 | 0.6101 |
| **Final fixed ensemble** | **0.9397** | **0.7092** | **0.9165** | **0.6304** |

The main useful changes were relative ball/player motion and contact features,
then training weights that give short possessions more influence. Uniform frame
loss underweights those possessions even though the official spell metric counts
each as one object. Weighting uses labels from training matches only.

We evaluated 66 configurations across these bounded experiments, including
baselines and duplicate settings. These are not independent statistical trials.
The learned turnover-boundary adjustment did not help. Making switches cheaper
created fragmentation. The player-only dead-ball head exposed dependence on the
source's frozen-ball artifact: its clean validation macro F1 was 0.8597, below
the target, although it provides a useful transfer diagnostic.

The final ensemble is one fixed 50/50 combination, not a fitted stacking model or
a sweep of mixture weights. Its gain over the weighted member also includes
changing that member's uniform transition cost to fixed 3/8 costs; averaging's
isolated contribution was not measured.

Detailed experiments: [physical features](REFINE_FEATURES.md),
[decoder](REFINE_DECODE.md), [dead-ball head](REFINE_DOMAIN.md),
[training weights](REFINE_WEIGHTED.md), [capacity](REFINE_SCALE.md), and
[ensemble](REFINE_ENSEMBLE.md).

## How much headroom remains

[Validation-only oracle diagnostics](HEADROOM.md) separate possible error sources;
they are not deployable predictions or attainable-performance guarantees.

- Correcting only dead/live status barely changes the original model's spell
  score: 0.6108 becomes 0.6090. Sportec already makes D unusually easy.
- Correcting H/A attribution while retaining every dead/live error reaches
  0.9958 spell F1. 98.6% of the original model's erroneous validation frames
  confuse home and away.
- Perfect labels sampled at 5 Hz still score approximately 0.975–0.994 spell F1,
  depending on grid and upsampling. Sampling does not explain the observed gap.
- Earlier error analysis found short possessions disproportionately missed:
  under-three-second spells account for 68.2% of misses but only 7.9% of live
  time. That motivated the training-weight experiment, not test-specific rules.

There is **no demonstrated intrinsic ceiling near 0.71**. We found diminishing
returns for the decoder and dead-ball blend, while capacity still gave about
0.016 additional validation spell F1. That larger fit took about nine minutes on
one CPU and remained worse than the smaller weighted model. The final combined
candidate added about 0.010 spell F1; its missing-ball gain over the weighted
member was only about 0.001.

This is a practical stopping point for this round of models and this development
set, not proof that further architectures or better tracking cannot improve.
More independent labelled matches are needed to distinguish small gains from
adaptation to the single validation match. Competition tracking quality is a
separate bottleneck; clean Sportec results cannot establish its performance.

## Evaluation protocol

Training remained J03WMX/J03WN1/J03WOH, with J03WOY for validation. J03WQQ was
reserved for final confirmation of locally trained models. Previously evaluated
J03WPY/J03WR9 were not reused to select these refinements. J03WQQ was validation
for the pretrained PathCRF checkpoint, so it is not an independent test of that
checkpoint; PathCRF is excluded from the new external comparison.

The J03WQQ converter sidecar's aggregate class counts were incidentally visible
during a metadata read after the fitting experiments had been declared. No
sequence labels or scores were inspected before freeze; those counts did not
change fitting, candidate definitions, or selection. The final ensemble was
chosen from validation results only.

Before central freeze, every selected validation score was reproduced from raw
tracking, with labels excluded from inference. Model, source, data, selection
report and transfer-artifact hashes bind the evaluation to those decisions.
The runner refuses repeated held-out scoring and refuses to regenerate frozen
competition predictions after the public check begins.

Sportec's dead-ball coordinate freeze remains a major caveat. Match-level
separation also does not imply unseen-team or unseen-provider evaluation.
Bootstrap intervals are conditional on the evaluated match or window.

## Competition transfer

The full 199,319-frame competition inputs were processed. PathCRF required a
generic tracking-only roster adapter because the input has fragmented tracklets
and occasionally more than eleven observations per team. Identity drift was
found and corrected using geometry and synthetic checks, before scoring any
competition labels. No manual tracking correction was made.

The adapter interpolates 7.75% of reconstructed player-slot frames; the maximum
missing interval is 145 seconds. Predictions across such gaps are uncertain.
See [CHALLENGE_TRANSFER.md](CHALLENGE_TRANSFER.md). Automatic SmartTagging
intervals were audited, but fixed 35-second restart context windows were not
treated as exact stoppage boundaries or touch ground truth.

Only the public five minutes have available reference labels. The
[competition rules](../RULES.md) reserve these for pipeline checks, not training
or tuning. All transfer candidates were declared before scoring. Public results
cannot select a candidate or support full-match accuracy claims. Per-frame
predictions and organizer assets remain under ignored `data/`.

The public check covers **7,500 graded frames and 17 possessions**, not the full
match. Scores below describe all predeclared candidates; they were not used to
replace the model selected on external validation.

| Frozen candidate | Frame macro F1 | Spell F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Organizer smooth50 | 0.6207 | 0.3256 | 71.32% |
| Starting model | 0.6504 | 0.4000 | 73.85% |
| Externally selected refinement | 0.6487 | 0.4300 | 74.92% |
| Player-only dead-ball diagnostic | 0.7874 | 0.3429 | 79.31% |
| PathCRF teams + original dead-ball head | 0.6795 | 0.5067 | 77.47% |
| PathCRF teams + refined dead-ball head | 0.6665 | 0.4957 | 76.97% |

Dead-ball F1 falls from **0.9989 externally to 0.3392** for the selected
refinement on this window, with dead-ball recall only 20.4%. This is consistent
with the source-artifact concern identified before testing. The player-only
diagnostic's dead-ball F1 is 0.7537, but its spell score is worse; it is not a
validated replacement. PathCRF attributes the correct team on 90.75% of
reference-live frames, a conditional accuracy figure, not whole-match macro F1.

Uncertainty is large: the selected model's conditional 90% bootstrap interval is
**[0.468, 0.785]** for frame macro F1 and **[0.300, 0.550]** for spell F1. The
original PathCRF hybrid's spell interval is **[0.391, 0.613]**. These intervals
cannot establish which model wins across complete games.

The selected model's home-possession estimate on this window is 56.56%, versus
57.27% reference: only 0.72 percentage points error despite weak timeline scores.
An accurate aggregate percentage can conceal substantial frame and spell errors.

No model or threshold was changed after the public check. More tuning on clean
Sportec does not address the observed transfer gap. The next useful evidence is
independent labelled broadcast tracking with realistic stoppages and missing
observations; better ball/player tracking and reliable dead-ball evidence remain
necessary. A true performance ceiling on the competition match is unknown.

The selected submission candidate is
`data/experiments/refinement/competition/selected_refinement.csv`. It has all
199,319 rows and passes the official validator; it has not been submitted to the
organizers. All other predeclared predictions are retained beside it.

## Reproduction

First reproduce the original models and PathCRF setup using [README.md](README.md)
and [PATHCRF.md](PATHCRF.md). Then run:

```sh
python experiments/audit_dataset.py --matches J03WQQ
cp experiments/results/refinement_protocol.json data/experiments/refinement_protocol.json
python experiments/refine_features.py
python experiments/refine_decode.py
python -m experiments.refine_combined
python experiments/refine_domain.py
python experiments/refine_weighted.py
python experiments/refine_scale.py
python -m experiments.refine_ensemble
python -m experiments.challenge
python -m experiments.challenge_pathcrf
python -m unittest discover -s experiments/tests -v
python -m experiments.refinement freeze
python -m experiments.refinement test
python -m experiments.refinement challenge
python -m experiments.refinement competition-check
python -m experiments.refinement_errors
```

The fixed protocol is tracked in `results/refinement_protocol.json` and the
final manifest. Training caches and frozen outputs are not overwritten; use a
fresh checkout for a reproduction.

Tracked aggregate artifacts are `results/refinement_manifest.json`,
`results/refinement_test.json`, and `results/refinement_competition.json`.
