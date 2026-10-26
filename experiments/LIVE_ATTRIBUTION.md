# Live-only team attribution with frozen GRU/HGB dead-ball probabilities

The spell-weighted binary H/A head has a **small, mixed development gain**:
clean spell F1 rises from .5952 to .5980, while macro F1 falls from .8687 to
.8655. Live-only training without weights is worse. Keep the adopted GRU/HGB
baseline; retain the weighted head as an experimental comparator pending stronger
confirmation. The external experiment used no organizer inputs/reference labels;
the subsequent user-requested public pipeline check is recorded below.

## Measured result

These are five historical external development matches, with **pooled** GRU/HGB
D checkpoints and equal provider weights. They are a different view from the
historical .5746 D-provider-holdout score; compare rows within this table.

| Clean team head | Spell F1 | Macro F1 | 70/30 summary |
| --- | ---: | ---: | ---: |
| Original three-class HGB baseline | .5952 | **.8687** | .6772 |
| Live-only binary HGB | .5885 | .8648 | .6714 |
| Live-only + fixed short-spell weights | **.5980** | .8655 | **.6783** |

The weighted head ranks first under the frozen spell-first rule, by **.00289**
(0.29 percentage points). This does not establish an upgrade: its conditional
paired match-bootstrap 95% interval is **[-.00173, +.00542]**, already spanning
zero despite the interval omitting unseen-provider uncertainty and uncertainty
from the singleton Sportec/Metrica sources. Its clean macro F1 falls by .00317.
No promotion or change to the packaged baseline was made.

| Provider, clean inputs | Baseline spell | Live-only spell | Weighted live-only spell |
| --- | ---: | ---: | ---: |
| Sportec J03WOY | .5930 | .5741 | .5962 |
| Metrica game 3 | .7006 | .6999 | .7118 |
| SkillCorner mean of three | .4919 | .4916 | .4862 |

The clean gain comes from Sportec and Metrica; the weighted candidate loses
.00575 spell F1 on SkillCorner. Raw conditional H/A accuracy on reference-live
frames falls from .8986 to .8937 overall, and from .8502 to .8407 on SkillCorner.
Its overall decoded D F1 also falls from .8887 to .8851, even though P(D) is
identical: joint decoding responds to changes in team confidence.

| Synthetic ball-gap condition | Spell F1 | Macro F1 | 70/30 summary |
| --- | ---: | ---: | ---: |
| Original baseline | .5407 | **.8589** | .6362 |
| Live-only binary HGB | .5367 | .8560 | .6325 |
| Live-only + short-spell weights | **.5501** | .8559 | **.6418** |

The weighted stress gain is .00934, driven by **+.03172 on Sportec**; Metrica
changes by -.00052 and SkillCorner by -.00318. Its conditional interval
[+.00534, +.01206] cannot establish generalization across providers: only the
three SkillCorner matches vary in this bootstrap, while Sportec/Metrica each
have one match. The mask is crude synthetic missingness, not a broadcast test.

Weighting does recover additional 1–3 second possessions, with costs elsewhere.
The following descriptive recall counts pool the five matches; they do not
replace the source-balanced primary metric.

| True duration, clean inputs | Baseline matched / true | Weighted matched / true |
| --- | ---: | ---: |
| <1 second | 63 / 274 (23.0%) | 62 / 274 (22.6%) |
| 1–<3 seconds | 170 / 509 (33.4%) | 192 / 509 (37.7%) |
| 3–<10 seconds | 434 / 727 (59.7%) | 441 / 727 (60.7%) |
| >=10 seconds | 392 / 453 (86.5%) | 387 / 453 (85.4%) |

Matches use the official one-to-one IoU .5 assignment. The weighted head
predicts 1,900 spells versus baseline's 1,768, against 1,963 reference spells.
Thus increased short-spell recall accompanies extra segmentation; the mean
spell F1 gain remains small. These results support investigating the loss
weighting, but do not support replacing the team head based on this run.

All 30 evaluations completed. The original baseline reproduces its historical
pooled spell/macro/D scores **exactly on all five matches**. All prediction,
input, source, and checkpoint hashes were verified, and each match uses one
identical D-probability array across all six comparisons. All **53** synthetic
and contract tests pass, including five new training/inference contracts; the
bundled submission passes the official format validator. That external run
opened or reset no organizer gate. Aggregates: [development result](results/live_attribution_development.json)
and [verification receipt](results/live_attribution_verification.json).

## Subsequent organizer public pipeline check

At the user's explicit request, both already frozen alternatives were checked
once against the organizer's public reference. All three full-length CSVs
contain 199,319 frames and pass the official format validator. The regenerated
baseline CSV matches its historically checked file **byte for byte**; its
saved scores are reused, and only the two new heads are scored. The model,
input, source, shared P(D), and CSV hashes were fixed before reference access.

This is the identical **7,500 graded frames and 17 reference possessions**.

| Frozen team head | Spell F1 | Macro F1 | Dead-ball F1 | 70/30 summary | Predicted spells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adopted baseline (historical score, identical reproduced CSV) | .4889 | .8568 | .8630 | .5993 | 19 |
| Live-only binary HGB | .4914 | .8541 | .8574 | .6002 | 18 |
| Live-only + short-spell weights | .4649 | .8492 | .8571 | .5802 | 20 |

Live-only's observed spell difference is **+.00254** (0.25 percentage points).
The weighted head's difference is **-.02402** (-2.40 percentage points), with
macro F1 lower by .00762. Its small external gain did not appear on this window.
Neither candidate exceeds the historical PathCRF hybrid's .5067 public spell
point estimate. These are descriptive checks on a small, previously inspected
window, not fresh confirmation or evidence of a full-match winner. The results
do not select, tune, promote, or change any candidate. The baseline remains
GRU/HGB, as decided before this check.

`live_attribution_challenge.py` reuses the existing exclusive one-shot check
guard in `temporal_challenge.py`, under a separate user-requested check receipt.
Earlier guards and historical predictions are preserved. The original commands
are retained for provenance; completed checks must not be reset or rerun:

```sh
python -m experiments.live_attribution_challenge generate
python -m experiments.live_attribution_challenge check
```

Tracking inputs, CSVs and raw output stay under ignored
`data/experiments/live_attribution/organizer_check/challenge/`. The tracked
[public-check aggregate](results/live_attribution_public_check.json) omits
frame locations. All **57** current synthetic/contract tests pass, including
four new checks that fail before public-reference access for missing inputs,
partial comparisons, changed sources, or an already completed report.

## Hypothesis and controls

The current conditional H/A distribution comes from a three-class H/A/D HGB,
renormalized over H/A, even though the GRU/ordered-HGB ensemble supplies P(D).
Train a binary H/A model on live training frames so it spends its tree capacity
on the remaining team decision. Test one fixed weighting ablation that raises
the influence of short training possessions, which frame loss underweights.

There are exactly three candidates, with no adaptive search:

- Packaged contextual HGB baseline, unchanged.
- `live_only`: H/A targets only, uniform weights within provider.
- `live_spell_weighted`: the same binary model, with training-frame weight
  `clip(sqrt(5 / spell_duration_seconds), .25, 4)` before provider normalization.

Both new heads use the existing 137 physical/contextual features, 63 leaves,
260 boosting iterations, learning rate .08, minimum leaf size 35, L2 penalty 1,
no early stopping, and seed 17. Binary HGB uses one tree per iteration rather
than the multiclass baseline's three. This holds the per-class settings fixed,
not the total number of trees.
Each binary fit uses 655,952 augmented live rows; fits took 122.5 and 127.9
seconds respectively on one CPU thread, after feature preparation.

Sampling is unchanged: take the existing roughly 4-Hz graded training rows,
then retain H/A rows. Augmentation stays clean plus the existing Markov ball-gap
view, with team swaps for both; no new corruption recipe is introduced. Weight
durations are computed from training labels only, split at X/D, half changes,
and timestamp discontinuities. Each provider receives equal total training
weight. Both new heads are frozen before any development scoring.

Every candidate receives the exact same packaged 50/50 GRU + ordered HGB P(D)
and unchanged Viterbi cost 3. The supplied D probability is unchanged by the
new team head; joint Viterbi decoding can still alter final D intervals when
team likelihoods change. This tests replacing team attribution in the current
pipeline, rather than retraining the D detector.

An earlier two-stage classifier was tried on Sportec in `SUPERVISED.md`, and
short-spell weighting helped a later Sportec physical-feature model in
`REFINE_WEIGHTED.md`. Those are prior art for the hypothesis, not results for
this pooled binary head combined with the packaged GRU/HGB D detector.

## Evaluation scope

Training uses the established 13 matches: Sportec J03WMX/J03WN1/J03WOH,
Metrica games 1/2, and the first eight SkillCorner H/A/D matches. The five
whole-match development cases are Sportec J03WOY, Metrica game 3, and
SkillCorner 1996435/1996436/2006229. Historical Sportec test matches are excluded.
SkillCorner retains the existing detected-position masks and X filtering below
12 visible players; its interruption labels have ungraded boundary margins.
Training weights describe graded live runs after this filtering, which can
split one longer possession into several visible pieces.

The D checkpoints are **pooled**, so this run is whole-match development with
all providers represented in training. Its baseline must be compared with the
other rows in this report, not the historical .5746 D-provider-holdout score.
None of the five matches is fresh confirmation. Metrica/SkillCorner targets
are event-derived silver labels, and Sportec has a frozen-ball D shortcut.

The primary view uses clean inputs. The supplementary view applies the existing
41% Markov ball-mask recipe, mean gap one second, independently fixed seed
112026, before recomputing features. All team heads receive identical masked
inputs; player-only P(D) remains identical across conditions. This is a ball-gap
sensitivity check, not a realistic broadcast corruption benchmark.

Scoring uses the unchanged official evaluator, with half/time-gap separation.
Average match scores within provider, then average the three providers equally.
Rank by spell F1, using macro F1 only for exact ties; also show the existing
70% spell / 30% macro auxiliary score. Report D F1/precision/recall/boundaries,
raw conditional live-frame team accuracy, spell counts, and short-spell recall at
IoU .5. Paired whole-match bootstrap intervals condition on the observed sources;
singleton Sportec/Metrica cases contribute no match-sampling uncertainty.
Only one training seed is used; this does not establish seed robustness.

## Reproduce

Restore the external NPZs and frozen model package as described in `BASELINE.md`.
The organizer inputs are unnecessary. Install the experiment requirements and
CPU PyTorch 2.8.0, then run from the repository root:

```sh
python -m experiments.live_attribution \
  --out data/experiments/live_attribution_reproduction \
  --report experiments/results/live_attribution_reproduction.json
python -m unittest discover -s experiments/tests -v
python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
```

Use fresh output paths; frozen reports/specifications are not overwritten.
Provider feature caches are prepared independently in three processes; model
fits each use one CPU thread. The predeclaration records input, checkpoint,
source, and metric hashes and package versions. Models, frame predictions, and
feature caches remain under ignored `data/`; only aggregate results are tracked.
