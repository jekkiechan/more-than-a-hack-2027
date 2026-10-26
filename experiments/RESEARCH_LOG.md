# Possession experiment approaches

Updated 2026-10-26. The current development head is the externally selected
[50/50 GRU + ordered HGB baseline](BASELINE.md), with contextual HGB H/A
attribution and Viterbi cost 3. Its external D-provider-holdout spell/macro is
.5746/.8449; the frozen organizer five-minute check is .4889/.8568. The older
PathCRF hybrid scored .5067/.6795 on that same small window, so competition
superiority is unproven. Full-match performance remains unknown.

The [temporal comparison](LEARNED_TEMPORAL.md),
[ensemble results](TEMPORAL_ENSEMBLE.md),
[PathCRF retention ablation](PATHCRF_ENSEMBLE.md), and
[evaluation provenance](eval_set/README.md) are now available. TCN and BiGRU
have been measured; LSTM remains untested. Realistic tracking corruption is a
[proposed next experiment](DIRECTIONS.md), not part of the packaged head.

The remainder preserves the earlier 2026-10-25 contextual experiment and its
then-proposed follow-ups. That separate implementation remains on
[`possession/robust-pipeline`](https://github.com/jekkiechan/more-than-a-hack-2027/tree/possession/robust-pipeline)
in [draft PR #8](https://github.com/jekkiechan/more-than-a-hack-2027/pull/8).
That draft implementation is not included in the current baseline promotion.
No candidate has been submitted to the organizers.

The completed experiment is pinned to commit
[`7433fef`](https://github.com/jekkiechan/more-than-a-hack-2027/commit/7433fef721a810a2f8af383ca29fad7d13fa28d1).
Its [methods and reproduction](https://github.com/jekkiechan/more-than-a-hack-2027/blob/7433fef721a810a2f8af383ca29fad7d13fa28d1/experiments/PIPELINE.md),
[external results](https://github.com/jekkiechan/more-than-a-hack-2027/blob/7433fef721a810a2f8af383ca29fad7d13fa28d1/experiments/results/pipeline_validation.json)
and [public check](https://github.com/jekkiechan/more-than-a-hack-2027/blob/7433fef721a810a2f8af383ca29fad7d13fa28d1/experiments/results/pipeline_competition.json)
remain available there. Raw inputs, models and frame-level predictions stay in
ignored `data/`.

## Completed: repair, control context, ensembling and decoding

The question was whether the existing data supports better possession inference
through feature engineering and model combination before collecting more data.

| Approach | What was tested |
| --- | --- |
| Observation repair | Respect observation masks, remove isolated impossible position spikes and bridge short plausible player gaps. Never extrapolate across long gaps or period/time breaks. |
| Contextual model | Add 31 control-context features to 137 existing geometry/motion features, then fit a gradient-boosted H/A/D classifier. |
| Team ensembling | Average baseline and contextual H/A probabilities conditional on live play. |
| Player-only D head | Learn stoppages from collective speed, slow/fast fractions, motion coherence, pitch shape and player visibility, without ball inputs. |
| D blending and decoding | Combine full-model and player-only D probabilities; compare fixed team/dead switching costs of 3/3 and 3/8 with Viterbi decoding. |

The contextual model already uses both past and future tracking. Its features
include centered motion/control averages, continuous control duration, previous
and upcoming team-control evidence with a five-second expiry, agreement across
ball gaps, support near the ball and repair indicators. The classifier predicts
each frame from those summaries. Viterbi then chooses a label sequence using
fixed switching penalties; this is not a learned recurrent model.

The selected configuration uses the contextual classifier's own H/A/D outputs,
averaged with its team-swapped prediction, followed by costs 3/3. The best
D-blended alternative uses:

```text
q = 0.5 * P_baseline(H/A | live) + 0.5 * P_context(H/A | live)
d = 0.25 * P_baseline(D) + 0.25 * P_context(D) + 0.5 * P_player(D)
P(H/A) = (1 - d) * q
P(D) = d
```

That alternative also changes H/A attribution and decoder costs, so its
comparison with the contextual model does not isolate the effect of D.

### External evidence

Eight predeclared candidates produced 80 evaluations across three providers,
five validation matches, and clean/degraded tracking. Each provider is excluded
from training when evaluated. All candidates share training rows, augmentation,
match/provider weighting and booster capacity. Selection averages official
spell F1 equally across providers and conditions, with matches equally weighted
within each provider; macro F1 breaks ties.

| Candidate | Mean spell F1 | Mean macro F1 |
| --- | ---: | ---: |
| Contextual model, selected | 0.5512 | 0.8147 |
| Team ensemble + half player-only D, costs 3/8 | 0.5455 | 0.8398 |
| Team ensemble + half player-only D, costs 3/3 | 0.5448 | 0.8390 |
| Team ensemble, full-model D | 0.5446 | 0.8141 |
| Baseline with identical training | 0.5259 | 0.8113 |
| Preprocessing only | 0.5251 | 0.8105 |
| Team ensemble + player-only D, costs 3/8 | 0.5075 | 0.8123 |
| Team ensemble + player-only D, costs 3/3 | 0.5033 | 0.8110 |

Control context gains 0.0253 spell F1 over this experiment's baseline, mostly
through Metrica improvements. Preprocessing alone does not improve the aggregate.
The D blend helps SkillCorner: on its clean validation view, D F1 rises from
0.3637 to 0.6341 and spell F1 from 0.4056 to 0.4256. It loses spell performance
on the full-view sources. D transfer and H/A attribution both remain gaps.

These are reused development matches and one seed, not independent confirmation.
The 0.0058 aggregate spell gap between context and the best D blend does not
establish a reliable ranking. The baseline here is not a common-protocol
head-to-head comparison against the previous full submission pipeline. Final
contextual fitting used 13 training matches and excluded all five validation
matches.

### Frozen public pipeline check

One check of the externally selected contextual model scored 0.4190 spell F1,
0.6455 macro F1 and 72.08% accuracy on the public five minutes. D F1 was 0.3945,
with 29.07% recall. There were 25 predicted possessions against 17 reference
possessions. The conditional 90% spell bootstrap interval was [0.2952, 0.5333].

The earlier selected refinement's recorded point estimates were 0.4300 spell
and 0.6487 macro. The new external gain has not demonstrated improvement on the
competition segment or full match. This small window cannot establish a model
ranking and, under [RULES.md](../RULES.md), must not be used to train, tune,
select a candidate or adjust thresholds. No settings changed after the check.
Read the saved aggregates rather than repeatedly scoring the public window.

## Proposed: isolate and improve D

The following comparisons are discussed but have not been implemented or run
in this experiment series.

1. **Hold contextual H/A fixed and vary only D.** Blend contextual D with the
   existing player-only head on a small predeclared external-validation grid.
   Initially keep the decoder fixed. This avoids diluting contextual team
   attribution and measures the D contribution directly.
2. **Add stopping and restarting evidence.** Compare past versus future player
   activity, collective deceleration/restart acceleration, duration and restart
   formations. The current head already measures activity levels and rolling
   averages; the hypothesis is that temporal order adds information.
3. **Improve representative training examples.** Prioritize real live-play
   occlusions, slow-but-live possessions, and reliable dead-onset/restart labels
   from other matches. More clean Sportec alone can reinforce its frozen-ball
   shortcut. Audit label definitions and visibility artifacts for each source.
4. **Evaluate decoding separately.** Test different D-entry and D-exit costs
   and soft duration priors after isolating the D head. Report spell F1, D
   precision/recall, boundary timing and short-possession performance together;
   higher D frame F1 alone does not ensure better possession spells.

Automatic Smart Tagging and observation quality are potential soft evidence.
Tagging intervals are not exact stoppage boundaries and must not be converted
directly into D labels.

## Proposed: learned temporal context

The [challenge](../CHALLENGE.md) provides the full match before inference, so
past and future tracking are valid inputs. A sequence can distinguish players
slowing into a stoppage from players accelerating out of a restart, even when
their average speeds are similar. Such patterns are uncertain evidence, not
hard rules. Future possession labels must never enter the inputs.

Compare richer engineered temporal features with a small temporal convolutional
network (TCN) and a small bidirectional GRU or LSTM. Start with an independently
controlled D head and keep contextual H/A fixed. Use geometry/motion features
plus missingness masks and time since observation, rather than treating missing
positions as stationary players. Overlapping windows of roughly 10–20 seconds
are a starting hypothesis, not a selected setting. Preserve output coverage and
boundary resolution, and reset context at half/time breaks.

A GRU/LSTM learns ordered memory; bidirectional inference uses both sides of
the frame. A TCN learns temporal patterns with convolutions and can be easier
to parallelize. Neither has demonstrated a gain in this experiment series.
Millions of correlated frames do not replace diversity in matches and stoppages.

Predeclare the comparison before fitting. Split by whole match and provider,
never random overlapping windows. Compare on the same external protocol,
repeat promising stochastic models across seeds, and seek fresh independent
match confirmation. Existing PathCRF work remains a separate learned sequence
baseline; its native model has no referee D head (see [PATHCRF.md](PATHCRF.md)).

## 2026-10-26: composite corruption training

The fixed [corruption ablation](REALISTIC_CORRUPTION.md) completed 120 evaluations
across four training arms, five development matches, pooled/D-provider-held-out
training and three input views. Augmenting both GRU/HGB D heads and HGB team
attribution raises D-provider-held-out corrupted spell/macro from .4205/.7699
to .4564/.7993, with stress spell gains on all three providers. Clean spell/macro
falls from .5746/.8449 to .5617/.8399. Team-only augmentation improves stressed
conditional H/A accuracy by 3.13 percentage points but also costs clean spell F1.

This supports investigating corruption training, rather than promoting this
recipe. The adopted baseline is unchanged. The comparison reproduces archived
clean baseline scores, preserves original grading masks under corruption, and
does not read organizer reference labels. The recipe mixes several failure
mechanisms, so it cannot identify which causes the clean loss. Next isolate
corruption groups and reduce their training weight with a fixed budget. Fresh
labeled broadcast confirmation remains unavailable.
