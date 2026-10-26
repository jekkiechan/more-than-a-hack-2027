# Next possession research directions

**Retain `gru_hgb_ballgap25_v1`. These are proposed experiments, not selected
models or a baseline promotion.** The completed comparisons are in
[POSSESSION_GAPS.md](POSSESSION_GAPS.md).

## First dependency: trustworthy aligned labels

Use the acquired CC0 Belmont–Wakefield recording for development. Its full audio
has been processed, but it has neither possession ground truth nor calibrated
player/ball tracking. Establish source-video/audio time mapping, inspect cuts
and replays, extract tracking, and calibrate pitch coordinates. Native container
PTS provenance does not establish alignment to any existing provider's tracking.

Start with a fixed ten-minute gameplay annotation pilot. Two annotators should
independently label D/live, H/A possession under the project's last-control rule,
uncertain intervals, and whistle onsets/source/type from the raw video/audio.
Adjudicate disagreements. Keep whistle labels separate from actual play-state
labels: a permission whistle may precede the restart and many restarts are
unwhistled. The proposal-assisted packet is suitable for development review;
independent interval labels require the separate template without model hints.

Reserve the different-producer MRVTV Lake Region–Harwood recording for later
confirmation before fitting or choosing thresholds. Its current exposure is
metadata/probe only. More independent matches will be needed for strong transfer
claims; one new match cannot characterize all broadcast domains.

## 1. State-aware whistle and tracking fusion

**Hypothesis:** audio supplies stoppage evidence missing from player motion.
Perfect observed D/live labels raised the diagnostic spell score from .57402 to
.69094, whereas the player-only auxiliary head barely changed possession scores.
This motivates a new signal, not another sweep of its transition penalty.

Freeze these comparisons before examining confirmation labels:

1. Tracking-only baseline with its current H/A head and decoder.
2. An audio-only D/live or directional-boundary diagnostic with the same H/A head.
3. Tracking plus whistle scores/onsets and audio-quality/missingness indicators.
4. The same fusion with audio displaced by a fixed offset within continuous
   segments, scored on common support, as a timing-information control.

Fit thresholds, lag distributions and fusion weights on development sources
only. Model stoppage and restart separately; treat audio as evidence rather than
a hard state toggle. Include crowd whistles, repeated blasts, commentary, music,
silent intervals and replay audio as negative/uncertain examples. The current
robot-microphone detector and its >.65 threshold are uncalibrated starting points.

**Readout:** whistle precision/recall/onset latency, D/live boundary precision/
recall at .5 and 1 second, and downstream official spell F1. A higher whistle
score or more detected fragments is not a success criterion. Require a downstream
benefit over the matched tracking baseline and displaced-audio control.

## 2. Better evidence for brief H/A control changes

**Hypothesis:** brief controlled touches need a discriminative control-change
signal. Perfect team evidence reached .85165 with baseline P(D), indicating
more remaining team-attribution headroom than D/live correction alone.

Build short control/turnover examples around independently annotated transitions.
Pair genuine brief possession changes with difficult negatives: same-team passes,
nearest-player changes, opponent deflections without control, and contested balls
that retain the last controlling team. Combine ball/player relative motion and
contact evidence with short video crops when tracking is ambiguous. Use a short
event representation with longer context, supervised jointly with team state.

This extends the information available to the model. The older learned turnover
adjustment, simple duration priors and stronger frame weights have already been
tested; repeating those alone is lower priority.

**Readout:** subsecond and 1–3-second recall alongside spell precision, false
possessions per minute, long-spell recall and official spell F1. Evaluate the
team head under full provider holdout: pooled-H/A results overstated clean
transfer by about four spell-F1 points in the latest comparison.

## 3. Reliability-aware multimodal fusion

**Hypothesis:** the useful observation source changes with ball visibility,
tracking quality and audio quality. An unconditional 25% ball-hidden average
hurt clean performance; that does not establish the value of a quality-aware gate.

Compare the unchanged baseline, the fixed mixture already tested, and a gate
using only observation-quality features: missing-ball duration, detected versus
extrapolated tracks, player visibility, calibration uncertainty, camera changes,
and audio confidence/quality. Route team-control and D/live evidence separately.
Preserve strong ball evidence when reliable; use contextual or audio evidence
when its observation is missing or uncertain.

Train the gate with cross-fitted predictions from whole training matches and
source-only preprocessing. Prevent missingness from becoming a shortcut for the
provider's label censoring. Validate on genuine recorded gaps, with synthetic
corruption as a secondary stress test rather than its own confirmation set.

**Readout:** clean/stress tradeoffs, conditional team accuracy with coverage,
calibration, and per-visibility-tier spell F1. A smaller clean/stress gap caused
by lower clean accuracy does not count as robustness improvement.

## 4. Precision-aware event training

**Hypothesis:** training needs to distinguish genuine short controls from
fragmentation, rather than merely making short predictions cheaper.
Inverse-duration weighting improved targeted recall but lost clean spell F1;
duration priors produced 2,356–2,695 predictions against 1,963 reference spells.

Test a bounded event-supervision or structured-loss arm with explicit hard
negative transitions and a penalty for unsupported extra spells. Preserve the
training budget, source mass and negative examples. Use accurate complete spells
for duration supervision; censored fragments remain unknown. Select settings
inside the source-training split, not from the confirmation match.

**Readout:** official spell F1 and precision/recall together, predicted/reference
spell counts, duration bins and timing. Improving short recall while degrading
total spell F1 is another tradeoff, not an adoption result.

## Evaluation and adoption gate

- Exclude the target provider from both H/A and D fitting, calibration and learned
  preprocessing. Retain the old D-only/pooled-H/A view as separately labeled context.
- Preserve native grading masks and unknown labels. New annotations should
  include representative dead time; retained SkillCorner cores cannot establish
  complete dead-ball or boundary performance.
- Freeze finite arms and choices before independent confirmation. Group all
  derived clips, synthetic views and audio channels by their original match.
- Use the unchanged official spell score first and macro F1 only for exact ties;
  retain the additional 70/30 summary without making it the ranking rule.
- Report every provider and all meaningful tradeoffs. Intervals on the current
  five exposed games do not establish uncertainty over unseen providers.
- Keep organizer labels/public gates closed. Keep the existing baseline until a
  future separately authorized adoption decision has supporting evidence.

The first practical study is aligned whistle fusion plus its annotation pilot.
The H/A control-change study is the next major modeling direction. Broader model
capacity and hyperparameter sweeps come after those evidence gaps are addressed.
