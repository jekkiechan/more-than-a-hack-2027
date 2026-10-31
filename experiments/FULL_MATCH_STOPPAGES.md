# Full-match and stoppage coverage study proposal

Status: proposed; no new fitting or scoring has started. A bounded
silver-supervised development comparison can use the existing SoccerNet event
stream and already exposed external development matches. Independent continuous
gold remains a stronger optional confirmation path.

## Question

Does continuous coverage of stoppages and restarts improve H/A/D/X possession
recovery beyond the effect of adding more labeled time? A later, separate question
is whether longer temporal context helps once coverage is held fixed. Keep the
current five-second context halo for the first comparison; do not combine a
context change with the coverage test.

This distinction matters because the adopted SoccerNet-GAR dense silver labels
are dense only within observed records. In the 45 TRAIN games they cover 46.9
hours, but consist of 11,249 merged contiguous records with median length about
11 seconds. Tracking-invalid gaps split records, clip boundaries censor long
states, and records are not full-match timelines. `full_y` in those records is
native-rate label data inside each record, not continuous match truth. GAR's
current action clips are 144 frames (about 4.8 seconds) around event points and
cannot supply the long live-ball context or complete stoppage/restart sequences
needed by this question.

## Local source availability and limits

- A dated restore receipt records that `OpenSportsLab/SoccerNet-GAR`,
  `tracking-full`, revision `32c840210b7d08d06976e7f9457dfff79fb2da99`, had
  seven TRAIN files hash-verified on 30 October 2026. The current checkout does
  not contain those source bytes, so their present local availability has not
  been verified. Dense labels are derived from the event stream; they are silver
  rules, not independently annotated continuous H/A/D truth.
  The held-out GAR valid/test clips are likewise short clip records, not a
  continuous full-match asset. Do not infer complete stoppage coverage from the
  87.4% known-frame fraction inside TRAIN records.
- SoccerTrack v2 match 117093 has a verified small public sample: BAS action
  points, ball NPZs, metadata and player-node event rows. The complete first- and
  second-half panorama videos and GSR tracking files are listed, but have not
  been downloaded. BAS and player-node rows are event points, not possession
  intervals. The ball paths are event-interpolated, so they are not independent
  observed-ball tracks. This is a feasible non-broadcast annotation pilot after
  video/tracking acquisition and alignment; it cannot establish TV-broadcast
  transfer.
- The legacy SoccerNet-GAR whole-match tracking/video route and PFF/Gradient
  World Cup release are access-gated. The existing request remains unresolved;
  no fresh request or data access is implied by this proposal. A reply must
  clarify matching-video rights and permission for ML training/evaluation.
- Action spotting supplies sparse events and event-team semantics, not
  continuous possession truth. Do not turn event anchors into uninterrupted
  gold labels or count their clip windows as independent labeled hours. The
  project already has continuously labeled, full-match external development
  fixtures, but these five matches are exposed and cannot provide fresh
  independent confirmation.

## Proposed first comparison

Start with a silver-supervised development comparison: use SoccerNet-GAR
TRAIN event streams to derive labels under an explicit touch/restart rule, then
construct matched training views that differ in clip-limited versus broader
temporal coverage. Use the already exposed external development fixtures to
check pipeline behavior and score the fixed comparison under the existing
whole-match/provider protocol. Treat this as development evidence only. Keep
unknown labels unknown; do not propagate sparse event anchors into unsupported
continuous H/A/D targets. Require the touch/restart semantics to support the
intended D and restart labels, and retain provenance and censored boundaries.

If stronger confirmation becomes feasible, add a new independently labeled
full-match source. Labelers should see video only; hide action events, tracker
predictions and model outputs until raw annotations are locked. Define H/A as
last controlling team, D as dead play, and X as unresolved; preserve boundary
uncertainty, filmed gaps, periods and clock breaks. A double-labeled shared
slice with adjudication can quantify agreement, but this gold step is optional
and is not a prerequisite for the silver development comparison. Human labels
on the competition match remain prohibited.

Freeze a source/match manifest before viewing outcomes. Use whole-match
exclusions for every learned model and normalizer. Keep the existing model,
features, decoder, five-second context, training steps and sampling recipe fixed.
The first design is a coverage-by-label-hours comparison:

1. **Clip-limited control:** fixed amount of labeled live-play time sampled from
   the source's ordinary action-centered windows, retaining their natural
   missingness and censoring.
2. **Coverage-matched control:** the same number of labeled seconds from the
   same matches, but sampled as complete temporal blocks that deliberately
   include the beginning/end of stoppages and restarts. This isolates coverage
   composition from total labeled duration.
3. **More-hours arm:** the same complete-coverage sampling rule with a
   predeclared larger labeled-time budget. This estimates the effect of adding
   hours after coverage has been represented.

Use identical match folds, exposure accounting, updates and total training
compute. Keep all windows with their source match and time block. Report the
number of labeled hours, continuous match hours, fraction of stoppages/restarts
fully covered, censored interval counts, usable H/A/D/X seconds, and class and
boundary distributions per arm. Since neighboring windows are correlated,
uncertainty must be summarized by match, not by frame. A small one-match pilot
can validate annotation and coverage tooling, but cannot support an efficacy
claim.

Only after this comparison is complete should context be tested: compare the
fixed five-second halo with one predeclared longer context (for example 15
seconds) on the same complete-coverage labeled set, folds, training compute,
features and decoder. This keeps longer network context separate from both
labeled-hour volume and stoppage coverage. More context must not cross half,
match, clock discontinuities, or tracking gaps.

Primary readout should use official possession spell F1 on adjudicated,
continuous held-out intervals, with all five IoU thresholds, macro frame F1,
D precision/recall, short-spell recall, fragmentation, and boundary results.
Report complete versus left/right-censored spells separately; never score a
truncated event as a complete gold interval. Event-spotting metrics may be
reported as a separate auxiliary task and cannot stand in for possession
metrics. The public 7,500-frame organizer window is excluded from all training,
selection and tuning.

## Readiness gate and next actions

Before a silver development comparison, verify the relevant source terms and
existing artifact availability, audit the event-to-touch/restart semantics,
preserve native masks and period/timestamp metadata, and declare the whole-match
folds and exposure. A new source is needed only for an independent transfer or
gold-confirmation claim. Check license terms when acquiring or using any new
video/tracking asset; no separate written-approval process is presumed where
the existing terms already authorize the intended use. Validate clock
alignment before joining action points to tracking. For SoccerTrack 117093, the
verified BAS clock and current loader disagree about half-2 timing; resolve
against the videos and raw period metadata before using events as silver
anchors.

For the existing-source development study, first verify/restore the declared
SoccerNet bytes from the recorded source and publish a coverage audit with
actual union-of-time
coverage, gap distributions, stoppage/restart completeness and edge censoring.
Do not treat merged GAR clip duration or event-point count as full-match
coverage. If full broadcast access remains blocked, the SoccerTrack panorama
pilot can test annotation workflow and interval scoring only, with its
non-broadcast limitation stated prominently.
