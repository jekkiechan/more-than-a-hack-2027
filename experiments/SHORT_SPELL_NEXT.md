# Proposed next short-spell study

**Proposed only. No fitting, scoring, tuning, or model change was performed.**
Independent interval labels are the next recommended investment before more
short-spell fits on the current five-match cohort. The adopted combined-objective
GRU already reweights complete H/A spell frames and trains a balanced adjacent-pair
transition surrogate; another adjacent-pair turnover loss would overlap that
objective. A different temporal representation remains possible, but these
results do not establish a high-value recipe for it. Recent native-rate,
contact-memory, window-overlap, and corrected turnover-cost experiments did not
improve the clean spell-F1 screen. The turnover-cost correction raised short
recall but reduced total spell F1 and increased false extra spells. Existing
short-spell weighting and duration priors also trade recall for overall score.

## Recommended question

**Hypothesis:** under-three-second spells will remain the largest share of
unmatched complete references when the same pipeline is judged against
blind, manually adjudicated H/A/D/X intervals from a new match. If they do not,
the apparent short-spell headroom in the exposed, partly event-derived silver
cohort is not a sound basis for another model objective.

This first tests whether the target error is real under independent interval
labels. The current audit reports under-three-second recall of .384 clean and
.203 feed-like; about 39–40% of complete short misses have correct-team evidence
but insufficient interval overlap. These diagnostics come from exposed matches,
and SkillCorner has substantial edge censoring. They do not show that another
transition loss will fix the span.

## Minimal proposed protocol

- Use SoccerTrack v2 match 117093 as one sealed, new-provider panorama match.
  Start with one fixed, continuous ten-minute first-half slice; if annotation
  agreement and workload support it, extend to the full half before scoring.
- Have two football-literate annotators independently label H/A/D/X from the
  video alone, applying the last-controlled-team rule. Record boundary
  uncertainty, confidence, reason, and annotator; adjudicate only after both
  raw label files are frozen. Keep events, ball files, and model outputs hidden
  until then.
- Treat the match as held out from all training and model/threshold choices;
  no provider fold is meaningful for this single new provider. Preserve the
  half boundary, video cuts/replays, X intervals, and timestamp discontinuities;
  never bridge them when forming spans.
- Before any score, verify the video PTS, half-local clock, XML timing, and
  tracking schema. The published BAS JSON and loader disagree about second-half
  clock origin. Use BAS only as event-silver after reconciliation. Exclude the
  ball NPZ from model input because its positions are interpolated from event
  anchors.
- After labels are frozen and the existing baseline's features are mapped
  without label access, make one baseline-only official-score pass on the
  annotated interval. Report all five IoUs, duration-bin recall, unmatched
  spells, and boundary-uncertain coverage. This is a data-quality and
  mechanism check, not a promotion test; one match cannot establish broadcast
  transfer.

The ten-minute pilot requires two independent annotation passes and adjudication
but zero model fits. The full-half pass and one sealed baseline score are
conditional follow-on steps under a separate declaration. The source, license,
clock discrepancy, acquisition state, and limits are documented in
[the independent-data search](INDEPENDENT_DATA_SEARCH.md).

## Evidence checked

- The adopted loss already uses complete-spell duration weights and an
  adjacent-pair H/A and D/live transition surrogate
  ([objective report](OBJECTIVE_ENSEMBLE.md)).
- Native-rate residuals failed the clean screen
  ([report](NATIVE_SHORT_SPELL.md)); window overlap and contact memory also
  failed their clean screens ([window report](WINDOW_OVERLAP.md),
  [contact report](CONTACT_MEMORY.md)).
- The turnover decoder's natural-weight head was inert; corrected weight scale
  increased short recall but lowered clean/feed-like spell F1 by .00158/.00297
  and added about 6.9 false extra spells per match
  ([v2 report](../TURNOVER_DECODER_V2.md)).
- The recoverability audit shows both emission absence and insufficient span
  overlap, but all evaluation matches are previously exposed and two providers
  use event-derived silver labels
  ([audit](SHORT_SPELL_RECOVERABILITY.md)).
