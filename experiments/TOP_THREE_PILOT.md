# First three research directions: tracking pilot

This batch implements the first three priorities in [DIRECTIONS.md](DIRECTIONS.md):
state-conditioned development views, broadcast ball recovery, and gap-context /
de-duplicated team features. The adopted `gru_hgb_ballgap25_v1` package and all
closed organizer evaluation gates are preserved. The new code is experimental.

**Outcome:** all 80 historical development evaluations completed. Gap context
and close-observation suppression show no overall spell-F1 gain in the matched
pilot. Keep the adopted baseline. Broadcast fusion adds observations but still
has substantial inter-feed ball disagreement, and the simulated player defects
need further calibration.

## Challenge-like development views

`tracking_research.py` fits a ball-gap onset model from **unlabeled organizer
half one**. Its inputs are current/past ball speed and direction, pitch-edge
distances, nearby player counts and visible-player count. Future gap duration
is sampled after onset from similar observed start states; it is never an
onset-predictor input. Existing natural gaps are retained, and the added mask
has a budget based on the target missing fraction. Possession targets never
place, accept or reject masks.

On organizer half two the onset model has **0.6334 AUROC** on 9,413 eligible
5 Hz steps, including 78 gap onsets. Log loss is 0.04730 versus 0.04815 for
the first-half constant-rate predictor. This is modest missingness prediction,
not evidence about possession accuracy or the causes of real occlusion.

The full tactical audit measures 40.69% ball missingness during the published
halves, 150 ball gaps with a 10.72 s median, and a 35.52 s median observed
player-track run. Same-team suppression within 1 m affects 12.26% of active
frames but only 0.76% of visible player observations. Proximity does not prove
that two detections represent the same player.

The view also permutes same-team slots across empirical-duration segments,
breaks derivatives at identity changes, and adds short close copies in spare
slots. It preserves native frame grids, period/time boundaries and grading
masks. Copy localization noise and whole-team permutations are assumptions;
this prototype does not reproduce every detector/identity failure. The edge
features are camera-independent; camera-specific near/far-side effects remain
unmodeled.

Calibration is split by organizer half. External training and evaluation use
separate random draws, with evaluation seeds 2026/2027/2028. They still share
a simulator family. Independent labeled broadcast confirmation remains needed.

The player component is still too aggressive. In evaluation seed 2026,
Sportec/Metrica synthetic tracklet medians are **12.80 / 15.76 s**, versus
35.52 s in the organizer tracking; close-pair frame fractions are **20.7% /
18.1%**, versus 12.3%. These are stress views, not calibrated replicas of the
organizer defects. Natural gaps and close-player geometry differ by source,
and short synthetic copies also affect the tracklet distribution. This limits
transfer claims and argues for calibrating player defects separately.

## Broadcast ball recovery

`broadcast_fusion.py` parses the additional organizer ATD export and aligns
player constellations by same-team one-to-one geometric matching. It finds
two offset clusters, **4,014 and 26,920 frames**, without possession labels or
manual anchors. Constant offsets are used only near supported anchors; each
mapped frame must also pass the player-geometry test, and isolated alignments
are rejected. This assumes shared pitch coordinates and team conventions.

The audit accepts 82,541 broadcast frames (50.18%). It recovers **4,419 of
64,643 tactical ball-missing active frames, or 6.84%**, reducing missingness
from **40.69% to 37.91%**. Existing tactical ball observations are preserved.

There is a material quality limitation: on 61,927 frames where both feeds
observe the ball, median/p90 disagreement is **0.76 / 5.25 m**, and 10.53%
disagree by more than 5 m. Neither feed is coordinate ground truth. The
alignment gates do not guarantee that recovered ball coordinates are correct;
fusion is not adopted as a default preprocessing step on coverage alone.

The separate `broadcast_quality.py` audit reports all six predeclared proxy
gates (player error 0.5/1/2 m, with/without a 65 m/s motion and 0.2 s persistence
check). Even the strictest recovers only 3,660 gap frames while **10.25%** of
overlapping ball observations still disagree by more than 5 m. Simple player
alignment and motion gates do not remove most of the ball disagreements.
No gate was chosen using possession scores, and gap-coordinate accuracy is
still unknown. See the [quality audit](results/broadcast_fusion_quality.json).

A full-length experimental fusion CSV was generated and format-validated using
the adopted model, without reading or scoring organizer possession labels.
Its local path is `data/experiments/broadcast_fusion/research_candidate.csv`.
It is not a submitted or promoted candidate.

## Matched external feature comparison

`research_directions.py` freezes the protocol before fitting or scoring and
uses the existing 13 training / five development matches across Sportec,
Metrica and SkillCorner. The three matched HGB team arms are:

| Arm | Team inputs |
| --- | --- |
| `matched` | Existing 137 physical features |
| `gap` | Those features plus 14 gap descriptors: endpoint positions, duration/progress, implied endpoint speed and same-team endpoint distances |
| `gap_dedup` | The same gap features after suppressing close same-team observations before all team feature extraction |

All three use 160 iterations, 31 leaves, learning rate .08, 2 Hz sampled
training rows, equal provider/match/view weights, team symmetry and identical
clean/conditioned training views. GRU/HGB D probabilities and decoder cost 3
are shared. The adopted model is an additional comparator with its original
training recipe; feature attribution comes from the matched arms.

These are **pooled historical development comparisons**, not unseen-provider
transfer. No evaluation match enters training, but both trainable branches
have seen training matches from every provider. Only evaluation-view seed
variation is measured; training uses one frozen draw. Official spell F1 is
primary, macro F1 breaks exact ties, and 70/30 is auxiliary. The report also
includes H/A accuracy within ball gaps, transition F1 at 0.5/1 s, short-spell
recall and per-provider results. D timing on SkillCorner remains ungradable
where X margins remove reference boundaries.

### Results

Each number is an equal-provider mean, with matches and the three conditioned
view seeds equally weighted within provider. These pooled-D numbers are not
directly comparable to the earlier D-provider-holdout headline scores.

| Team model | Clean spell | Clean macro | Conditioned spell | Conditioned macro |
| --- | ---: | ---: | ---: | ---: |
| Adopted package | **0.5938** | **0.8675** | **0.5356** | **0.8482** |
| Matched control | 0.5779 | 0.8646 | 0.5243 | 0.8428 |
| Gap context | 0.5761 | 0.8652 | 0.5216 | 0.8429 |
| Gap context + suppression | 0.5745 | 0.8651 | 0.5223 | 0.8432 |

Against the matched control, gap context changes spell F1 by **−0.0018 clean /
−0.0028 conditioned**; adding suppression gives **−0.0034 / −0.0020**.
Those small differences do not establish that endpoint context is intrinsically
unhelpful, but they provide no basis for adopting these particular arms.

Clean natural-gap H/A accuracy rises from 0.5335 to 0.6124 with gap context,
while conditioned-gap accuracy falls from 0.6551 to 0.6363. The clean gap
summary excludes Sportec, which has no missing-ball live frames; the sources
and eligible counts differ between conditions. The clean summary covers only
322 Metrica and 1,039 SkillCorner live frames. Transition F1 at 1 s falls from
0.4442 to 0.4389 clean and 0.3920 to 0.3895 conditioned. Better average team
labels in some gaps therefore did not translate into better possession spells.

The adopted package also leads the three newly fitted arms, but that broader
comparison changes capacity, training rate and augmentation. The matched-arm
differences isolate features. There is no independent broadcast confirmation,
and neither the new models nor fusion are promoted. The next useful work is
separate calibration of player defects and independent ball-confidence evidence,
before integrating preprocessing into the adopted model or widening searches.

The [complete development report](results/top_three_development.json) contains
all 80 cells, seed variation, per-source metrics, view profiles and model hashes.

## Reproduction

Restore the private input/model cache as described in [BASELINE.md](BASELINE.md).
Restore all 18 fixed external match NPZs into their usual ignored paths. The
broadcast ATD XML/TXT come from the original organizer shared data folder;
neither video is required by this implementation. Keep all organizer assets,
calibration state, models, fused tracking and frame outputs under ignored
`data/` or the authorized private cache.

```sh
python -m pip install -r experiments/requirements-temporal.txt
python -m experiments.broadcast_fusion --out data/experiments/broadcast_fusion
python -m experiments.broadcast_quality
python -m experiments.research_directions all --out data/experiments/top_three
python -m unittest discover -s experiments/tests -v
python evaluation/validate_submission.py data/experiments/broadcast_fusion/research_candidate.csv
```

The fusion command writes fused inputs and an alignment audit. Generate its
experimental CSV with `experiments.current_baseline predict --inputs
data/experiments/broadcast_fusion/fused_inputs.npz --output
data/experiments/broadcast_fusion/research_candidate.csv`. Neither command
opens organizer reference labels. Use new output directories to repeat a run;
the runner guards the experiment specification, source/input hashes, adopted
baseline manifest and fitted checkpoints. `prepare --matches`, `fit --arms`
and `evaluate --matches --report-name development_<shard>.json` support
independent workers; run `fit` after all fitting workers, then `merge` after
evaluation workers to reject incomplete/duplicate cohorts.

The team-only artifact archive is in private `mtah-data` branch
`research/top-three-directions`, bound by `top_three_cache_manifest.json`.
That manifest restores the broadcast files, calibration, three pilot models,
complete reports, experimental fusion CSV, adopted package and all 18 external
match inputs. The matched training feature caches are reproducible and omitted.
The synthetic suite passes **72 tests**, including nine new tracking/fusion
contract tests, and the full-length fusion CSV passes the official validator.

Aggregate evidence: [tracking audit](results/broadcast_fusion_tracking_audit.json),
[calibration](results/top_three_calibration.json) and
[frozen protocol](results/top_three_protocol.json).
