# Tracking-based football possession with a tactical H/A expert

More Than A Hack 2027, Initial Challenge · 16 November 2026

Submission package: `joint_gru_sn_dense_spell_frontier_ball_exit_tactical_ensemble_v1`.
The team-named CSV has 199,319 consecutive `frame,label` rows at 25 Hz.
Its SHA-256 is
`8e26f823a3e933f1feca2a64dfe6d7f728a4741548c8c7cd95d51746b6a355b9`.
The team name is supplied in the submission form and upload filenames.

## Approach and rationale

We infer H (home), A (away) and D (dead ball) from tactical tracking. Inference
uses both past and future observations; this is an offline full-match system.
The adopted pipeline combines learned temporal possession, a geometric dead-ball
boundary correction, and a direction-aware team-attribution expert.

1. **Tracking features and temporal models.** Tactical ATD coordinates are scaled
   to pitch metres. At 5 Hz we build 137 team/ball features and 67 player-only
   features: nearest-player distances, contact history, ball motion and gaps,
   player observation masks, team speeds and rolling summaries. Bidirectional
   GRUs use 10 s cores with 5 s context on either side. Conditional H/A and
   player-only dead-ball paths separate team attribution from stoppage detection.
   Hidden widths are 96 for H/A and 24 for the teacher's D path.
2. **Frozen temporal ensemble.** Three H/A experts (CE, calibrated and adopted)
   are combined with weights .5/.375/.125. Each has seeds 2026/2027. We average
   seeds within each native 5 Hz sampling phase, then phases, and interpolate to
   the native 25 Hz timeline. Ten checkpoint roles implement these experts and
   their paired teachers; identical teacher bytes are retained under both role
   names. A cost-3 decoder controls H/A switching. The predecessor's decoded
   dead-ball mask is preserved before the geometric correction.
3. **Ball-exit correction (Fable).** A frozen zero-margin pitch-exit rule snaps
   dead-ball boundaries toward observed ball exits, with a 3 s advance window
   and a 5 s delay window. It is tracking-only and adds no learned model or
   match-specific frame corrections.
4. **Tactical H/A expert.** A HistGradientBoosting classifier predicts P(H | live)
   from 97 tracking features: each team's attacking direction per half, oriented
   ball position/velocity/progress, ball relative to team centroids and lines,
   players ahead of the ball, team spreads and motion. Direction is inferred
   from player locations. Direct and team-swapped predictions are averaged.
   We add .25 times its log-odds to the temporal conditional H/A log-odds,
   protecting adopted spells shorter than 5 s, and decode H/A again inside the
   unchanged Fable D mask.

Nearest-player heuristics and per-frame classifiers miss sustained possession
through ball gaps. Temporal context helps recover those intervals; the tactical
expert contributes team shape and attacking direction that the local temporal
features lack. The player-only D path avoids provider-specific shortcuts in
ball visibility. The decoder suppresses flicker because the ranking metric is
whole-possession `spellF1`, rather than only frame accuracy.

The two pooled CE weights were reconstructed in a separately declared study
because the historical files were unavailable. They are new checkpoint bytes,
not the missing originals. The frozen recipe, sampling and normalization were
verified, and these exact reconstructed files are in the released package.

## Assets used and dropped

| Organizer asset | Use |
| --- | --- |
| `BAR-ATH_tactical_atd.xml` and `.txt` | The only model inputs: pitch geometry, track-slot teams, players and ball. |
| Tactical video | Inspected to understand the feed; not a model input or a source of hand labels/corrections. |
| Broadcast video/tracking | Not used; different timeline and incomplete geometry. |
| Homographies | Not used; tactical ATD already supplies pitch coordinates. |
| Smart Tagging | Audited as an automatic prior, then dropped; phases are not possession labels. |
| Organizer public 5-minute labels | Post-selection pipeline diagnostics only. Never used for fitting, tuning or candidate selection. The window was reused across historical checks and is not independent confirmation. |

No organizer-match hand labels, human-produced events, corrected tracking or
private ground truth enter training or inference. Missing players and ball are
masked; no person corrected trajectories, identities or transitions.

## External training data and labels

All learned components use other matches. We made no manual match annotations.
Provider labels and algorithmically derived silver labels are distinguished below.

| Dataset | Training amount | Label source and producer | Access |
| --- | --- | --- | --- |
| Sportec Open DFL data, Bassek et al., v1, CC BY 4.0 | 3 matches: J03WMX/J03WN1/J03WOH | Sportec Solutions/DFL provider `BallStatus` and `BallPossession` channels. | [figshare](https://doi.org/10.6084/m9.figshare.28196177.v1) |
| Metrica Sports sample data | Games 1 and 2 | Fixed H/A/D conversion of Metrica's human-annotated event streams. | [sample-data](https://github.com/metrica-sports/sample-data) |
| SkillCorner open data, MIT | 8 A-League matches | Provider `possession.group` for H/A; provider `game_interruption` events for D, with 1 s unknown margins. | [opendata](https://github.com/SkillCorner/opendata), revision `4340d27` |
| SoccerNet-GAR `tracking-full` | 45 TRAIN games | Fixed touch-stream rules produce dense H/A/D silver labels from dataset event columns; not human frame-level possession annotation. | [OpenSportsLab/SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR), revision `32c8402`, gated access |

The temporal package uses these 13 original TRAIN matches and 45 GAR TRAIN
games. The initial dense training uses complete-spell/transition losses and
per-frame supervision, with 50% GAR windows. Later CE continuation uses 512
updates at `3e-4` followed by 512 at `1e-4`, 32-window batches, a 50/50
original/GAR mix, distinct-possession sampling and a frozen four-view schedule.
The tactical expert uses only the 13 original TRAIN matches: 5 Hz live frames,
clean plus one synthetic v2 view, direct and team-swapped examples, and equal
provider/match/view weighting. It uses no GAR, development or organizer labels.

Development uses five other matches (Sportec J03WOY, Metrica Game 3,
SkillCorner 1996435/1996436/2006229), with whole-provider exclusions for the
held-out comparison. These matches were reused extensively for selection.
The 9 selected-on GAR VALID games and 10 repeatedly evaluated GAR TEST games
are separate evaluation sets, not training data. The three additional Sportec
evaluation matches J03WPY/J03WQQ/J03WR9 are not training matches.

Source code, dataset acquisition, split details and training recipes are in
`experiments/SOCCERNET_DATA_ACCESS.md`, `SOCCERNET_DENSE.md`,
`FRONTIER_POOLED_RECONSTRUCTION.md` and `TACTICAL_HA_V1.md` in the repository.
Released checkpoints make retraining unnecessary for exact CSV reproduction.

## Possession rules and tracking defects

- **Loose or contested ball:** SoccerNet's silver-label rule retains the last
  controlling team across uncontrolled opponent touches, challenges and rebounds;
  control changes at the next controlled opponent touch.
- **Restarts:** an `OUT` event in external training labels starts D until the
  next touch. In deployment, learned D and the pitch-exit correction provide
  stoppage boundaries; no event stream from the challenge game is read.
- **Missing/noisy tracks:** explicit masks, visible-player counts, contact
  history and bidirectional context preserve evidence across gaps. Training
  includes track-slot dropout and synthetic ball-gap views. The decoder carries
  an owner until competing evidence supports a switch; this is not a guarantee
  that every contested interval follows the rule correctly.
- **Symmetry:** team swaps in training and the tactical expert's two inference
  passes reduce home/away bias.
- **Abstention:** X is written outside the two tracking-defined half blocks.
  Every frame inside those blocks receives H, A or D. No score labels determine
  the half boundaries.

## Saved results and limitations

These are saved evaluations; preparing the submission does not rescore or tune
on the organizer public labels. The full-match organizer score is unknown.

| Protocol | Final spell F1 | Comparison / interpretation |
| --- | ---: | --- |
| Five exposed development matches, clean, equal-provider mean | .681742 | Fable .681508 |
| Same development, historical feed-like v1 | .552349 | Fable .534731 |
| Same development, feed-like v2 | .603079 | Fable .598690 |
| Three additional Sportec matches, clean / four v2 views | .744783 / .662575 | Fable .742889 / .654618 |
| GAR TEST10 exact package, clean / four v2 views, equal-game means | .716648 / .715728 | Fable .709186 / .709176; 9/10 games improve in each view |
| Organizer public diagnostic, 7,500 graded frames / 17 possessions | .623529 | Macro F1 .870218; D-F1 .889372. Public spell unchanged from Fable. |

Protocols have different labels and aggregation; their scores cannot be pooled.
Development tuning included approximately 56 exploratory tactical fusion arms.
A strict clean short-spell recall guard fails by .000151, and the historical
+.01 clean-improvement screen is unmet. Adoption follows the user's explicit
spell-F1 preference, with those failures disclosed. GAR labels are silver,
segments are discontinuous and TEST has repeated exposure. Synthetic v2 gaps
are model-conditioned and use development labels when constructing corruption;
that limits what transfer gains establish. Sportec is also a training provider.
None of these sets establishes independent continuous-match gold performance.

Short possessions, ball loss and cross-provider domain shift remain failure
modes. The model can retain a wrong owner through long gaps and its temporal
smoothing can merge brief spells. The geometric rule can miss unseen exits.
The final full-match file contains 281 predicted possessions under the official
validator's counting convention. With more time, we would acquire independently
annotated continuous matches from other games and improve automatic tracking.

## Exact reproduction

Public source: [jekkiechan/more-than-a-hack-2027](https://github.com/jekkiechan/more-than-a-hack-2027).
Use release tag `submission-2026-v1` for a stable source and weight package.
[Release downloads](https://github.com/jekkiechan/more-than-a-hack-2027/releases/tag/submission-2026-v1)
require no private-repository access. Only the eleven inference checkpoint roles,
a manifest and a readme are released; organizer assets and predictions are excluded.

Use Linux with Python 3.12, CPU PyTorch 2.8.0, and the pinned dependencies:

```sh
git clone --branch submission-2026-v1 https://github.com/jekkiechan/more-than-a-hack-2027.git
cd more-than-a-hack-2027
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r submission/requirements.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

Put the organizer's `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt` in `data/tactical/` (or use `MTAH_DATA_DIR` for the
organizer bundle root). Then, from the repository root:

```sh
bash submission/make_submission.sh 'YOUR_REGISTERED_TEAM_NAME'
```

The wrapper downloads missing weights, checks archive and checkpoint SHA-256,
prepares tracking inputs from raw ATD without reading reference labels, predicts,
runs the official format validator and requires the sealed CSV hash above.
It fixes CPU thread counts to one. The output is
`data/submission/YOUR_REGISTERED_TEAM_NAME.csv`; existing CSVs are never overwritten.
`submission/model_manifest.json` freezes the submitted recipe independently of
future research defaults. `submission/weights.json` pins release bytes and hashes.
The downloader can also use an offline archive with `--archive /path/to/archive`.
