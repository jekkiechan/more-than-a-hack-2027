# tactical_ha_v1: direction-aware H/A expert on top of Fable snap_m0.0

Status: promoted on explicit user instruction ("promote if net positive for spell F1"),
15 November 2026. Package `joint_gru_sn_dense_spell_frontier_ball_exit_tactical_ensemble_v1`.
The ten GRU checkpoints and the dead-ball mask are unchanged; one new role, `tactical_ha`.

## Why this, and why it should transfer

Oracle H/A inside the predicted live spans lifts development spell F1 by +.20 to +.25;
an oracle D mask lifts it by +.085. Team attribution is the headroom. The GRU's inputs
are local to the ball (nearest players, speeds, contact evidence) and contain no
attacking direction, ball progress or team shape. Those cues are provider-independent
football structure and stay observable when ball tracking degrades.

## Method

- `tactical_ha_features`: each team's attacking direction per half from where its players
  stand; then oriented ball position/velocity/progress, ball relative to each team's
  centroid, deep line and front, share of each team ahead of the ball, centroid and line
  speeds, spreads, distances to the ball. Tracking only. 97 columns.
- `tactical_ha` model: one HistGradientBoosting classifier for P(H | live), fitted on the
  13 original TRAIN matches (5 Hz live frames; clean plus one `feed_like_v2` view; direct
  and team-swapped rows; equal provider/match/view weight). No GAR, development or
  organizer data. Inference averages the direct and team-swapped passes.
- Fusion (`tactical_ha_runtime`): decode the adopted pipeline unchanged; add
  `0.25 x logit(P_tactical)` to the conditional H/A mixture log-odds, except inside
  adopted spells shorter than 5 s; re-decode H/A with the unchanged cost 3 inside the
  unchanged D mask.

## Evidence (spell F1; control = adopted Fable package)

| Set | View | Control | Candidate | Delta |
|---|---|---:|---:|---:|
| Development, 35 exposed cells, expert held out by provider | clean | .681508 | .681742 | +.000234 |
| | feed_like_v1 | .534731 | .552349 | +.017618 |
| | feed_like_v2 | .598690 | .603079 | +.004389 |
| Sportec J03WPY/J03WQQ/J03WR9, exact pooled package | clean | .742889 | .744783 | +.001894 |
| | feed_like_v2 (4 fresh realizations) | .654618 | .662575 | +.007957 |
| GAR TEST10, exact pooled package | clean | .709186 | .716648 | +.007462 |
| | v2 (4 realizations) | .709176 | .715728 | +.006552 |

- Development: macro F1 -.000077 / +.011789 / +.001779; short-spell recall -.000151 /
  +.014418 / +.010445; D-F1 identical. v2 providers +.0027 / +.0025 / +.0079
  (Metrica/SkillCorner/Sportec); v1 +.0114 / +.0186 / +.0229; clean -.0011 / +.0010 / +.0008.
  26 cells up, 9 down. The v2 mean passes the long-standing >.60 target.
- Sportec: 14 cells up, 1 tied, 0 down. Pass rule fixed before scoring.
- GAR: 9 of 10 games up in each view; paired whole-game bootstrap 95% intervals
  [+.0043, +.0108] clean and [+.0037, +.0094] v2; macro +.0034 / +.0034; short recall
  +.0084 / +.0088. Control replays the saved adopted GAR scores with zero error. GAR
  shows about 12 players per frame, so the expert works with partial visibility.
- Challenge match, label-free: 1,263 of 158,857 graded frames change (0.80%), 272 -> 281
  possessions, D mask identical. The expert agrees with the adopted team label on 89.8%
  of live frames, in line with its held-out accuracy elsewhere.

## Limits and disclosures

- One strict repository guard fails: development clean short-spell recall -.000151.
  The historical +.01 clean screen is not met. Adoption follows the user's spell-F1 rule.
- The fusion arm was chosen on the development clean and v2 cells after about 56
  exploratory arms (uniform weights; ball-gap, uncertainty, sustained-uncertainty and
  near-gap gates; short-spell/edge protection; a spell-relabel audit that was rejected).
  Differences between the protected arms are within noise. The v1 cells, the Sportec
  matches and GAR TEST were scored once each, after the arm was frozen.
- Sportec is a training provider (the three matches are not training matches; two were
  scored by an earlier study). GAR labels are silver and TEST has repeated exposure.
- The organizer public window was not scored. No continuous-match gold confirmation.
- `tactical_ha.joblib` is a scikit-learn 1.8.0 pickle from our own code.

## Reproduce

This repository contains the code; restore the package from the private cache.
this package with its versioned cache manifest, then:

```sh
python -m experiments.current_baseline predict --manifest data/models/joint_gru_sn_dense_spell_frontier_ball_exit_tactical_ensemble_v1/baseline_manifest.json --output <fresh.csv>
```

Expected CSV SHA-256 `8e26f823a3e933f1feca2a64dfe6d7f728a4741548c8c7cd95d51746b6a355b9`.
`source_snapshot/study/` holds the runners as executed; `evidence/` holds aggregates.
