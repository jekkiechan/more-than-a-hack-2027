# Tracking-based football possession — Douki Douki

More Than A Hack 2027 · 16 November 2026

```mermaid
flowchart TD
    A["Tactical tracking: players, ball, pitch"] --> B["Tracking features and observation masks"]
    B --> C["5 Hz temporal GRU ensemble: ownership and dead ball"]
    B --> D["Tactical expert: attacking direction and team shape"]
    C --> E["25 Hz probabilities and possession decoding"]
    A --> F["Geometric ball-exit correction"]
    E --> F
    E --> G["Tactical H/A fusion and final decoding"]
    D --> G
    F -->|Fixed dead-ball mask| G
    G --> H["199,319 frame labels: H / A / D / X"]
```

Temporal and tactical models combine with a ball-exit rule to produce labels.
This offline system uses past and future context. H/A denote home/away possession, D dead ball, and X frames
outside the tracking-defined halves.

## Data

Challenge-match inputs are `BAR-ATH_tactical_atd.xml` and `.txt`:
pitch geometry, team assignments, players and ball. Tactical video was inspected
to understand the feed, without creating labels or corrections. Broadcast
video/tracking and homographies were unused; Smart Tagging was audited and
dropped because its phases are not possession labels. No manual annotations,
human-produced events or private ground truth from this match enter the system.

All learning uses other matches:

| Dataset and source | Training amount | Label producer and construction |
| --- | --- | --- |
| [Sportec Open DFL, v1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0 | 3 matches: J03WMX/J03WN1/J03WOH | Sportec Solutions/DFL `BallStatus` and `BallPossession` channels. |
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | Games 1–2 | Fixed H/A/D conversion of Metrica's human-annotated events. |
| [SkillCorner open data](https://github.com/SkillCorner/opendata), MIT, revision `4340d27` | 8 A-League matches | Provider `possession.group` for H/A and `game_interruption` for D; 1 s unknown margins. |
| [SoccerNet-GAR tracking-full](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR), revision `32c8402`, gated access | 45 TRAIN games | Algorithmic dense H/A/D silver labels from dataset touch/event columns; not human frame-level possession annotations. |

The temporal ensemble uses all 13 original matches and 45 GAR TRAIN games.
The tactical expert uses only the 13 original matches. Unknown targets and
match, half and segment breaks are preserved. Organizer public labels are used
only for a post-selection diagnostic, never training, tuning or selection.

## Features

ATD coordinates are converted to pitch metres. At 5 Hz, 137 team/ball features
capture nearest-player distances, contact history, ball motion and gaps;
67 player-only features capture observation masks, player counts, speeds and
rolling motion summaries. Raw missingness masks are retained; tactical features
fill short ball gaps. Training includes player-slot dropout and synthetic gaps.

The tactical expert uses 97 features: inferred attacking direction per team and
half; oriented ball position, velocity and progress; ball position relative to
team centroids and lines; players ahead of the ball; team spread and motion.
Direction comes from player locations.

## Modeling

The model has two complementary sources of ownership evidence: temporal tracking
context and team-level tactical structure. The diagram shows their outputs and
how the postprocessing below combines them.

```mermaid
flowchart TD
    F["204 temporal features at 5 Hz"] --> C
    F --> L
    F --> V
    subgraph GRU["Temporal GRUs: two seeds per branch"]
        C["Cross-entropy continuation"]
        L["Dense-label training"]
        V["Diversified possession sampling"]
    end
    C -->|0.50| M["H/A probability blend at 25 Hz"]
    L -->|0.375| M
    V -->|0.125| M
    V --> D["Initial dead-ball mask"]
    D --> S["Ball-exit correction: fixed D mask"]
    T["97 tactical features"] --> B["Gradient-boosted trees: direct and team-swapped passes"]
    B --> P["Tactical probability of home ownership"]
    M --> I["Initial H/A decode within fixed D mask"]
    S --> I
    I -->|Protect spells shorter than 5 s| U["Add weighted tactical log-odds"]
    M --> U
    P --> U
    U --> O["Final H/A decode"]
    S -->|Keep D unchanged| O
```

- **Separate ownership and dead-ball paths.** Bidirectional gated recurrent units
  (GRUs) use a 10 s core with 5 s context on either side. Past and future evidence
  helps bridge tracking gaps. H/A ownership receives hidden width 96; the teacher's
  smaller D path uses width 24 and only player features, avoiding provider-specific
  shortcuts in ball visibility during stoppages.
- **Blend temporal evidence.** The three branches use different training objectives
  and possession sampling, so the final prediction need not depend on a single
  recipe. H/A probabilities use weights 0.5/0.375/0.125; the diversified branch
  supplies the initial D mask. Each branch uses seeds 2026/2027. Predictions are
  interpolated to 25 Hz and averaged across seeds within sampling offsets, then
  offsets, reducing sensitivity to initialization and the 5 Hz sampling phase.
- **Train for whole possessions and imperfect feeds.** Frame, complete-possession
  and transition supervision address both label accuracy and possession boundaries.
  Equal original/GAR sampling prevents the larger GAR set from dominating;
  distinct-possession sampling limits long spells' influence. Four fixed corruption
  views expose the model to feed defects. Cross-entropy continuation uses 512
  updates at `3e-4`, then 512 at `1e-4`, with 32-window batches.
- **Add tactical context.** A HistGradientBoosting classifier predicts home
  ownership given live play from attacking direction and team shape, cues absent
  from the local temporal features. Boosted trees capture nonlinear relationships
  in these tabular features. It trains on 5 Hz live frames, clean plus one synthetic
  `feed_like_v2` view, with equal provider/match/view weights. Team-swapped training
  and averaged direct/swapped inference reduce dependence on home/away assignment.

Released checkpoints freeze the exact model used for the submitted file.

## Postprocessing

Weights, decoder cost and timing windows were selected on external development
data and frozen before final package checks.

- **Suppress possession flicker.** A temporal decoder applies an H/A switch cost
  of 3. This discourages brief unsupported changes that split one possession into
  several and hurt the whole-possession ranking metric.
- **Align stoppages with observed exits.** The ball-exit rule moves dead-ball
  starts toward tracking-observed pitch exits, with zero pitch margin, a 3 s
  advance window and a 5 s delay window. This uses direct geometric evidence
  to refine boundaries inferred from slower-changing player motion.
- **Fuse cautiously.** Add 0.25 times tactical log-odds to temporal H/A log-odds,
  except inside initially decoded possessions shorter than 5 s. The modest weight
  adds tactical context while short-spell protection avoids directly reweighting
  already brief possessions. Final decoding keeps the corrected D mask fixed:
  the tactical expert can change the owning team, but cannot create stoppages.
- **Apply the possession convention.** Loose balls, challenges and rebounds stay
  with the last controlling team until controlled opponent contact. GAR silver
  labels encode this; an OUT event starts D until the next touch. Inference
  estimates these boundaries from tracking alone. Frames inside the two
  tracking-defined halves receive H/A/D; outside frames receive X. There are
  no manual frame corrections.

## Eval

Whole matches are separated for fitting and evaluation. Development uses
Sportec J03WOY, Metrica Game 3 and SkillCorner 1996435/1996436/2006229.
The tactical expert is also held out by provider; upstream components do not
constitute a fully nested provider holdout. We average matches within each
provider, then providers equally. Each match is evaluated on clean tracking and
four fixed `feed_like_v2` stress realizations. This simulator uses label-free
challenge-feed statistics to calibrate ball gaps, player dropout, tracking noise
and identity switches, with missingness concentrated around stoppages.

The exact deployed pooled package is checked on three additional Sportec
matches (J03WPY/J03WQQ/J03WR9) and ten GAR TEST games. Neither set enters
fitting. Corruption scores average four realizations within each game, then
games equally. Nine selected-on GAR VALID games remain separate from TEST.

```mermaid
flowchart TD
    A["External matches: split whole matches before fitting"] --> B["Fit: 13 original matches and 45 GAR TRAIN games"]
    A --> C["Development: 5 other matches; tactical expert held out by provider"]
    A --> D["Deployed package checks: 3 Sportec matches and 10 GAR TEST games"]
    C --> E["Clean tracking and four fixed stress realizations"]
    D --> E
    E --> F["Official metrics; report each evaluation set separately"]
```

The official ranking metric, spell F1, matches H/A possessions one-to-one by
temporal intersection over union (IoU), averaging F1 at thresholds 0.3, 0.4,
0.5, 0.6 and 0.7. Macro F1 averages frame-level F1 across H/A/D. Final results:

| Evaluation set | Clean spell F1 | Feed-like v2 spell F1 |
| --- | ---: | ---: |
| Five exposed development matches | 0.681742 | 0.603079 |
| Three additional Sportec matches | 0.744783 | 0.662575 |
| GAR TEST, ten games | 0.716648 | 0.715728 |

The organizer diagnostic covers 7,500
frames and 17 possessions: spell F1 **0.623529**, macro F1 **0.870218**,
D-F1 **0.889372**. The spell bootstrap 5–95% interval is [0.458824, 0.776471].
Full-match organizer performance is unknown. Sources:
[tactical evaluation](https://github.com/jekkiechan/more-than-a-hack-2027/blob/submission-2026-v1/experiments/TACTICAL_HA_V1.md)
and [public diagnostic](https://github.com/jekkiechan/more-than-a-hack-2027/blob/submission-2026-v1/experiments/TACTICAL_PUBLIC_V1.md).

Development selection included about 56 tactical fusion variants.
The clean short-possession recall guard was not satisfied;
adoption prioritized overall spell F1. GAR has silver labels, discontinuous
segments and repeated TEST exposure. Sportec is also a training provider;
two extra matches had prior evaluation exposure. V2 corruption is
model-conditioned and uses development labels in its construction. The reused
public window is small. These protocols cannot be pooled or treated as
independent continuous-match gold confirmation.

Long ball gaps can preserve a wrong owner, smoothing can merge brief possessions,
and unseen exits can defeat geometric correction. Further work needs independent,
continuously annotated matches and improved automatic tracking.

## How to reproduce

The frozen package is
`joint_gru_sn_dense_spell_frontier_ball_exit_tactical_ensemble_v1`.
[Public source and weights](https://github.com/jekkiechan/more-than-a-hack-2027/releases/tag/submission-2026-v1)
are pinned by `submission/model_manifest.json` and `submission/weights.json`.
On Linux with Python 3.12:

```sh
git clone --branch submission-2026-v1 https://github.com/jekkiechan/more-than-a-hack-2027.git
cd more-than-a-hack-2027
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r submission/requirements.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

Place the organizer's tactical ATD `.xml` and `.txt` files in `data/tactical/`, then run:

```sh
bash submission/make_submission.sh 'Douki Douki'
```

The wrapper verifies downloaded weights, prepares tracking without reference
labels, predicts with one CPU thread, validates format and checks the hash.
Clean-environment reproduction is verified. Output:
`data/submission/Douki Douki.csv`, 199,319 rows and 281 predicted possessions;
existing CSVs are not overwritten. Expected SHA-256:

```text
8e26f823a3e933f1feca2a64dfe6d7f728a4741548c8c7cd95d51746b6a355b9
```
