# <team_name> — frame-level possession for BAR–ATH

More than a Hack 2027, Phase 1. Submission file: `<team_name>.csv` (199,319 rows,
`frame,label`). Model package: `joint_gru_sn_dense_ensemble_v1`.

## 1. Approach

We treat possession as a sequence-labelling problem on tracking geometry alone.
The pipeline has three stages:

1. **Per-frame features from the tactical ATD.** We compute 137 hand-designed team
   features and 67 player-only dead-ball features, giving 204 features per step. The
   team features include:
   - distance from the ball to the nearest one to three players of each team,
     and player counts within 2/5/10 m;
   - home-versus-away distance advantage, and the last and next contact by each
     team within 1/2/3 m with its age;
   - ball speed, the time since and until the ball was last/next observed, and
     the ball's distance to the pitch edge;
   - squad-level physical state: median and upper-quartile speed and the
     stationary and running fractions, with 1 s and 3 s means;
   - rolling 0.25–2.5 s means of the above and a nearest-player vote.
2. **A temporal model on 5 Hz windows.** Each window is a 10 s core with 5 s of
   context on each side. A model has two parts:
   - a conditional, bidirectional **H/A GRU** (hidden width 96) that predicts which team owns
     the ball, given that the ball is live;
   - a separate **D path** (bidirectional GRU, hidden width 24) that predicts dead ball from
     player-only features.

   The ensemble has two independently seeded members (seeds 2026 and 2027).
   Their H/A/D probabilities are upsampled to 25 Hz and averaged.
3. **Decoding.** A three-state Viterbi decoder runs separately in each half,
   with a fixed switch cost of 3. It turns the probabilities into H/A/D runs. The
   switch cost suppresses one-frame flickers, which the ranking metric
   (`spellF1`) punishes far more than per-frame accuracy does.

**Why this design and not something else**

- Simple heuristics topped out early. Nearest-player rules with smoothing (the
  published baselines) and gradient-boosted per-frame models plateau well below
  a learned temporal model on our development matches. The GRU was the first
  family to clear our +0.01 `spellF1` screen by a wide margin (+0.093 over the
  previous best).
- **Team assignment is the bottleneck, not dead ball.** On a validation match,
  98.6% of wrong frames were H/A confusion, so model capacity goes into the H/A
  head.
- **The D path uses no ball features.** Every external provider has a dead-ball
  shortcut in its ball channel:
  - Sportec freezes the ball's coordinates;
  - Metrica blanks the ball;
  - SkillCorner's visibility filter drops most dead frames.

  A ball-based D head therefore does not transfer to automatic, uncorrected
  tracking. Player motion (a stationary squad, restart formations) does transfer.
- We optimise for `spellF1` directly. Training uses complete-spell and
  transition losses as well as per-frame cross-entropy, so the model is rewarded
  for recovering whole possessions.

## 2. Assets used and dropped

| Asset | Used? | Notes |
|---|---|---|
| Tactical ATD tracking (`BAR-ATH_tactical_atd.{xml,txt}`) | **Yes, the only input** | Team membership of each track slot comes from the ATD metadata. Players and ball are scaled to metres. Missing ball/players are masked, not imputed. |
| Tactical video | No | Not needed for geometry-only features. We inspected it only to understand the data. |
| Broadcast video and tracking | No | Not synchronised to the tactical timeline, and only 65.8% of frames have a usable homography. |
| Homographies | No | Tracking is already in pitch coordinates. |
| Smart Tagging | No | We audited it as a prior (team-tagged phases, restarts). It is automatically inferred and describes phases rather than possessions, so it was left out. |
| Public 5-minute ground truth | Pipeline check only | It was never used to train, tune or select anything. We scored it once, after the model was chosen, as a format and alignment check. |

## 3. Training data

**None of the training data comes from the provided match.** All labels on other
matches were produced by their providers or by a fixed rule from provider event
data. We made no manual annotation of any kind.

| Dataset | Matches used for training | Labels: source and who produced them | Access |
|---|---|---|---|
| Sportec Open DFL data (Bassek et al., v1, CC BY 4.0) | 3 (J03WMX, J03WN1, J03WOH) | Provider `BallStatus` / `BallPossession` channels, produced by the data provider (Sportec Solutions / DFL) | [figshare DOI 10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1) |
| Metrica Sports sample data | 2 (Games 1, 2) | H/A/D derived by a fixed rule from Metrica's own human-annotated event stream for those games | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| SkillCorner open data (MIT) | 8 A-League matches | Provider `possession.group` for H/A. D comes from provider `game_interruption` events, with 1 s unknown margins | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata) @ `4340d27` |
| SoccerNet-GAR, `tracking-full` | 45 TRAIN games | Dense H/A/D *silver* labels from a fixed touch-stream rule over the dataset's event columns (§4). Not hand-annotated per frame. | [huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR), revision `32c8402`, gated |

**Volume:** 13 tracked matches and 45 SoccerNet games. SoccerNet makes up 50% of
each training batch, and training runs 2,048 optimiser updates per member.

**Model selection** used five more held-out external matches with full provider
holdout: no learned component saw the held-out provider. Those matches were:
Sportec J03WOY, Metrica Game 3, and SkillCorner 1996435, 1996436 and 2006229.

## 4. Possession rules implemented

The rules are encoded in how the training labels are built and in the decoder.
There are no frame-specific rules or constants for this match.

- **Three states only.** A possession is a maximal run of H or A. D is a
  stoppage, never a possession.
- **Loose and contested balls stay with the last controlling team.** In the
  SoccerNet label rule, an uncontrolled opponent touch (a challenge, rebound or
  clearance) does not transfer possession. Possession changes only at the
  opponent's next *controlled* touch.
- **Restarts.** The ball is dead from an `OUT` event until the next touch. The
  next player to touch it after a stoppage takes possession, which matches the
  challenge's "clear possessor at a restart" definition.
- **Missing ball.** The ball is absent on 41% of graded frames, so the features
  include the time since and until the ball was last/next seen. During a ball
  gap, the GRU's context and the decoder's switch cost carry the current owner
  until the geometry shows a change.
- **Abstention.** `X` is written only on frames outside both halves (before
  kick-off, half-time, after the final whistle). Those frames are ungraded, so
  this costs nothing. Every frame inside play gets H, A or D.

## 5. Handling missing and noisy tracking

- **Masking, not imputation.** Unobserved players and ball are NaN-masked, and
  the model receives explicit observation flags and visible-player counts.
- **Training-time corruption.** We randomly drop out player track slots and
  inject ball gaps calibrated to the challenge feed's ball-missing fraction. For
  reference, the feed has 1.4 ball gaps per minute with a median length of
  10.7 s. We also train on team-swapped
  copies so that home and away are symmetric.
- **Stress testing.** Every candidate was also scored on a synthetic "feed-like"
  corruption of the held-out matches. A model had to pass that guardrail to be
  adopted.

## 6. Results (development; the full-match score is unknown to us)

| Evaluation | spellF1 | macroF1 |
|---|---|---|
| Five held-out external matches, clean tracking | 0.6472 | — |
| Same matches, synthetic feed-like corruption | 0.5058 | — |
| Organiser public 5-minute window (17 possessions; one-time format check, never used for selection) | 0.6303 | 0.8514 |
| Published floor `baseline_smooth50`, public window | 0.3256 | — |

The submitted file predicts 264 live possessions, with a mean length of 13.2 s.

## 7. Limitations

- **Domain shift.** Every training match comes from another provider. Automatic,
  uncorrected tracking with frequent ball loss is only approximated by our
  synthetic corruption, so the full-match score could fall nearer the feed-like
  number.
- **Short possessions.** Spells under 2 s are recalled poorly: about 22–33% at
  IoU 0.5 on validation. The switch cost trades these for fewer false splits.
- **Silver labels.** The SoccerNet labels come from a rule over event metadata,
  and the providers' definitions of possession differ slightly from the
  challenge's corrected-tracking definition.
- **No independent confirmation.** All five development matches were used
  repeatedly during research. There is no fresh hand-labelled test set.

**With more time** we would:
- fit the label rule to each provider's possession definition and add
  source-conditioned heads;
- build an independent, blind-labelled test match on a different game;
- improve the tracking itself, by reconnecting fragmented tracks and
  interpolating ball gaps.

## 8. Reproduction

Environment: Python 3.12 or 3.13, CPU only. It was verified on 2026-10-31 with
Python 3.13.18, torch 2.8.0+cpu and scikit-learn 1.8.0.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r experiments/requirements-temporal.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

1. Put the organiser files `BAR-ATH_tactical_atd.xml` and
   `BAR-ATH_tactical_atd.txt` in `data/tactical/`.
2. Put the four model weights in `data/models/joint_gru_sn_dense_ensemble_v1/`:
   `dense_2026.joblib`, `teacher_2026.joblib`, `dense_2027.joblib` and
   `teacher_2027.joblib`. *(Download location: <weights_url>.)* Their SHA-256
   values are checked automatically against `experiments/baseline_manifest.json`.
3. Run:

```bash
submission/make_submission.sh <team_name>
```

This prepares the inputs from the raw ATD, runs inference, validates the file
and prints its hash. The expected SHA-256 of the CSV is
`57ca788056ccf2f41839730fba492f15885c69f5788f35146ba1610b80b5b22b`.
Inference is deterministic: we regenerated the file from the raw ATD in a fresh
environment and the hash matched byte for byte.

**Retraining from scratch** is optional, because the weights are trained only on
the external data above. The recipe and runners are documented in
`experiments/SOCCERNET_DENSE.md` and `experiments/BASELINE.md`, and the
SoccerNet download steps are in `experiments/SOCCERNET_DATA_ACCESS.md`.
