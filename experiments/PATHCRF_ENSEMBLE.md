# PathCRF retention test with the adopted GRU HGB baseline

**Keep GRU + HGB as the adopted baseline. Exclude PathCRF from the default
pipeline, retaining its code/checkpoint as an archived optional comparator.**
The fixed PathCRF integrations tested here lower both spell F1 and macro F1,
while adding about 13–15 minutes of inference per match on one CPU thread.

## What was compared

PathCRF predicts H/A attribution, not referee dead/alive status. It cannot be
averaged as another D head. Every candidate below uses identical 50/50 GRU +
ordered HGB P(D), the same Viterbi cost of 3, and one of three team sources:

1. The adopted baseline's fixed contextual HGB conditional H/A probabilities.
2. PathCRF H/A attribution replacing HGB attribution.
3. A fixed 50/50 HGB and PathCRF conditional-team average.

PathCRF shares are the original CRF's triangular overlap team vote fractions,
not calibrated posterior probabilities. Their argmax was checked against the
original adapter on synthetic tracking and matches exactly. No blend weights,
decoder settings, thresholds, or models were fitted during this comparison.
The original adapter, checkpoint, and historical experiments were preserved.

The checkpoint is trial 140 at upstream revision
`1b7c6f85a6b78c92e4543fde72ca2b86a933f997`, SHA-256
`55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9`.
All 3,431,031 parameters load strictly with `weights_only=True`. Model inference
uses geometry only, without ball or target fields, teacher forcing, or true live
boundaries. Window/stride remain the original adapter's fixed 10/5 seconds at
5 Hz. Roster preparation uses the existing challenge adapter within continuous
nonzero-period blocks; empty period-zero rows receive neutral team shares.

## Matched external results

The cohort is **Metrica game 3 and SkillCorner 1996435/1996436/2006229**.
Sportec J03WOY was explicitly excluded because PathCRF trained on it; J03WQQ
was its validation match. These four matches are outside the checkpoint's
Sportec training/validation sources, but remain historical development matches
for our research. This is not a fresh confirmation set.

All four full-match PathCRF outputs were frozen before new comparison scoring.
The unchanged official scorer produced 24 match-level evaluations: three team
variants, four matches, and pooled/provider-holdout D views. The main view below
holds each evaluation provider out of D-head training; HGB H/A stays pooled.
Average matches within source, then give the two sources equal weight.

| Team attribution with fixed GRU HGB D | Spell F1 | Macro F1 | 70/30 summary |
| --- | ---: | ---: | ---: |
| HGB baseline | **.5744** | **.8346** | **.6524** |
| 50/50 HGB plus PathCRF | .5414 | .8213 | .6254 |
| PathCRF alone | .4993 | .8090 | .5922 |

The blended variant loses **.03298 spell F1**, and the replacement loses
**.07510**. Both also lose macro F1, so official spell-first ordering and the
auxiliary weighted summary agree. These baseline numbers use **two providers**;
do not mix them with the earlier three-provider baseline score of .5746/.8449.

| D-head holdout source | HGB baseline spell | HGB plus PathCRF spell | PathCRF spell |
| --- | ---: | ---: | ---: |
| Metrica | .6837 | .6517 | .5984 |
| SkillCorner mean of three | .4650 | .4311 | .4002 |

Pooled-D results agree: baseline .5962 spell/.8650 macro, blended .5604/.8523,
and PathCRF .5140/.8421. This conclusion does not depend on the D training view.
Conditional H/A accuracy on graded reference-live frames falls from .8985 to
.8765 for the blend and .8675 for PathCRF replacement. The D probabilities
are fixed, but Viterbi's decoded D intervals can still change when team
likelihoods change.

## Missing observations and cost

SkillCorner needs **50.3%, 60.4%, and 53.1%** of player-slot samples filled in
the complete roster. Metrica needs essentially none, yet PathCRF is still worse
there. The observed loss cannot be attributed solely to partial-camera inputs.

The visibility diagnostic checks H/A on live frames with observed versus missing
ball. Only one SkillCorner game shows a missing-ball gain: on 665 such frames,
the team blend scores .6932 H/A accuracy versus .6586 for baseline. It is worse
on the other two SkillCorner matches and Metrica. Missing-ball group sizes are
small (178–665 live frames per match), so this does not establish a reliable
occlusion advantage or justify a visibility-based routing rule.

Inference took **800.5–911.1 seconds per match**, plus up to 53 seconds for
geometry preparation, with four one-thread processes running concurrently.
Strict loading, normalized vote shares, original-adapter argmax parity, empty
period exclusion, and unchanged D marginals are verified. All project tests
pass with the optional neural dependencies installed.

The conclusion applies to this pinned checkpoint, tracking adapter, and fixed
fusion. Learned calibration or fine-tuning were not tested. Keep the archived
component for future representative-match research; the current evidence does
not support paying its default runtime/dependency cost. The earlier organizer
PathCRF score used a different D head and only 17 possessions, and cannot select
an integration for the new baseline. The organizer gate was not reopened.

## Artifacts and reproduction

- [Full external comparison](results/pathcrf_ensemble_development.json)
- [Ball visibility diagnostic](results/pathcrf_visibility.json)
- [Baseline promotion and restore commands](BASELINE.md)

Raw tracking, reconstructed geometry, checkpoint weights, and frame outputs
remain under ignored `data/`. Only code, hashes, and aggregate scores are tracked.
The derived adapter retains MPL-2.0 attribution; see `licenses/pathcrf-MPL-2.0.txt`.

After preparing the same external data and pinned checkout as `PATHCRF.md`,
install the optional packages in `requirements-pathcrf.txt`, then run:

```sh
python -m experiments.pathcrf_ensemble freeze
python -m experiments.pathcrf_ensemble predict --match metrica_3
python -m experiments.pathcrf_ensemble predict --match skillcorner_had_1996435
python -m experiments.pathcrf_ensemble predict --match skillcorner_had_1996436
python -m experiments.pathcrf_ensemble predict --match skillcorner_had_2006229
python -m experiments.pathcrf_ensemble score
python -m experiments.pathcrf_visibility
```

These commands have completed; frozen inference/results are not overwritten.
Use a new declared experiment for different models or fusion choices.
