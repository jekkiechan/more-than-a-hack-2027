# Ball-exit correction of the frozen dead-ball mask

Study `ball_exit_dmask_v1` · 11 November 2026 · complete; one development candidate
retained; the >0.60 v2 target is still unmet.

## Result and decision

**Retain `snap_m0.0` as a development candidate. Do not change the adopted package.**
It moves the start of each predicted dead-ball run to the moment the ball is observed
leaving the pitch. Nothing is fitted and under 1% of frames change.

| Candidate | Clean spell | V1 spell | V2 spell | Short recall clean / v1 / v2 | Retention |
| --- | ---: | ---: | ---: | --- | --- |
| Approved frontier (control, exact replay) | 0.670137 | 0.532128 | 0.593619 | 0.409 / 0.220 / 0.289 | — |
| `delay_m0.0` (round 1) | 0.679156 | 0.534334 | 0.596884 | 0.450 / 0.230 / 0.307 | passes |
| `advance_m0.0` (round 2) | 0.672530 | 0.532528 | 0.595445 | 0.411 / 0.221 / 0.289 | passes |
| **`snap_m0.0` (round 2)** | **0.681508** | **0.534731** | **0.598690** | 0.452 / 0.230 / 0.308 | passes |
| `snap_m0.5` (round 2) | 0.675232 | 0.534661 | 0.596201 | 0.436 / 0.229 / 0.299 | passes |
| `force_m0.0` (round 1) | 0.659908 | 0.527044 | 0.585594 | 0.414 / 0.227 / 0.298 | fails all three spell means |
| `both_m0.0` (round 1) | 0.669028 | 0.529339 | 0.588881 | 0.456 / 0.237 / 0.317 | fails all three spell means |

`snap_m0.0` versus the frontier: clean **+0.011371**, v1 **+0.002602**, v2 **+0.005071**
spell; macro +0.0043 / +0.0012 / +0.0027; D-F1 +0.0046 / +0.0007 / +0.0036. Every
retention check passes. V2 spell is **0.0013 below 0.60**.

### Provider spread

| `snap_m0.0` Δ spell | Sportec | Metrica | SkillCorner |
| --- | ---: | ---: | ---: |
| Clean | +0.0202 | +0.0219 | -0.0080 |
| V1 | +0.0043 | +0.0090 | -0.0055 |
| V2 | +0.0115 | +0.0089 | -0.0052 |

All eight Sportec and Metrica v2 cells improve (+0.006 to +0.015). All twelve
SkillCorner v2 cells decline (-0.001 to -0.008). With SkillCorner unchanged the v2 mean
would be 0.6004; that is a description of where the gap sits, not a result.

The SkillCorner loss follows its label convention. Its event-derived silver labels start
dead ball before the ball physically leaves: the first outside observation comes a median
0.4 s (75th percentile 1.8 s) after the labelled start, versus 0.1–0.2 s before it on
Sportec and Metrica. Frames the delay edit re-opens are live in 97–100% of graded cases
on Sportec and Metrica and in 22–32% on SkillCorner. Which convention the challenge
labels follow is not established here.

## Why the dead-ball mask

Every frontier study since the phase-average package has kept the predecessor's decoded
D mask, which comes from a player-only D path. Diagnostics on the exposed cells:

- An oracle D mask with unchanged H/A adds 0.07–0.09 spell F1 per inspected cell.
- On Sportec 17–21% of reference possessions are lost because the mask predicts D over
  them, more than are lost to team confusion (10–13%). Most are short spells just before
  a stoppage. Their pooled P(D) has a median of 0.40–0.47.
- The ball is observed outside the pitch on 0.5–1% of observed live frames and 59–76% of
  observed dead frames (Sportec, Metrica).
- The challenge feed shows the ball outside the pitch on 4.4% of observed frames, in 104
  episodes (inputs only; no labels read).

Two cheaper families were explored first and are not candidates: 42 uniform D-edge
shifts and 24 P(D)-thresholded edge trims. None improved the three spell means; the
frozen mask is the optimum of the uniform-shift family.

## Rule

Inputs are ball position and pitch size only. Ball speed and stillness are never read,
so a provider's frozen stoppage coordinates cannot trigger an edit.

- **delay**: a predicted D run that starts while the ball is observed inside the pitch,
  followed within 5 s by an outside observation, starts at that outside observation.
- **advance**: a predicted D run starts at the earliest outside observation within 3 s
  before it, when the ball is not observed inside again in between.
- **snap** = delay then advance. `m0.0` / `m0.5` is the distance beyond the line that
  counts as outside.
- **force** (round 1 only): every outside observation is dead, held up to 3 s.

H/A is then decoded inside the edited live spans with the unchanged frontier mixture
(0.5/0.375/0.125, H/A cost 3).

`force` failed because isolated outside sightings are dead in only 54–55% of graded
Metrica frames, and outside sightings right after a predicted D end are restarts (0%).
Outside sightings just before a predicted D start are dead in 75–88% of graded frames on
all three providers, which is what `advance` keeps.

## Design and evidence limits

Both rounds were frozen and committed before scoring:
[round 1](results/ball_exit_dmask_v1_declaration.json),
[round 2](results/ball_exit_dmask_v1_round2_declaration.json). Scorecards:
[round 1](results/ball_exit_dmask_v1.json), [round 2](results/ball_exit_dmask_v1_round2.json).
The control replays exactly (0.6701365499557103 / 0.532128306931943 / 0.5936187198896373).

- The failure was located with development labels on these five exposed matches, and
  round 2 was designed after round 1's results and label-based precision checks. Ten
  candidates were scored in total. Treat the gain as exposed-development evidence.
- Sportec keeps ball coordinates through stoppages, so `force` could profit from
  persistent outside positions; `delay` and `advance` act only at a run's start.
- `feed_like_v2` removes ball observations at state-dependent rates placed with
  development labels. The clean view has no such placement and shows the larger gain.
- A restart cue (ball re-entering) was checked and not pursued: it is precise only when
  the predicted D run was about to end anyway.
- No GAR VALID/TEST, organizer or public scoring. The exact pooled deployment has not
  been run with this rule.

Stop ball-exit rules, margins and constants on this cohort. Next: check `snap_m0.0`
once, unchanged, on the ten GAR TEST games with the exact deployment package, then
decide on adoption. A learned D path with boundary-relative ball features is the
natural follow-up if that check holds.

## Reproduction

Restore as for [`sc2020_realfeed_v1`](SC2020_REALFEED_V1.md#reproduction), then:

```sh
python -m experiments.sc2020_realfeed_prepare cells
python -m experiments.ball_exit_dmask_v1
BALL_EXIT_ROUND=2 python -m experiments.ball_exit_dmask_v1
```
