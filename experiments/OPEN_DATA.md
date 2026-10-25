# More open data, closer to the challenge

Goal: training and validation data that look like the challenge, a Metrica
automatic track of a full match at 25 Hz with the ball missing on 41% of graded
frames, labelled H/A/D under "loose ball stays with the last team". The fork
validated on one Sportec match, whose frozen dead-ball coordinates make D
nearly free. This adds two labelled H/A/D sources and a cross-source protocol.

## New sources

| Source | Matches | Hz | Format | Labels (how) | Ball missing, graded |
| --- | ---: | ---: | --- | --- | ---: |
| Sportec Open DFL (existing) | 4 used | 25 | DFL XML | official BallStatus/BallPossession | ~0 (frozen in D) |
| **Metrica sample games** | 3 | 25 | CSV ×2, **FIFA EPTS ×1** (the challenge's format) | **H/A/D from Metrica's human events** | 32–41% |
| **SkillCorner open data, H/A/D** | 11 (was 5, H/A only) | 10 | JSONL | H/A provider possession; **D from `game_interruption` events** | 35–48% |

### Metrica sample data (`metrica_open.py`)

[metrica-sports/sample-data](https://github.com/metrica-sports/sample-data),
pinned at `e706dd5`. Same provider as the challenge; Game 3 ships in the same
FIFA EPTS metadata/text pair as `BAR-ATH_tactical_atd.*`. No licence file; the
README asks for responsible use and acknowledgement of Metrica Sports.

Labels come from the human event stream only; tracking is never consulted:

- **Live:** team of the latest PASS / CARRY / SHOT / SET PIECE / RECOVERY /
  BALL OUT event. BALL LOST and CHALLENGE do not change the team: a loose ball
  stays with the last team until the opponent's RECOVERY, which is the
  challenge's rule.
- **D:** from a stoppage trigger (BALL OUT end, shot OUT/GOAL end, OFFSIDE,
  HAND BALL, FAULT RECEIVED) to the next restart event. A set piece with no
  recorded trigger is dead from the end of the previous event (11–12 per match).
- **X:** before the first and after the last event of each half.

The fork's audit had set Metrica aside as "circular" because labels would
come from a nearest-player rule. Building them from the event stream avoids that.

**Independent check of the D timing.** Metrica's sample tracking blanks the
ball exactly while it is out of play. The event-derived D intervals match the
ball-absent intervals at frame IoU **0.918 / 0.969 / 0.991** (games 1/2/3).
Median boundary offset is one frame at the start and zero at the end. The
labels and the blanking come from different pipelines, so this confirms the
derivation.

| Game | Graded min | H / A / D share | Spells | Spell s, mean / median | Spells < 3 s |
| ---: | ---: | --- | ---: | --- | ---: |
| 1 | 94.7 | .31 / .29 / .41 | 354 | 9.6 / 5.0 | 33% |
| 2 | 93.4 | .31 / .29 / .40 | 321 | 10.4 / 5.9 | 30% |
| 3 | 95.1 | .35 / .33 / .32 | 382 | 10.2 / 5.4 | 33% |

**Caveat: this is a shortcut source for D.** The ball is missing on
0.4–1.3% of live frames but 93–99.8% of dead frames, so here "ball missing"
means D. The challenge's missingness comes from occlusion during play
([DATA.md](../DATA.md)). Use Metrica for D only through player-only features,
or after re-masking the ball during live play. Use it freely for H/A
attribution at 25 Hz and for spell-length priors.

### SkillCorner with D labels (`skillcorner_had.py`)

The pinned [SkillCorner open data](https://github.com/SkillCorner/opendata)
commit now holds 11 A-League 2024/25 matches with `dynamic_events.csv`.
`datasets.py` used 5 matches and H/A only, because tracking carries no
ball-status field. Player-possession events carry `game_interruption_after`
/ `game_interruption_before` (throw-in, goal kick, corner, free kick, goal),
which bound every stoppage:

- **D core:** end of the possession tagged with an interruption-after, to the
  start of the next player possession, shrunk by 1 s at each end. The margins
  are X: the ball may still be rolling out, or the restart pass may be in flight.
- **Live:** provider `possession.group`; nulls inside live play keep the last
  team (challenge rule).

70–105 stoppages per match; D share .38–.56 (D runs median ~22 s, because
the restart is timed at the first reception). Silver labels, as before.

**Caveat: a broadcast-camera shortcut.** During D the broadcast cuts to
close-ups and replays, so only ~4 of 22 players are detected in D frames versus
~14 live, and the ball ~25% vs ~91%. The challenge's reference camera is the
tactical feed (no cuts, 20.5 tracked objects on average). Experiments below
blank undetected positions (challenge NaN convention) and drop frames with
fewer than 12 detected players from both training and scoring.

## Every source has its own D shortcut

| Source | What gives D away | In the challenge's tactical feed? |
| --- | --- | --- |
| Sportec | ball XY frozen during stoppages | not known; tracking is automatic and noisy |
| Metrica sample | ball blanked during stoppages | no: 41% missing, mostly occlusion in play |
| SkillCorner | broadcast cuts away; few players visible | no: tactical camera, no cuts |

Selecting a model on any one of these rewards that source's shortcut. This fits
the fork's public-window diagnostic: the Sportec-selected D head scored D F1
0.34–0.37, against 0.75 for a player-only stoppage head
([AGENTS.md](../AGENTS.md)). That was a pipeline check, not a selection.

## Result 1: a player-only D detector transfers across providers

`dead_transfer.py` fits the fork's player-only stoppage head
(`refine_domain.features`: collective speed, shape and touchline occupancy, no
ball) on every mixture of sources and scores frame D F1 / AUROC on each source's
held-out matches. Each source gets equal total weight. Splits were fixed before
fitting: Sportec J03WOY, Metrica game 3, and the last 3 SkillCorner IDs. Cells
in parentheses include the evaluation source in training.

| Train \ eval | Sportec | Metrica | SkillCorner |
| --- | --- | --- | --- |
| sportec | (0.913 / 0.979) | 0.849 / 0.951 | 0.654 / 0.862 |
| metrica | 0.879 / 0.964 | (0.840 / 0.957) | 0.659 / 0.860 |
| skillcorner | 0.700 / 0.927 | 0.712 / 0.896 | (0.754 / 0.925) |
| sportec+metrica | (0.907 / 0.978) | (0.848 / 0.956) | 0.671 / 0.871 |
| sportec+skillcorner | (0.901 / 0.975) | 0.848 / 0.955 | (0.751 / 0.923) |
| metrica+skillcorner | 0.872 / 0.966 | (0.836 / 0.956) | (0.754 / 0.922) |
| all three | (0.907 / 0.976) | (0.849 / 0.958) | (0.750 / 0.922) |

- Between the two full-pitch trackers the player-only signal transfers almost
  losslessly: Sportec→Metrica 0.849 against Metrica→Metrica 0.840. The labels
  come from different annotation pipelines (official BallStatus vs human events).
- Partial broadcast views are a different domain: SkillCorner is ~0.10 lower
  in-source and transfers worse both ways.
- Pooling costs at most 0.006 F1 against the in-source specialist on every
  split, so the pooled head is the safe choice for an unseen provider.
  The challenge's tactical feed shows 20.5 of 22 players on average, closer to
  the full-pitch sources.

One evaluation match per source (three for SkillCorner); these are point
estimates, not intervals.

## Result 2: full H/A/D, official metric, three providers

`multisource_eval.py` trains the refined model family (`refine_features`,
clean + 41%-masked ball, team swap, HGB 63/260, equal weight per source). It
scores each training mixture with the official metric on every evaluation
split. D comes either from the full model or from a player-only head trained
on the same sources. Cells are frame macro F1 / spell F1 / D F1, averaged over
a source's evaluation matches, at switch cost 3. `*` marks a source that is in
the training mixture.

| Train | D head | Sportec J03WOY | Metrica game 3 | SkillCorner ×3 |
| --- | --- | --- | --- | --- |
| sportec | full model | *.936 / .672 / .999 | .887 / .648 / .874 | .565 / .392 / .228 |
| sportec+metrica | full model | *.937 / .680 / .999 | *.953 / .769 / .976 | .610 / .399 / .352 |
| sportec+skillcorner | full model | *.930 / .658 / .999 | .901 / .664 / .903 | *.837 / .500 / .864 |
| all three | full model | *.935 / .678 / .999 | *.950 / .769 / .970 | *.836 / .500 / .865 |
| sportec | player-only | *.875 / .576 / .924 | .876 / .621 / .858 | .731 / .406 / .671 |
| sportec+metrica | player-only | *.871 / .576 / .920 | *.887 / .675 / .865 | .744 / .418 / .699 |
| sportec+skillcorner | player-only | *.864 / .560 / .917 | .877 / .631 / .864 | *.794 / .474 / .773 |
| all three | player-only | *.873 / .586 / .924 | *.887 / .665 / .865 | *.793 / .477 / .774 |

All 80 rows (both heads, costs 3 and 8, per match) are in
`results/multisource_eval.json`.

What it says:

1. **The single-match validation picks the wrong D head.** On Sportec J03WOY
   the full model wins clearly: macro .936 against .875, D F1 .999. On
   SkillCorner, a provider it never saw, the same full model's D collapses
   (D F1 .23, macro .565), while the player-only head holds (.67 / .731). That
   is the same reversal as the public-window diagnostic on the challenge
   (full-model D F1 .34–.37, player-only .75).
2. **More sources help on providers the model has not seen, at no cost in-source.**
   - Unseen SkillCorner: adding Metrica raises macro by +.045 (full model) and
     +.013 (player-only). At cost 8, player-only spell F1 rises from .432 to .451.
   - Unseen Metrica: adding SkillCorner gives +.014 macro and +.016 spell.
   - Sportec validation stays within ±.015 spell for every mixture, and the
     pooled model is +.006 over Sportec-only.
3. **Team attribution is the spell-F1 bottleneck on broadcast data.**
   - Even with D fixed, SkillCorner spell F1 stays at or below .52, with
     team accuracy on live frames .83–.85.
   - On Metrica (25 Hz, full view) team accuracy on live frames is .94–.95.
   - The SkillCorner views are partial and 10 Hz, with ball positions NaN when
     undetected. That is closer to the challenge's 41% missing ball than
     Metrica's clean live ball.
4. **The in-source Metrica numbers are optimistic.** "Ball missing" means D
   there (see the caveat above). The full model's .976 D F1 on Metrica
   game 3 after training on Metrica largely reflects that shortcut. Use the
   unseen-source columns to judge.

Practical consequences:
- Choose D heads, blend weights and decoder costs on unseen-source cells.
- Pool all sources for the final model.
- Spend the next modelling effort on H/A attribution under ball loss. Spend
  data effort on a large broadcast-like set (PFF World Cup 2022, see
  [DIRECTIONS.md](DIRECTIONS.md)).

Limits: one evaluation match for Sportec and Metrica, three for SkillCorner;
no seed repeats; SkillCorner labels are silver. The Sportec-only rows were
recovered from the first run's log, so they are rounded to 4 decimals.

## Reproduce

```sh
python experiments/metrica_open.py --download       # data/external/processed/metrica_{1,2,3}.npz
python experiments/skillcorner_had.py --download    # data/external/processed/skillcorner_had_*.npz
python experiments/audit_dataset.py --matches J03WMX J03WN1 J03WOH J03WOY
python experiments/dead_transfer.py                 # results/dead_transfer.json
python experiments/multisource_eval.py              # results/multisource_eval.json
```

All downloads and arrays stay under the gitignored `data/`.
