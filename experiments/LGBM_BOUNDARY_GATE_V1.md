# Gated boundary refiner: development gain does not transfer to GAR TEST

13 November 2026. **Not promoted; keep reconstructed Fable adopted.** A
confidence-gated, boundary-only use of the LightGBM dead-ball refiner passes
every development guard and leaves public spell F1 unchanged, but lowers spell
F1 on nine of ten GAR TEST games. The predeclared GAR rule fails.

## Question

[The pooled refiner](LGBM_PUBLIC_V1.md) improved development spell F1 and lost
.054 public spell F1. Can a deployment rule keep the development gain without
moving the public result, and does it hold on GAR TEST?

## Candidate

`boundary_conf0.6`, implemented in
[lgbm_boundary_gate_runtime.py](lgbm_boundary_gate_runtime.py). The unrestricted
refiner proposes a D mask (50/50 P(D) blend, frozen decoder and snap). An edit is
applied only if it moves the start or end of an existing baseline D run and the
refiner's mean P(D) inside the edit is at least .6 (live to D) or at most .4
(D to live). H/A decoding is unchanged.

Known limitation of the evaluated rule: it does not strictly preserve the D-run
count. An edit touching a block edge, or a confident extension and trim at the
same boundary, can add a run. Both cases are covered by unit tests.

## Development search (exposed data)

Six Haiku agents and one combination round scored 43 fixed arms on the 35
exposed development cells with provider-held-out refiners; 15 pass all guards.
[All arm aggregates](results/lgbm_boundary_gate_v1_development.json). Deltas
are against the adopted control. Challenge agreement is spell F1 against the
adopted prediction on the full challenge match, computed without labels.

| Arm | Spell clean / v1 / v2 | v2 D-F1 | v2 spell gain kept | Challenge agreement | Guards failed |
| --- | --- | ---: | ---: | ---: | ---: |
| Unrestricted refiner | +.0116 / +.0041 / +.0149 | +.0263 | 100% | .9409 | 2 |
| Confident edits (0.6) | +.0117 / +.0078 / +.0130 | +.0225 | 87% | .9501 | 0 |
| **Confident boundary shifts (0.6)** | +.0077 / +.0060 / +.0093 | +.0169 | 63% | .9730 | 0 |
| D-run end shifts only | +.0067 / +.0054 / +.0065 | +.0128 | 43% | .9856 | 0 |

Edit-level findings on development: adding dead ball is right only 17–38% of
the time on the v1 view and drives the v1 guard failures; removing dead ball is
right 75–88% of the time. On the challenge match the refiner mostly removes dead
ball (72% of edited frames), the opposite of every development view, and its
ball gaps are much longer (90th percentile 34 s against 15–23 s).

The arm was chosen from 43 on five exposed matches, so its development gain
carries selection optimism.

## Public regression check

Requested by the user on 12 November as a no-regression check, not an
evaluation. One sealed prediction, one gate, official evaluator, 7,500 graded
frames and 17 possessions. [Aggregate](results/lgbm_boundary_gate_v1_public.json).

| Metric | Adopted | Unrestricted refiner | Candidate |
| --- | ---: | ---: | ---: |
| Spell F1 | .623529 | .569697 | .623529 |
| Macro F1 | .864234 | .886569 | .860928 |
| D-F1 | .889372 | .924716 | .877950 |
| D precision / recall | .8182 / .9741 | .9666 / .8863 | .8697 / .8863 |

Spell F1 is unchanged; macro F1 and D-F1 are slightly lower. No tuning or
repeat scoring followed.

## GAR TEST10

[Declaration](results/lgbm_boundary_gate_gar_v1_declaration.json) frozen before
scoring; [runner](lgbm_boundary_gate_gar_v1.py); [result](results/lgbm_boundary_gate_gar_v1.json).
Exact adopted package and the pooled public refiner, ten TEST games, clean plus
the four fixed v2 realizations, realizations averaged within game and games
equally. The control replays the saved adopted scores within 1e-12 in all 50
game-views. Intervals are paired whole-game bootstraps, 2,000 draws.

| View | Metric | Adopted | Candidate | Delta | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Clean | Spell F1 | .709186 | .705303 | −.003883 | −.0060 to −.0018 |
| Clean | Macro F1 | .784717 | .780845 | −.003873 | −.0066 to −.0011 |
| Clean | D-F1 | .593243 | .587968 | −.005275 | −.0124 to +.0021 |
| V2 | Spell F1 | .709176 | .704097 | −.005080 | −.0079 to −.0021 |
| V2 | Macro F1 | .784865 | .779091 | −.005774 | −.0089 to −.0027 |
| V2 | D-F1 | .596512 | .585903 | −.010609 | −.0178 to −.0032 |

Spell F1 falls in nine of ten games in both views; only game 3855 improves
(+.0009 clean, +.0025 v2). The direction of error differs from the challenge
match: on GAR the candidate adds dead ball (D recall +.050 to +.067, precision
−.038), which is the edit direction that was unreliable on the development v1
view. About 2.1–2.2% of frames change.

The declared rule required spell F1 not down and macro/D-F1 not down more than
.001 in both views. **It fails on every term.**

## Decision

No promotion. The active manifest and decision policy are unchanged. The
development result on `feed_like_v2` did not predict GAR TEST for this
mechanism: the refiner's behaviour depends on the feed (it adds dead ball on
development and GAR, removes it on the challenge match), and the gate reduces
the size of that effect without correcting its direction.

Evidence limits: GAR uses generated silver labels on discontinuous segments at
about 29.97 Hz, while the refiner was trained on 25 Hz development matches.
TEST10 has repeated descriptive exposure. The public window has 17 possessions.

Next, if this line continues: test the D-to-live-only gated arm on GAR, since
that direction was reliable on development and GAR's loss comes from added dead
ball. That would be a new declared study; nothing was rerun after this result.

## Validation

Ten unit tests cover the gate. The runtime reproduces the sealed public
candidate CSV byte-for-byte. Prepared GAR records match the adopted package's
earlier run hash-for-hash. The report replays all 100 saved prediction scores
before aggregating. Native predictions and the exploration harness remain in
local ignored storage; no private upload is claimed. The research progress site
was not updated from this session.
