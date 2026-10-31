# Provider touch-definition calibration

## Result

The corrected, TRAIN-only Sportec study selected `hold` ownership through an explicitly non-changing duel and a 0-second out-tail grace in all three leave-one-match-out folds. This is a source-specific, deterministic rule comparison against Sportec's own `BallPossession` labels. It does not establish that the GAR touch-run rule should change, and it is not a full-provider transfer result: Sportec contributes only three exposed TRAIN matches, and the GAR-specific CL single-frame and censored-run dimensions have no direct counterpart in these event records.

The fixed-candidate results below are paired per-match diagnostics already present in the frozen v2 report's inner-fold scores. Each match appears twice there and the values match exactly. They are not additional folds, and they did not select the outer-fold configuration. `known` is the fraction of predicted frames labeled H/A/D rather than X; `X` is the predicted-unknown count. Reference coverage is 100% for every match, which is distinct from prediction coverage. The official spell scorer fills X within reference spells, so spell F1 must be read together with known rate and macro F1.

| Match | Fixed rule | Spell F1 | Macro F1 | Predicted known | Predicted X |
| --- | --- | ---: | ---: | ---: | ---: |
| J03WMX | hold, 0 s | 0.4447 | 0.8018 | 95.68% | 6,312 |
| J03WMX | hold, 3 s | 0.4422 | 0.8021 | 97.07% | 4,280 |
| J03WMX | literal challenger, 0 s | 0.3703 | 0.8013 | 95.70% | 6,278 |
| J03WMX | literal challenger, 3 s | 0.3626 | 0.8016 | 97.07% | 4,278 |
| J03WN1 | hold, 0 s | 0.2721 | 0.6997 | 94.99% | 7,090 |
| J03WN1 | hold, 3 s | 0.2703 | 0.6982 | 96.58% | 4,846 |
| J03WN1 | literal challenger, 0 s | 0.2564 | 0.6995 | 95.02% | 7,051 |
| J03WN1 | literal challenger, 3 s | 0.2498 | 0.6980 | 96.58% | 4,842 |
| J03WOH | hold, 0 s | 0.4650 | 0.7539 | 95.73% | 5,863 |
| J03WOH | hold, 3 s | 0.4539 | 0.7551 | 97.33% | 3,662 |
| J03WOH | literal challenger, 0 s | 0.3520 | 0.7533 | 95.75% | 5,834 |
| J03WOH | literal challenger, 3 s | 0.3370 | 0.7545 | 97.33% | 3,658 |

Within the hold rule, the 0-second tail has about 1.39–1.60 percentage points lower predicted coverage than 3 seconds, while spell F1 is 0.0017–0.0111 higher and macro F1 changes by at most 0.0016. This is a coverage tradeoff, not evidence of an unqualified quality gain. Holding through non-changing duels has higher spell F1 than assigning the challenger literally on all three matches; macro F1 changes by less than 0.001. The outer-fold selected scores are J03WMX 0.4447/0.8018, J03WN1 0.2721/0.6997, and J03WOH 0.4650/0.7539 (spell/macro F1). The large match-to-match variation, particularly the lower J03WN1 score, argues against claiming a robust improvement.

## Source mapping and limits

The initial [field inventory](results/provider_touch_feasibility_v1.json) did not
inspect Sportec's raw event XML. Its field-name-based feasibility conclusion is
superseded by the native-event mapping and corrected v2 comparison below.
Different provider field names or different fixtures alone do not rule out
calibrating on each provider's own paired events and possession references.

The Sportec raw event XML contains event-level `Play`, `OtherBallAction`, `BallClaiming`, `TacklingGame`, and restart/stoppage records. The adapter maps `Play`, `OtherBallAction`, `BallClaiming`, and `TacklingGame` with `PossessionChange=true` to an owner anchor at the event time. For `TacklingGame` with `PossessionChange=false`, it identifies the challenger from the participant whose role is `withoutBallControl`; the two tested interpretations either keep the current owner or assign the challenger at that frame. Restart `DecisionTimestamp` marks dead-ball onset; restart `EventTime` marks the restart, with team assigned only when the nested action has an explicit team. Foul, offside, final whistle, disallowed goal, wide shot, successful shot, and goal events mark dead-ball onset. Event timestamps are placed on the nearest processed tracking timestamp, at most 0.5 seconds away.

The v2 mapping intentionally does not reproduce the complete SoccerNet-GAR dense-label recipe. Sportec event records are not contiguous GAR touch runs: they cannot support CH/RE/CL run-ending behavior, CL's single-frame exception, or right-censored touch-run handling. This study tests a defensible subset of the rule family—owner anchors, non-changing opponent duels, and an out-tail grace—using Sportec's own event and label definitions. The event adapter never uses target labels to make anchors, classify event types, select timestamp origins, or choose alignment thresholds. No learned predictor, normalization, or model head was fit.

Metrica Games 1–2 provide event rows with team, type, and start/end frames, but their dense target is derived from those same event semantics; scoring an event-derived mapping against it would be circular. Game 3 is selection data and was excluded. SkillCorner's event stream contains possession windows and interruption-before/after annotations, but those do not identify GAR-style controlled or uncontrolled touch runs; its selection matches were excluded. The three Sportec TRAIN fixtures are not paired with SoccerNet fixtures, and no cross-fixture pairing was attempted. All Sportec selection/test, Metrica, SkillCorner, SoccerNet validation/test, and organizer labels were excluded from selection and scoring.

The restored provider data used here were not present in this session's initial checkout. The five Sportec/Metrica processed NPZs were restored to the private `mtah-data` cache; event-only raw files were restored under ignored local `data/` paths from documented source receipts. This is a session-local availability statement, not a claim that these datasets were never historically acquired. Only the three Sportec TRAIN NPZs and matching raw event/match-information XML were inputs to v2.

## Clock correction and frozen evidence

The first attempt, v1, used `KickoffTime` as the event clock origin. That is incorrect for these processed Sportec tracking arrays, whose timestamps begin at the first observed position frame. V1 is invalid and must not be cited as a calibration result. The pinned Figshare positions-file prefixes show the first observed frame (`N=10000`) and timestamp below; v2 uses each timestamp as time zero. The offset column is first-frame time minus kickoff time.

| Match | Figshare positions file | First frame UTC | Offset from kickoff |
| --- | ---: | --- | ---: |
| J03WMX | 51643514 | 2023-05-27 13:30:12.600 | +0.38 s |
| J03WN1 | 51643517 | 2023-05-27 13:31:06.400 | −2.24 s |
| J03WOH | 51643520 | 2022-08-26 16:32:09.720 | +1.62 s |

V1 declaration, report, and source snapshot hashes are respectively `50685cbbf136acf99ddc2c07cd041db50b55b4c7b2c8f4452ddbeaab22b22908`, `23858874fc7fef863efd35f7033dd0fac2d1524921019eb96f405d29785e639b`, and `db29d94c834c2f3252490f78e297dda0cc961bdf2bf1157cc61fefec0f9c476e`. The invalid result is explicitly marked in [the v1 alignment correction](results/provider_touch_sportec_train_nested_v1_alignment_correction.json), and its exact source is retained under ignored `data/experiments/provider_touch_sportec_train_nested_v1/`.

The corrected v2 protocol was frozen before scoring. Its declaration hash is `3b1f41e52ca5ef2cdf7bdbfb0e6848b15bd3c1c4ce0a7226e98ebda26541fed8`, runner hash is `75baf21990f1e9b52451e704609de48226db2c5ee11f4da5aa794a4c9ef89e81`, and scored report hash is `0cd85035d62fb5a36f230eda134ee808c4cc65759db76aedeaaa1b89a0c02caa`. The report includes SHA-256 and byte counts for all 12 inputs (three event XMLs, three match-information XMLs, three processed reference NPZs, and three source receipts), plus hashes for the manifest and scoring/mapping code. The pre-score code snapshot is under ignored `data/experiments/provider_touch_sportec_train_nested_v2/`.

The [public aggregate summary](results/provider_touch_sportec_train_nested_v2_summary.json)
omits the official evaluator’s per-interval match arrays. The byte-identical full
v1/v2 scored outputs remain in ignored local storage, with their original hashes
above; the [publication receipt](results/provider_touch_publication_receipt.json)
records the relocation and both summary hashes. No scores were recomputed to
make these summaries. The full private output is local-only, not remotely archived.

Reproduce the frozen run from the repository root with:

```bash
python -m experiments.provider_touch_calibration_v2 run \
  --data-dir data \
  --source-root /workspace/mtah-data \
  --declaration experiments/results/provider_touch_sportec_train_nested_v2_declaration.json \
  --output /tmp/provider_touch_sportec_train_nested_v2.reproduced.json
```

The runner verifies frozen source/input hashes before scoring. It uses three whole-match outer folds; within each fold, it evaluates the four predeclared fixed candidates on the other two complete matches and selects by equal-match mean spell F1, with macro F1 as tie-break. This is nested selection among fixed rules, not nested-CV training of a predictor. The study is exploratory, limited to previously exposed development matches, and does not authorize model promotion or a claim of fresh confirmation.

Synthetic adapter and selector contracts are in `tests/test_provider_touch_sportec_adapter.py`, `tests/test_provider_touch_calibration.py`, and `tests/test_provider_touch_feasibility.py`.
