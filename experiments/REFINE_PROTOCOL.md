# Fresh confirmation for local model refinements

The separate protocol in `data/experiments/refinement_protocol.json` reserves **J03WQQ** for one final test of new local refinements. It does not change the original benchmark split.

| Role | Sportec match IDs |
|---|---|
| Training | J03WMX, J03WN1, J03WOH |
| Validation and model selection | J03WOY |
| Fresh local-model test | J03WQQ |
| Previously viewed; excluded from fresh claims | J03WPY, J03WR9 |

Freeze the model and configuration before scoring J03WQQ. Select only on J03WOY: seek per-frame macro F1 of at least 0.90, then maximize official spell F1 among eligible configurations; otherwise select the highest validation macro F1. Do not inspect test targets, derive input features or intervals from them, or retune after seeing the final result. Reusing the validation match for many experiments can overfit it; the new test exists to check that risk.

**J03WQQ is not an independent test for the published PathCRF trial 140 checkpoint.** It was that checkpoint's validation match. The fresh-test claim here applies to local models fitted only on the three declared training matches and selected on J03WOY. The already viewed J03WPY and J03WR9 cannot establish fresh held-out performance after further refinement.

## Prepared data

The existing `experiments/audit_dataset.py` downloader and converter prepared `data/external_audit/sportec/processed/J03WQQ.npz`: 142,345 frames at 25 Hz. The raw files and provenance remain in the same Sportec directory. The equivalent standalone command is:

```bash
python experiments/audit_dataset.py --matches J03WQQ --workers 2
```

Source: [Sportec Figshare version 1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0. Metadata file `51643484` has MD5 `5979209af316a914ce57347d0633df6c`; observed tracking file `51643529` has MD5 `9105bad987191d3c3c362c1fce9a4fcd`. Sizes and MD5 hashes were verified by the existing downloader. The converter's output was redirected to an unread local log to avoid exposing test-label summaries during preparation. No predictions or scores were computed in this preparation step.

The original `data/external_audit/sportec/split.json` retained SHA-256 `b945a12df761cf5f1dddc0900b99f829841fd11cf07342538ea7fdc7da7401fc` before and after conversion. The new protocol's SHA-256 at creation is `eab9db9f239d7c748f2cef7970594994fa7897f742448a6452ee08b600ed17d3`.

Inference should consume an allowlist of coordinates, team assignments, observation masks, pitch dimensions, periods, timestamps, and FPS. Keep `labels`, raw BallStatus, and raw BallPossession out of the model input. Any use of Sportec's upstream frozen-ball dead-period artifact must remain explicit; another strong score on this source would still not establish generalization to broadcast ball tracking.

## Automatic Smart Tagging audit

The supplied challenge Smart Tagging XML was inspected only as an automatic input source. No challenge ground-truth file was opened, no possession accuracy was calculated, and no frame-specific correction or rule was created. Aggregate findings are saved locally in `data/experiments/smart_tagging_audit.json`.

Each SportsCode instance has `ID`, `start`, `end`, `code`, and grouped textual `label` values. The export contains no confidence score, model version, or model checkpoint. Its 638 instances use 14 codes:

| Code | Count |
|---|---:|
| PLAYERS IN THE BOX | 97 |
| BALL IN FINAL THIRD | 93 |
| BALL IN THE BOX | 72 |
| PROGRESSION | 66 |
| DEFENDING IN MIDDLE THIRD | 66 |
| SET PIECES | 59 |
| CREATING CHANCES | 30 |
| DEFENDING IN DEFENSIVE THIRD | 30 |
| LONG BALL | 29 |
| ATTACKING TRANSITION | 24 |
| DEFENSIVE TRANSITION | 24 |
| BUILD UP | 22 |
| DEFENDING IN ATTACKING THIRD | 22 |
| GOALS | 4 |

Tag groups are `Team`, `Half`, `Type`, `Max Players in the box`, `Side`, and `Direction of ball entry`. Team values are `Barcelona` (314 instances), the literal spelling `Atletic Club` (288), and `N/A` (36). A defensive phase's Team identifies the defending side, so it cannot be interpreted directly as the possessing side.

The repository documents timestamps as seconds from the first tactical-video frame, converted with `round(seconds * 25)` to half-open frame intervals. They are not timestamps on the broadcast feed. The loaded intervals are all within the tactical-video duration. Their union covers 70.03% of all video frames and 82.27% of the published half blocks. Multiple tags overlap on 58.08% of half-block frames; both team names occur simultaneously on 42.17%. These overlaps often reflect paired attacking and defending descriptions, not a detector conflict: all 22 build-up intervals, all 66 progression intervals, and all 24 attacking-transition intervals have an identically timed defensive counterpart naming the opposite team. Creating-chance/defending-third pairs match this way in 28 of 30 cases.

Restart timing is too broad for a direct dead-ball label:

| Type | Count | Interval duration | Known team |
|---|---:|---:|---:|
| Throw-in | 26 | 35 s | 0 |
| Corner Kick | 12 | 35 s | 12 |
| Goal Kick | 10 | 35 s | 10 |
| Free Kick | 5 | 35 s | 5 |
| Kick Off (After Goal) | 4 | 5 s | 0 |
| Kick Off (Start) | 2 | 5 s | 0 |

Thus only 27 of 59 restarts provide a team, and the 53 non-kickoff intervals are fixed 35-second windows. The interval boundaries alone do not establish the instant play stops or restarts. Across all tag types, interval durations range from 5 to 250.72 seconds. Copying an entire tag interval into possession or dead-ball labels would confuse tactical context with state duration.

Generic soft evidence remains plausible: attacking/defending phase semantics can support team attribution during ball gaps, while set-piece context can raise the probability of a nearby stoppage/restart when coordinate motion agrees. Its reliability and exact temporal relationship have not been measured here. Do not invent offsets from individual challenge examples or force interval-wide labels. The open Sportec benchmark does not provide the same Metrica tags, so it cannot validate the incremental value of this feature without another permitted source of equivalent automatic outputs.

## Licensing and reproducibility

The Metrica export is explicitly automatic and is an allowed challenge input under `RULES.md`. It is a supplied output, not an open-source possession model: no generating checkpoint or training data is distributed here. The XML and challenge tracking/video are licensed for the competition only, and must remain outside the public repository. Do not redistribute the XML or frame-level derived exports. The starter loader and repository code are MIT licensed; that code license does not extend to the assets. Any final method using the tags must document that dependency and regenerate its results from the authorized asset bundle.
