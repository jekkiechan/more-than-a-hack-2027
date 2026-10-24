# Possession challenge work plan

Status: 2026-10-24. Based on upstream commit `f8f2629`.

## Goal and inputs

Phase 1 asks for one possession label for each of the tactical video's 199,319
frames at 25 fps. The match is Barcelona (home) against Athletic Club (away).
Possession percentages are a possible summary; the competition evaluates the
underlying timeline, primarily as complete possession spells.

Metrica provides automatic player and ball tracking, team membership, tactical
tags, two videos, and camera-to-pitch homographies. Start with these assets.
Rebuilding detection and team classification is optional, not a prerequisite.

The main challenges are missing ball observations (41% of graded frames), noisy
tracking, control changes, and distinguishing stoppages from occlusion.

## Output and evaluation contract

- CSV header: `frame,label`; frames `0` through `199318`, in order.
- `H`: home in possession while play is live.
- `A`: away in possession while play is live.
- `D`: dead ball, regardless of restart ownership.
- `X`: abstention; allowed, but not a separate ground-truth possession state.
- Loose and contested balls retain the last controlled team's possession until
  the opponent gains control. Do not add a FIFA-style contested output class.
- Primary metric: `spellF1`, matching possession intervals by temporal IoU and
  averaging F1 over thresholds 0.3, 0.4, 0.5, 0.6, and 0.7.
- First tie-break: per-frame `macroF1` over H/A/D.
- The temporal scorer fills X within each half, so abstaining across a change of
  possession can extend the wrong label. MacroF1 counts graded abstentions as misses.

See [CHALLENGE.md](../CHALLENGE.md), [EVALUATION.md](../EVALUATION.md), and
[SUBMISSION.md](../SUBMISSION.md) for authoritative details.

## Verified setup

- [x] Clone the public upstream with its Git history.
- [x] Preserve upstream as the `upstream` remote.
- [x] Create a local `possession/research` branch.
- [x] Run the official scorer on all four bundled baselines.
- [x] Validate the bundled `baseline_smooth50.csv` submission format.
- [x] Connect the personal GitHub fork as `origin` and push `possession/research`.
- [x] Retrieve the tactical tracking, metadata, Smart Tagging, and homography from
  the participant data bundle.
- [x] Verify tracking and homography frame alignment and the tagging time base.
- [x] Regenerate all four reference baselines from the actual tracking assets.
- [x] Confirm every regenerated CSV matches the corresponding upstream baseline
  byte for byte, passes validation, and reproduces the public scores below.

Working fork: [jekkiechan/more-than-a-hack-2027](https://github.com/jekkiechan/more-than-a-hack-2027),
branch `possession/research`. `upstream` remains the organizers' repository.

Four tactical input files (about 219 MB combined) are available locally in
`data/tactical/`. They are ignored by Git. The multi-gigabyte videos and the
broadcast inputs have not been downloaded; the reference baseline needs only the
tactical tracking metadata and raw tracking file.

The scorer runs in the current environment without additional packages:
Python 3.12.14, NumPy 2.3.5, pandas 2.2.3. No GPU is needed for this baseline.

```bash
python starter/baseline.py --out data/generated_baselines
python evaluation/validate_submission.py data/generated_baselines/baseline_smooth50.csv
python evaluation/run_evaluation.py --results data/generated_baselines --out data/baseline_evaluation --no-detail
```

On the delivered tactical assets, the checks confirmed 199,319 consecutive ATD
records (IDs 1 through 199,319), 46 team-assigned track slots, 638 tactical-tag
instances, and 199,319 valid, aligned homography rows. The ball is observed in
94,214 of 158,857 graded frames: 40.69% are missing. Both tagged kickoff intervals
contain the starter's published kickoff anchors. These reproduce the data
description; no human labels or parameter tuning were used.

The baseline regeneration completed in about two seconds in this CPU environment.
Generated CSVs, evaluation reports, and the data-check report stay local in
gitignored directories. This verifies the data-to-submission pipeline, not a new
method or an improvement over the reference baseline.

Observed scores on the public five-minute window (7,500 graded frames, 17
possession spells). Accuracy was independently calculated from the regenerated
CSVs, masking out ground-truth X frames, and agrees with the official scorer:

| Baseline | Correct frames | Accuracy | macroF1 | spellF1 |
| --- | ---: | ---: | ---: | ---: |
| baseline_raw | 4,884 | 65.12% | 0.5972 | 0.0378 |
| baseline_smooth25 | 5,184 | 69.12% | 0.6200 | 0.2868 |
| baseline_smooth50 | 5,349 | 71.32% | 0.6207 | 0.3256 |
| baseline_smooth125 | 5,456 | 72.75% | 0.5812 | 0.4242 |

Always predicting the most common class (home) would achieve 46.05% accuracy on
this window. For the published reference `baseline_smooth50`, recall is 88.04%
for home, 77.38% for away, and 21.38% for dead ball. Its team-label accuracy on
ground-truth live frames is 83.49%; incorrect D or X predictions count as misses
there too. It predicts 26 possession spells against 17 in the public ground truth.

On the same window, the reference baseline estimates home possession at 62.12%,
versus 57.27% in the public ground truth (4.85 percentage points of absolute
error). Each share uses its own H/(H+A) denominator, excluding its D/X frames.
The local aggregate report is `data/baseline_evaluation/accuracy_summary.json`.

These match the upstream README. The organizer-reported full-match floor is
`baseline_smooth50` with spellF1 **0.2433**. The private full-match scores were not
reproduced locally. Do not select a model or smoothing window using the public
scores: the public labels are explicitly not a tuning set.

## First implementation direction

1. Load tactical tracking and automatic Smart Tagging. Check frame alignment,
   coordinate units, team mapping, missingness, and track-slot fragmentation.
   Raw ATD frame IDs are one-based; submission frames are zero-based. Smart
   Tagging uses seconds from the tactical video's start. The broadcast feed has
   a separate timeline and should be deferred until there is a reason to align it.
2. Build evidence for H/A/D: nearest team distances, ball and player motion,
   persistence of control, pitch boundaries, tactical phases, and restart cues.
   Missing ball observations alone must not imply a stoppage. Tags are noisy
   evidence, not ground truth, and defensive tags need their proper interpretation.
3. Infer a coherent timeline with a state machine or temporal sequence model.
   Preserve possession through passes and uncertain periods; require supporting
   evidence for turnovers. Estimate stoppages separately from team control.
4. Choose parameters using other matches, synthetic cases, or unlabeled methods.
   Use this match's public labels only to check pipeline correctness.
5. Generate the complete CSV deterministically, validate it, and document the
   method, dependencies, external data, limitations, and reproduction command.

This direction is now implemented in the [external experiment suite](../experiments/README.md).
Four tracking-based approaches were compared on fixed match-level partitions.
The validation-selected gap-augmented boosting model reached **0.9298 frame macro
F1 / 0.6232 spell F1** on the held-out Sportec match, and **0.9090 / 0.5355** with
synthetic observation gaps. Sportec's frozen dead-ball coordinates make its
stoppage classification unusually easy; these results do not establish challenge
accuracy. A complete challenge CSV was generated and validated without reading
challenge ground truth. See the linked report for reproduction and limitations.

The [subsequent refinement round](../experiments/REFINEMENT.md) raised a fresh
external test to **0.9408 frame macro F1 / 0.7059 spell F1**, compared with
**0.9303 / 0.6654** for the prior model on the same match. The frozen competition
pipeline check then exposed poor transfer: the selected model scored
**0.6487 / 0.4300** on the public five minutes; the predeclared PathCRF hybrid
scored **0.6795 / 0.5067**. No model or threshold was adjusted using those public
results. Reliable stoppage detection and independent broadcast validation remain
unresolved; no full-match accuracy or intrinsic ceiling is established.

## Reusable research

- [Automatic event detection in football using tracking data](https://arxiv.org/abs/2202.00804):
  a published approach that infers possession from tracking before deriving events.
  More directly relevant to this task than training a whole-frame image classifier.
- [SkillCorner open data](https://skillcorner.github.io/opendata/): other matches
  with player/ball tracking and possession labels. A candidate external source for
  development, subject to checking its licence, schema, and differing label definitions.
- [Wyscout possession definition](https://dataglossary.wyscout.com/ball_possession/):
  useful context on control and stoppages. The hackathon's definition remains authoritative.
- [Roboflow Sports](https://github.com/roboflow/sports/tree/main/examples/soccer):
  optional detection/tracking components if investigation shows the supplied tracking
  needs supplementation. Not required to get the first solution running.

## Competition constraints

The [rules](../RULES.md) prohibit hand labels or manual tracking corrections for
this match, including labels for training or validation. They also prohibit using
human-produced event/possession data about this match from outside providers.
Labels from other matches are allowed under the documented access requirements.

The match assets and reconstructive derived data must not be published. Keep the
bundle in the existing gitignored `data/` directory, or set `MTAH_DATA_DIR` to an
external directory. Public source code, upstream public ground truth, and the
bundled baselines are already covered by the repository's licence.

The final deliverables are a possession CSV, a public reproducible code repository,
and a short technical document. The published deadline is **November 17, 2026**.
