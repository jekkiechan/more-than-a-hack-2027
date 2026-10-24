# Running an existing pretrained possession-path model

We downloaded and ran the official [PathCRF source and checkpoint](https://github.com/hyunsungkim-ds/pathcrf/tree/1b7c6f85a6b78c92e4543fde72ca2b86a933f997). This is a tracking-to-team-possession adaptation of a player-to-player event model. It is not a replication of the paper's in-play event benchmark.

The source is pinned to `1b7c6f85a6b78c92e4543fde72ca2b86a933f997` and carries [MPL 2.0](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/LICENSE). Our adapter is also marked MPL 2.0 because its feature/decode conventions follow that implementation; the license text is available at that link. We use trial 140's default evaluation checkpoint, `saved/140/model/state_dict_best_acc.pt`, SHA-256 `55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9`. It has approximately 3.43 million trainable parameters. State-dict loading is strict and uses PyTorch's `weights_only=True`; there is no fine-tuning, replacement output head, or random initialization of missing weights.

## Independent match and comparison scope

Trial 140 trained on J03WMX, J03WN1, J03WOH, J03WOY, and J03WPY; it used J03WQQ for validation and J03WR9 for testing. Therefore our original held-out J03WPY cannot test this checkpoint independently. We use **J03WR9 only** for this separate pretrained-model comparison, leaving the primary split unchanged. The checkpoint and adapter are fixed before its labels are scored. See [PATHCRF_DATA.md](PATHCRF_DATA.md) for independently audited data provenance and roster conversion.

When compared with our frozen custom estimator on J03WR9, the test match and scorer are the same, but training budgets differ: the checkpoint used five Sportec matches, while our estimator used three. Both results remain same-provider transfer estimates, not evidence of broadcast-tracking performance.

## Measured comparison

The checkpoint and both mappings were fixed before this match's labels were
scored. All 146,810 frames were predicted, with 370 reference possession spells.
No model or adapter was selected again from these scores.

| Frozen method | Whole-match H/A/D macro F1 | Possession-spell F1 | H/A macro F1 on reference-live frames |
| --- | ---: | ---: | ---: |
| Pretrained PathCRF, H/A only | 0.4427 | 0.4508 | **0.9148** |
| Pretrained PathCRF + geometric dead-ball cue | **0.9382** | 0.6015 | 0.9113 |
| Our preselected gap-augmented boosting + Viterbi | 0.9320 | **0.6122** | 0.8986 |

The H/A-only row cannot predict D, so its whole-match score is not a measure of a
native three-class capability. Its reference-live diagnostic uses the ground
truth mask only for scoring. The hybrid predicts D from ball geometry using the
same previously frozen thresholds as the primary baseline; it does not use true
live boundaries. Its estimated home share is 62.92%, versus 62.40% reference,
an error of 0.52 percentage points. Our model estimates 63.19%, error 0.79 points.

**The pretrained model is a credible candidate.** It has higher frame F1 here;
our small estimator has slightly higher spell F1. Gaps of 0.006 and 0.011 on one
match do not establish a reliable winner. The experiment supports reusing
PathCRF rather than starting another deep architecture immediately, while keeping
the inexpensive custom estimator as a baseline. Improving short turnovers and
robust dead-ball inference remains useful for either approach. A future hybrid
with a learned dead-ball model would require a new declared development protocol;
it was not selected or evaluated in this comparison.

Full PathCRF inference took **413.8 seconds on two CPU threads**. Its player-only
input means its native team attribution does not depend on ball detections. The
hybrid's D cue does depend on ball geometry and on Sportec's frozen-coordinate
artifact; its high clean score cannot establish robustness to real camera gaps.

Aggregate results: [pathcrf_comparison.json](results/pathcrf_comparison.json).
Frozen comparison: [pathcrf_comparison_manifest.json](results/pathcrf_comparison_manifest.json).

## Input audit and fixed adapter

`pathcrf_eval.py` loads only `players`, `teams`, `period`, `fps`, `timestamps`, and `pitch`. It does not deserialize the NPZ labels and does not use ball coordinates. All original model teacher-forcing arguments remain `None`; the model runs in evaluation and inference mode.

The eight features match the checkpoint: team indicator, goalkeeper flag, x, y, vx, vy, speed, and acceleration. Coordinates are metres on a 105 × 68 m pitch. Each period is rotated, when needed, using geometric goalkeeper estimates so home defends the left side, matching the source loader's `STATIC_HOME_AWAY` orientation. Velocity and acceleration are computed before downsampling, using the upstream 15-sample and 9-sample Savitzky–Golay conventions and player-speed outlier interpolation above 12 m/s. Internal coordinate gaps are interpolated within a player's observed span. No derivatives cross period or timestamp discontinuities.

Phases change only at period/time breaks and players' first/last coordinate observations. Each team receives eleven padded slots, followed by four fixed outside nodes. Short substitution phases remain in the output. The adapter rejects more than eleven simultaneously active players rather than silently truncating them.

Inference uses 10-second windows at 5 Hz with a fixed 5-second stride. It runs the original learned edge head **and dynamic sparse CRF**. Each window is decoded independently, and overlapping predictions use triangular weights that remain positive at the ends. This differs from the upstream full-episode evaluation for a necessary reason: the stock wrapper builds episodes from true live/dead labels, while its outside CRF states are absorbing. Feeding an entire half into that decoder would prevent a new possession after an outside state. Fixed windows avoid supplying true restart boundaries.

The decoded edge source means the previous possessing player. Its team supplies H/A, retaining the passing team's possession while the ball travels. An outside-source self-loop retains the last predicted team within the window. If a window begins outside, the original macro source head's probability summed over real players provides the initial team. The separately saved macro-head diagnostic is not used to select a better test result.

**The pretrained model has no referee dead/alive head.** Its training excludes dead-ball episodes, and outside nodes are not interchangeable with the challenge's D class. Report its H/A F1 on reference-live frames explicitly as a conditional diagnostic. Those reference labels affect scoring only, never model coverage, windows, or features. A separately named whole-match hybrid may combine these frozen H/A predictions with the already frozen geometric dead-ball cue. That hybrid is not an all-in-one pretrained H/A/D classifier.

Sportec's raw source freezes ball XY during dead periods. A geometric D cue can exploit that provider-specific shortcut. High hybrid D performance would not establish reliable dead-ball recognition from broadcast tracking.

## Reproduction

All checkpoints, dependencies, match data, and frame predictions stay in ignored `data/`. The environment is CPU-only; no CUDA or custom torch-scatter/torch-sparse installation is needed for this set-transformer checkpoint.

```bash
git clone https://github.com/hyunsungkim-ds/pathcrf.git data/external_models/pathcrf
git -C data/external_models/pathcrf checkout 1b7c6f85a6b78c92e4543fde72ca2b86a933f997
python -m pip install --target data/external_models/python_packages \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --target data/external_models/python_packages --no-deps \
  torch-geometric==2.6.1 xxhash aiohttp aiosignal attrs frozenlist multidict \
  yarl propcache aiohappyeyeballs tqdm requests pyparsing
python experiments/audit_dataset.py --matches J03WR9
python -m unittest discover -s experiments/tests -v
python -m experiments.freeze_pathcrf
# Requires the already frozen custom models from the primary experiment.
python -m experiments.compare_pathcrf prepare
python -m experiments.pathcrf_eval \
  data/external_audit/sportec/processed/J03WR9.npz \
  --out data/experiments/pathcrf --threads 2 --batch-size 4
python -m experiments.compare_pathcrf score
```

The repository's usual numpy/scipy and related requirements must already be installed. The adapter automatically finds the local dependency directory. The CLI is prediction-only: it has no scoring or tuning option and produces an inference metadata JSON plus an NPZ of predictions. The metadata records the checkpoint hash, input fields, inference timing, coverage, and that labels were not read.

A 60-second training-match smoke test completed in 5.7 seconds with two CPU threads and batch size four. Strict checkpoint loading, full frame coverage, finite features, and invariance to poisoned possession/ball-state fields passed. An independent reviewer also ran six contract tests before inference.

The full J03WR9 run completed in **413.8 seconds** on two CPU threads: 146,810 frames, nine geometry-defined phases, and 1,170 windows. Every frame received H/A, including the 17-frame substitution phase. The pretrained CRF produced an outside source in 6,703 of 58,454 decoded window steps; the fixed carry-forward rule handled these, including overlap. These counts describe inference behavior, not accuracy. Predictions and audit metadata are in `data/experiments/pathcrf/J03WR9_predictions.npz` and `J03WR9_inference.json`. The adapter SHA-256 frozen before the run was `26a6a192d555fdf6ebea0b2d174fac5c670775ea9782f92b5f20d338a2df09d5` and remained unchanged afterward. The separate comparator scores these saved predictions; inference never read the test labels.
