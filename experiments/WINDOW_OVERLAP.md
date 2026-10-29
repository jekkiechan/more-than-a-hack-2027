# Shifted-core inference experiment

Status: completed; retain current baseline. V4 ran one zero-fit inference arm.
The v1 attempt stopped before scoring because the scalar upsampling helper received a
three-class array. V2 corrected that call and passed the preflight, then stopped
at first-cell control digest parity because a recomputed original pass was
within 1e-6 but not byte-identical to the archived result. The immutable
declarations and deviation receipts are preserved at the [v1 declaration](results/window_overlap_declaration.json),
[v1 receipt](results/window_overlap_v1_prescore_deviation.json),
[v2 declaration](results/window_overlap_declaration_v2.json), and
[v2 receipt](results/window_overlap_v2_prescore_deviation.json). V3 passed the
feature and probability preflight, then stopped before scoring because its
preflight archive reader omitted one required seed array; see the [v3 declaration](results/window_overlap_declaration_v3.json)
and [v3 receipt](results/window_overlap_v3_prescore_deviation.json). V4 fixes
that archive-key contract and keeps the same arm, cohort, checkpoints, weights,
and score criteria. It uses the saved per-seed original probability arrays for
exact control parity and runs the shifted tiling on the pinned cached feature
records. A separate pre-score diagnostic found current Torch
re-inference differs from the archived member probabilities by less than
7e-7 despite byte-identical cached and reconstructed float32 features; see
[the replay-drift receipt](results/window_overlap_v2_replay_drift.json).

This study tests whether averaging a second inference tiling improves possession
spell F1 for the adopted `joint_gru_combined_ensemble_v1`. The two trained
checkpoints, five-second context, ten-second retained cores, 5 Hz feature grid,
player-only dead-ball path, equal seed weights, and cost-3 decoder remain fixed.
The candidate averages original and five-second-shifted native H/A/D
probabilities equally wherever the shifted tiling has a retained-core
prediction. At each continuous block's first 25 sampled rows, the original
tiling supplies the prediction because the shifted grid has not started.
Partial final cores are retained, and all windows clip context inside their
own period/time-continuous block.

The matched control reuses the exact saved original-tiling probabilities and
score rows from the combined-objective comparison on the five previously exposed
full-provider development matches: Sportec J03WOY, Metrica Game 3, and
SkillCorner 1996435, 1996436, and 2006229. Each match has one clean record and
two fixed synthetic `feed_like` records. Every learned component, including
normalization, excludes the evaluated provider. The two seeds are scored
separately and as an actual 50/50 probability ensemble. No fitting or public
diagnostic is part of this experiment.

Earlier work tested longer context as a training change, and architecture
ensemble work averaged different model families. The historical PathCRF
adapter uses overlapping windows for a separate pretrained H/A model. None of
those studies changes only the inference phase of the adopted GRU's retained
cores, so they do not duplicate this question.

The current clean spell screen is +.01 equal-provider F1. The `feed_like`
screen is no worse than -.01 versus the active ensemble; each provider's clean
D-F1 and macro-F1 must stay within .02. These are reported as pass/fail
diagnostics; the study cannot promote a model automatically. Primary ranking
uses unrounded official spell F1, macro F1 only for exact ties, and the fixed
70/30 summary is auxiliary. The readout includes all five IoU thresholds,
short-spell recall, D precision/recall, counts, fragmentation, censored edges,
complete-boundary timing, per-seed spread, and descriptive seam probability
disagreement.

The frozen v4 protocol and hashes are in
[the declaration](results/window_overlap_declaration_v4.json). A first-cell
inference/parity preflight checked all-X feature reconstruction, exact saved
control blending, and shifted probability validity before scoring. Results are
in [the aggregate](results/window_overlap_v4.json), with the private artifact
receipt at [window_overlap_artifacts_v4.json](results/window_overlap_artifacts_v4.json).
The original-tiling control must reproduce all fifteen saved cell probabilities,
decodes, and aggregate rows before the shifted arm is considered. All five
matches were previously exposed, `feed_like` is synthetic, and this result
cannot establish independent or full-match performance.

## Readout

Retain the adopted ensemble. The shifted-core candidate raises clean spell F1
by **0.000734**, but misses the current +0.01 clean screen. Feed-like spell F1
declines **0.000500**, within the -0.01 floor. All three providers pass clean
D-F1 and macro-F1 guards. No automatic promotion or public diagnostic occurred.

| View | Original spell / macro | Shifted spell / macro | Spell / macro change | Auxiliary 70/30 change |
| --- | ---: | ---: | ---: | ---: |
| Clean | .633205 / .854226 | .633939 / .854850 | +.000734 / +.000625 | +.000701 |
| Feed-like | .498830 / .811387 | .498330 / .811308 | -.000500 / -.000080 | -.000374 |

The per-provider spell changes are mixed. Clean moves by +.003363 on Sportec,
-.001095 on Metrica, and -.000067 on SkillCorner. Feed-like moves by +.002196,
-.004453, and +.000758 respectively. Clean D-F1 changes are -.000161,
+.002393, and +.000702; clean macro changes are +.000383, +.001247, and
+.000243. The largest provider-specific stress loss is Metrica.

| View | IoU .3 | IoU .4 | IoU .5 | IoU .6 | IoU .7 | Under-3s recall change at .5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean spell-F1 change | -.000831 | -.002527 | +.002512 | +.000320 | +.004195 | +.003642 |
| Feed-like spell-F1 change | +.000083 | -.003698 | -.002901 | +.000945 | +.003072 | +.004084 |

The strict-IoU gains do not create a clean-screen-sized spell improvement. D
precision/recall changes are -.000465/+.002050 clean and -.001019/+.003234
feed-like. In the clean match rows, the 1,963 reference spells produce 1,998
original predictions and 2,001 candidate predictions. Per-match counts, all
five IoU scores, fragmentation, graded-edge censoring, and complete-boundary
errors are in the aggregate. Candidate fragmentation averages 9.22 reference
spells per clean cell versus 9.00 for control; feed-like is 8.50 for both.
Reference spells touching grading-run edges are unchanged. Mean complete-boundary
start/end timing changes from -0.569/+0.256 s to -0.588/+0.260 s clean, and
from -0.690/+0.548 s to -0.788/+0.547 s feed-like. These timings do not show a
consistent boundary correction.

Seed effects disagree. Seed 2026 changes by +.002820 clean and +.005039
feed-like; seed 2027 changes by -.001060 and -.001312. The true probability
ensemble produces only a small clean gain and a slight feed-like loss. The
shifted grid has retained-core support on 99.82% of native rows; leading edges
use the original probability. Descriptive total-variation disagreement is
slightly higher within one second of original seams than elsewhere (.02804 vs
.02633 clean, .03362 vs .03201 feed-like), which indicates modest seam
sensitivity but is not a selection metric.

All 15 cells completed with exact saved original member/ensemble probability,
decode, and score parity. The first-cell preflight sealed the shifted candidate
probability before labels were used for scoring. The original model replay
diagnostic remains separate: current Torch re-inference differs from saved
member probabilities by under 7e-7 despite identical float32 feature values;
the saved arrays are the exact control. The eight targeted synthetic contracts
pass. Evidence uses five previously exposed full-provider development matches;
feed-like is synthetic, and independent broadcast/full-match performance is
unknown.

Private run outputs are locally verified under
`data/experiments/window_overlap_v4/`; independent remote restoration remains
pending. The aggregate links the full spell diagnostics and paired cells.

```sh
/workspace/venv/bin/python -m experiments.window_overlap_experiment check
/workspace/venv/bin/python -m experiments.window_overlap_experiment preflight
/workspace/venv/bin/python -m experiments.window_overlap_experiment run
/workspace/venv/bin/python -m experiments.window_overlap_experiment report
```

Independent replay of all 15 candidate/control cells verifies official decodes, score rows, exact archived controls, aggregation and guardrails. See the [independent verification receipt](results/window_overlap_independent_verification.json).

The [shared artifact receipt](results/spell_f1_followup_artifacts.json) records the locally verified archive. Remote upload was blocked by automatic approval review pending explicit authorization; no fresh remote restore is claimed.
