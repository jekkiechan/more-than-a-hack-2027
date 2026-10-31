# Matched temporal neural architecture comparison

Status: completed and independently verified locally (2026-10-28).

Under the declared fixed budget, retain the historical slot-dropout GRU
family. All three challengers failed the declared
both-seed screen under this fixed training budget. Mean clean/feed-like spell
F1 was .6132/.4798 for GRU, .6008/.4704 for TCN, .5976/.4656 for BiLSTM and
.5253/.4323 for the small Transformer. These are separate-seed averages on
five reused development matches, not a prediction ensemble or fresh confirmation.
The Transformer improved diagnostic composite stress (.3881 vs .3141), but
lost on the active clean/feed-like criteria. This does not rule out differently
tuned LSTMs or Transformers. A plain RNN was not tested.

The current adopted package is `joint_gru_seed_ensemble_v1`, with a separately
measured actual-ensemble clean/feed-like result of .6265/.4874 in the later
[seed-ensemble study](SEED_GAP_CONTEXT.md). This architecture comparison used
the former `joint_gru_slotdrop_v1` recipe as its reference; its GRU .6132/.4798
is the arithmetic mean of two separate fits, not the adopted ensemble score.
The current ensemble was not an arm in this comparison.

The next evidence gap is an independent labeled broadcast match. These results
did not justify changing the then-adopted package or reopening organizer
scoring.

Implementation and frozen runner remain on the local working branch
`possession/temporal-architecture-comparison`: implementation snapshot
`c31afbcc0ec7d05da151348ce2e98439b6fef5ff`, final comparison snapshot
`9f023deed33ec8605312eb7e97de67c32c1204e3`. Those commits are not published
on the remote repository and are not included in this documentation PR.
The [frozen declaration](results/temporal_architecture_declaration.json) binds
all inputs, sources, architecture settings and comparator checkpoints.

The user requested three neural challengers to the then-adopted slot-dropout GRU,
implemented, tested and verified by Luna agents at xhigh reasoning. This study
compares a bidirectional LSTM, a small noncausal Transformer and a noncausal TCN.
The question is whether replacing the temporal encoder improves joint H/A/D
possession recovery under the current training and feed-like stress protocol.
It does not search each architecture's optimal hyperparameters.

## Declaration

The reference is `joint_gru_slotdrop_v1`, evaluated through its six archived
slot-dropout full-provider-holdout checkpoints, not its pooled deployment fit.
Two seeds (2026, 2027) and three provider exclusions give six fits per challenger,
18 new fits total. Every challenger uses the existing factorized likelihood:
H/A conditional on live, and a player-only D path using shared encoder weights.
The D path masks both non-player values and their finite-value indicators.

| Encoder | Fixed design | Parameters at 408 normalized inputs |
| --- | --- | ---: |
| Archived BiGRU | 24 hidden units per direction | 62,643 |
| BiLSTM | 18 hidden units per direction, nearest GRU parameter budget | 61,743 |
| Transformer | 48 channels, 4 heads, 2 encoder layers, 96 FFN units; fixed sinusoidal positions | 57,699 |
| TCN | 56 channels; four residual kernel-3 convolutions, dilations 1/2/4/8 | 60,931 |

Capacity is chosen from parameter counts before results, not from development
scores. Network dropout is zero in all challengers; raw player-slot dropout
remains the common observation augmentation. The TCN receptive field is 31
samples (radius 3 seconds), within the five-second context halo. Recurrent and
attention models can use the whole input window; equal input windows do not
imply equal effective receptive fields. No halo, derivative or decoder change
is included. Transformer attention uses explicit positional encoding and the
same feature-level missingness indicators as the recurrent networks.

Training preserves the adopted recipe: 13 external TRAIN matches, clean/legacy
views, approximately 25% raw player-slot dropout in the second view, 25%
whole-window ball-gap replacement, H/A symmetry, training-only normalization,
equal provider/match loss mass, unknown-target zero weight, 5 Hz sampling,
10-second cores and five-second halos. Use eight fixed epochs, batch size 64,
AdamW learning rate .001 and weight decay .0001, gradient clipping at 1.
All derived windows remain within their source match/period/time block.
All learned components exclude the held-out provider.

The [evaluation manifest](eval_set/manifest.json) supplies exact TRAIN and
development IDs, NPZ hashes, label provenance and exposure. The five reused
development matches are Sportec J03WOY, Metrica game 3, and SkillCorner
1996435/1996436/2006229. Sportec uses provider labels; Metrica and SkillCorner
include derived silver labels. Preserve native unknown labels, masks and
SkillCorner visibility filtering. No new confirmation set is claimed.

Evaluate each model/seed on clean inputs, two fixed `feed_like` views, and two
fixed `composite` views: 25 match/view cells per model/seed, 200 cells across
the three challengers and archived GRU. Use the existing corruption calibration
and `corrupt_1`/`corrupt_2` seeds. Feed-like retains ball gaps, coordinate defects,
slot gaps and ID swaps; composite additionally includes regional blanking and
whole-frame sampling loss. Composite is diagnostic only, consistent with the
current [policy](model_decision_policy.json). Its historical guardrail change
was made after composite results were known; this new comparison declares the
current policy before fitting.

Use the unchanged cost-3 decoder and official possession spell F1. Average
stress replicates within match, matches within provider, then providers equally.
Report each fit seed separately; their mean is not a prediction ensemble.
Among eligible candidates rank on unrounded clean spell F1, with macro F1 for
exact ties; the 70% spell / 30% macro summary is auxiliary only.

For each seed, compare with the corresponding archived dropout GRU: require
at least +.01 clean spell F1, no provider's clean D F1 or macro F1 more than .02
lower, and feed-like spell F1 no more than .01 lower. Both seeds must pass.
These are conservative development screens, not significance tests or automatic
adoption. Report clean/feed-like/composite results even when a screen fails.
Include D precision/recall, all five spell IoUs, short-spell recall, false
possessions, fragmentation, boundary diagnostics where supported, and runtime.

Restore and hash-check existing artifacts before training. The executable
declaration must bind source, configuration, input, comparator, policy and
environment versions. Freeze every checkpoint before any new score is read.
Preserve native probability/prediction files and hashes for independent replay;
verify comparator parity against saved results on identical cells. No additional
arms, epoch selection, seed selection, tuning or public scoring belongs to this
run. Independent agents verify contracts, cohort completeness and saved outputs.

## Execution and artifacts

The runner is `experiments.temporal_architecture_experiment`; its default output
is ignored `data/experiments/temporal_architecture_v1`. Restore the external
arrays and six comparator checkpoints from the private temporal robustness
cache; LFS pointers are not restored data. Checkpoint, source and data access
failures are execution blockers, never model results.

The [restore receipt](results/temporal_architecture_restore.json) verifies 18
NPZ inputs, six checkpoints and their sidecars, 30 saved probability outputs,
and archive metadata. Clean/composite parity uses those saved arrays;
feed-like parity uses exact saved per-cell metrics from the stress-component
report. The optional historical feed-like probability archive was not restored.

```sh
python -m experiments.temporal_architecture_experiment stage
python -m experiments.temporal_architecture_experiment freeze
python -m experiments.temporal_architecture_experiment prepare
for architecture in bilstm transformer tcn; do
  for source in sportec metrica skillcorner; do
    for seed in 2026 2027; do
      python -m experiments.temporal_architecture_experiment fit --architecture "$architecture" --source "$source" --seed "$seed"
    done
  done
done
python -m experiments.temporal_architecture_experiment seal
python -m experiments.temporal_architecture_experiment evaluate
python -m experiments.temporal_architecture_experiment report
```

Use the pinned CPU PyTorch environment. Staging computes features only and
may run independently by provider; freeze binds every staged cache before
fitting. Operational fit concurrency is limited to two workers in this run.

## Results (development-only; 2026-10-28)

All 18 challenger fits were sealed before scoring. The frozen report contains 200 unique cells: 40 Sportec, 40 Metrica and 120 SkillCorner, covering the archived slot-dropout GRU, BiLSTM, Transformer and TCN at seeds 2026/2027 across the three full-provider holdouts and clean, feed-like and composite views. The report’s integrity checks accepted all 200 cells, and all 175 tests passed. Independent verification passed for all 200 cells and their metrics/diagnostics, 50 baseline cells, and nine checkpoint prediction replays.

None of the three challengers passed the declared guardrails for both seeds. The equal-provider two-seed arithmetic means (not ensembles) were clean spell/macro F1 0.6132/0.8489 for the archived GRU, 0.5976/0.8398 for BiLSTM, 0.5253/0.8260 for Transformer and 0.6008/0.8475 for TCN. No model or baseline was promoted; the study made no changes to the then-adopted package or organizer-scoring gates. The current `joint_gru_seed_ensemble_v1` was adopted later in a separate study and was not evaluated here.

### Archived-probability tolerance deviation

The frozen runner’s archived probability-array comparison used absolute tolerance `atol=1e-6` with `rtol=0`. A separate, pre-resume declaration authorized `atol=1e-5` only for the 12 archived clean/composite probability-array cells for Sportec and Metrica (six cells per source: two seeds across clean and the two composite replicates). The declaration binds all 30 archived probability artifacts; the other 18 SkillCorner cells retain the original `1e-6` check. Feed-like parity checks are unchanged.

For those 12 cells, decoded predictions had to match exactly and the frozen metric-parity checks were delegated unchanged; all 12 passed (float metrics retain the original `1e-10` comparison tolerance, and counts are exact). The largest archived/current probability difference was `1.4007091522216797e-6` for Sportec (99 scalar entries above `1e-6`) and `1.0728836059570312e-6` for Metrica (14 scalar entries above `1e-6`). No input, feature, period or timestamp mismatch was found in the read-only Sportec replay. The frozen study sources, declaration and checkpoints were not changed. The tolerance declaration and per-cell receipts remain local under ignored `data/experiments/temporal_architecture_v1/`; only aggregate evidence and provenance receipts are published here, with no model weights, probability arrays or frame-level outputs.

The [aggregate development report](results/temporal_architecture_development.json), [tolerance declaration](results/temporal_architecture_probability_tolerance_deviation.json), [Sportec tolerance receipt](results/temporal_architecture_probability_tolerance_sportec.json), [Metrica tolerance receipt](results/temporal_architecture_probability_tolerance_metrica.json), [operational event log](results/temporal_architecture_operational_events.json) and [independent verification receipt](results/temporal_architecture_independent_verification.json) record the run provenance. The [local artifact inventory](results/temporal_architecture_artifact_manifest.json) records file hashes and availability. Its `protocol_and_result_files` entry binds the original work-run prose at local commit `9f023de`; this published copy adds only current-baseline and publication-status clarifications, so its prose hash differs. All frozen JSON declarations, aggregates and receipt files remain byte-identical. Model checkpoints, prediction arrays, and detailed local receipts remain in ignored `data/experiments/temporal_architecture_v1/` and are not published.


Development-only results from the completed frozen protocol. The two-seed arithmetic means below are summaries of separate fits, **not an ensemble**. Composite corruption is diagnostic; feed-like corruption is the active stress guardrail. These results are not fresh confirmation and do not automatically promote a model.

### Equal-provider headline summary

Seed means are arithmetic averages across the two fitted seeds, **not ensembles**. The 70/30 column is auxiliary (0.7 × clean spell F1 + 0.3 × clean macro-F1).

| Model | Seed | Clean spell F1 | Clean macro-F1 | Clean 70/30 | Feed-like spell F1 | Composite spell F1 |
|---|---|---:|---:|---:|---:|---:|
| slot_dropout_gru | 2026 | 0.609 | 0.843 | 0.679 | 0.474 | 0.318 |
| slot_dropout_gru | 2027 | 0.618 | 0.854 | 0.689 | 0.485 | 0.310 |
| slot_dropout_gru | 2-seed mean (not ensemble) | 0.613 | 0.849 | 0.684 | 0.480 | 0.314 |
| bilstm | 2026 | 0.602 | 0.843 | 0.674 | 0.469 | 0.311 |
| bilstm | 2027 | 0.593 | 0.836 | 0.666 | 0.463 | 0.279 |
| bilstm | 2-seed mean (not ensemble) | 0.598 | 0.840 | 0.670 | 0.466 | 0.295 |
| transformer | 2026 | 0.532 | 0.821 | 0.619 | 0.435 | 0.396 |
| transformer | 2027 | 0.519 | 0.830 | 0.612 | 0.430 | 0.380 |
| transformer | 2-seed mean (not ensemble) | 0.525 | 0.826 | 0.615 | 0.432 | 0.388 |
| tcn | 2026 | 0.604 | 0.848 | 0.677 | 0.477 | 0.302 |
| tcn | 2027 | 0.598 | 0.847 | 0.673 | 0.463 | 0.349 |
| tcn | 2-seed mean (not ensemble) | 0.601 | 0.848 | 0.675 | 0.470 | 0.325 |

### Frozen development screen

A screen passes only when that seed clears every declared clean, provider, and feed-like guardrail.

| Candidate | Seed | Seed passes | Clean spell Δ vs GRU | Feed-like spell Δ vs GRU | Worst provider clean D-F1/macro-F1 Δ | Both seeds pass |
|---|---:|---|---:|---:|---:|---|
| bilstm | 2026 | no | -0.007 | -0.006 | -0.002 | no |
| bilstm | 2027 | no | -0.024 | -0.023 | -0.084 | no |
| transformer | 2026 | no | -0.077 | -0.039 | -0.030 | no |
| transformer | 2027 | no | -0.099 | -0.056 | -0.031 | no |
| tcn | 2026 | no | -0.005 | 0.003 | -0.000 | no |
| tcn | 2027 | no | -0.020 | -0.022 | -0.023 | no |

### Clean metrics by held-out provider

Each metric cell is `seed 2026 / seed 2027 / two-seed arithmetic mean`; the mean is not an ensemble.

| Model | Held-out provider | Clean spell F1 | Macro-F1 | D precision | D recall | D F1 |
|---|---|---:|---:|---:|---:|---:|
| slot_dropout_gru | sportec | 0.619 / 0.603 / 0.611 | 0.858 / 0.868 / 0.863 | 0.944 / 0.898 / 0.921 | 0.849 / 0.929 / 0.889 | 0.894 / 0.913 / 0.904 |
| slot_dropout_gru | metrica | 0.748 / 0.786 / 0.767 | 0.912 / 0.915 / 0.913 | 0.895 / 0.924 / 0.910 | 0.886 / 0.856 / 0.871 | 0.891 / 0.889 / 0.890 |
| slot_dropout_gru | skillcorner | 0.459 / 0.463 / 0.461 | 0.761 / 0.780 / 0.770 | 0.845 / 0.837 / 0.841 | 0.624 / 0.698 / 0.661 | 0.716 / 0.761 / 0.739 |
| bilstm | sportec | 0.601 / 0.571 / 0.586 | 0.858 / 0.853 / 0.855 | 0.922 / 0.885 / 0.904 | 0.884 / 0.920 / 0.902 | 0.903 / 0.902 / 0.902 |
| bilstm | metrica | 0.749 / 0.756 / 0.753 | 0.910 / 0.911 / 0.910 | 0.898 / 0.918 / 0.908 | 0.885 / 0.861 / 0.873 | 0.891 / 0.889 / 0.890 |
| bilstm | skillcorner | 0.456 / 0.453 / 0.455 | 0.762 / 0.745 / 0.754 | 0.807 / 0.841 / 0.824 | 0.660 / 0.567 / 0.613 | 0.725 / 0.676 / 0.701 |
| transformer | sportec | 0.500 / 0.491 / 0.495 | 0.828 / 0.840 / 0.834 | 0.939 / 0.917 / 0.928 | 0.823 / 0.883 / 0.853 | 0.877 / 0.899 / 0.888 |
| transformer | metrica | 0.636 / 0.623 / 0.630 | 0.882 / 0.884 / 0.883 | 0.924 / 0.901 / 0.913 | 0.812 / 0.842 / 0.827 | 0.864 / 0.871 / 0.868 |
| transformer | skillcorner | 0.459 / 0.443 / 0.451 | 0.755 / 0.767 / 0.761 | 0.801 / 0.841 / 0.821 | 0.621 / 0.657 / 0.639 | 0.699 / 0.737 / 0.718 |
| tcn | sportec | 0.591 / 0.563 / 0.577 | 0.868 / 0.864 / 0.866 | 0.924 / 0.941 / 0.933 | 0.902 / 0.881 / 0.891 | 0.913 / 0.910 / 0.911 |
| tcn | metrica | 0.774 / 0.782 / 0.778 | 0.912 / 0.909 / 0.911 | 0.898 / 0.947 / 0.923 | 0.883 / 0.828 / 0.855 | 0.891 / 0.883 / 0.887 |
| tcn | skillcorner | 0.447 / 0.448 / 0.447 | 0.764 / 0.768 / 0.766 | 0.915 / 0.867 / 0.891 | 0.611 / 0.642 / 0.627 | 0.732 / 0.738 / 0.735 |

### Official spell F1 at all five IoU thresholds

Each cell shows `seed 2026 / seed 2027 / two-seed arithmetic mean`; replicates are averaged within match, then matches within provider, then providers equally.

| Model | View | IoU 0.3 | IoU 0.4 | IoU 0.5 | IoU 0.6 | IoU 0.7 |
|---|---|---:|---:|---:|---:|---:|
| slot_dropout_gru | clean | 0.708 / 0.711 / 0.710 | 0.674 / 0.677 / 0.675 | 0.619 / 0.627 / 0.623 | 0.559 / 0.573 / 0.566 | 0.483 / 0.500 / 0.492 |
| slot_dropout_gru | feed_like | 0.592 / 0.597 / 0.594 | 0.541 / 0.550 / 0.546 | 0.478 / 0.487 / 0.482 | 0.413 / 0.430 / 0.422 | 0.347 / 0.362 / 0.354 |
| slot_dropout_gru | composite | 0.434 / 0.431 / 0.432 | 0.372 / 0.365 / 0.368 | 0.316 / 0.302 / 0.309 | 0.258 / 0.251 / 0.255 | 0.210 / 0.202 / 0.206 |
| bilstm | clean | 0.698 / 0.689 / 0.694 | 0.668 / 0.655 / 0.661 | 0.617 / 0.610 / 0.614 | 0.553 / 0.536 / 0.544 | 0.475 / 0.475 / 0.475 |
| bilstm | feed_like | 0.585 / 0.575 / 0.580 | 0.538 / 0.528 / 0.533 | 0.468 / 0.470 / 0.469 | 0.404 / 0.400 / 0.402 | 0.347 / 0.341 / 0.344 |
| bilstm | composite | 0.424 / 0.389 / 0.407 | 0.365 / 0.324 / 0.345 | 0.308 / 0.274 / 0.291 | 0.253 / 0.223 / 0.238 | 0.207 / 0.184 / 0.195 |
| transformer | clean | 0.655 / 0.649 / 0.652 | 0.605 / 0.594 / 0.600 | 0.539 / 0.523 / 0.531 | 0.466 / 0.449 / 0.457 | 0.393 / 0.380 / 0.386 |
| transformer | feed_like | 0.560 / 0.558 / 0.559 | 0.506 / 0.502 / 0.504 | 0.437 / 0.433 / 0.435 | 0.368 / 0.360 / 0.364 | 0.303 / 0.296 / 0.300 |
| transformer | composite | 0.514 / 0.506 / 0.510 | 0.462 / 0.443 / 0.453 | 0.401 / 0.382 / 0.392 | 0.332 / 0.314 / 0.323 | 0.272 / 0.255 / 0.263 |
| tcn | clean | 0.698 / 0.695 / 0.696 | 0.666 / 0.662 / 0.664 | 0.615 / 0.611 / 0.613 | 0.555 / 0.546 / 0.550 | 0.485 / 0.475 / 0.480 |
| tcn | feed_like | 0.582 / 0.577 / 0.580 | 0.547 / 0.528 / 0.538 | 0.487 / 0.464 / 0.476 | 0.415 / 0.408 / 0.412 | 0.355 / 0.339 / 0.347 |
| tcn | composite | 0.418 / 0.478 / 0.448 | 0.360 / 0.414 / 0.387 | 0.294 / 0.348 / 0.321 | 0.242 / 0.284 / 0.263 | 0.195 / 0.222 / 0.208 |

### Short-spell, fragmentation, false-possession, and boundary diagnostics

Cells show `seed 2026 / seed 2027 / two-seed arithmetic mean`. Short-spell recall is for reference possessions under 3 seconds at IoU 0.5. Boundary tolerance labels use the nearest available sampled-frame tolerance to nominal 0.5 s and 1.0 s, selected separately for each match.

| Model | View | Short recall <3s | False possessions/min | Fragmented GT spells/match | D-boundary P/R/F1 @~0.5s | Mean abs latency @~0.5s | D-boundary F1 @~1s | Mean abs latency @~1s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| slot_dropout_gru | clean | 0.335 / 0.336 / 0.336 | 2.106 / 1.850 / 1.978 | 9.333 / 8.667 / 9.000 | 0.163/0.142/0.152 / 0.194/0.168/0.180 / 0.178/0.155/0.166 | 0.224 / 0.226 / 0.225 | 0.290 / 0.274 / 0.282 | 0.469 / 0.400 / 0.435 |
| slot_dropout_gru | feed_like | 0.185 / 0.188 / 0.187 | 2.244 / 2.020 / 2.132 | 9.556 / 8.167 / 8.861 | 0.160/0.144/0.152 / 0.181/0.159/0.168 / 0.170/0.151/0.160 | 0.231 / 0.222 / 0.227 | 0.276 / 0.268 / 0.272 | 0.467 / 0.415 / 0.441 |
| slot_dropout_gru | composite | 0.102 / 0.099 / 0.101 | 2.126 / 2.068 / 2.097 | 20.722 / 21.944 / 21.333 | 0.036/0.044/0.039 / 0.033/0.044/0.038 / 0.035/0.044/0.038 | 0.237 / 0.226 / 0.232 | 0.082 / 0.080 / 0.081 | 0.499 / 0.481 / 0.490 |
| bilstm | clean | 0.314 / 0.315 / 0.315 | 1.886 / 2.130 / 2.008 | 9.556 / 9.667 / 9.611 | 0.170/0.142/0.155 / 0.148/0.135/0.141 / 0.159/0.139/0.148 | 0.245 / 0.250 / 0.248 | 0.278 / 0.271 / 0.275 | 0.469 / 0.485 / 0.477 |
| bilstm | feed_like | 0.168 / 0.169 / 0.169 | 2.069 / 2.205 / 2.137 | 10.333 / 8.722 / 9.528 | 0.170/0.145/0.156 / 0.145/0.135/0.140 / 0.158/0.140/0.148 | 0.249 / 0.252 / 0.251 | 0.269 / 0.262 / 0.265 | 0.458 / 0.486 / 0.472 |
| bilstm | composite | 0.100 / 0.100 / 0.100 | 2.039 / 2.295 / 2.167 | 16.444 / 19.333 / 17.889 | 0.040/0.044/0.042 / 0.021/0.026/0.023 / 0.030/0.035/0.033 | 0.257 / 0.211 / 0.234 | 0.086 / 0.053 / 0.070 | 0.507 / 0.516 / 0.511 |
| transformer | clean | 0.210 / 0.205 / 0.208 | 2.274 / 2.283 / 2.279 | 11.222 / 13.000 / 12.111 | 0.120/0.118/0.119 / 0.125/0.126/0.125 / 0.122/0.122/0.122 | 0.258 / 0.230 / 0.244 | 0.220 / 0.250 / 0.235 | 0.473 / 0.488 / 0.481 |
| transformer | feed_like | 0.140 / 0.147 / 0.143 | 2.358 / 2.248 / 2.303 | 10.611 / 12.222 / 11.417 | 0.128/0.129/0.128 / 0.129/0.131/0.130 / 0.129/0.130/0.129 | 0.265 / 0.227 / 0.246 | 0.227 / 0.251 / 0.239 | 0.470 / 0.489 / 0.480 |
| transformer | composite | 0.106 / 0.116 / 0.111 | 2.309 / 2.184 / 2.246 | 13.778 / 16.111 / 14.944 | 0.079/0.084/0.081 / 0.085/0.082/0.081 / 0.082/0.083/0.081 | 0.241 / 0.235 / 0.238 | 0.164 / 0.154 / 0.159 | 0.498 / 0.481 / 0.490 |
| tcn | clean | 0.358 / 0.335 / 0.346 | 2.381 / 2.329 / 2.355 | 11.556 / 10.556 / 11.056 | 0.170/0.157/0.163 / 0.164/0.145/0.153 / 0.167/0.151/0.158 | 0.249 / 0.243 / 0.246 | 0.298 / 0.285 / 0.292 | 0.492 / 0.483 / 0.487 |
| tcn | feed_like | 0.213 / 0.181 / 0.197 | 2.424 / 2.431 / 2.427 | 11.500 / 11.333 / 11.417 | 0.144/0.138/0.141 / 0.150/0.138/0.143 / 0.147/0.138/0.142 | 0.246 / 0.249 / 0.247 | 0.276 / 0.261 / 0.268 | 0.507 / 0.483 / 0.495 |
| tcn | composite | 0.111 / 0.105 / 0.108 | 2.813 / 2.632 / 2.722 | 32.389 / 34.000 / 33.194 | 0.031/0.055/0.040 / 0.028/0.053/0.037 / 0.029/0.054/0.038 | 0.279 / 0.258 / 0.268 | 0.077 / 0.085 / 0.081 | 0.503 / 0.544 / 0.523 |

Effective per-match boundary tolerances: nominal 0.5 s uses 0.48–0.5 s; nominal 1.0 s uses 1–1 s.

### Fit and inference runtime

Fit time is read from checkpoint receipts and averaged across held-out folds; inference is the recorded model prediction, upsampling, and decoder time per match, averaged equally by provider on clean views. Feature-preparation time is shown separately. Archived GRU timings come from a different historical environment, so runtime differences are descriptive and are not a controlled speed benchmark.

| Model | Seed | Fit seconds (fold mean; range) | Inference seconds/match | Feature seconds/match |
|---|---|---:|---:|---:|
| slot_dropout_gru | 2026 | 572.0 (318.3–702.8) | 1.916 | 23.497 |
| slot_dropout_gru | 2027 | 533.2 (378.1–653.1) | 1.940 | 23.497 |
| slot_dropout_gru | 2-seed mean (not ensemble) | 552.6 (318.3–702.8) | 1.928 | 23.497 |
| bilstm | 2026 | 392.6 (209.6–500.5) | 2.370 | 23.497 |
| bilstm | 2027 | 388.5 (231.1–486.3) | 2.341 | 23.497 |
| bilstm | 2-seed mean (not ensemble) | 390.5 (209.6–500.5) | 2.355 | 23.497 |
| transformer | 2026 | 465.8 (274.5–566.8) | 2.569 | 23.497 |
| transformer | 2027 | 475.6 (306.8–588.3) | 2.716 | 23.497 |
| transformer | 2-seed mean (not ensemble) | 470.7 (274.5–588.3) | 2.642 | 23.497 |
| tcn | 2026 | 255.5 (146.1–326.2) | 1.974 | 23.497 |
| tcn | 2027 | 250.5 (139.4–314.4) | 1.849 | 23.497 |
| tcn | 2-seed mean (not ensemble) | 253.0 (139.4–326.2) | 1.912 | 23.497 |

Verified 200 unique report cells across 3 held-out providers; views: clean, feed_like, composite. Replicate aggregation follows the frozen report contract.
