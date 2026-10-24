# Separate data for the published PathCRF checkpoint

The published PathCRF trial 140 checkpoint was trained on all five Sportec matches in our original benchmark. Its leave-one-match-out split uses **J03WQQ for validation and J03WR9 for testing**. Therefore J03WR9 is a separate test of that pretrained model; evaluating it on our original J03WPY test would overlap the checkpoint's training data.

This document covers data preparation and inference-format auditing. It does not report test scores or select parameters on J03WR9.

## Source and reproducible preparation

- Dataset: [Sportec open dataset, Figshare version 1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0.
- Match: J03WR9, Fortuna Düsseldorf–1. FC Kaiserslautern, 2022-11-11.
- Model source: [hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf/tree/1b7c6f85a6b78c92e4543fde72ca2b86a933f997), commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`, MPL 2.0.
- Metadata file ID `51643490`, MD5 `5e2d580d3d8ed97e32959de90d04992b`.
- Observed tracking file ID `51643532`, MD5 `e060bb7d7a5a3bfbd15bbf9bc631024d`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/audit_dataset.py --matches J03WR9 --workers 2
```

The downloader checks both file size and MD5 against the versioned article metadata. Raw files are under `data/external_audit/sportec/raw/`; converted tracking is `data/external_audit/sportec/processed/J03WR9.npz`, with conversion metadata in the adjacent JSON. No event XML is required for geometry-only inference.

The existing five-match split is unchanged. Its SHA-256 before and after this download was `b945a12df761cf5f1dddc0900b99f829841fd11cf07342538ea7fdc7da7401fc`. J03WR9 was added only as an extra raw/processed match; it was not added to that split.

## Geometry schema and roster

The NPZ contains 146,810 observations at 25 Hz, two periods, a 105 × 68 m pitch, and 40 roster slots. Ball and player XY coordinates are metres shifted from the source's centred coordinates to `[0,105] × [0,68]`. Missing player coordinates remain NaN; the converter does not fill them. Period 2 begins at array index 68,354. Timestamps preserve the halftime gap.

Player slots are sorted by `(team, PersonId)`, where home is 0 and away is 1. They are not in XML document order. Recover the mapping by parsing `Teams/Team/Players/Player` in the match-information XML and sorting those pairs. Metadata `PlayingPosition="TW"` identifies the starting goalkeepers: home slot 8 (`DFL-OBJ-0028FW`, shirt 33) and away slot 20 (`DFL-OBJ-00003W`, shirt 1).

An auxiliary, target-free map is saved at `processed/J03WR9_input_metadata.json`. It includes roster IDs, shirts, goalkeeper flags, and half/substitution boundaries inferred from coordinate presence only:

```text
[0, 68354, 107317, 114873, 114890, 121324, 129587, 129956, 135440, 146810]
```

These are half-open array-index intervals. There are eleven observed players per side throughout. One interval between two substitutions contains only 17 frames; an adapter must handle it explicitly and retain it in the evaluation denominator.

The supplied NPZ includes `labels` solely for external scoring. Inference must receive an allowlisted dictionary without that array. BallStatus, BallPossession, event annotations, and target-derived intervals must not enter the inference pipeline.

## Upstream input requirements and leakage exclusions

Trial 140 uses 26 nodes: eleven players per side and four outside nodes. Its eight node features are team indicator, goalkeeper indicator, x, y, vx, vy, speed, and acceleration. Positions are metres and velocities are metres per second; the dataset code does not apply an external normalization. The standard checkpoint has no ball-position input. Its 25 Hz feature stream is subsampled by five to 5 Hz, with 10-second windows.

The team nearer the left goal is ordered first, with team indicator −1; the opposite side is +1. Outside nodes have zero indicators and zero motion features, positioned at `(0,34)`, `(105,34)`, `(52.5,0)`, and `(52.5,68)`. Goalkeepers receive indicator 1. The upstream Sportec loader also transforms to static home/away orientation; raw NPZ coordinates preserve the original orientation, so adapters must account for this consistently. Home defends the left goal in the first half and the right goal in the second half of this match.

Upstream motion features use Savitzky–Golay smoothing, order 2: 15 samples for differences of positions multiplied by FPS, then 9 samples for differences of velocities multiplied by FPS. Acceleration is the magnitude of the resulting x/y acceleration. Player speed outliers above 12 m/s are interpolated within observed spans. An adapter must avoid differentiating across half or substitution boundaries.

Calling the upstream high-level preprocessing unchanged would leak information about the test targets:

- `label_frames_and_episodes` forms episodes from the supplied `ball_state == "alive"` flag.
- `label_phases` also filters to true alive frames and discards phases with fewer than 100 such observations.
- `calculate_running_features` invokes those helpers if their identifiers are absent.
- `PossessionDataset` filters episodes using ground-truth possessing-player availability.
- Inference may inspect target possession when `evaluate=True`, and optional episode-end correction may use the next labelled event.

For a whole-match geometry-only test, use half/coordinate-presence boundaries and fixed windows instead. Call the lower-level inference path with `evaluate=False` and no event-derived `last_node`; disable episode-end correction. Keep every evaluation frame, including short substitution intervals and dead-ball frames. Topology-only CRF allowed-edge masks are model constraints, not target possession.

The checkpoint predicts player-to-player path edges, whereas this benchmark evaluates H/A/D states. Any conversion from edges to team possession and any dead-ball detector must be specified separately and frozen before scoring J03WR9. This is an adapted whole-match evaluation, not a reproduction of the paper's in-play player-edge metric.

## Source limitation

The original Sportec XML freezes ball XY during dead-ball periods. This behavior was verified in the raw files of the original train/validation matches and is not introduced by the converter. A geometry-based dead-ball detector can exploit this dataset-specific cue; success with that detector does not establish dead-ball recognition from broadcast tracking. PathCRF's standard player-only input does not itself use the ball XY cue.
