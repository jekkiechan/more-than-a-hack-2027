"""Apply frozen external models to fragmented challenge tracking, without GT.

The new adapter changes only tracking identity/missingness representation. Its
Hungarian assignment and interpolation never consume possession labels or ball
positions. The original PathCRF adapter and model checkpoint remain unchanged.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments import pathcrf_eval, robust
from experiments.challenge import write_submission

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = {"slots_per_team": 11, "speed_limit_m_s": 12.0,
              "position_tolerance_m": 2.0, "track_id_bonus_m": 1.0,
              "prediction_horizon_s": 0.5,
              "continuous_track_priority_s": 0.5,
              "keeper_policy": "maximum abs(mean_x-pitch_midpoint) times observed fraction"}
INPUT_KEYS = robust.INPUT_KEYS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in INPUT_KEYS if key in source}


def _assign_direction(xy, anchor, initial_tracks, fps, direction, output):
    """Match observations to persistent slots; dummy columns allow missingness."""
    n, n_tracks, _ = xy.shape
    last_xy = xy[anchor, initial_tracks].copy()
    n_slots = len(initial_tracks)
    last_frame = np.full(n_slots, anchor, np.int64)
    last_track = initial_tracks.copy()
    velocity = np.zeros((n_slots, 2), dtype=np.float64)
    discarded = reassigned = observed = 0
    frames = range(anchor + direction, n if direction > 0 else -1, direction)
    for frame in frames:
        tracks = np.flatnonzero(np.isfinite(xy[frame]).all(axis=1))
        if len(tracks) == 0:
            continue
        age = np.abs(frame-last_frame) / fps
        predicted = last_xy + velocity * np.minimum(age, ASSIGNMENT["prediction_horizon_s"])[:, None]
        distances = np.linalg.norm(predicted[:, None] - xy[frame, tracks][None], axis=2)
        gate = ASSIGNMENT["position_tolerance_m"] + ASSIGNMENT["speed_limit_m_s"] * age
        costs = distances - ASSIGNMENT["track_id_bonus_m"] * (last_track[:, None] == tracks)
        costs[distances > gate[:, None]] = 1e9
        continuous = ((last_track[:, None] == tracks) & (distances <= gate[:, None])
                      & (age[:, None] <= ASSIGNMENT["continuous_track_priority_s"]))
        # A stale slot must not steal a still-visible, consistent tracklet merely
        # because its larger motion gate makes leaving it unmatched expensive.
        costs[continuous] = -1e6
        costs = np.c_[costs, np.full((n_slots, n_slots), 105.0)]
        slots, columns = linear_sum_assignment(costs)
        matched = columns < len(tracks)
        slots, tracks = slots[matched], tracks[columns[matched]]
        observed += len(slots)
        discarded += int(np.isfinite(xy[frame]).all(axis=1).sum()) - len(slots)
        reassigned += int(np.count_nonzero(last_track[slots] != tracks))
        positions = xy[frame, tracks]
        new_velocity = (positions-last_xy[slots]) / age[slots, None]
        speed = np.linalg.norm(new_velocity, axis=1)
        new_velocity *= np.minimum(1, ASSIGNMENT["speed_limit_m_s"] / np.maximum(speed, 1e-8))[:, None]
        velocity[slots] = 0.75 * velocity[slots] + 0.25 * new_velocity
        last_xy[slots] = positions
        last_frame[slots] = frame
        # A tracklet has exactly one owner, even after an identity handoff.
        old_owners = np.isin(last_track, tracks)
        old_owners[slots] = False
        last_track[old_owners] = -1
        last_track[slots] = tracks
        output[frame, slots] = positions
    return {"discarded_observations": discarded, "slot_track_changes": reassigned,
            "assigned_observations": observed}


def stabilize_team(xy, fps, pitch_length=105.0):
    """Create eleven identity-consistent approximate trajectories from tracklets.

    A persistently extreme player initializes a reserved goalkeeper slot. The
    first frame with ten other observations initializes the field slots. With extra
    tracks, retain the most frequently observed identities in this half. Forward
    and backward gated matching share that anchor. Interpolation is deliberately
    explicit: this model expects complete trajectories and has no missingness
    feature. It is not evidence that an occluded player's position is known.
    """
    xy = np.asarray(xy, dtype=np.float64)
    observed = np.isfinite(xy).all(axis=2)
    coverage = observed.sum(axis=0)
    mean_x = np.divide(np.nansum(xy[:, :, 0], axis=0), coverage,
                       out=np.full(xy.shape[1], pitch_length/2), where=coverage > 0)
    keeper = int(np.argmax(np.abs(mean_x-pitch_length/2) * coverage))
    field_tracks = np.flatnonzero(np.arange(xy.shape[1]) != keeper)
    field_xy = xy[:, field_tracks]
    field_observed = observed[:, field_tracks]
    anchors = np.flatnonzero(field_observed.sum(axis=1) >= 10)
    if not len(anchors):
        raise ValueError("No frame has ten observed field players; cannot initialize pretrained roster")
    anchor = int(anchors[0])
    tracks = np.flatnonzero(field_observed[anchor])
    tracks = tracks[np.argsort(-coverage[field_tracks[tracks]], kind="stable")[:10]]
    tracks = tracks[np.lexsort((field_xy[anchor, tracks, 1], field_xy[anchor, tracks, 0]))]
    field_output = np.full((len(xy), 10, 2), np.nan)
    field_output[anchor] = field_xy[anchor, tracks]
    forward = _assign_direction(field_xy, anchor, tracks, fps, 1, field_output)
    backward = _assign_direction(field_xy, anchor, tracks, fps, -1, field_output)
    output = np.concatenate([xy[:, keeper:keeper+1], field_output], axis=1)
    assigned = np.isfinite(output).all(axis=2)
    missing_runs = []
    frames = np.arange(len(xy))
    for slot in range(11):
        valid = np.flatnonzero(assigned[:, slot])
        if not len(valid):
            raise AssertionError("An initialized roster slot has no observations")
        gaps = np.diff(np.r_[-1, valid, len(xy)]) - 1
        missing_runs.extend((gaps[gaps > 0] / fps).tolist())
        for axis in (0, 1):
            output[:, slot, axis] = np.interp(frames, valid, output[valid, slot, axis])
    metadata = {
        "anchor_frame_within_half": anchor,
        "geometric_keeper_track": keeper,
        "geometric_keeper_mean_x": float(mean_x[keeper]),
        "geometric_keeper_observed_fraction": float(coverage[keeper]/len(xy)),
        "original_visible_count_quantiles": np.quantile(observed.sum(axis=1), [0, .1, .5, .9, 1]).tolist(),
        "imputed_slot_frame_fraction": float(1-assigned.mean()),
        "missing_run_seconds_quantiles": np.quantile(missing_runs, [0, .5, .9, .99, 1]).tolist() if missing_runs else [0]*5,
        "discarded_observations": forward["discarded_observations"] + backward["discarded_observations"] + max(0, field_observed[anchor].sum()-10),
        "slot_track_changes": forward["slot_track_changes"] + backward["slot_track_changes"],
    }
    metadata["discarded_observations"] = int(metadata["discarded_observations"])
    return output.astype(np.float32), assigned, metadata


def prepare_half(data, indices):
    fps = float(data["fps"])
    player_groups, observed_groups, report = [], [], {}
    for team in (0, 1):
        players = data["players"][indices][:, data["teams"] == team]
        stable, observed, metadata = stabilize_team(players, fps, float(data["pitch"][0]))
        player_groups.append(stable)
        observed_groups.append(observed)
        report[str(team)] = metadata
    prepared = {"players": np.concatenate(player_groups, axis=1),
                "teams": np.repeat([0, 1], 11),
                "period": data["period"][indices], "fps": data["fps"],
                "timestamps": data["timestamps"][indices], "pitch": data["pitch"]}
    return prepared, np.concatenate(observed_groups, axis=1), report


def synthetic_contract():
    """A fragmented, shuffled synthetic roster must preserve physical motion."""
    n, fps = 100, 25
    base = np.c_[np.linspace(5, 95, 11), np.linspace(8, 60, 11)]
    truth = base[None] + np.arange(n)[:, None, None] * np.array([.02, .005])
    xy = np.full((n, 22, 2), np.nan)
    # Keeper stays a tracklet, while all ten field identities fragment midway.
    xy[:, 0] = truth[:, 0]
    xy[:50, 1:11] = truth[:50, 1:]
    xy[50:, 12:] = truth[50:, 1:]
    xy[25:35, 3] = np.nan
    stable, observed, report = stabilize_team(xy, fps)
    assert np.max(np.abs(stable-truth)) < 1e-4
    assert np.isfinite(stable).all() and stable.shape == (n, 11, 2)
    assert not observed[25:35, 3].any()
    # Observation order does not determine the physical assignment.
    shuffled, _, _ = stabilize_team(xy[:, np.arange(22)[::-1]], fps)
    assert np.allclose(shuffled, stable)
    return {"fragmented_identity_recovery": True, "missing_gap_interpolation": True,
            "track_column_permutation": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/challenge/inputs.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/challenge_pathcrf")
    parser.add_argument("--robust-model", type=Path, default=ROOT / "data/experiments/robust/model.joblib")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original_frozen = json.loads((ROOT / "data/experiments/pathcrf/frozen_config.json").read_text())
    robust_frozen = json.loads((ROOT / "data/experiments/robust/frozen_config.json").read_text())
    assert sha256(pathcrf_eval.__file__) == original_frozen["adapter_sha256"]
    assert sha256(pathcrf_eval.UPSTREAM / pathcrf_eval.CHECKPOINT) == original_frozen["checkpoint_sha256"]
    assert sha256(args.robust_model) == robust_frozen["model_sha256"]
    contracts = synthetic_contract()
    data = load_inputs(args.inputs)
    halves = []
    geometry_report = {}
    for half in sorted(set(data["period"]) - {0}):
        indices = np.flatnonzero(data["period"] == half)
        prepared, observed, report = prepare_half(data, indices)
        np.savez_compressed(args.out / f"half_{half}_geometry.npz", **prepared,
                            player_observed=observed)
        halves.append((int(half), indices, prepared))
        geometry_report[str(half)] = report
        print(f"Prepared half {half}: {len(indices)} frames", flush=True)
    (args.out / "geometry_audit.json").write_text(json.dumps(geometry_report, indent=2) + "\n")
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(args.inputs),
        "adapter_sha256": sha256(__file__),
        "original_adapter_sha256": sha256(pathcrf_eval.__file__),
        "checkpoint_commit": pathcrf_eval.COMMIT,
        "checkpoint_sha256": sha256(pathcrf_eval.UPSTREAM / pathcrf_eval.CHECKPOINT),
        "robust_model_sha256": sha256(args.robust_model),
        "assignment": ASSIGNMENT, "contracts": contracts,
        "candidates": ["frozen_robust_had", "pathcrf_ha", "pathcrf_team_robust_dead"],
        "selection": "predeclared transfer candidates; never select using challenge public GT",
        "challenge_ground_truth_read": False, "period_zero_label": "X",
        "pathcrf_window_seconds": 10, "pathcrf_stride_seconds": 5,
        "threads": 2, "batch_size": 4,
    }
    (args.out / "frozen_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        return
    started = time.monotonic()
    baseline = robust.predict(data, joblib.load(args.robust_model))
    baseline[data["period"] == 0] = "X"
    model = pathcrf_eval.load_model(threads=2)
    native = np.full(len(data["period"]), "X", dtype="U1")
    macro = native.copy()
    inference = {}
    for half, indices, prepared in halves:
        pred, macro_pred, stats = pathcrf_eval.predict(prepared, model, batch_size=4)
        native[indices] = pred
        macro[indices] = macro_pred
        inference[str(half)] = stats
    hybrid = native.copy()
    hybrid[baseline == "D"] = "D"
    outputs = {"frozen_robust_had": baseline, "pathcrf_ha": native,
               "pathcrf_team_robust_dead": hybrid}
    output_hashes = {}
    for name, prediction in outputs.items():
        path = args.out / f"{name}.csv"
        write_submission(prediction, path)
        output_hashes[name] = sha256(path)
    np.savez_compressed(args.out / "predictions.npz", **outputs, macro_diagnostic=macro)
    metadata = {"inference": inference, "elapsed_seconds": time.monotonic()-started,
                "n_frames": len(native), "n_half_frames": int(np.count_nonzero(data["period"])),
                "n_outside_half_frames": int(np.count_nonzero(data["period"] == 0)),
                "output_sha256": output_hashes, "challenge_ground_truth_read": False,
                "frozen_manifest": "frozen_config.json"}
    (args.out / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
