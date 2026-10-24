# SPDX-License-Identifier: MPL-2.0
# Upstream feature/decode conventions: https://github.com/hyunsungkim-ds/pathcrf
# Source revision and its MPL-2.0 license are retained in the ignored checkout.
"""Tracking-only transfer adapter for the official pretrained PathCRF trial 140.

This is a possession-team diagnostic, not a replication of the paper's event
benchmark. Never call the upstream episode/phase builders: they use true alive
labels. Prediction loads only an explicit geometry whitelist from NPZ files.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data/external_models/pathcrf"
PACKAGES = ROOT / "data/external_models/python_packages"
COMMIT = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
CHECKPOINT = "saved/140/model/state_dict_best_acc.pt"
GEOMETRY_KEYS = ("players", "teams", "period", "fps", "timestamps", "pitch")


def load_geometry(path):
    """Do not even deserialize labels, ball status, events, or possession."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in GEOMETRY_KEYS if key in source}


def blocks(data):
    breaks = np.diff(data["period"]) != 0
    if "timestamps" in data:
        dt = np.diff(data["timestamps"])
        breaks |= (dt <= 0) | (dt > 1.5 / float(data["fps"]))
    bounds = np.r_[0, np.flatnonzero(breaks) + 1, len(data["period"])]
    return list(zip(bounds[:-1], bounds[1:]))


def _smooth(value, width):
    if len(value) < 3:
        return value
    width = min(width, len(value) if len(value) % 2 else len(value) - 1)
    return savgol_filter(value, width, polyorder=min(2, width - 1), axis=0)


def physical_features(xy, fps):
    """Match upstream 25Hz derivative smoothing, independently of all labels.

    Input is one player's observed active span within a period. Interpolate
    internal missing coordinates; remove >12m/s instantaneous player outliers,
    then reproduce upstream velocity/acceleration alignment and smoothing.
    """
    xy = np.array(xy, dtype=np.float64, copy=True)
    frames = np.arange(len(xy))
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return np.zeros((len(xy), 6), np.float32)
    for axis in (0, 1):
        xy[:, axis] = np.interp(frames, frames[finite], xy[finite, axis])
    if len(xy) == 1:
        return np.c_[xy, np.zeros((1, 4))].astype(np.float32)
    outlier = np.r_[False, np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps > 12]
    good = ~outlier
    for axis in (0, 1):
        xy[:, axis] = np.interp(frames, frames[good], xy[good, axis])
    vel = _smooth(np.diff(xy, axis=0) * fps, 15)
    accel = _smooth(np.diff(vel, axis=0) * fps, 9)
    result = np.zeros((len(xy), 6), np.float32)
    result[:, :2] = xy
    result[1:, 2:4] = vel
    result[0, 2:4] = vel[0]
    result[:, 4] = np.linalg.norm(result[:, 2:4], axis=1)
    result[1:-1, 5] = np.linalg.norm(accel, axis=1)
    return result


def feature_phases(data):
    """Yield geometry-only roster phases: (start, end, 26-node tensor, teams).

    Substitutions are first/last observations, never event annotations. Missing
    roster slots remain zero, as in the original padded training tensors.
    """
    players = np.asarray(data["players"], dtype=np.float64)
    teams = np.asarray(data["teams"])
    fps = float(data["fps"])
    pitch = np.asarray(data.get("pitch", [105, 68]))
    players = players * (np.array([105, 68]) / pitch)
    for period_start, period_end in blocks(data):
        pxy = players[period_start:period_end].copy()
        # Match the source loader's STATIC_HOME_AWAY orientation, using only
        # player positions. Rotate both axes when home defends the right goal.
        observed_count = np.isfinite(pxy[:, :, 0]).sum(axis=0)
        mean_x = np.divide(np.nansum(pxy[:, :, 0], axis=0), observed_count,
                           out=np.full(pxy.shape[1], np.nan), where=observed_count > 0)
        keepers = []
        for team in (0, 1):
            candidates = np.flatnonzero((teams == team) & (observed_count > 0))
            keepers.append(candidates[np.argmax(np.abs(mean_x[candidates] - 52.5))])
        if mean_x[keepers[0]] > mean_x[keepers[1]]:
            pxy = np.array([105, 68]) - pxy
        physical = np.zeros((*pxy.shape[:2], 6), np.float32)
        spans = []
        boundaries = {0, len(pxy)}
        for player in range(pxy.shape[1]):
            observed = np.flatnonzero(np.isfinite(pxy[:, player]).all(axis=1))
            if not len(observed):
                spans.append((-1, -1))
                continue
            first, stop = int(observed[0]), int(observed[-1]) + 1
            spans.append((first, stop))
            boundaries.update((first, stop))
            physical[first:stop, player] = physical_features(pxy[first:stop, player], fps)
        boundaries = sorted(boundaries)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            active = [i for i, (first, stop) in enumerate(spans) if first <= start < stop]
            selected = [[i for i in active if teams[i] == team] for team in (0, 1)]
            if not all(selected):
                raise ValueError("A geometry phase contains no players for one team")
            if max(map(len, selected)) > 11:
                raise ValueError("More than 11 active players: inspect geometry-only roster conversion")
            # Same goalkeeper heuristic as upstream utils.detect_keepers.
            mean_x = {i: physical[start:end, i, 0].mean() for i in active}
            gks = [max(team, key=lambda p: abs(mean_x[p] - 52.5)) for team in selected]
            left = 0 if mean_x[gks[0]] < mean_x[gks[1]] else 1
            node_teams = np.full(26, -1, np.int8)
            feature = np.zeros((end - start, 26, 8), np.float32)
            for side, team in enumerate((left, 1-left)):
                for slot, player in enumerate(selected[team]):
                    node = side * 11 + slot
                    feature[:, node, 0] = -1 if side == 0 else 1
                    feature[:, node, 1] = player == gks[team]
                    feature[:, node, 2:] = physical[start:end, player]
                    node_teams[node] = team
            feature[:, 22:, 2:4] = [[0, 34], [105, 34], [52.5, 0], [52.5, 68]]
            yield period_start + start, period_start + end, feature, node_teams


def load_model(upstream=UPSTREAM, threads=1):
    if PACKAGES.exists():
        sys.path.insert(0, str(PACKAGES))
    sys.path.insert(0, str(upstream))
    import torch
    from models.utils import build_model
    torch.set_num_threads(threads)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Unexpected upstream revision: {revision}")
    args = json.loads((upstream / "saved/140/args.json").read_text())
    if args.get("ball_feature", False) or args["node_in_dim"] != 8:
        raise ValueError("Adapter expects the audited player-only eight-feature checkpoint")
    model = build_model(args, device=torch.device("cpu"))
    model.load_state_dict(torch.load(upstream / CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def predict(data, model, batch_size=4, progress=True):
    """Frozen 10s / 5s stride windows, CRF decode, source-team aggregation.

    The upstream CRF treats outside nodes as absorbing states. Decode windows
    independently so a whole half cannot get trapped after an out-of-play event.
    Overlaps combine triangular weighted team votes. Outside decoded sources
    carry the last predicted player team; a leading outside source uses the
    pretrained macro source distribution over real player nodes.
    """
    import torch
    fps = float(data["fps"])
    step = int(round(fps / 5))
    if not np.isclose(fps / step, 5):
        raise ValueError("Adapter currently requires a source FPS divisible by 5")
    total = len(data["period"])
    predictions = np.full(total, "X", dtype="U1")
    macro_predictions = predictions.copy()
    stats = {"phases": 0, "windows": 0, "outside_source_steps": 0, "decoded_steps": 0}
    began = time.monotonic()
    with torch.inference_mode():
        for begin, end, full_features, node_teams in feature_phases(data):
            stats["phases"] += 1
            down_idx = np.arange(0, len(full_features), step)
            features = full_features[down_idx]
            n = len(features)
            width = min(50, n)
            starts = list(range(0, max(1, n-width+1), 25))
            if starts[-1] != n-width:
                starts.append(n-width)
            votes = np.zeros((n, 2), np.float64)
            macro_votes = np.zeros_like(votes)
            weight_total = np.zeros(n)
            weights = 1 + np.minimum(np.arange(width), np.arange(width)[::-1])
            valid_players = np.flatnonzero(node_teams >= 0)
            for chunk_start in range(0, len(starts), batch_size):
                chunk = starts[chunk_start:chunk_start+batch_size]
                tensor = torch.from_numpy(np.stack([features[s:s+width] for s in chunk]))
                # All optional target/teacher-forcing inputs remain None.
                macro, micro, edges = model(tensor)
                valid_edges = model.valid_orig_edge_ids
                decoded = model.crf.decode(micro.index_select(2, valid_edges),
                                           edges.index_select(2, valid_edges))
                sources = model.comp_src[decoded].numpy()
                macro_prob = torch.softmax(macro[:, :, valid_players, 0], dim=-1).numpy()
                team_prob = np.stack([macro_prob[:, :, node_teams[valid_players] == team].sum(axis=-1)
                                      for team in (0, 1)], axis=-1)
                for row, start in enumerate(chunk):
                    source_team = node_teams[sources[row]].copy()
                    stats["outside_source_steps"] += int(np.count_nonzero(source_team < 0))
                    stats["decoded_steps"] += width
                    for frame in range(width):
                        if source_team[frame] < 0:
                            source_team[frame] = (source_team[frame-1] if frame else
                                                  int(np.argmax(team_prob[row, frame])))
                    votes[start:start+width] += np.eye(2)[source_team] * weights[:, None]
                    macro_votes[start:start+width] += team_prob[row] * weights[:, None]
                    weight_total[start:start+width] += weights
                stats["windows"] += len(chunk)
            if np.any(weight_total == 0):
                raise AssertionError("Uncovered tracking frames")
            up_idx = np.minimum(np.arange(end-begin) // step, n-1)
            predictions[begin:end] = np.array(["H", "A"])[votes.argmax(axis=1)[up_idx]]
            macro_predictions[begin:end] = np.array(["H", "A"])[macro_votes.argmax(axis=1)[up_idx]]
            if progress:
                print(f"PathCRF phase {stats['phases']}: {end}/{total} frames, "
                      f"{stats['windows']} windows, {time.monotonic()-began:.1f}s", flush=True)
    stats["inference_seconds"] = time.monotonic() - began
    return predictions, macro_predictions, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/pathcrf")
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, help="Geometry-only smoke test, never a scored subset")
    args = parser.parse_args()
    data = load_geometry(args.match)
    if args.max_frames:
        for key in ("players", "period", "timestamps"):
            if key in data:
                data[key] = data[key][:args.max_frames]
    model = load_model(args.upstream, args.threads)
    pred, macro, stats = predict(data, model, args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.match.stem}_predictions.npz", prediction=pred,
                        macro_diagnostic=macro, period=data["period"], fps=data["fps"])
    metadata = {"upstream_commit": COMMIT, "checkpoint": CHECKPOINT,
                "checkpoint_sha256": hashlib.sha256((args.upstream / CHECKPOINT).read_bytes()).hexdigest(),
                "match": args.match.stem, "n_frames": len(pred), "geometry_keys": list(data),
                "window_seconds": 10, "stride_seconds": 5, "target_fps": 5,
                "orientation": "geometry-inferred STATIC_HOME_AWAY; rotate both axes",
                "uses_true_live_boundaries": False, "native_dead_head": False,
                "test_labels_read": False, "batch_size": args.batch_size, **stats}
    (args.out / f"{args.match.stem}_inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
