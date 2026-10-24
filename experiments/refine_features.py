"""Two bounded physical-feature refinements, selected on validation only.

Adds ball/player relative motion and contact context to the frozen baseline
feature implementation. No ownership, possession, event, or state metadata is
available to features or prediction. Existing models/sources are never changed.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import robust, supervised  # noqa: E402

CAPACITIES = ((31, 160), (63, 260))
SWITCH_COSTS = (0, 1, 3, 8)
ALLOWED_TRAIN = {"J03WMX", "J03WN1", "J03WOH"}
ALLOWED_VALIDATION = {"J03WOY"}


def _gradient(values, period, fps):
    output = np.full_like(values, np.nan, dtype=np.float64)
    for start, stop in supervised.blocks(period):
        if stop - start > 1:
            output[start:stop] = np.gradient(values[start:stop], axis=0) * fps
    return output


def _vector_limit(values, maximum):
    result = values.copy()
    result[np.linalg.norm(result, axis=-1) > maximum] = np.nan
    return result


def features(data):
    """Tracking-only features; all team-specific columns use H_/A_ prefixes."""
    data = robust.tracking_inputs(data)
    base, names = supervised.features(data)
    names = list(names)
    columns = [base]
    fps = float(data["fps"])
    period = supervised.model_periods(data)
    raw_ball = np.asarray(data["ball"], dtype=np.float64)
    ball = supervised._short_fill(raw_ball, period, fps)
    players = np.asarray(data["players"], dtype=np.float64)
    teams = np.asarray(data["teams"])
    available = np.isfinite(ball).all(axis=1)
    width = max(1, 2 * round(0.08 * fps) + 1)
    ball_velocity = supervised._rolling(_gradient(ball, period, fps), width, period)
    ball_velocity = _vector_limit(ball_velocity, 65)
    ball_velocity[~available] = np.nan
    player_velocity = supervised._rolling(_gradient(players, period, fps), width, period)
    player_velocity = _vector_limit(player_velocity, 13)
    player_velocity[~np.isfinite(players).all(axis=2)] = np.nan
    ball_accel_vec = _gradient(ball_velocity, period, fps)
    ball_accel = np.minimum(np.linalg.norm(ball_accel_vec, axis=1), 150)
    player_accel = np.minimum(np.linalg.norm(_gradient(player_velocity, period, fps), axis=2), 30)
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    player_speed = np.linalg.norm(player_velocity, axis=2)

    def add(name, values):
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32)[:, None])

    def context(name, values, seconds):
        add(f"{name}_mean_{seconds}s", supervised._rolling(
            values, 2 * round(seconds * fps) + 1, period))

    add("physical_ball_acceleration", ball_accel)
    add("physical_ball_speed", ball_speed)
    for seconds in (0.25, 1.0):
        context("physical_ball_acceleration", ball_accel, seconds)
    lag = max(1, round(0.2 * fps))
    turn = np.full(len(ball), np.nan)
    for start, stop in supervised.blocks(period):
        if stop - start <= 2 * lag:
            continue
        before, after = ball_velocity[start:stop-2*lag], ball_velocity[start+2*lag:stop]
        denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        turn[start+lag:stop-lag] = 1 - np.divide(
            np.sum(before * after, axis=1), denom,
            out=np.full(len(denom), np.nan), where=denom > 0.25)
    add("physical_ball_turn", turn)
    context("physical_ball_turn", turn, 0.25)
    pitch = np.asarray(data.get("pitch", [105, 68]), dtype=float)
    edge_distances = np.column_stack((ball[:, 0], pitch[0] - ball[:, 0],
                                     ball[:, 1], pitch[1] - ball[:, 1]))
    nearest_edge = np.argmin(np.where(np.isfinite(edge_distances), edge_distances, np.inf), axis=1)
    outward = np.column_stack((-ball_velocity[:, 0], ball_velocity[:, 0],
                               -ball_velocity[:, 1], ball_velocity[:, 1]))
    normal_velocity = outward[np.arange(len(ball)), nearest_edge]
    add("boundary_outward_ball_speed", normal_velocity)
    context("boundary_outward_ball_speed", normal_velocity, 0.25)

    offset = ball[:, None, :] - players
    distance = np.linalg.norm(offset, axis=2)
    relative_velocity = ball_velocity[:, None, :] - player_velocity
    relative_speed = np.linalg.norm(relative_velocity, axis=2)
    closing = -np.sum(offset * relative_velocity, axis=2) / np.maximum(distance, 0.1)
    cosine = np.sum(player_velocity * ball_velocity[:, None, :], axis=2) / np.maximum(
        player_speed * ball_speed[:, None], 0.25)
    parallel = np.sum(player_velocity * ball_velocity[:, None, :], axis=2) / np.maximum(
        ball_speed[:, None], 0.5)
    metrics = {"relative_speed": relative_speed, "radial_closing": closing,
               "speed_alignment": cosine, "player_speed": player_speed,
               "parallel_speed": parallel, "player_acceleration": player_accel}

    for team, prefix in enumerate(("H", "A")):
        indices = np.flatnonzero(teams == team)
        if not len(indices):
            raise ValueError("Expected player slots from both teams")
        team_distance = distance[:, indices]
        ranked = np.argsort(np.where(np.isfinite(team_distance), team_distance, np.inf), axis=1)
        first = indices[ranked[:, 0]]
        row = np.arange(len(ball))
        nearest_distance = distance[row, first]
        for rank in (0, 1):
            selected = indices[ranked[:, min(rank, len(indices)-1)]]
            for metric, value in metrics.items():
                selected_value = value[row, selected].copy()
                selected_value[~np.isfinite(distance[row, selected])] = np.nan
                add(f"{prefix}_physical_nearest{rank+1}_{metric}", selected_value)
                if rank == 0 and metric in ("relative_speed", "radial_closing", "speed_alignment"):
                    for seconds in (0.25, 1.0):
                        context(f"{prefix}_physical_nearest1_{metric}", selected_value, seconds)
        # Smooth, bounded contact evidence: proximity plus shared movement.
        closest_rel = relative_speed[row, first]
        control = np.exp(-0.5 * (nearest_distance / 1.8) ** 2) * np.exp(-0.5 * (closest_rel / 5) ** 2)
        impulse = np.exp(-0.5 * (nearest_distance / 2.5) ** 2) * np.minimum(ball_accel / 20, 3)
        add(f"{prefix}_physical_control_evidence", control)
        add(f"{prefix}_physical_contact_impulse", impulse)
        for seconds in (0.25, 1.0):
            context(f"{prefix}_physical_control_evidence", control, seconds)
            context(f"{prefix}_physical_contact_impulse", impulse, seconds)
        for radius in (3, 8):
            near = (team_distance <= radius) & np.isfinite(team_distance)
            for metric, value in (("speed", player_speed), ("relative_speed", relative_speed),
                                  ("closing", closing)):
                observed = near & np.isfinite(value[:, indices])
                count = observed.sum(axis=1)
                total = np.where(observed, value[:, indices], 0).sum(axis=1)
                average = np.divide(total, count, out=np.full(len(ball), np.nan), where=count > 0)
                add(f"{prefix}_physical_within{radius}m_{metric}", average)
        # Collective speed is still observable during masked ball gaps.
        for quantile in (0.5, 0.9):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value = np.nanquantile(player_speed[:, indices], quantile, axis=1)
            add(f"{prefix}_physical_player_speed_q{quantile}", value)

    matrix = np.column_stack(columns).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("Invalid feature schema")
    return matrix, names


def probabilities_from_features(matrix, names, model):
    if names != model["feature_names"]:
        raise ValueError("Feature schema mismatch")
    with threadpool_limits(limits=1):
        direct = model["estimator"].predict_proba(matrix)
        reverse = model["estimator"].predict_proba(supervised.swap_team_features(matrix, names))
    classes = np.asarray(model["classes"])
    result = np.zeros((len(matrix), 3))
    for column, label in enumerate(classes):
        opposite = 1-label if label < 2 else label
        index = int(np.flatnonzero(classes == opposite)[0])
        result[:, label] = (direct[:, column] + reverse[:, index]) / 2
    return result


def predict_probabilities(data, model):
    matrix, names = features(robust.tracking_inputs(data))
    return probabilities_from_features(matrix, names, model)


def predict(data, model):
    inputs = robust.tracking_inputs(data)
    return supervised.decode(predict_probabilities(inputs, model), supervised.model_periods(inputs),
                             float(inputs["fps"]), model["config"]["switch_cost"])


def self_test():
    n, fps = 80, 10.
    t = np.arange(n) / fps
    ball = np.column_stack((20 + t, np.full(n, 20.)))
    players = np.stack((ball + [0.6, 0], ball + [4, 2], ball + [-0.9, 0], ball + [-5, 1]), axis=1)
    data = {"ball": ball, "players": players, "teams": np.array([0, 0, 1, 1]),
            "period": np.r_[np.ones(40), np.full(40, 2)], "fps": np.array(fps),
            "timestamps": t, "pitch": np.array([105, 68]), "labels": np.full(n, "H"),
            "ball_observed": np.ones(n, dtype=bool), "owner": np.zeros(n)}
    original, names = features(data)
    changed, changed_names = features(dict(data, labels=np.full(n, "D"), owner=np.ones(n)))
    np.testing.assert_array_equal(original, changed)
    assert names == changed_names
    swapped, swapped_names = features(dict(data, teams=1-data["teams"]))
    np.testing.assert_allclose(swapped, supervised.swap_team_features(original, names),
                               rtol=1e-5, atol=1e-5, equal_nan=True)
    assert swapped_names == names
    index = {name: i for i, name in enumerate(names)}
    np.testing.assert_allclose(original[:, index["H_physical_nearest1_relative_speed"]], 0, atol=1e-5)
    np.testing.assert_allclose(original[:, index["H_physical_nearest1_speed_alignment"]], 1, atol=1e-5)
    # Complete gaps preserve target independence and do not invent motion.
    masked = robust.masked_inputs(data, 2026)
    masked_x, _ = features(masked)
    masked_swap, _ = features(dict(masked, teams=1-masked["teams"]))
    np.testing.assert_allclose(masked_swap, supervised.swap_team_features(masked_x, names),
                               rtol=1e-5, atol=1e-5, equal_nan=True)
    print("CHECKS target-independence, team-swap equivariance, shared-velocity physics passed", flush=True)


def prepare_training(paths):
    xs, ys, manifest, names = [], [], [], None
    for i, path in enumerate(paths):
        data = supervised.load(path)
        labels = np.asarray(data["labels"])
        stride = max(1, round(float(data["fps"]) / 4))
        rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
        y = np.array([np.flatnonzero(supervised.LABELS == label)[0] for label in labels[rows]], dtype=np.int8)
        for condition, view in (("clean", robust.tracking_inputs(data)),
                                ("masked", robust.masked_inputs(data, 2026+i))):
            print(f"FEATURES {path.stem} {condition}", flush=True)
            x, current_names = features(view)
            if names is not None and names != current_names:
                raise ValueError("Inconsistent feature schemas")
            names = current_names
            x = x[rows]
            xs.extend((x, supervised.swap_team_features(x, names)))
            ys.extend((y, np.where(y < 2, 1-y, y)))
        manifest.append({"path": str(path), "sha256": robust.sha256(path),
                         "labelled_sampled_frames": len(rows), "stride": stride, "mask_seed": 2026+i})
    return np.concatenate(xs), np.concatenate(ys), names, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "data/external_audit/sportec/split.json")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/external_audit/sportec/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_features")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.check:
        return
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "frozen_config.json").exists():
        raise ValueError("Already frozen; use a new output directory")
    split = json.loads(args.split.read_text())
    if set(split["train"]) != ALLOWED_TRAIN or set(split["validation"]) != ALLOWED_VALIDATION:
        raise ValueError("This bounded experiment is restricted to the predeclared train/validation matches")
    # Test IDs may occur in the manifest, but no held-out path is constructed.
    x, y, names, manifest = prepare_training([args.processed / f"{name}.npz" for name in split["train"]])
    validation, val_features = [], {}
    for name in split["validation"]:
        data = supervised.load(args.processed / f"{name}.npz")
        validation.append((name, data))
        val_features[name] = {}
        for condition, view in (("clean", robust.tracking_inputs(data)),
                                ("masked", robust.masked_inputs(data, 2026))):
            print(f"FEATURES {name} {condition}", flush=True)
            matrix, these_names = features(view)
            if names != these_names:
                raise ValueError("Validation schema differs")
            val_features[name][condition] = (view, matrix)
    print(f"TRAIN rows={len(y)} features={len(names)}", flush=True)
    candidates, best, best_model, best_probabilities = [], None, None, None
    for leaves, iterations in CAPACITIES:
        print(f"FIT leaves={leaves} iterations={iterations}", flush=True)
        estimator = HistGradientBoostingClassifier(
            max_iter=iterations, max_leaf_nodes=leaves, learning_rate=0.08,
            min_samples_leaf=35, l2_regularization=1, early_stopping=False, random_state=17)
        with threadpool_limits(limits=1):
            estimator.fit(x, y)
        model = {"estimator": estimator, "feature_names": names, "classes": estimator.classes_.tolist(),
                 "training": manifest,
                 "config": {"leaves": leaves, "iterations": iterations, "switch_cost": 0,
                            "train_fps": 4, "two_stage": False, "random_state": 17,
                            "augmentation": "clean + 41% block-masked; each with team swap",
                            "features_api": "experiments.refine_features.features",
                            "prediction_api": "experiments.refine_features.predict"}}
        probabilities = {name: {condition: (view, probabilities_from_features(matrix, names, model))
                               for condition, (view, matrix) in views.items()}
                         for name, views in val_features.items()}
        for cost in SWITCH_COSTS:
            candidate = robust.evaluate_cost(validation, probabilities, cost)
            candidate.update({"leaves": leaves, "iterations": iterations})
            candidates.append(candidate)
            print(json.dumps({"leaves": leaves, "iterations": iterations, "switch_cost": cost,
                  **{condition: {key: candidate[condition][key] for key in ("mean_macro_f1", "mean_spell_f1")}
                     for condition in ("clean", "masked")}}), flush=True)
            if best is None or robust.selection_key(candidate) > robust.selection_key(best):
                best = candidate
                best_model = dict(model, config=dict(model["config"], switch_cost=cost))
                best_probabilities = probabilities
        (args.out / "validation_progress.json").write_text(json.dumps(candidates, indent=2) + "\n")
    model_path = args.out / "model.joblib"
    joblib.dump(best_model, model_path)
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "config": best_model["config"],
              "selection": robust.SELECTION, "capacities": CAPACITIES, "switch_costs": SWITCH_COSTS,
              "training": manifest, "split": split, "classes": best_model["classes"],
              "feature_count": len(names), "training_rows_after_augmentation": len(y),
              "model_sha256": robust.sha256(model_path), "stress_used_for_selection": False,
              "test_evaluated": False, "validation_mask_seed": 2026,
              "source_sha256": {str(p.relative_to(ROOT)): robust.sha256(p) for p in (
                  ROOT / "experiments/refine_features.py", ROOT / "experiments/robust.py",
                  ROOT / "experiments/supervised.py", ROOT / "experiments/stress.py",
                  ROOT / "experiments/metrics.py")},
              "validation_data_sha256": {name: robust.sha256(args.processed / f"{name}.npz")
                                         for name in split["validation"]}}
    (args.out / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.out / "validation.json").write_text(json.dumps({"selected": best, "candidates": candidates}, indent=2) + "\n")
    for name, conditions in best_probabilities.items():
        for condition, (view, probability) in conditions.items():
            np.savez_compressed(args.out / f"{name}_{condition}_probabilities.npz",
                                probabilities=probability.astype(np.float32), period=view["period"],
                                timestamps=view["timestamps"], fps=view["fps"])
    print("FROZEN", json.dumps({"config": frozen["config"], "model_sha256": frozen["model_sha256"],
                                "seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
