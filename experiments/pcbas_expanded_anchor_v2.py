"""Expanded, frozen-package PCBAS TRAIN action-anchor evaluation.

This is a sparse event-anchor portability study, not possession-spell evaluation.
The target column is excluded from every inference read; all target joins occur
only after the complete, hash-sealed prediction cohort has been produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = "pcbas_expanded_anchor_v2"
SALT = "pcbas_expanded_anchor_v2"
DATA = ROOT / "data/independent/sn_pcbas_2026"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / "experiments/results" / f"{STUDY}_declaration.json"
RESULT = ROOT / "experiments/results" / f"{STUDY}_result.json"
MANIFEST = ROOT / "experiments/baseline_manifest.json"
H5 = DATA / "train_tactical_data.h5"
EVENTS = DATA / "playbyplay_train.json"
PACKAGE = "joint_gru_sn_dense_original_diversity_phase_ensemble_v1"
N_GAMES = 12
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261102
KNOWN_VAL_GAMES = {"game_18", "game_24", "game_47"}
VISIBILITY_BINS = ((0, 5, "0-5"), (6, 11, "6-11"), (12, None, "12+"))
CLASSES = {1: "drive", 2: "pass", 3: "cross", 5: "shot"}

# Inference call graph modules whose implementation can affect this study.
# This is intentionally explicit; unrelated experiment files are not frozen.
SOURCE_FILES = (
    "experiments/pcbas_expanded_anchor_v2.py",
    "experiments/pcbas_event_anchor_v1.py",
    "experiments/current_baseline.py",
    "experiments/original_full_phase_baseline.py",
    "experiments/original_full_baseline.py",
    "experiments/feed_like_v2_public.py",
    "experiments/feed_benchmark_audit_v1.py",
    "experiments/soccernet_extension.py",
    "experiments/soccernet_capacity.py",
    "experiments/soccernet_capacity_model.py",
    "experiments/sn_dense_baseline.py",
    "experiments/sn_dense_train.py",
    "experiments/live_gap_experiment.py",
    "experiments/tracking_corruption.py",
    "experiments/supervised.py",
    "experiments/learned_temporal.py",
    "experiments/had_gru.py",
    "experiments/factorized_had.py",
    "experiments/objective_losses.py",
    "experiments/refine_features.py",
    "experiments/refine_domain.py",
    "experiments/robust.py",
    "experiments/stress.py",
    "experiments/stress_components.py",
    "experiments/eval_set/protocol.py",
    "experiments/calibrated_dense_baseline.py",
    "experiments/feed_spell_more.py",
    "experiments/feed_spell_push.py",
    "experiments/decoder_refinement.py",
    "experiments/decoder_search.py",
)


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def atomic_exclusive(path: Path, writer) -> None:
    """Write a complete artifact then claim its final name without overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        writer(temp_path)
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def rank_games(game_ids):
    """Stable SHA256 ranking independent of labels, predictions or file order."""
    ranked = []
    for game_id in sorted(set(map(str, game_ids))):
        rank = hashlib.sha256(f"{SALT}:{game_id}".encode("utf-8")).hexdigest()
        ranked.append(dict(game_id=game_id, rank_sha256=rank))
    return sorted(ranked, key=lambda row: (row["rank_sha256"], row["game_id"]))


def cohort_from_keys(keys):
    halves_by_game = {}
    for key in keys:
        if "_H" not in key:
            continue
        game, half_text = key.rsplit("_H", 1)
        if half_text not in ("1", "2"):
            continue
        halves_by_game.setdefault(game, set()).add(int(half_text))
    eligible = sorted(game for game, halves in halves_by_game.items()
                      if halves == {1, 2} and game not in KNOWN_VAL_GAMES)
    if len(eligible) != 48:
        raise ValueError(f"Expected all 48 complete PCBAS TRAIN games; found {len(eligible)}")
    ranked = rank_games(eligible)
    return ranked[:N_GAMES], ranked[N_GAMES:]


def _load_runtime():
    from experiments import current_baseline
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("package_version") != PACKAGE:
        raise ValueError("The active package changed; declare a new study")
    models = current_baseline.load_models(manifest_path=MANIFEST)
    return manifest, models


def _data_inventory():
    import h5py
    for path in (H5, EVENTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    with h5py.File(H5, "r") as source:
        keys = sorted(source.keys())
    selected, reserve = cohort_from_keys(keys)
    all_halves = sorted(keys)
    return selected, reserve, all_halves


def freeze():
    if DECLARATION.exists():
        raise FileExistsError(f"Frozen declaration already exists: {DECLARATION}")
    manifest, _ = _load_runtime()  # verifies every active checkpoint hash
    selected, reserve, all_halves = _data_inventory()
    data_hashes = {str(path.relative_to(ROOT)): sha(path) for path in (H5, EVENTS)}
    import h5py
    with h5py.File(H5, "r") as source:
        half_feature_hashes = {key: array_sha(source[key][:, :13])
                               for row in selected for key in (f"{row['game_id']}_H1", f"{row['game_id']}_H2")}
    source_hashes = {path: sha(ROOT / path) for path in SOURCE_FILES}
    model_hashes = dict(manifest["models_sha256"])
    model_paths = {name: sha(ROOT / manifest["model_directory"] / name) for name in model_hashes}
    if model_paths != model_hashes:
        raise ValueError("Current model files do not match active manifest hashes")
    declaration = dict(
        study=STUDY, status="frozen", created_utc=datetime.now(timezone.utc).isoformat(),
        question="Does the exact adopted package agree with acting-team labels at reliable action anchors across a deterministic 12-game PCBAS TRAIN sample?",
        interpretation="Sparse event-anchor portability only; no possession-spell or dead-ball evaluation.",
        package=PACKAGE, baseline_manifest_sha256=sha(MANIFEST), model_sha256=model_hashes,
        model_paths_sha256=model_paths, source_sha256=source_hashes, input_sha256=data_hashes,
        dataset=dict(repo_id="SoccerNet/SN-PCBAS-2026", revision="a16e373c8c8d26b1af4874f4208810baf337e709",
                     partition="TRAIN", partition_role="benchmark train; matches are held out from this project's frozen weights",
                     hdf5_inventory=all_halves, games_available=len({k.rsplit("_H", 1)[0] for k in all_halves}),
                     known_val_game_ids_excluded=sorted(KNOWN_VAL_GAMES & {k.rsplit("_H", 1)[0] for k in all_halves})),
        selected_half_feature_sha256=half_feature_hashes,
        cohort=dict(salt=SALT, selection="ascending SHA256('pcbas_expanded_anchor_v2:' + game_id) among complete H1/H2 keys; first 12 selected; remainder reserved",
                    selected=selected, reserve=reserve, selected_games=N_GAMES,
                    halves=[f"{row['game_id']}_H{half}" for row in selected for half in (1, 2)]),
        inference=dict(fps=25, ball="all missing; observation mask false", player_visibility="v1 adapt_rows proxy: finite coordinates and finite ROI columns; nonobserved positions masked",
                       target_separation="HDF5 columns [:, :13] only during predict; action class and event JSON are not read until all predictions are sealed",
                       coordinates="same standardized 105x68 adapter assumption as PCBAS v1; no tuning/clipping", fits=0,
                       model="unchanged exact adopted package and decoder", parity="direct deployed API vs v1 adapter on first 750 available frames of first selected half"),
        anchors=dict(classes=CLASSES, primary="Drive equal-game decoded team agreement; D/X count incorrect",
                     secondary="Pass, Cross and Shot per-class accuracy and predicted-D rate; per-game and pooled counts",
                     visibility="At selected anchor frames, stratify by visible players: 0-5, 6-11, 12+",
                     baseline="Per-game majority direction alias; descriptive only, computed from that game's anchors"),
        uncertainty=dict(method="paired whole-game percentile bootstrap", replicates=BOOTSTRAPS,
                         seed=BOOTSTRAP_SEED, reported=["equal-game Drive accuracy 95% interval",
                         "equal-game Drive accuracy minus per-game majority alias 95% interval"],
                         unit="game; halves and event anchors remain clustered", no_frame_bootstrap=True,
                         scope="conditional on this one PCBAS source and selected 12 games; not provider-level uncertainty"),
        exclusions=["replay events", "unselected classes", "opposing-team events on same frame", "duplicate anchors", "anchors without exact tracking frame"],
        no_training_or_selection=True, no_threshold_or_offset_search=True, no_organizer_labels=True,
        limitations=["Action anchors are sparse and have no duration; official possession spell F1 is not measurable",
                     "No dead-ball reference, so dead precision/recall is not measurable",
                     "All ball positions are missing: severe input-regime shift",
                     "Only one data source; 12-game intervals quantify match variation within this source, not unseen-provider variation",
                     "Dataset IDs are anonymized; fixture overlap cannot be exhaustively certified",
                     "Player positions include source-imputed tracks; visibility is proxied by ROI fields",
                     "Standardized 105x68 pitch scale is assumed, not measured"],
        commands=["python -m experiments.pcbas_expanded_anchor_v2 freeze",
                  "python -m experiments.pcbas_expanded_anchor_v2 parity",
                  "python -m experiments.pcbas_expanded_anchor_v2 predict --half GAME_H1",
                  "python -m experiments.pcbas_expanded_anchor_v2 seal",
                  "python -m experiments.pcbas_expanded_anchor_v2 score"],
        output_directory=str(OUT.relative_to(ROOT)))
    if len(declaration["cohort"]["halves"]) != 24:
        raise ValueError("Selected cohort does not contain 24 halves")
    write_once(DECLARATION, declaration)
    print("FROZEN", sha(DECLARATION), flush=True)


def verify_declaration(verify_external_inputs=False):
    declaration = json.loads(DECLARATION.read_text())
    for path, expected in declaration["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen source hash mismatch: {path}")
    if sha(MANIFEST) != declaration["baseline_manifest_sha256"]:
        raise ValueError("Active model manifest changed")
    manifest, _ = _load_runtime()
    if manifest["models_sha256"] != declaration["model_sha256"]:
        raise ValueError("Active checkpoint inventory changed")
    if {name: sha(ROOT / manifest["model_directory"] / name) for name in manifest["models_sha256"]} != declaration["model_paths_sha256"]:
        raise ValueError("Active checkpoint files changed")
    if verify_external_inputs:
        for path, expected in declaration["input_sha256"].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Frozen input hash mismatch: {path}")
    return declaration


def _half_keys(declaration):
    return list(declaration["cohort"]["halves"])


def _load_half_features(source, key, expected_feature_hash):
    from experiments.pcbas_event_anchor_v1 import adapt_rows
    if key not in source:
        raise ValueError(f"Missing selected half {key}")
    rows = source[key][:, :13]  # Never materialize event class / action targets.
    if array_sha(rows) != expected_feature_hash:
        raise ValueError(f"Target-free input rows differ from frozen hash: {key}")
    inputs, frames, audit = adapt_rows(rows, int(key[-1]))
    return inputs, frames, audit, array_sha(rows)


def parity():
    declaration = verify_declaration()
    if (OUT / "parity_receipt.json").exists():
        raise FileExistsError("Parity receipt already exists; preserve the frozen run")
    import h5py
    import torch
    from threadpoolctl import threadpool_limits
    from experiments import original_full_phase_baseline
    torch.set_num_threads(2)
    manifest, models = _load_runtime()
    key = _half_keys(declaration)[0]
    with h5py.File(H5, "r") as source:
        rows = source[key][:, :13]
    if array_sha(rows) != declaration["selected_half_feature_sha256"][key]:
        raise ValueError("Parity rows differ from the frozen target-free input hash")
    frames = np.unique(rows[:, 0].astype(np.int64))[:750]
    inputs, native_frames, audit = __import__("experiments.pcbas_event_anchor_v1", fromlist=["adapt_rows"]).adapt_rows(
        rows[np.isin(rows[:, 0].astype(np.int64), frames)], int(key[-1]))
    with threadpool_limits(limits=2):
        direct = original_full_phase_baseline.predict_labels(inputs, models, manifest)
        wrapped = __import__("experiments.pcbas_event_anchor_v1", fromlist=["infer"]).infer(inputs, models, manifest)
    if not np.array_equal(direct, wrapped):
        raise ValueError("Adapter output differs from deployed exact-package API")
    control = OUT / "parity_control.npz"
    def save_control(path):
        with path.open("wb") as handle:
            np.savez_compressed(handle, frames=native_frames, prediction=direct)
    atomic_exclusive(control, save_control)
    receipt = dict(declaration_sha256=sha(DECLARATION), selected_half=key, exact_equal=True,
                   control_sha256=sha(control), native_frames=len(native_frames), input_audit=audit,
                   target_events_read=False, input_action_class_read=False)
    write_once(OUT / "parity_receipt.json", receipt)
    print("PARITY PASS", len(native_frames), flush=True)


def predict(half_key: str):
    declaration = verify_declaration()
    if half_key not in _half_keys(declaration):
        raise ValueError(f"Unselected half: {half_key}")
    parity_path = OUT / "parity_receipt.json"
    if not parity_path.is_file():
        raise ValueError("Run parity before prediction")
    parity_receipt = json.loads(parity_path.read_text())
    control_path = OUT / "parity_control.npz"
    if (parity_receipt.get("declaration_sha256") != sha(DECLARATION)
            or not parity_receipt.get("exact_equal") or not control_path.is_file()
            or parity_receipt.get("control_sha256") != sha(control_path)):
        raise ValueError("Parity receipt/control does not verify")
    out_path = OUT / f"{half_key}.npz"
    receipt_path = OUT / f"{half_key}.receipt.json"
    if out_path.exists() or receipt_path.exists():
        if not (out_path.is_file() and receipt_path.is_file()):
            raise ValueError(f"Incomplete existing output for {half_key}; inspect before resuming")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("declaration_sha256") != sha(DECLARATION)
                or receipt.get("half") != half_key
                or receipt.get("output_sha256") != sha(out_path)
                or receipt.get("model_sha256") != declaration["model_sha256"]
                or receipt.get("target_free_feature_rows_sha256") != declaration["selected_half_feature_sha256"][half_key]
                or receipt.get("target_events_read") is not False
                or receipt.get("input_action_class_read") is not False):
            raise ValueError(f"Existing half receipt does not verify: {half_key}")
        print("ALREADY COMPLETE", half_key, flush=True)
        return
    import h5py
    import torch
    from threadpoolctl import threadpool_limits
    from experiments.pcbas_event_anchor_v1 import infer
    torch.set_num_threads(2)
    manifest, models = _load_runtime()
    with h5py.File(H5, "r") as source:
        inputs, frames, audit, feature_hash = _load_half_features(
            source, half_key, declaration["selected_half_feature_sha256"][half_key])
    start = time.monotonic()
    with threadpool_limits(limits=2):
        prediction = infer(inputs, models, manifest)
    if prediction.shape != frames.shape or not np.isin(prediction, ("H", "A", "D")).all():
        raise ValueError("Invalid full-half predictions")
    def save_prediction(path):
        with path.open("wb") as handle:
            np.savez_compressed(handle, frames=frames, prediction=prediction,
                                visible_players=inputs["player_observed"].sum(axis=1))
    atomic_exclusive(out_path, save_prediction)
    receipt = dict(study=STUDY, half=half_key, declaration_sha256=sha(DECLARATION),
                   input_sha256=declaration["input_sha256"][str(H5.relative_to(ROOT))],
                   target_free_feature_rows_sha256=feature_hash,
                   baseline_manifest_sha256=declaration["baseline_manifest_sha256"],
                   model_sha256=declaration["model_sha256"], output_sha256=sha(out_path),
                   frames=len(frames), input_audit=audit, seconds=time.monotonic() - start,
                   target_events_read=False, input_action_class_read=False)
    write_once(receipt_path, receipt)
    print("PREDICTED", half_key, len(frames), round(receipt["seconds"], 2), flush=True)


def seal():
    declaration = verify_declaration()
    parity_path = OUT / "parity_receipt.json"
    if not parity_path.is_file():
        raise ValueError("Missing parity receipt")
    parity_receipt = json.loads(parity_path.read_text())
    parity_control = OUT / "parity_control.npz"
    if (parity_receipt.get("declaration_sha256") != sha(DECLARATION) or not parity_receipt.get("exact_equal")
            or not parity_control.is_file() or parity_receipt.get("control_sha256") != sha(parity_control)):
        raise ValueError("Parity receipt mismatch")
    for key in _half_keys(declaration):
        if not ((OUT / f"{key}.npz").is_file() and (OUT / f"{key}.receipt.json").is_file()):
            raise ValueError(f"Prediction incomplete: {key}")
    import h5py
    receipts = {}
    with h5py.File(H5, "r") as source:
        for key in _half_keys(declaration):
            rows = source[key][:, :13]
            if array_sha(rows) != declaration["selected_half_feature_sha256"][key]:
                raise ValueError(f"Selected half input changed before seal: {key}")
            expected_frames = np.unique(rows[:, 0].astype(np.int64))
            output = OUT / f"{key}.npz"
            with np.load(output, allow_pickle=False) as arrays:
                frames, prediction, visible = arrays["frames"], arrays["prediction"], arrays["visible_players"]
            if (frames.shape != expected_frames.shape or not np.array_equal(frames, expected_frames)
                    or prediction.shape != frames.shape or visible.shape != frames.shape
                    or not np.isin(prediction, ("H", "A", "D")).all()
                    or not np.issubdtype(frames.dtype, np.integer)
                    or (len(frames) > 1 and np.any(np.diff(frames) <= 0))):
                raise ValueError(f"Prediction arrays do not match native half frames: {key}")
    for key in _half_keys(declaration):
        output, receipt_file = OUT / f"{key}.npz", OUT / f"{key}.receipt.json"
        if not (output.is_file() and receipt_file.is_file()):
            raise ValueError(f"Prediction incomplete: {key}")
        receipt = json.loads(receipt_file.read_text())
        if (receipt.get("declaration_sha256") != sha(DECLARATION)
                or receipt.get("half") != key or receipt.get("output_sha256") != sha(output)
                or receipt.get("input_sha256") != declaration["input_sha256"][str(H5.relative_to(ROOT))]
                or receipt.get("target_free_feature_rows_sha256") != declaration["selected_half_feature_sha256"][key]
                or receipt.get("model_sha256") != declaration["model_sha256"]
                or receipt.get("target_events_read") is not False
                or receipt.get("input_action_class_read") is not False):
            raise ValueError(f"Prediction receipt mismatch: {key}")
        receipts[key] = dict(receipt_path_sha256=sha(receipt_file), **receipt)
    seal_value = dict(study=STUDY, declaration_sha256=sha(DECLARATION),
                      parity_receipt_sha256=sha(parity_path), halves=receipts,
                      target_events_read=False, input_action_class_read=False)
    seal_path = OUT / "prediction_seal.json"
    write_once(seal_path, seal_value)
    print("SEALED", sha(seal_path), flush=True)


def visibility_summary(anchors, prediction, visible_players):
    prediction = np.asarray(prediction).astype(str)
    visible_players = np.asarray(visible_players, dtype=int)
    result = {}
    for lo, hi, name in VISIBILITY_BINS:
        current = [a for a in anchors if visible_players[a["index"]] >= lo and
                   (hi is None or visible_players[a["index"]] <= hi)]
        if not current:
            result[name] = dict(n=0, correct=0, accuracy=None, predicted_d_rate=None)
            continue
        correct = sum(prediction[a["index"]] == ("H" if a["team"] == 0 else "A") for a in current)
        result[name] = dict(n=len(current), correct=int(correct), accuracy=correct / len(current),
                            predicted_d_rate=sum(prediction[a["index"]] == "D" for a in current) / len(current))
    return result


def _majority_accuracy(cell):
    counts = cell["drive"]["target_direction_team_counts"]
    n = cell["drive"]["n"]
    if n <= 0:
        raise ValueError("Every selected game must have at least one eligible Drive anchor")
    return max(counts.values()) / n


def bootstrap_game_intervals(game_rows):
    """Paired, equal-game bootstrap; games are the resampling units."""
    eligible = [row for row in game_rows if row["drive_accuracy"] is not None]
    if len(eligible) != N_GAMES:
        raise ValueError(f"Expected Drive anchors in every selected game; found {len(eligible)}")
    accuracy = np.asarray([row["drive_accuracy"] for row in eligible], dtype=float)
    deltas = np.asarray([row["drive_accuracy"] - row["majority_direction_accuracy"] for row in eligible], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_idx = rng.integers(0, len(eligible), size=(BOOTSTRAPS, len(eligible)))
    acc_samples = accuracy[sample_idx].mean(axis=1)
    delta_samples = deltas[sample_idx].mean(axis=1)
    def interval(values):
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]
    return dict(method="paired whole-game percentile bootstrap", replicates=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
                equal_game_drive_accuracy=float(accuracy.mean()), equal_game_drive_accuracy_95_interval=interval(acc_samples),
                equal_game_drive_minus_majority_accuracy=float(deltas.mean()),
                equal_game_drive_minus_majority_95_interval=interval(delta_samples),
                game_count=len(eligible), no_frame_or_anchor_bootstrap=True,
                interpretation="Conditional on these selected matches from one PCBAS source; not an unseen-provider interval.")


def score():
    declaration = verify_declaration()
    seal_path = OUT / "prediction_seal.json"
    if not seal_path.is_file():
        raise ValueError("Predictions are not sealed")
    seal_value = json.loads(seal_path.read_text())
    if (seal_value.get("declaration_sha256") != sha(DECLARATION)
            or set(seal_value.get("halves", {})) != set(_half_keys(declaration))
            or seal_value.get("target_events_read") is not False
            or seal_value.get("input_action_class_read") is not False):
        raise ValueError("Prediction seal declaration mismatch")
    for key in _half_keys(declaration):
        path = OUT / f"{key}.npz"
        receipt = seal_value["halves"].get(key)
        receipt_path = OUT / f"{key}.receipt.json"
        if (not receipt or receipt.get("output_sha256") != sha(path)
                or not receipt_path.is_file()
                or receipt.get("receipt_path_sha256") != sha(receipt_path)):
            raise ValueError(f"Sealed prediction mismatch: {key}")
    score_gate = OUT / "score_gate_claimed.json"
    write_once(score_gate, dict(declaration_sha256=sha(DECLARATION),
                               prediction_seal_sha256=sha(seal_path),
                               target_input_sha256=declaration["input_sha256"][str(EVENTS.relative_to(ROOT))],
                               target_access_starts_after_predictions=True))
    if sha(EVENTS) != declaration["input_sha256"][str(EVENTS.relative_to(ROOT))]:
        raise ValueError("Frozen action annotation file changed")
    from experiments.pcbas_event_anchor_v1 import select_anchors, anchor_counts, summarize
    truth = json.loads(EVENTS.read_text())  # First event-target read, after 24 half outputs are sealed.
    selected_games = [row["game_id"] for row in declaration["cohort"]["selected"]]
    selected_halves = _half_keys(declaration)
    if not set(selected_halves) <= set(truth.get("keys", [])) or not set(selected_halves) <= set(truth.get("events", {})):
        raise ValueError("Selected annotation keys do not match the frozen cohort")
    half_counts, excluded, half_visibility = {}, {}, {}
    game_counts = {}
    for key in selected_halves:
        game_id = key.rsplit("_H", 1)[0]
        with np.load(OUT / f"{key}.npz", allow_pickle=False) as prediction_file:
            frames = prediction_file["frames"]
            prediction = prediction_file["prediction"]
            visible = prediction_file["visible_players"]
        anchors, excluded[key] = select_anchors(truth["events"][key], frames)
        half_counts[key] = anchor_counts(anchors, prediction)
        half_visibility[key] = visibility_summary(anchors, prediction, visible)
        game_counts.setdefault(game_id, []).append(half_counts[key])
    # Pool the two half counts before treating each match as one cluster.
    game_rows = []
    for game_id in selected_games:
        counts = Counter()
        per_class = {}
        for class_name in CLASSES.values():
            n = correct = 0
            pred_counts, team_counts = Counter(), Counter()
            for half_cell in game_counts[game_id]:
                item = half_cell[class_name]
                n += item["n"]
                correct += item["correct"]
                pred_counts.update(item["predicted_state_counts"])
                team_counts.update(item.get("target_direction_team_counts", {}))
            per_class[class_name] = dict(n=n, correct=correct, accuracy=correct / n if n else None,
                                         predicted_state_counts=dict(pred_counts),
                                         target_direction_team_counts=dict(team_counts),
                                         predicted_d_rate=pred_counts.get("D", 0) / n if n else None)
        drive_n = per_class["drive"]["n"]
        game_rows.append(dict(game_id=game_id, classes=per_class,
                              drive_accuracy=per_class["drive"]["accuracy"],
                              majority_direction_accuracy=_majority_accuracy({"drive": per_class["drive"]}),
                              drive_n=drive_n))
    bootstrap = bootstrap_game_intervals(game_rows)
    # The same v1 aggregate function supplies pooled/equal-game class summaries;
    # game-level and visibility detail are added explicitly below.
    summary = summarize(half_counts)
    result = dict(study=STUDY, package=PACKAGE, declaration_sha256=sha(DECLARATION),
                  prediction_seal_sha256=sha(seal_path), selected_games=selected_games,
                  reserve_games=[row["game_id"] for row in declaration["cohort"]["reserve"]],
                  summary=summary, game_results=game_rows, half_counts=half_counts,
                  visibility_by_half=half_visibility, excluded=excluded, bootstrap=bootstrap,
                  calibration="Assumed standardized105x68; physical calibration unverified",
                  ball_missing_fraction=1.0,
                  interpretation="PCBAS TRAIN sparse action-anchor portability check for the frozen exact package. No continuous H/A/D supervision, possession-spell metric, dead-ball accuracy, fitting, tuning or promotion.")
    write_once(RESULT, result)
    print(json.dumps(dict(summary=summary, bootstrap=bootstrap), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("freeze", "parity", "predict", "seal", "score"))
    parser.add_argument("--half", help="Selected half key, for predict")
    args = parser.parse_args()
    if args.step == "predict":
        if not args.half:
            parser.error("predict requires --half KEY")
        predict(args.half)
    elif args.half:
        parser.error("--half is only valid with predict")
    else:
        globals()[args.step]()


if __name__ == "__main__":
    main()
