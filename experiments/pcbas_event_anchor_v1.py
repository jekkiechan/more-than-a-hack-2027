"""Frozen-package portability diagnostic at FOOTPASS VAL touch anchors only.

No fitting, synthetic possession intervals, ball reconstruction or organizer scoring.
Coordinates use a declared standardized 105x68 pitch, not measured pitch dimensions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = "pcbas_event_anchor_v1"
DATA = ROOT / "data/independent/sn_pcbas_2026"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / "experiments/results" / f"{STUDY}_declaration.json"
RESULT = ROOT / "experiments/results" / f"{STUDY}_result.json"
MANIFEST = ROOT / "experiments/baseline_manifest.json"
HALVES = tuple(f"game_{game}_H{half}" for game in (18, 24, 47) for half in (1, 2))
CLASSES = {1: "drive", 2: "pass", 3: "cross", 5: "shot"}
PACKAGE = "joint_gru_sn_dense_original_diversity_phase_ensemble_v1"
PITCH = np.array([105., 68.])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def adapt_rows(rows, half):
    """Accept only 13 non-target columns; keep native times and ROI visibility."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 13 or half not in (1, 2):
        raise ValueError("Expected 13 feature columns and an explicit half")
    for column in (0, 1, 2):
        if not np.isfinite(rows[:, column]).all() or not np.equal(rows[:, column], np.floor(rows[:, column])).all():
            raise ValueError("Noninteger frame/player/team identity")
    frames, fi = np.unique(rows[:, 0].astype(np.int64), return_inverse=True)
    ids, pi = np.unique(rows[:, 1].astype(np.int64), return_inverse=True)
    if not len(frames) or not np.isin(rows[:, 2], (0, 1)).all():
        raise ValueError("Empty data or unknown direction team")
    if len(np.unique(fi * len(ids) + pi)) != len(rows):
        raise ValueError("Duplicate frame/player rows")
    teams = np.empty(len(ids), dtype=np.int8)
    for slot in range(len(ids)):
        values = np.unique(rows[pi == slot, 2])
        if len(values) != 1:
            raise ValueError("A player changes direction team within a half")
        teams[slot] = values[0]
    if set(teams) != {0, 1}:
        raise ValueError("Both direction teams are required")
    observed = np.isfinite(rows[:, 5:7]).all(axis=1) & np.isfinite(rows[:, 9:13]).all(axis=1)
    players = np.full((len(frames), len(ids), 2), np.nan, dtype=np.float64)
    mask = np.zeros((len(frames), len(ids)), dtype=bool)
    players[fi[observed], pi[observed]] = rows[observed, 5:7] * PITCH
    mask[fi[observed], pi[observed]] = True
    inputs = dict(ball=np.full((len(frames), 2), np.nan), players=players,
                  teams=teams, period=np.full(len(frames), half, dtype=np.int8),
                  fps=np.array(25.), timestamps=frames / 25., pitch=PITCH.copy(),
                  ball_observed=np.zeros(len(frames), dtype=bool), player_observed=mask)
    audit = dict(frames=len(frames), first_frame=int(frames[0]), last_frame=int(frames[-1]),
                 frame_step_counts={str(k): int(v) for k, v in Counter(np.diff(frames)).items()},
                 source_rows=len(rows), unique_player_slots=len(ids),
                 visible_player_rows=int(observed.sum()), source_unobserved_or_imputed_rows=int((~observed).sum()),
                 visible_player_frame_mean=float(mask.sum(axis=1).mean()),
                 ball_observed_frames=0, coordinate_calibration="standardized_105x68_assumption",
                 no_coordinate_clipping=True)
    return inputs, frames, audit


def select_anchors(events, frames):
    """Exact native-frame anchors, without replay, background or ambiguous teams."""
    frames = np.asarray(frames, dtype=np.int64)
    if len(frames) and np.any(np.diff(frames) <= 0):
        raise ValueError("Frame mapping must be strictly increasing")
    rows = np.asarray(events)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("Expected frame/team/shirt/class/bbox/replay events")
    if not np.isfinite(rows).all() or not np.equal(rows, np.floor(rows)).all():
        raise ValueError("Noninteger event fields")
    rows = rows.astype(np.int64)
    if not np.isin(rows[:, 1], (0, 1)).all() or not np.isin(rows[:, 5], (0, 1)).all():
        raise ValueError("Unknown event team/replay code")
    team_by_frame = {}
    for frame, team, _, cls, _, replay in rows:
        if not replay and 1 <= cls <= 8:
            team_by_frame.setdefault(int(frame), set()).add(int(team))
    seen = set()
    selected, excluded = [], Counter()
    for frame, team, shirt, cls, bbox, replay in rows:
        if int(cls) not in CLASSES:
            excluded["nonselected_class"] += 1
            continue
        if replay:
            excluded["replay"] += 1
            continue
        if len(team_by_frame[int(frame)]) != 1:
            excluded["opposing_team_same_frame"] += 1
            continue
        token = (int(frame), int(team), int(shirt), int(cls))
        if token in seen:
            excluded["exact_duplicate"] += 1
            continue
        seen.add(token)
        index = int(np.searchsorted(frames, frame))
        if index == len(frames) or frames[index] != frame:
            excluded["no_exact_tracking_frame"] += 1
            continue
        selected.append(dict(frame=int(frame), index=index, team=int(team),
                             shirt=int(shirt), class_id=int(cls), bbox_flag=int(bbox)))
    return selected, dict(excluded)


def anchor_counts(anchors, prediction):
    prediction = np.asarray(prediction).astype(str)
    if not np.isin(prediction, ("H", "A", "D", "X")).all():
        raise ValueError("Unknown predicted state")
    out = {}
    for class_id, name in CLASSES.items():
        selected = [row for row in anchors if row["class_id"] == class_id]
        states = [str(prediction[row["index"]]) for row in selected]
        correct = sum(state == ("H" if row["team"] == 0 else "A") for row, state in zip(selected, states))
        out[name] = dict(n=len(selected), correct=int(correct), predicted_state_counts=dict(Counter(states)),
                         target_direction_team_counts=dict(Counter(str(row['team']) for row in selected)),
                         actor_bbox_absent=sum(row["bbox_flag"] == 0 for row in selected))
    return out


def summarize(cells):
    result = {}
    for name in CLASSES.values():
        game_counts = {}
        for key, cell in cells.items():
            game = key.rsplit("_H", 1)[0]
            total = game_counts.setdefault(game, dict(n=0, correct=0, predicted_state_counts={}, target_direction_team_counts={}))
            total["n"] += cell[name]["n"]
            total["correct"] += cell[name]["correct"]
            for state, n in cell[name]["predicted_state_counts"].items():
                total["predicted_state_counts"][state] = total["predicted_state_counts"].get(state, 0) + n
            for team, n in cell[name].get("target_direction_team_counts", {}).items():
                total["target_direction_team_counts"][team] = total["target_direction_team_counts"].get(team, 0) + n
        for row in game_counts.values():
            row["team_agreement"] = row["correct"] / row["n"] if row["n"] else None
            row["predicted_dead_at_live_anchor_rate"] = row["predicted_state_counts"].get("D", 0) / row["n"] if row["n"] else None
            row["majority_direction_alias_fraction"] = max(row["target_direction_team_counts"].values()) / row["n"] if row["target_direction_team_counts"] else None
        n = sum(v["n"] for v in game_counts.values())
        correct = sum(v["correct"] for v in game_counts.values())
        result[name] = dict(n=n, correct=correct, pooled_team_agreement=correct / n if n else None,
                            equal_game_team_agreement=float(np.mean([v["team_agreement"] for v in game_counts.values() if v["n"]])) if n else None,
                            by_game=game_counts)
    return result


def infer(inputs, models, manifest):
    from experiments import original_full_phase_baseline, tracking_corruption
    clean = tracking_corruption.inputs_only(inputs)
    if np.isfinite(clean["ball"]).any() or clean["ball_observed"].any():
        raise ValueError("This study requires completely missing ball coordinates")
    return original_full_phase_baseline.predict_labels(clean, models, manifest)


def freeze():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["package_version"] != PACKAGE:
        raise ValueError("Adopted package changed")
    paths = [DATA / "val_tactical_data.h5", DATA / "playbyplay_val.json", DATA / "structural_audit.json",
             DATA / "tactical_data_format.txt"]
    source_paths = sorted((ROOT / "experiments").glob("*.py"))
    source_paths.append(ROOT / "experiments/tests/test_pcbas_event_anchor_v1.py")
    declaration = dict(study=STUDY, status="frozen", created_utc=datetime.now(timezone.utc).isoformat(),
        question="Does the exact adopted package agree with the acting team at reliable live control anchors when the ball is wholly unavailable?",
        hypothesis="Decoded Drive anchor team agreement exceeds the two-team uninformative 0.5 reference; descriptive portability diagnostic only, no promotion gate.",
        package=PACKAGE, baseline_manifest_sha256=sha(MANIFEST), models_sha256=manifest["models_sha256"],
        arms=["adopted_package_native_player_visibility_ball_all_missing"], fits=0,
        halves=list(HALVES), fps=25, seeds="unchanged 2026/2027 ensemble", cpu_threads=2,
        calibration=dict(pitch=[105, 68], status="inferred standardized pitch; NOT verified physical calibration",
            approval="Root pre-score implementation decision explicitly accepted standardized105x68, no sweep", transform="x*105,y*68; no clipping; no supplied speed fields"),
        team_mapping="Independent half-local aliases: source left_to_right0->H,1->A; GT team uses same direction codes. Aliases do not mean native home/away; no linking halves.",
        visibility="ROI x,y,width,height all finite AND finite x,y => observed; otherwise player coordinates NaN. Offscreen role-imputed positions excluded. Ball NaN/false throughout.",
        time="Absolute zero-based video frame/25, exact joins only. Separate halves; native missing frames retained as time breaks, never filled.",
        feature_exclusions=["action class", "event actor", "event team", "GT bbox/replay", "role", "source velocities"],
        anchor_classes=CLASSES, primary="drive equal-game decoded team agreement; D/X count incorrect",
        secondary="Separate pass/cross/shot agreement and predicted D rates at live action moments only; raw counts and per-game values",
        exclusions=["replay==1", "nonselected classes", "opposing teams at any nonreplay onball event on same frame", "exact duplicate anchors", "no exact tracking frame"],
        aggregation="Sum halves within game, equal mean across3 games; pooled numerators/denominators also reported. No frame bootstrap or interval confidence claims.",
        class_balance_reference="Per-game target direction-team counts and majority-direction-alias fraction, descriptive only. 0.5 is uniform guessing, not a majority-team baseline or adoption threshold.",
        parity_control="First750 native frames of game18H1, no events/class read. Save direct original_full_phase_baseline labels, replay via study wrapper, require exact equality; no scores.",
        no_sweeps=True, no_time_tolerance=True, no_spell_metrics=True, no_dead_gold=True,
        no_training_or_selection=True, no_organizer_access=True,
        exposure="New provider release, no known exact-package training use; anonymous matches cannot prove disjoint fixture identity. VAL used once for this diagnostic, not a fresh full-pipeline provider-holdout claim.",
        stop_conditions=["input/source/model hash mismatch", "unknown teams or noninteger frames", "duplicate frame/player rows", "player team changes inside half", "nonfinite inference or wrong output length"],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        limitations=["Sparse manually annotated touch moments, no possession durations", "No negative/dead-ball reference; cannot estimate D precision/recall", "100% ball missing is severe distribution shift", "Standardized pitch scale is assumed", "Most positions are imputed; ROI mask proxies observation", "Anonymous fixture identity prevents exhaustive overlap audit", "No comparison to original protocol scores or model adoption"],
        tests="python -m unittest experiments.tests.test_pcbas_event_anchor_v1 -v",
        commands=["python -m experiments.pcbas_event_anchor_v1 freeze", "python -m experiments.pcbas_event_anchor_v1 parity", "python -m experiments.pcbas_event_anchor_v1 predict", "python -m experiments.pcbas_event_anchor_v1 score"],
        output_directory=str(OUT.relative_to(ROOT)))
    write_once(DECLARATION, declaration)
    print("FROZEN", sha(DECLARATION), flush=True)


def verify_declaration():
    declaration = json.loads(DECLARATION.read_text())
    for group in ("input_sha256", "source_sha256"):
        for path, expected in declaration[group].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Frozen {group} mismatch: {path}")
    if sha(MANIFEST) != declaration["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest mismatch")
    return declaration


def predict():
    declaration = verify_declaration()
    parity_receipt = json.loads((OUT / "parity_control.json").read_text())
    if parity_receipt['declaration_sha256'] != sha(DECLARATION) or not parity_receipt['exact_equal']:
        raise ValueError('Missing matching exact-package parity control')
    import h5py
    import torch
    from threadpoolctl import threadpool_limits
    from experiments import current_baseline
    torch.set_num_threads(2)
    manifest = json.loads(MANIFEST.read_text())
    models = current_baseline.load_models(manifest_path=MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(OUT / "prediction_gate_claimed.json", dict(declaration_sha256=sha(DECLARATION), target_events_not_read=True))
    receipts = {}
    with threadpool_limits(limits=2), h5py.File(DATA / "val_tactical_data.h5", "r") as source:
        if set(source.keys()) != set(HALVES):
            raise ValueError("Incomplete cohort")
        for key in HALVES:
            start = time.monotonic()
            # Explicitly exclude source action column13 at HDF5 read time.
            inputs, frames, audit = adapt_rows(source[key][:, :13], int(key[-1]))
            prediction = infer(inputs, models, manifest)
            if prediction.shape != frames.shape or not np.isin(prediction, ("H", "A", "D")).all():
                raise ValueError("Invalid native prediction")
            target = OUT / f"{key}.npz"
            with target.open("xb") as handle:
                np.savez_compressed(handle, frames=frames, prediction=prediction,
                                    visible_players=inputs["player_observed"].sum(axis=1))
            receipts[key] = dict(output_sha256=sha(target), seconds=time.monotonic()-start, input_audit=audit)
            print("PREDICTED", key, len(frames), round(receipts[key]["seconds"], 2), flush=True)
    write_once(OUT / "prediction_seal.json", dict(declaration_sha256=sha(DECLARATION), receipts=receipts,
        model_sha256=declaration["models_sha256"], target_events_read=False, input_action_class_read=False))


def parity():
    verify_declaration()
    import h5py
    import torch
    from threadpoolctl import threadpool_limits
    from experiments import current_baseline, original_full_phase_baseline
    torch.set_num_threads(2)
    manifest = json.loads(MANIFEST.read_text())
    models = current_baseline.load_models(manifest_path=MANIFEST)
    with h5py.File(DATA / 'val_tactical_data.h5', 'r') as source:
        data = source[HALVES[0]][:, :13]
    frames = np.unique(data[:, 0])[:750]
    inputs, native, audit = adapt_rows(data[np.isin(data[:, 0], frames)], 1)
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        control = original_full_phase_baseline.predict_labels(inputs, models, manifest)
        control_path = OUT / 'direct_api_parity_control.npz'
        with control_path.open('xb') as handle:
            np.savez_compressed(handle, frames=native, prediction=control)
        measured = infer(inputs, models, manifest)
    with np.load(control_path, allow_pickle=False) as saved:
        equal = bool(np.array_equal(saved['prediction'], measured))
    receipt = dict(declaration_sha256=sha(DECLARATION), exact_equal=equal,
                   control_sha256=sha(control_path), native_frames=len(native), input_audit=audit,
                   target_events_read=False, input_action_class_read=False)
    write_once(OUT / 'parity_control.json', receipt)
    if not equal:
        raise ValueError('Wrapper differs from saved direct exact-package control')
    print('PARITY PASS', len(native), flush=True)


def score():
    verify_declaration()
    seal = json.loads((OUT / "prediction_seal.json").read_text())
    if seal["declaration_sha256"] != sha(DECLARATION):
        raise ValueError("Prediction declaration mismatch")
    for key in HALVES:
        if sha(OUT / f"{key}.npz") != seal["receipts"][key]["output_sha256"]:
            raise ValueError("Prediction seal mismatch")
    write_once(OUT / "anchor_score_gate_claimed.json", dict(prediction_seal_sha256=sha(OUT / "prediction_seal.json")))
    truth = json.loads((DATA / "playbyplay_val.json").read_text())
    if set(truth["keys"]) != set(HALVES) or set(truth["events"]) != set(HALVES):
        raise ValueError("Incomplete event cohort")
    cells, excluded = {}, {}
    for key in HALVES:
        with np.load(OUT / f"{key}.npz", allow_pickle=False) as source:
            anchors, excluded[key] = select_anchors(truth["events"][key], source["frames"])
            cells[key] = anchor_counts(anchors, source["prediction"])
    result = dict(study=STUDY, package=PACKAGE, declaration_sha256=sha(DECLARATION),
                  prediction_seal_sha256=sha(OUT / "prediction_seal.json"),
                  summary=summarize(cells), half_counts=cells, excluded=excluded,
                  calibration="Assumed standardized105x68; physical calibration unverified", ball_missing_fraction=1.,
                  interpretation="Sparse action-anchor portability diagnostic; no continuous gold, spellF1, dead-ball accuracy, fitting, tuning or promotion.")
    write_once(RESULT, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("freeze", "parity", "predict", "score"))
    globals()[parser.parse_args().step]()
