"""Exact-package GAR TEST10 check of the confidence-gated boundary refiner.

Fixed comparison: adopted reconstructed Fable (control) versus the same package
with `lgbm_boundary_gate_runtime.predict_labels` and the pooled public refiner.
Clean tracking plus the four fixed v2 realizations of `gar_deployment_eval_v2`.
The control must replay the saved adopted GAR scores within 1e-12 per game/view
before the candidate is scored. No fitting, tuning or organizer access.

    python -m experiments.lgbm_boundary_gate_gar_v1 run --games 3850 3854
    python -m experiments.lgbm_boundary_gate_gar_v1 report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import ball_exit_runtime as adopted_runtime
from experiments import current_baseline, spell_frontier_baseline as frontier
from experiments import lgbm_boundary_gate_runtime as gate
from experiments import spell_frontier_gar as historical
from experiments import spell_frontier_gar_sixphase as sixphase
from experiments import v2_iterative_runtime as fast

ROOT = Path(__file__).resolve().parents[1]
STUDY = "lgbm_boundary_gate_gar_v1"
OUT = ROOT / "data/experiments" / STUDY
RECORDS = OUT / "records"
RES = ROOT / "experiments/results"
DECL = RES / f"{STUDY}_declaration.json"
RESULT = RES / f"{STUDY}.json"
PARENT_DECL = RES / "ball_exit_reconstructed_v1_gar_declaration.json"
PARENT_RESULT = RES / "ball_exit_reconstructed_v1_gar.json"
MANIFEST = ROOT / "experiments/baseline_manifest.json"
REFINER = ROOT / "data/experiments/lgbm_public_v1/refiner.txt"
VIEWS = ("clean", "v2_1", "v2_2", "v2_3", "v2_4")
KEYS = historical.MATRIX_KEYS
TOLERANCE = 1e-12
BOOTSTRAP = 2000


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sources():
    modules = (gate, adopted_runtime, frontier, historical, sixphase, fast, current_baseline)
    out = {str(Path(m.__file__).relative_to(ROOT)): sha(m.__file__) for m in modules}
    out[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    return out


def declare():
    parent = read(PARENT_DECL)
    tracking = historical.dense.tracking_root(historical.dense.DEFAULT_RAW)
    for row in parent["raw_files"]:
        if sha(tracking / row["path"]) != row["sha256"]:
            raise ValueError("GAR source hash mismatch: " + row["path"])
    fit = read(RES / "lgbm_public_v1_fit.json")
    if sha(REFINER) != fit["model_sha256"]:
        raise ValueError("Pooled refiner differs from the sealed public fit")
    with fast.installed():
        historical.dense.prepare_split("test", output=RECORDS, workers=2)
    records = {str(p.relative_to(ROOT)): sha(p) for p in sorted(RECORDS.rglob("*")) if p.is_file()}
    write_once(DECL, {
        "study": STUDY, "candidate": "boundary_conf0.6", "threshold": gate.THRESHOLD, "blend": gate.BLEND,
        "question": "Does the gated boundary refiner keep or improve the adopted package's GAR TEST spell F1?",
        "games": parent["games"], "views": VIEWS, "raw_files": parent["raw_files"],
        "manifest_sha256": sha(MANIFEST), "refiner_sha256": sha(REFINER),
        "calibration_sha256": sha(historical.heldout.CALIBRATION),
        "parent_result_sha256": sha(PARENT_RESULT), "record_files": records, "sources": sources(),
        "control": "Adopted reconstructed Fable; saved per-game scores must replay within tolerance",
        "control_tolerance": TOLERANCE,
        "aggregation": "Realizations within game, then equal games; paired whole-game bootstrap",
        "bootstrap_draws": BOOTSTRAP, "bootstrap_seed": 2026,
        "decision_rule": ("Good if v2 and clean spell F1 deltas are both >= 0 and macro F1 and D-F1 "
                          "deltas are both >= -0.001 in v2 and clean."),
        "prior_exposure": ("Arm chosen on five exposed development matches; one sealed public "
                           "diagnostic already run. TEST10 has repeated descriptive exposure; this is "
                           "not fresh confirmation. Refiner training used no GAR data."),
        "selection_or_tuning": False, "organizer_reference_access": False, "fitting": False})
    print("Declared", len(records), "record files", flush=True)


def check():
    declaration = read(DECL)
    if (declaration["sources"] != sources() or declaration["manifest_sha256"] != sha(MANIFEST)
            or declaration["refiner_sha256"] != sha(REFINER)
            or declaration["parent_result_sha256"] != sha(PARENT_RESULT)):
        raise ValueError("Frozen source, package, refiner or control changed")
    return declaration


def run(games):
    import lightgbm as lgb
    import torch
    torch.set_num_threads(1)
    declaration = check()
    unknown = set(games) - set(declaration["games"])
    if unknown:
        raise ValueError(f"Not TEST games: {sorted(unknown)}")
    manifest = read(MANIFEST)
    models = current_baseline.load_models(manifest_path=MANIFEST)
    for role, model in models.items():
        trained = {str(g).removeprefix("soccernet_") for g in model.get("training", {}).get("soccernet_games", [])}
        if trained.intersection(declaration["games"]):
            raise ValueError("TEST exposure in checkpoint: " + role)
    parent = adopted_runtime.frontier_manifest(manifest)
    booster = lgb.Booster(model_file=str(REFINER))
    saved = read(PARENT_RESULT)["games"]
    calibration = read(historical.heldout.CALIBRATION)["calibration"]
    for game in games:
        dest = OUT / "games" / f"{game}.json"
        if dest.exists():
            print("Already complete", game, flush=True)
            continue
        records = historical.heldout.load_game_records("test", game, RECORDS)
        with fast.installed():
            captured = historical.heldout.capture_inputs("test", game)
        inputs, labels, saved_x, names = historical.heldout.build_game(records, captured)
        rows, predictions = {}, {}
        for view in VIEWS:
            native = inputs if view == "clean" else historical._corrupt_view(
                inputs, labels, calibration, game, int(view[-1]))[0]
            with fast.installed():
                phases = historical.heldout.phase_set(native, names, float(records[0]["fps"]))
            if view == "clean":
                np.testing.assert_array_equal(phases[0]["x"], saved_x)
            with sixphase.installed():
                components = frontier.components_from_records(native, phases, models, parent)
            control = adopted_runtime.decode_components(native, components)
            control_metrics = historical._score(labels, native, control)
            expected = saved[game]["views"][view]["snap"]
            for metric in KEYS:
                if abs(control_metrics[metric] - expected[metric]) > TOLERANCE:
                    raise ValueError(f"Control replay mismatch: {game}/{view}/{metric}")
            candidate = gate.predict_labels(native, components, booster)
            rows[view] = {"control": control_metrics,
                          "candidate": historical._score(labels, native, candidate),
                          "frames_changed": float(np.mean(candidate != control))}
            predictions[f"{view}__control"] = control.astype("S1")
            predictions[f"{view}__candidate"] = candidate.astype("S1")
            print("Scored", game, view, flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (OUT / "games" / f"{game}.npz").open("xb") as stream:
            np.savez_compressed(stream, labels=labels.astype("S1"), period=inputs["period"],
                                timestamps=inputs["timestamps"], fps=inputs["fps"], **predictions)
        write_once(dest, {"game": game, "views": rows, "declaration_sha256": sha(DECL),
                          "prediction_sha256": sha(OUT / "games" / f"{game}.npz")})
        print("Completed GAR TEST", game, flush=True)


def report():
    declaration = check()
    games = {g: read(OUT / "games" / f"{g}.json") for g in declaration["games"]}
    if len(games) != 10:
        raise ValueError("Expected all ten TEST games")
    for game, row in games.items():
        path = OUT / "games" / f"{game}.npz"
        if row["declaration_sha256"] != sha(DECL) or row["prediction_sha256"] != sha(path):
            raise ValueError("Saved game artifact changed: " + game)
        with np.load(path, allow_pickle=False) as z:
            inputs = {k: z[k] for k in ("period", "timestamps", "fps")}
            for view in VIEWS:
                for arm in ("control", "candidate"):
                    replay = historical._score(z["labels"].astype("U1"), inputs, z[f"{view}__{arm}"].astype("U1"))
                    for metric, value in replay.items():
                        if abs(value - row["views"][view][arm][metric]) > TOLERANCE:
                            raise ValueError("Saved score replay failed")
    order = list(declaration["games"])
    rng = np.random.default_rng(declaration["bootstrap_seed"])
    draws = rng.integers(0, len(order), size=(declaration["bootstrap_draws"], len(order)))
    aggregate = {}
    for name, members in (("clean", VIEWS[:1]), ("v2", VIEWS[1:])):
        aggregate[name] = {}
        per_game = {arm: {m: np.array([np.mean([games[g]["views"][v][arm][m] for v in members]) for g in order])
                          for m in KEYS} for arm in ("control", "candidate")}
        for arm in per_game:
            aggregate[name][arm] = {m: float(per_game[arm][m].mean()) for m in KEYS}
        delta = {m: per_game["candidate"][m] - per_game["control"][m] for m in KEYS}
        aggregate[name]["delta"] = {m: float(delta[m].mean()) for m in KEYS}
        aggregate[name]["delta_ci95"] = {m: [float(x) for x in np.percentile(delta[m][draws].mean(axis=1), [2.5, 97.5])]
                                         for m in ("spell_f1", "macro_f1", "d_f1", "short_spell_recall_iou_0_5_lt3s")}
        aggregate[name]["games_spell_up_flat_down"] = [int((delta["spell_f1"] > 0).sum()),
            int((delta["spell_f1"] == 0).sum()), int((delta["spell_f1"] < 0).sum())]
        aggregate[name]["per_game_spell_delta"] = {g: float(x) for g, x in zip(order, delta["spell_f1"])}
        aggregate[name]["frames_changed"] = float(np.mean([np.mean([games[g]["views"][v]["frames_changed"]
                                                                    for v in members]) for g in order]))
    good = all(aggregate[n]["delta"]["spell_f1"] >= 0 and aggregate[n]["delta"]["macro_f1"] >= -0.001
               and aggregate[n]["delta"]["d_f1"] >= -0.001 for n in ("clean", "v2"))
    write_once(RESULT, {"study": STUDY, "declaration_sha256": sha(DECL), "aggregate": aggregate,
        "decision_rule_passed": bool(good), "games": games, "independent_prediction_replay_passed": True,
        "test_exposure": "Repeated descriptive check, not fresh confirmation",
        "organizer_reference_access": False, "selection_or_tuning": False})
    print(json.dumps({k: {a: v for a, v in aggregate[k].items() if a != "per_game_spell_delta"}
                      for k in aggregate}, indent=1))
    print("decision_rule_passed", good)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run", "report"))
    parser.add_argument("--games", nargs="*", default=[])
    args = parser.parse_args()
    if args.action == "run":
        run(args.games)
    else:
        globals()[args.action]()
