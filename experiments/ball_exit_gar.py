"""Fixed exact-package TEST10 regression check, preserving prior GAR runners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import ball_exit_delivery as delivery, ball_exit_runtime as candidate
from experiments import current_baseline, spell_frontier_baseline as frontier
from experiments import spell_frontier_gar as historical
from experiments import spell_frontier_gar_sixphase as sixphase
from experiments import v2_iterative_runtime as fast
from experiments.ball_exit_followup import sha, write_once

ROOT = delivery.ROOT
OUT = delivery.OUT / "gar"
RECORDS = OUT / "records"
DECL = ROOT / "experiments/results/ball_exit_promotion_v1_gar_declaration.json"
VIEWS = ("clean", "v2_1", "v2_2", "v2_3", "v2_4")


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    delivery.check_manifest()  # Do not acquire/prepare a substitute package.
    parent = read(historical.prior.DECLARATION)
    tracking = historical.dense.tracking_root(historical.dense.DEFAULT_RAW)
    for row in parent["raw_files"]:
        if row["path"].startswith("test/") and sha(tracking / row["path"]) != row["sha256"]:
            raise ValueError("GAR source hash mismatch: " + row["path"])
    with fast.installed():
        historical.dense.prepare_split("test", output=RECORDS, workers=2)
    source_paths = {**delivery.sources(), **{str(Path(m.__file__).relative_to(ROOT)): sha(m.__file__)
        for m in (historical, historical.heldout, historical.dense, sixphase, fast)}}
    source_paths[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    record_files = {str(p.relative_to(ROOT)): sha(p) for p in RECORDS.rglob("*") if p.is_file()}
    write_once(DECL, {"study": delivery.STUDY, "manifest_sha256": sha(delivery.STAGED),
        "sources": source_paths, "record_files": record_files,
        "raw_files": [r for r in parent["raw_files"] if r["path"].startswith("test/")],
        "games": parent["match_ids"]["test"], "views": VIEWS,
        "baseline_result_sha256": sha(historical.RESULT),
        "calibration_sha256": sha(historical.heldout.CALIBRATION),
        "corruption": "Identical gar_deployment_eval_v2 seeds, four realizations per game",
        "selection_or_tuning": False, "repeated_test_exposure": True,
        "scope": "Ten silver-label TEST games only; VALID remains separate; no continuous-gold claim",
        "control_tolerance": 1e-12, "aggregation": "Realizations within game, then equal games"})


def check():
    manifest = delivery.check_manifest()
    declaration = read(DECL)
    if declaration["manifest_sha256"] != sha(delivery.STAGED):
        raise ValueError("GAR manifest changed")
    for rel, digest in {**declaration["sources"], **declaration["record_files"]}.items():
        if sha(ROOT / rel) != digest:
            raise ValueError("GAR source/record changed: " + rel)
    tracking = historical.dense.tracking_root(historical.dense.DEFAULT_RAW)
    for row in declaration["raw_files"]:
        if sha(tracking / row["path"]) != row["sha256"]:
            raise ValueError("GAR raw inputs changed")
    if sha(historical.RESULT) != declaration["baseline_result_sha256"] or sha(historical.heldout.CALIBRATION) != declaration["calibration_sha256"]:
        raise ValueError("GAR baseline/calibration changed")
    return manifest, declaration


def run():
    manifest, declaration = check()
    models = current_baseline.load_models(manifest_path=delivery.STAGED)
    parent = candidate.frontier_manifest(manifest)
    for role, model in models.items():
        games = {str(g).removeprefix("soccernet_") for g in model.get("training", {}).get("soccernet_games", [])}
        if games.intersection(declaration["games"]):
            raise ValueError("TEST exposure in checkpoint: " + role)
    baseline = read(historical.RESULT)["splits"]["test"]["scores"]
    calibration = read(historical.heldout.CALIBRATION)["calibration"]
    for game in declaration["games"]:
        dest = OUT / f"{game}.json"
        if dest.exists():
            saved = read(dest)
            if saved["declaration_sha256"] != sha(DECL) or sha(OUT / f"{game}.npz") != saved["prediction_sha256"]:
                raise ValueError("Saved game artifact changed")
            continue
        records = historical.heldout.load_game_records("test", game, RECORDS)
        with fast.installed():
            captured = historical.heldout.capture_inputs("test", game)
        inputs, labels, saved_x, names = historical.heldout.build_game(records, captured)
        rows, predictions = {}, {}
        for view in VIEWS:
            native = inputs if view == "clean" else historical._corrupt_view(inputs, labels, calibration, game, int(view[-1]))[0]
            with fast.installed():
                phases = historical.heldout.phase_set(native, names, float(records[0]["fps"]))
            if view == "clean":
                np.testing.assert_array_equal(phases[0]["x"], saved_x)
            with sixphase.installed():
                components = frontier.components_from_records(native, phases, models, parent)
            control = frontier.decode_components(native, components)
            control_metrics = historical._score(labels, native, control)
            expected = baseline[view]["per_game"][game]["spell_frontier"]
            for metric in historical.MATRIX_KEYS:
                if abs(control_metrics[metric] - expected[metric]) > 1e-12:
                    raise ValueError(f"GAR control replay mismatch: {game}/{view}/{metric}")
            prediction = candidate.decode_components(native, components)
            measured = historical._score(labels, native, prediction)
            rows[view] = {"control": control_metrics, "snap": measured}
            predictions[f"{view}__control"] = control.astype("S1")
            predictions[f"{view}__snap"] = prediction.astype("S1")
        with (OUT / f"{game}.npz").open("xb") as stream:
            np.savez_compressed(stream, labels=labels.astype("S1"), period=inputs["period"],
                                timestamps=inputs["timestamps"], fps=inputs["fps"], **predictions)
        write_once(dest, {"game": game, "views": rows, "declaration_sha256": sha(DECL),
                          "prediction_sha256": sha(OUT / f"{game}.npz")})
        print("Completed GAR TEST", game, flush=True)
    report()


def report():
    _, declaration = check()
    if len(declaration["games"]) != 10 or len(set(declaration["games"])) != 10:
        raise ValueError("Expected all ten TEST games")
    games = {g: read(OUT / f"{g}.json") for g in declaration["games"]}
    for game, row in games.items():
        if row["declaration_sha256"] != sha(DECL) or row["prediction_sha256"] != sha(OUT / f"{game}.npz"):
            raise ValueError("Saved GAR artifact changed")
        # Independent replay from saved native outputs before aggregating.
        with np.load(OUT / f"{game}.npz", allow_pickle=False) as saved:
            inputs = {k: saved[k] for k in ("period", "timestamps", "fps")}
            for view in VIEWS:
                for arm in ("control", "snap"):
                    metrics = historical._score(saved["labels"].astype("U1"), inputs, saved[f"{view}__{arm}"].astype("U1"))
                    for metric, value in metrics.items():
                        if abs(value - row["views"][view][arm][metric]) > 1e-12:
                            raise ValueError("Saved GAR score replay failed")
    aggregate = {}
    for view, members in (("clean", VIEWS[:1]), ("v2", VIEWS[1:])):
        aggregate[view] = {}
        for arm in ("control", "snap"):
            aggregate[view][arm] = {m: float(np.mean([np.mean([r["views"][v][arm][m] for v in members]) for r in games.values()])) for m in historical.MATRIX_KEYS}
        aggregate[view]["delta"] = {m: aggregate[view]["snap"][m] - aggregate[view]["control"][m] for m in historical.MATRIX_KEYS}
    write_once(delivery.GAR, {"study": delivery.STUDY, "manifest_sha256": sha(delivery.STAGED),
        "declaration_sha256": sha(DECL), "aggregate": aggregate, "games": games,
        "spell_regression": any(r["delta"]["spell_f1"] < 0 for r in aggregate.values()),
        "independent_prediction_replay_passed": True, "test_exposure": "Repeated descriptive check, not fresh confirmation",
        "organizer_reference_access": False, "selection_or_tuning": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    globals()[parser.parse_args().action]()
