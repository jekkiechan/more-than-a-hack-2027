"""Matched full-pipeline evaluation for the frozen data/capacity study.

This evaluator consumes the model fits declared by ``v2_data_capacity`` and
rebuilds each exposed development view from the archived native inputs. It
keeps controls-only replay separate from candidate scoring so an archived
control mismatch blocks the run before candidate results are written.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_like_v2 as v2
from experiments import preparation as prep
from experiments import sn_dense_train as sn
from experiments import soccernet_extension as old
from experiments import v2_data_capacity as training
from experiments import v2_spell_target_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_data_capacity_eval"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
REPLAY = ROOT / f"experiments/results/{STUDY}_control_replay.json"
SEAL = ROOT / f"data/experiments/{STUDY}/evaluation_seal.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
SEEDS = tuple(training.SEEDS)
FOLDS = tuple(training.FOLDS)
RECIPES = tuple(training.ARMS)
VIEWS = (("clean", (0,)), ("feed_like_v1", (1, 2)),
         ("feed_like_v2", (1, 2, 3, 4)))
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
           "predicted_spell_count", "reference_spell_count",
           "short_spell_recall_iou_0_5_lt3s")
def _decoder(ha):
    return dict(audit.CONFIG, dt=4, dead=3, ha=ha, ht=1.0)


DECODERS = {
    "pr82_blend_ha1_5": _decoder(1.5),
    "adopted_blend_ha3": _decoder(3.0),
    "standalone_ha3": _decoder(3.0),
    "adopted_dead_new_ha3": _decoder(3.0),
}
CONTROL_ARMS = ("adopted_dead_original_ha3",)
ARMS = tuple(f"{recipe}/{arm}" for recipe in RECIPES for arm in DECODERS)
PR82_V2_FLOOR = .5927491913246999
HISTORICAL_MAIN = ROOT / "experiments/results/v2_weight_edit_development_v1.json"
HISTORICAL_V1 = ROOT / "experiments/results/v2_weight_edit_development_v1_v1.json"
_FIT_MODEL_CACHE = {}
_TEACHER_CACHE = {}


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def equal_provider(values: dict[tuple[str, str, int], float]) -> tuple[float, dict[str, float]]:
    """Average replicas within match, matches within provider, providers equally."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for (provider, match, _rep), value in values.items():
        grouped.setdefault(provider, {}).setdefault(match, []).append(float(value))
    if not grouped or any(not matches for matches in grouped.values()):
        raise ValueError("Cannot aggregate an empty or malformed cohort")
    by_provider = {provider: float(np.mean([np.mean(reps) for reps in matches.values()]))
                   for provider, matches in grouped.items()}
    return float(np.mean(list(by_provider.values()))), by_provider


def hard_gate(candidate: dict, adopted: dict,
              pr82_v2_spell: float = .5927491913246999) -> dict:
    """Compute predeclared no-regression thresholds with no tolerance."""
    checks = {}
    for view in ("clean", "feed_like_v1", "feed_like_v2"):
        for metric in ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                       "short_spell_recall_iou_0_5_lt3s"):
            actual = float(candidate[view][metric])
            baseline = float(adopted[view][metric])
            checks[f"{view}/{metric}_vs_adopted"] = {
                "actual": actual, "minimum": baseline, "passed": actual >= baseline}
    value = float(candidate["feed_like_v2"]["spell_f1"])
    checks["feed_like_v2/spell_f1_vs_pr82"] = {
        "actual": value, "minimum": pr82_v2_spell, "passed": value >= pr82_v2_spell}
    return {"checks": checks, "all_passed": all(item["passed"] for item in checks.values())}


def _source_hashes() -> dict[str, str]:
    paths = (Path(__file__), Path(training.__file__), Path(runtime.frozen.__file__),
             Path(runtime.frozen.audit.__file__), Path(runtime.frozen.v2.__file__),
             Path(runtime.frozen.prep.__file__), Path(runtime.frozen.tc.__file__),
             Path(runtime.frozen.audit.decoder.__file__), Path(runtime.frozen.sn.__file__),
             Path(runtime.frozen.old.__file__), Path(runtime.frozen.supervised.__file__),
             Path(runtime.frozen.audit.metrics.__file__),
             Path(training.recipe.rf.sn.cap.__file__), Path(training.recipe.rf.sn.fh.__file__),
             Path(training.recipe.rf.sn.lt.__file__),
             Path(ROOT / "evaluation/possession_metrics.py"))
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
            for path in paths}


def _raw_cache(provider: str, match: str, view: str, rep: int):
    """Load archived native inputs, never a prepared feature record."""
    runtime.frozen._roots()
    filename = audit.FILE[match]
    if view == "clean":
        path = audit.EVAL_INPUTS / f"{filename}_clean_clean.joblib"
    elif view == "feed_like_v1":
        path = audit.EVAL_INPUTS / f"{filename}_feed_like_corrupt_{rep}.joblib"
    else:
        path = audit.EVAL_INPUTS / f"{filename}_clean_clean.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Verified archived raw development input is missing: {path}")
    cached = joblib.load(path)
    labels = np.asarray(cached["labels"]).astype(str)
    native = runtime.frozen.tc.inputs_only(cached["inputs"])
    if view == "feed_like_v2":
        calibration = json.loads(runtime.frozen.CALIBRATION.read_text())["calibration"]
        seed = int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
        native = v2.corrupt(native, labels, calibration, seed)
    # Every view recomputes features after its raw tracking state is fixed.
    record = prep.build_record(native, labels, provider, filename)
    phases = audit.phase_records(native, record)
    period = runtime.frozen.supervised.model_periods(native)
    return path, native, labels, phases, period


def _views():
    return [(p, m, name, int(rep)) for p, m, _ in runtime.frozen.audit.MATCHES
            for name, reps in VIEWS for rep in reps]


def _model_hashes():
    """Bind both archived control families, fold teachers, and new fits."""
    facts = {}
    frozen = runtime.frozen
    for provider in frozen.audit.FOLD:
        for seed in SEEDS:
            for family in ("adop", "cand"):
                path = frozen._model_path(family, provider, seed)
                facts[f"{family}/{provider}/{seed}"] = {"path": str(path), "sha256": sha(path)}
            path = frozen._teacher_path(provider, seed)
            facts[f"teacher/{provider}/{seed}"] = {"path": str(path), "sha256": sha(path)}
            train_fold = f"leave_out_{frozen.audit.FOLD[provider]}"
            train_path = training.recipe.rf.sc.initial_path(train_fold, seed)
            facts[f"capacity_teacher/{provider}/{seed}"] = {
                "path": str(train_path), "sha256": sha(train_path)}
    return facts


def _cell_cache_path(provider, match, view, rep, suffix="control") -> Path:
    safe_view = "clean" if view == "clean" else f"{view}_r{rep}"
    return OUT / "emissions" / f"{provider}__{match}__{safe_view}__{suffix}.npz"


def _candidate_cache_path(provider, match, view, rep, recipe) -> Path:
    safe_view = "clean" if view == "clean" else f"{view}_r{rep}"
    return OUT / "emissions" / f"{provider}__{match}__{safe_view}__{recipe}.npz"


def _save_cell_arrays(path: Path, row: dict, identity: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in row.items()
              if key in {"labels", "timestamps", "period", "model_period", "fps"}}
    for name, value in row["probabilities"].items():
        arrays[f"prob__{name}"] = np.asarray(value)
    for name, value in identity.items():
        arrays[f"identity__{name}"] = np.asarray(str(value))
    np.savez_compressed(path, **arrays)
    return sha(path)


def _load_cell_arrays(path: Path, expected_identity: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        actual = {key.removeprefix("identity__"): str(saved[key].item())
                  for key in saved.files if key.startswith("identity__")}
        expected = {key: str(value) for key, value in expected_identity.items()}
        if actual != expected:
            raise RuntimeError(f"Saved emission identity changed: {path}")
        return {key.removeprefix("prob__"): saved[key].copy()
                for key in saved.files if key.startswith("prob__")}


def _predict_new(view: dict, record: dict, provider: str, arm: str, seed: int) -> np.ndarray:
    fold = f"leave_out_{runtime.frozen.audit.FOLD[provider]}"
    model_key = (arm, fold, seed)
    if model_key not in _FIT_MODEL_CACHE:
        _FIT_MODEL_CACHE[model_key] = training.load_fit(arm, fold, seed)
    model = _FIT_MODEL_CACHE[model_key]
    if (fold, seed) not in _TEACHER_CACHE:
        _TEACHER_CACHE[(fold, seed)] = joblib.load(training.recipe.rf.sc.initial_path(fold, seed))
    teacher = _TEACHER_CACHE[(fold, seed)]
    record = old.inference_record(record)
    record["names"] = list(record["names"])
    cap = training.recipe.rf.sn.cap
    original_network = cap.network
    cap.network = training._network
    try:
        probs = sn.predict(record, model, teacher)
    finally:
        cap.network = original_network
    return old.upsample(probs, record, runtime.frozen.tc.inputs_only(view))


def _score_probability(labels, view, period, probability, config):
    cell = {"labels": labels, "period": view["period"],
            "timestamps": view["timestamps"], "fps": float(view["fps"])}
    pred = audit.decode(probability, period, cell["fps"], config)
    return audit.score(cell, pred)


def _score_prediction(labels, view, prediction):
    cell = {"labels": labels, "period": view["period"],
            "timestamps": view["timestamps"], "fps": float(view["fps"])}
    return audit.score(cell, prediction)


def _adopted_dead_prediction(probability, adopted_probability, period, fps, config):
    """Keep the adopted D mask and decode H/A only inside its live spans."""
    adopted = audit.decode(adopted_probability, period, fps, audit.CONFIG)
    live = adopted != "D"
    cond = np.asarray(probability, dtype=np.float64).copy()
    ha = cond[:, :2].sum(axis=1)
    if (ha <= 0).any():
        raise ValueError("Conditional H/A probabilities require positive H/A mass")
    cond[:, :2] /= ha[:, None]
    cond[:, 2] = 0.0
    result = adopted.copy()
    for a, b in runtime.frozen.supervised.blocks(period):
        i = a
        while i < b:
            if not live[i]:
                i += 1
                continue
            j = i + 1
            while j < b and live[j]:
                j += 1
            result[i:j] = audit.decode(cond[i:j], period[i:j], fps, config)
            i = j
    if not np.array_equal(result == "D", adopted == "D"):
        raise RuntimeError("Adopted dead-state mask changed during conditional H/A decode")
    return result


def _score_cell(provider, match, view_name, rep, model_arm=None, control_cache=None):
    path, view, labels, phases, period = _raw_cache(provider, match, view_name, rep)
    if len(phases) not in (2, 5):
        raise ValueError(f"Unexpected native phase count: {len(phases)}")
    def pooled(entries):
        return np.mean([np.mean([p for ph, _, p in entries if ph == phase], axis=0)
                        for phase in range(len(phases))], axis=0)
    per_family = {"adop": [], "cand": []}
    if control_cache is None:
        for phase, record in enumerate(phases):
            for seed in SEEDS:
                per_family["adop"].append((phase, seed, runtime.frozen.predict_seed(view, record, "adop", provider, seed)))
                per_family["cand"].append((phase, seed, runtime.frozen.predict_seed(view, record, "cand", provider, seed)))
        pooled_controls = {family: pooled(entries) for family, entries in per_family.items()}
    else:
        per_family = {family: [(phase, seed, control_cache[f"{family}_p{phase}_s{seed}"])
                               for phase in range(len(phases)) for seed in SEEDS]
                      for family in ("adop", "cand")}
        pooled_controls = {family: pooled(entries) for family, entries in per_family.items()}
    candidate = {name: [] for name in DECODERS} if model_arm is not None else None
    if model_arm is not None:
        for phase, record in enumerate(phases):
            for seed in SEEDS:
                new_prob = _predict_new(view, record, provider, model_arm, seed)
                calibrated = next(p for ph, sd, p in per_family["adop"] if ph == phase and sd == seed)
                adopted = next(p for ph, sd, p in per_family["cand"] if ph == phase and sd == seed)
                candidate["standalone_ha3"].append((phase, seed, new_prob))
                candidate["pr82_blend_ha1_5"].append((phase, seed, .5 * (new_prob + calibrated)))
                candidate["adopted_blend_ha3"].append((phase, seed, .5 * (new_prob + adopted)))
                candidate["adopted_dead_new_ha3"].append((phase, seed, new_prob))
    score = {}
    for family in ("adop", "cand"):
        score[family] = _score_probability(labels, view, period, pooled_controls[family], audit.CONFIG)
    control_prediction = _adopted_dead_prediction(pooled_controls["cand"], pooled_controls["cand"],
                                                  period, float(view["fps"]), _decoder(3.0))
    score["adopted_dead_original_ha3"] = _score_prediction(labels, view, control_prediction)
    predictions = {"adop": audit.decode(pooled_controls["adop"], period, float(view["fps"]), audit.CONFIG),
                   "cand": audit.decode(pooled_controls["cand"], period, float(view["fps"]), audit.CONFIG),
                   "adopted_dead_original_ha3": control_prediction}
    if candidate is not None:
        standalone_entries = candidate["standalone_ha3"]
        component_scores = {}
        for seed in SEEDS:
            seed_phases = []
            for phase in range(len(phases)):
                members = [prob for phase_i, seed_i, prob in standalone_entries
                           if phase_i == phase and seed_i == seed]
                if len(members) != 1:
                    raise RuntimeError(f"Missing standalone seed/phase emission {seed}/{phase}")
                seed_phases.append(members[0])
            seed_probability = np.mean(seed_phases, axis=0)
            seed_metrics = _score_probability(labels, view, period, seed_probability,
                                              DECODERS["standalone_ha3"])
            component_scores[str(seed)] = {metric: seed_metrics[metric]
                                           for metric in ("spell_f1", "macro_f1", "d_f1")}
        for arm, entries in candidate.items():
            probs = pooled(entries)
            if arm == "adopted_dead_new_ha3":
                pred = _adopted_dead_prediction(probs, pooled_controls["cand"], period,
                                                float(view["fps"]), DECODERS[arm])
                score[arm] = _score_prediction(labels, view, pred)
                if not np.array_equal(pred == "D", predictions["cand"] == "D"):
                    raise RuntimeError("New adopted-dead candidate changed adopted D mask")
                predictions[arm] = pred
            else:
                score[arm] = _score_probability(labels, view, period, probs, DECODERS[arm])
                predictions[arm] = audit.decode(probs, period, float(view["fps"]), DECODERS[arm])
    saved_probabilities = {"adop": pooled_controls["adop"], "cand": pooled_controls["cand"]}
    saved_probabilities.update({f"{family}_p{phase}_s{seed}": prob
                                for family, entries in per_family.items()
                                for phase, seed, prob in entries})
    if candidate is not None:
        for arm, entries in candidate.items():
            saved_probabilities[arm] = pooled(entries)
    return {"input_path": str(path), "input_sha256": sha(path), "phase_count": len(phases),
            "reference_spell_count": int(score["cand"]["reference_spell_count"]),
            "metrics": score, "predictions": predictions,
            "component_seed_diagnostics": component_scores if candidate is not None else {},
            "probabilities": saved_probabilities, "labels": labels,
            "timestamps": np.asarray(view["timestamps"]), "period": np.asarray(view["period"]),
            "model_period": np.asarray(period), "fps": float(view["fps"])}


def replay_controls() -> dict:
    """Replay archived control models on raw inputs, without candidate fit access."""
    declaration = _require_declaration()
    if REPLAY.exists() or SEAL.exists() or RESULT.exists():
        raise RuntimeError("Control replay or evaluation output already exists")
    from concurrent.futures import ProcessPoolExecutor
    rows, cache_inventory = {}, {}
    jobs = _views()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for job, row in zip(jobs, pool.map(_control_job, jobs)):
            provider, match, view, rep = job
            key = f"{provider}|{match}|{view}|{rep}"
            cache_path = _cell_cache_path(provider, match, view, rep)
            identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
                        "provider": provider, "match": match, "view": view, "replicate": rep,
                        "input_sha256": row["input_sha256"], "phase_count": row["phase_count"]}
            cache_inventory[key] = {"path": str(cache_path.relative_to(ROOT)),
                                    "sha256": _save_cell_arrays(cache_path, row, identity),
                                    "identity": identity}
            row = {k: v for k, v in row.items() if k not in {"probabilities", "labels", "timestamps", "period", "model_period", "fps"}}
            rows[key] = row
    aggregates = _aggregate_controls(rows)
    expected = _historical_expectations()
    checks = {}
    for view in ("clean", "feed_like_v1", "feed_like_v2"):
        for metric in METRICS:
            actual = aggregates[view]["cand"][metric]["equal_provider"]
            target = expected[view][metric]
            checks[f"{view}/{metric}"] = {"actual": actual, "expected": target,
                                          "absolute_error": abs(actual-target),
                                          "passed": abs(actual-target) <= 1e-8}
    passed = all(row["passed"] for row in checks.values())
    counts = {f"{view}/{rep}": sum(1 for _p, _m, name, r in _views() if name == view and r == rep)
              for view, reps in VIEWS for rep in reps}
    official_ref_counts = {f"{view}/{rep}": int(sum(
        row["metrics"]["adop"]["reference_spell_count"] for key, row in rows.items()
        if key.split("|")[2:] == [view, str(rep)])) for view, reps in VIEWS for rep in reps}
    receipt = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
               "controls_only": True, "candidate_scoring": False,
               "matches": rows, "aggregate": aggregates,
               "cell_counts": counts, "reference_spell_counts": official_ref_counts,
               "emission_cache": cache_inventory,
               "model_hashes": _model_hashes(), "control_replay_checks": checks,
               "tolerance": 1e-8, "passed": passed}
    _write_once(REPLAY, receipt)
    if not passed:
        raise RuntimeError(f"Archived original_full control replay failed: {checks}")
    return receipt


def _control_job(job):
    provider, match, view, rep = job
    return _score_cell(provider, match, view, rep)


def _require_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError(f"Freeze capacity evaluator first: {DECLARATION}")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != list(ARMS):
        raise RuntimeError("Evaluation declaration identity or arms changed")
    if declaration["provenance"]["source_sha256"] != _source_hashes():
        raise RuntimeError("Evaluator source changed after freeze")
    if training.check().get("study_id") != training.STUDY:
        raise RuntimeError("Capacity training declaration changed")
    if sha(training.DECLARATION) != declaration["provenance"]["training_declaration"]["sha256"]:
        raise RuntimeError("Capacity training declaration differs from evaluator freeze")
    if sha(ROOT / "experiments/baseline_manifest.json") != declaration["provenance"]["baseline_manifest_sha256"]:
        raise RuntimeError("Active model manifest changed after evaluator freeze")
    if sha(ROOT / "experiments/model_decision_policy.json") != declaration["provenance"]["decision_policy_sha256"]:
        raise RuntimeError("Decision policy changed after evaluator freeze")
    if sha(runtime.frozen.CALIBRATION) != declaration["provenance"]["feed_like_v2_calibration_sha256"]:
        raise RuntimeError("feed_like_v2 calibration changed after evaluator freeze")
    for key, fact in declaration["provenance"]["archived_inputs"].items():
        if sha(fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Archived input changed: {key}")
    for key, fact in declaration["provenance"]["control_models"].items():
        if sha(fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Archived control/teacher changed: {key}")
    return declaration


def seal() -> dict:
    declaration = _require_declaration()
    if SEAL.exists() or RESULT.exists():
        raise RuntimeError("Evaluation seal/result already exists")
    if not REPLAY.is_file():
        raise FileNotFoundError("Run controls-only replay before sealing candidate fits")
    replay = json.loads(REPLAY.read_text())
    if (replay.get("passed") is not True or
            replay.get("declaration_sha256") != sha(DECLARATION) or
            replay.get("candidate_scoring") is not False):
        raise RuntimeError("Controls-only replay did not pass")
    training_decl_sha = declaration["provenance"]["training_declaration"]["sha256"]
    model_facts = {}
    for recipe in RECIPES:
        for fold in FOLDS:
            for seed in SEEDS:
                model_path = training.fit_path(recipe, fold, seed)
                receipt_path = training.receipt_path(recipe, fold, seed)
                if not model_path.is_file() or not receipt_path.is_file():
                    raise FileNotFoundError(f"Missing declared model fit: {model_path}")
                receipt = json.loads(receipt_path.read_text())
                expected = {"arm": recipe, "fold": fold, "seed": seed,
                            "sha256": sha(model_path),
                            "declaration_sha256": training_decl_sha,
                            "heldout_provider": fold.removeprefix("leave_out_")}
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Fit receipt mismatch: {recipe}/{fold}/{seed}")
                training_matches = set(receipt.get("training_matches", ()))
                heldout_matches = {match for provider, match, _ in runtime.frozen.audit.MATCHES
                                   if runtime.frozen.audit.FOLD[provider] == fold.removeprefix("leave_out_")}
                if not training_matches or training_matches.intersection(heldout_matches):
                    raise RuntimeError(f"Held-out provider leak or incomplete fit receipt: {recipe}/{fold}/{seed}")
                model_facts[f"{recipe}/{fold}/{seed}"] = {
                    "path": str(model_path.relative_to(ROOT)), "sha256": expected["sha256"],
                    "receipt": str(receipt_path.relative_to(ROOT)),
                    "receipt_sha256": sha(receipt_path)}
    cache_facts = replay["emission_cache"]
    for key, fact in cache_facts.items():
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise RuntimeError(f"Archived control emission missing/changed: {key}")
    seal_value = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
                  "control_replay_sha256": sha(REPLAY), "models": model_facts,
                  "control_emissions": cache_facts,
                  "sealed_before_candidate_scoring": True,
                  "organizer_reference_access": False, "public_or_GAR_scoring": False}
    _write_once(SEAL, seal_value)
    return seal_value


def _require_seal():
    declaration = _require_declaration()
    if not SEAL.is_file():
        raise RuntimeError("Seal all capacity fits and control emissions before scoring")
    seal_value = json.loads(SEAL.read_text())
    if (seal_value.get("declaration_sha256") != sha(DECLARATION) or
            seal_value.get("control_replay_sha256") != sha(REPLAY)):
        raise RuntimeError("Evaluation seal binding changed")
    for fact in seal_value["models"].values():
        if sha(ROOT / fact["path"]) != fact["sha256"] or sha(ROOT / fact["receipt"]) != fact["receipt_sha256"]:
            raise RuntimeError("Sealed candidate model/receipt changed")
    for fact in seal_value["control_emissions"].values():
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError("Sealed control emission changed")
    return declaration, seal_value, json.loads(REPLAY.read_text())


def run() -> dict:
    declaration, seal_value, replay = _require_seal()
    if RESULT.exists():
        raise RuntimeError("Evaluation result already exists; scoring is one-shot")
    cell_rows, candidate_cache = {}, {}
    jobs = [(recipe, *cell, replay["emission_cache"][
        f"{cell[0]}|{cell[1]}|{cell[2]}|{cell[3]}"])
        for recipe in RECIPES for cell in _views()]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=3) as pool:
        future_map = {pool.submit(_candidate_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(future_map), 1):
            recipe, provider, match, view, rep, _ = future_map[future]
            key, row, fact = future.result()
            candidate_cache[f"{recipe}/{key}"] = fact
            cell_rows.setdefault(key, {"control": replay["matches"][key], "candidates": {}})
            cell_rows[key]["candidates"][recipe] = row
            print(f"SCORED {done}/{len(jobs)} {recipe}/{key}", flush=True)
    aggregates = _aggregate_candidates(cell_rows)
    seed_aggregates = _aggregate_component_seeds(cell_rows)
    adopted = {view: replay["aggregate"][view]["cand"] for view, _ in VIEWS}
    gates = {}
    deltas = {}
    for recipe in RECIPES:
        gates[recipe], deltas[recipe] = {}, {}
        for arm in DECODERS:
            candidate = {view: aggregates[view][f"{recipe}/{arm}"] for view, _ in VIEWS}
            gate_values = {view: {metric: candidate[view][metric]["equal_provider"]
                                  for metric in METRICS} for view, _ in VIEWS}
            adopt_values = {view: {metric: adopted[view][metric]["equal_provider"]
                                   for metric in METRICS} for view, _ in VIEWS}
            gates[recipe][arm] = hard_gate(gate_values, adopt_values, PR82_V2_FLOOR)
            deltas[recipe][arm] = {
                view: {metric: gate_values[view][metric] - adopt_values[view][metric]
                       for metric in METRICS} for view, _ in VIEWS}
    out = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
           "evaluation_seal_sha256": sha(SEAL), "control_replay_sha256": sha(REPLAY),
           "arms": list(ARMS), "aggregate": aggregates, "strict_success_gates": gates,
           "delta_vs_adopted": deltas, "component_seed_diagnostics": seed_aggregates,
           "matches": cell_rows,
           "candidate_emissions": candidate_cache,
           "hard_thresholds": {"pr82_v2_spell_f1": PR82_V2_FLOOR,
                               "zero_tolerance_vs_adopted": True},
           "organizer_reference_access": False, "public_or_GAR_scoring": False,
           "promotion": False}
    _write_once(RESULT, out)
    return out


def _candidate_job(job):
    recipe, provider, match, view, rep, control_fact = job
    controls = _load_cell_arrays(ROOT / control_fact["path"], control_fact["identity"])
    row = _score_cell(provider, match, view, rep, model_arm=recipe, control_cache=controls)
    candidate_path = _candidate_cache_path(provider, match, view, rep, recipe)
    identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
                "evaluation_seal_sha256": sha(SEAL), "recipe": recipe,
                "provider": provider, "match": match, "view": view,
                "replicate": rep, "input_sha256": row["input_sha256"]}
    fact = {"path": str(candidate_path.relative_to(ROOT)),
            "sha256": _save_cell_arrays(candidate_path, row, identity), "identity": identity}
    clean_row = {k: v for k, v in row.items() if k not in {
        "probabilities", "predictions", "labels", "timestamps", "period", "model_period", "fps"}}
    return f"{provider}|{match}|{view}|{rep}", clean_row, fact


def _historical_expectations() -> dict:
    main = json.loads(HISTORICAL_MAIN.read_text())["arms"]["deployed_original_full"]
    v1 = json.loads(HISTORICAL_V1.read_text())["arms"]["deployed_original_full"]
    result = {view: main[view] for view in ("clean", "feed_like_v2")}
    result["feed_like_v1"] = v1
    return {view: {metric: float(result[view][metric]["equal_provider_mean"])
                   for metric in METRICS} for view in result}


def _aggregate_controls(rows: dict) -> dict:
    out = {}
    for view, reps in VIEWS:
        out[view] = {}
        for family in ("adop", "cand", "adopted_dead_original_ha3"):
            out[view][family] = {}
            for metric in METRICS:
                values = {}
                for key, row in rows.items():
                    provider, match, cell_view, rep = key.split("|")
                    if cell_view == view and int(rep) in reps:
                        values[(provider, match, int(rep))] = row["metrics"][family][metric]
                aggregate, providers = equal_provider(values)
                out[view][family][metric] = {"equal_provider": aggregate, "providers": providers}
    return out


def _aggregate_candidates(cell_rows: dict) -> dict:
    result = {}
    for view, reps in VIEWS:
        result[view] = {}
        for arm in ARMS:
            recipe, decoder = arm.split("/", 1)
            values_by_metric = {metric: {} for metric in METRICS}
            for key, row in cell_rows.items():
                provider, match, cell_view, rep = key.split("|")
                if cell_view != view or int(rep) not in reps:
                    continue
                metrics = row["candidates"][recipe]["metrics"][decoder]
                for metric in METRICS:
                    values_by_metric[metric][(provider, match, int(rep))] = metrics[metric]
            result[view][arm] = {}
            for metric, values in values_by_metric.items():
                mean, providers = equal_provider(values)
                result[view][arm][metric] = {"equal_provider": mean, "providers": providers}
    return result


def _aggregate_component_seeds(cell_rows: dict) -> dict:
    """Aggregate each fit recipe/seed separately; never pool or select seeds."""
    result = {}
    for recipe in RECIPES:
        result[recipe] = {}
        for seed in SEEDS:
            result[recipe][str(seed)] = {}
            for view, reps in VIEWS:
                result[recipe][str(seed)][view] = {}
                for metric in ("spell_f1", "macro_f1", "d_f1"):
                    values = {}
                    for key, row in cell_rows.items():
                        provider, match, cell_view, rep = key.split("|")
                        if cell_view == view and int(rep) in reps:
                            values[(provider, match, int(rep))] = row["candidates"][recipe][
                                "component_seed_diagnostics"][str(seed)][metric]
                    mean, providers = equal_provider(values)
                    result[recipe][str(seed)][view][metric] = {
                        "equal_provider": mean, "providers": providers, "cells": len(values)}
    return result


def freeze() -> dict:
    """Freeze eval protocol and artifact identities; does not fit or score."""
    runtime.frozen._roots()
    if DECLARATION.exists() or REPLAY.exists() or SEAL.exists() or RESULT.exists():
        raise RuntimeError("Evaluation output already exists; preserve the frozen study ID")
    train_decl = training.DECLARATION
    if not train_decl.is_file():
        raise FileNotFoundError("Freeze v2_data_capacity before freezing its evaluator")
    train = training.check()
    if tuple(train.get("arms", ())) != RECIPES:
        raise RuntimeError("Training declaration's fit recipes do not match the evaluator")
    if not HISTORICAL_MAIN.is_file() or not HISTORICAL_V1.is_file():
        raise FileNotFoundError("Saved matched control metrics are required")
    raw_inputs = {}
    for provider, match, _ in runtime.frozen.audit.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                path, *_ = _raw_cache(provider, match, view, rep)
                raw_inputs[f"{provider}/{match}/{view}/{rep}"] = {
                    "path": str(path), "sha256": sha(path)}
    archived_models = _model_hashes()
    calibration = runtime.frozen.CALIBRATION
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    declaration = {
        "study_id": STUDY,
        "status": "frozen_before_control_replay_and_candidate_scoring",
        "question": "Do the fixed capacity/data recipes pass strict clean/v1/v2 no-regression against adopted original_full while the PR82-style blend reproduces or exceeds the eligible v2 score?",
        "arms": list(ARMS),
        "candidate_methods": {
            "standalone_ha3": "Mean the two seed probabilities within each native phase, equal-mean native phases, then decode using f062 H/A cost 3.",
            "pr82_blend_ha1_5": "Fixed 50/50 new-model and archived calibrated-model probability blend, seed mean within phase, native phase mean, then selected decoder dt=4/dead=3/ha=1.5/ht=1.",
            "adopted_blend_ha3": "Fixed 50/50 new-model and archived original_full probability blend, seed mean within phase, native phase mean, then f062 H/A cost 3.",
            "adopted_dead_new_ha3": "Keep the archived adopted original_full f062 D mask; decode only the candidate conditional H/A probabilities inside each contiguous predicted-live span, respecting model-period breaks, with H/A cost 3. No labels enter inference.",
        },
        "cohort": {"matches": [{"provider": p, "match": m} for p, m, _ in runtime.frozen.audit.MATCHES],
                   "views": [name for name, _ in VIEWS], "feed_like_v1_realizations": [1, 2],
                   "feed_like_v2_realizations": [1, 2, 3, 4],
                   "phases": "all native 5 Hz phases (five at 25 Hz, two at 10 Hz)",
                   "aggregation": "replicates within match, matches within provider, providers equally",
                   "heldout_rule": "all new fits, teachers, normalization and controls exclude evaluated provider"},
        "metrics": {"reported": list(METRICS),
                    "strict_success": "clean/v1/v2 spell, macro, D-F1, D-precision, D-recall and short-spell recall each >= adopted matched control; v2 spell additionally >= PR82 eligible 0.5927491913246999; zero tolerance",
                    "provider_guard": "report per-provider metric deltas; no hidden tolerance"},
        "thresholds": {"pr82_v2_spell_f1": PR82_V2_FLOOR,
                       "adopted": {"clean_spell_f1": .667556, "v1_spell_f1": .531576,
                                   "v2_spell_f1": .583571}},
        "controls": {"control_replay_required_before_fit_or_candidate_scoring": True,
                     "control_models": ["archived calibrated", "adopted original_full",
                                        "adopted_dead_original_ha3"],
                     "no_candidate_scoring_during_replay": True,
                     "replay_tolerance": 1e-8},
        "budget": {"recipes": list(RECIPES), "candidate_arms_per_recipe": list(DECODERS),
                   "additional_fixed_controls": list(CONTROL_ARMS),
                   "candidate_scored_cells_per_arm": len(runtime.frozen.audit.MATCHES) * sum(len(r) for _, r in VIEWS),
                   "fits": 0, "decoder_search": False},
        "provenance": {"training_declaration": {"path": str(train_decl.relative_to(ROOT)),
                                                  "sha256": sha(train_decl)},
                       "archived_inputs": raw_inputs,
                       "control_models": archived_models,
                       "feed_like_v2_calibration_sha256": sha(calibration),
                       "historical_control_sources": {
                           str(p.relative_to(ROOT)): sha(p) for p in (HISTORICAL_MAIN, HISTORICAL_V1)},
                       "source_sha256": _source_hashes(),
                       "baseline_manifest_sha256": sha(ROOT / "experiments/baseline_manifest.json"),
                       "decision_policy_sha256": sha(ROOT / "experiments/model_decision_policy.json")},
        "organizer_reference_access": False, "public_or_GAR_scoring": False,
        "promotion": False,
    }
    _write_once(DECLARATION, declaration)
    return declaration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay-controls", "seal", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "replay-controls": replay_controls,
     "seal": seal, "run": run}[args.action]()


if __name__ == "__main__":
    main()
