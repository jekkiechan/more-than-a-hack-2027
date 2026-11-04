"""Independently audit the sealed v2 capacity evaluation aggregates and gates.

This verifier consumes the evaluator's saved per-cell official metrics. It does
not infer, score organizer/GAR data, modify candidate outputs, or promote a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_data_capacity_eval"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
TRAIN_DECLARATION = ROOT / "experiments/results/v2_data_capacity_v1_declaration.json"
VERIFICATION = ROOT / f"experiments/results/{STUDY}_verification.json"
REPLAY = ROOT / f"experiments/results/{STUDY}_control_replay.json"
SEAL = ROOT / f"data/experiments/{STUDY}/evaluation_seal.json"
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
           "predicted_spell_count", "reference_spell_count",
           "short_spell_recall_iou_0_5_lt3s")
VIEWS = ("clean", "feed_like_v1", "feed_like_v2")
STRICT_METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                   "short_spell_recall_iou_0_5_lt3s")
PR82_V2_SPELL = 0.5927491913246999


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(a: float, b: float, atol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= atol


def equal_provider(values: dict[tuple[str, str, int], float]) -> tuple[float, dict[str, float]]:
    """Average reps within match, matches within provider, providers equally."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for (provider, match, _rep), score in values.items():
        grouped.setdefault(provider, {}).setdefault(match, []).append(float(score))
    if not grouped or any(not matches for matches in grouped.values()):
        raise ValueError("Cannot aggregate an empty or malformed cohort")
    providers = {
        provider: sum(sum(reps) / len(reps) for reps in matches.values()) / len(matches)
        for provider, matches in grouped.items()
    }
    return sum(providers.values()) / len(providers), providers


def recompute_aggregates(matches: dict, arms: list[str]) -> dict:
    expected = {}
    for view in VIEWS:
        expected[view] = {}
        for arm in arms:
            expected[view][arm] = {}
            for metric in METRICS:
                values = {}
                for key, cell in matches.items():
                    provider, match, cell_view, rep = key.split("|")
                    if cell_view == view:
                        values[(provider, match, int(rep))] = cell["metrics"][arm][metric]
                aggregate, provider_values = equal_provider(values)
                expected[view][arm][metric] = {
                    "equal_provider": aggregate,
                    "providers": provider_values,
                }
    return expected


def evaluate_gates(candidate: dict, adopted: dict, *, pr82_spell: float = PR82_V2_SPELL) -> dict:
    checks = {}
    for view in VIEWS:
        for metric in STRICT_METRICS:
            value, baseline = float(candidate[view][metric]), float(adopted[view][metric])
            checks[f"{view}/{metric}_vs_adopted"] = {
                "actual": value, "minimum": baseline, "passed": value >= baseline,
            }
    value = float(candidate["feed_like_v2"]["spell_f1"])
    checks["feed_like_v2/spell_f1_vs_pr82"] = {
        "actual": value, "minimum": pr82_spell, "passed": value >= pr82_spell,
    }
    return {"checks": checks, "all_passed": all(row["passed"] for row in checks.values())}


def verify_dead_mask(candidate_mask, adopted_mask) -> None:
    """Require the protected pipeline to preserve every adopted live/dead decision."""
    import numpy as np

    candidate = np.asarray(candidate_mask, dtype=bool)
    adopted = np.asarray(adopted_mask, dtype=bool)
    if candidate.shape != adopted.shape or not np.array_equal(candidate, adopted):
        raise ValueError("Protected decoder changed the frozen adopted D mask")


def historical_diagnostics(aggregate: dict, arm: str, adopted_arm: str = "cand",
                           adopted_aggregate: dict | None = None) -> dict:
    """Report historical policy guards separately from strict study success."""
    adopted_aggregate = adopted_aggregate or aggregate
    checks = {}
    for metric in ("macro_f1", "d_f1"):
        candidate = aggregate["clean"][arm][metric]["providers"]
        adopted = adopted_aggregate["clean"][adopted_arm][metric]["providers"]
        for provider, value in candidate.items():
            floor = adopted[provider] - 0.02
            checks[f"clean/{metric}/{provider}_vs_adopted_minus_0_02"] = {
                "actual": value, "minimum": floor, "passed": value >= floor,
            }
    candidate_v2 = aggregate["feed_like_v2"][arm]["spell_f1"]["equal_provider"]
    adopted_v2 = adopted_aggregate["feed_like_v2"][adopted_arm]["spell_f1"]["equal_provider"]
    checks["feed_like_v2/spell_f1_vs_adopted_minus_0_01"] = {
        "actual": candidate_v2, "minimum": adopted_v2 - 0.01,
        "passed": candidate_v2 >= adopted_v2 - 0.01,
    }
    return {"checks": checks, "all_passed": all(row["passed"] for row in checks.values()),
            "primary_gate": False}


def _verify_cohort(declaration: dict, matches: dict) -> None:
    cohort = declaration["cohort"]
    pairs = {(row["provider"], row["match"]) for row in cohort["matches"]}
    if len(pairs) != 5 or len({provider for provider, _ in pairs}) != 3:
        raise ValueError("Expected five exposed development matches across three providers")
    expected = set()
    for provider, match in pairs:
        expected.add(f"{provider}|{match}|clean|0")
        expected.update(f"{provider}|{match}|feed_like_v1|{rep}" for rep in (1, 2))
        expected.update(f"{provider}|{match}|feed_like_v2|{rep}" for rep in (1, 2, 3, 4))
    if set(matches) != expected:
        raise ValueError(f"Evaluation cell inventory mismatch: expected {len(expected)}, got {len(matches)}")


def _metric_aggregates(matches: dict, recipes: list[str], decoder_arms: list[str]) -> dict:
    out = {view: {} for view in VIEWS}
    for view in VIEWS:
        for recipe in recipes:
            for decoder in decoder_arms:
                arm_id = f"{recipe}/{decoder}"
                out[view][arm_id] = {}
                for metric in METRICS:
                    values = {}
                    for key, row in matches.items():
                        provider, match, cell_view, rep = key.split("|")
                        if cell_view != view:
                            continue
                        score = row["candidates"][recipe]["metrics"][decoder][metric]
                        values[(provider, match, int(rep))] = score
                    mean, providers = equal_provider(values)
                    out[view][arm_id][metric] = {"equal_provider": mean, "providers": providers}
    return out


def _load_emission(fact: dict) -> dict:
    import numpy as np

    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise ValueError(f"Sealed emission missing or changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        identity = {key.removeprefix("identity__"): str(archive[key].item())
                    for key in archive.files if key.startswith("identity__")}
        expected = {key: str(value) for key, value in fact["identity"].items()}
        if identity != expected:
            raise ValueError(f"Sealed emission identity mismatch: {path}")
        return {key: archive[key].copy() for key in archive.files
                if not key.startswith("identity__")}


def _protected_prediction(probability, adopted_probability, period, fps, config):
    import numpy as np
    from experiments import feed_benchmark_audit_v1 as audit
    from experiments import v2_spell_target_runtime as runtime

    adopted = audit.decode(adopted_probability, period, fps, audit.CONFIG)
    result = adopted.copy()
    conditional = np.asarray(probability, dtype=np.float64).copy()
    mass = conditional[:, :2].sum(axis=1)
    if (mass <= 0).any():
        raise ValueError("Saved probability has no conditional H/A mass")
    conditional[:, :2] /= mass[:, None]
    conditional[:, 2] = 0.0
    live = adopted != "D"
    for begin, end in runtime.frozen.supervised.blocks(period):
        cursor = begin
        while cursor < end:
            if not live[cursor]:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < end and live[stop]:
                stop += 1
            result[cursor:stop] = audit.decode(conditional[cursor:stop], period[cursor:stop], fps, config)
            cursor = stop
    verify_dead_mask(result == "D", adopted == "D")
    return result


def _replay_emission_metrics(arrays: dict, decoder: str, *, protected: bool = False) -> dict:
    import numpy as np
    from experiments import feed_benchmark_audit_v1 as audit

    labels = arrays["labels"].astype(str)
    period = arrays["period"]
    cell = {"labels": labels, "period": period, "timestamps": arrays["timestamps"],
            "fps": float(arrays["fps"].item())}
    model_period = arrays["model_period"]
    if decoder in ("adop", "cand"):
        config = audit.CONFIG
        probability = arrays[f"prob__{decoder}"]
        prediction = audit.decode(probability, model_period, cell["fps"], config)
    elif decoder == "adopted_dead_original_ha3":
        prediction = _protected_prediction(arrays["prob__cand"], arrays["prob__cand"],
                                           model_period, cell["fps"],
                                           dict(audit.CONFIG, dt=4, dead=3, ha=3.0, ht=1.0))
    else:
        config = dict(audit.CONFIG, dt=4, dead=3,
                      ha=1.5 if decoder == "pr82_blend_ha1_5" else 3.0, ht=1.0)
        probability = arrays[f"prob__{decoder}"]
        if decoder == "adopted_dead_new_ha3":
            prediction = _protected_prediction(probability, arrays["prob__cand"],
                                               model_period, cell["fps"], config)
        else:
            prediction = audit.decode(probability, model_period, cell["fps"], config)
    return audit.score(cell, prediction), prediction


def _verify_saved_metric_row(saved: dict, actual: dict, where: str) -> None:
    for metric in METRICS:
        if not _close(saved[metric], actual[metric]):
            raise ValueError(f"Official scalar replay mismatch for {where}/{metric}: "
                             f"{saved[metric]} != {actual[metric]}")


def _verify_control_aggregates(replay: dict) -> None:
    for view in VIEWS:
        for family in ("adop", "cand"):
            for metric in METRICS:
                values = {}
                for key, row in replay["matches"].items():
                    provider, match, cell_view, rep = key.split("|")
                    if cell_view == view:
                        values[(provider, match, int(rep))] = row["metrics"][family][metric]
                aggregate, providers = equal_provider(values)
                saved = replay["aggregate"][view][family][metric]
                if not _close(saved["equal_provider"], aggregate):
                    raise ValueError(f"Control aggregate mismatch: {view}/{family}/{metric}")
                if set(saved["providers"]) != set(providers) or any(
                        not _close(saved["providers"][p], value) for p, value in providers.items()):
                    raise ValueError(f"Control provider aggregate mismatch: {view}/{family}/{metric}")
    for view, reps in (("clean", (0,)), ("feed_like_v1", (1, 2)), ("feed_like_v2", (1, 2, 3, 4))):
        for rep in reps:
            key = f"{view}/{rep}"
            matching = [row for cell_key, row in replay["matches"].items()
                        if cell_key.split("|")[2:] == [view, str(rep)]]
            if replay["cell_counts"][key] != len(matching):
                raise ValueError(f"Control cell count changed: {key}")
            reference_count = sum(row["metrics"]["cand"]["reference_spell_count"] for row in matching)
            if replay["reference_spell_counts"][key] != reference_count:
                raise ValueError(f"Control reference spell count changed: {key}")


def _verify_native_emissions(result: dict, replay: dict, seal: dict) -> int:
    controls = seal["control_emissions"]
    candidates = result["candidate_emissions"]
    recipes = sorted({key.split("/", 1)[0] for key in candidates})
    replayed = 0
    for key, row in result["matches"].items():
        control_fact = controls[key]
        provider, match, view, rep = key.split("|")
        control_identity = control_fact["identity"]
        expected_control_identity = {"provider": provider, "match": match, "view": view,
                                     "replicate": int(rep), "declaration_sha256": sha(DECLARATION)}
        if any(str(control_identity.get(name)) != str(value)
               for name, value in expected_control_identity.items()):
            raise ValueError(f"Control emission identity differs from cell key: {key}")
        control_arrays = _load_emission(control_fact)
        control_row = replay["matches"][key]
        if row.get("control") != control_row:
            raise ValueError(f"Candidate result changed its control row: {key}")
        for decoder in ("adop", "cand", "adopted_dead_original_ha3"):
            actual, _ = _replay_emission_metrics(control_arrays, decoder)
            _verify_saved_metric_row(control_row["metrics"][decoder], actual, f"control/{key}/{decoder}")
        for recipe in recipes:
            candidate_fact = candidates[f"{recipe}/{key}"]
            candidate_identity = candidate_fact["identity"]
            provider, match, view, rep = key.split("|")
            expected_identity = {"recipe": recipe, "provider": provider, "match": match,
                                 "view": view, "replicate": int(rep),
                                 "evaluation_seal_sha256": sha(SEAL)}
            if any(str(candidate_identity.get(name)) != str(value)
                   for name, value in expected_identity.items()):
                raise ValueError(f"Candidate emission identity differs from cell key: {recipe}/{key}")
            if candidate_identity.get("input_sha256") != controls[key]["identity"]["input_sha256"]:
                raise ValueError(f"Candidate input binding differs from controls: {recipe}/{key}")
            candidate_arrays = _load_emission(candidate_fact)
            for name in ("labels", "timestamps", "period", "model_period", "fps"):
                import numpy as np
                if not np.array_equal(candidate_arrays[name], control_arrays[name]):
                    raise ValueError(f"Candidate evaluation cohort differs from controls: {recipe}/{key}/{name}")
            saved = row["candidates"][recipe]["metrics"]
            for decoder in ("adop", "cand", "adopted_dead_original_ha3",
                            "standalone_ha3", "pr82_blend_ha1_5", "adopted_blend_ha3",
                            "adopted_dead_new_ha3"):
                actual, prediction = _replay_emission_metrics(candidate_arrays, decoder)
                _verify_saved_metric_row(saved[decoder], actual, f"{recipe}/{key}/{decoder}")
                if decoder in ("adopted_dead_original_ha3", "adopted_dead_new_ha3"):
                    # The archived adopted full decoder supplies the immutable D mask.
                    from experiments import feed_benchmark_audit_v1 as audit
                    adopted_pred = audit.decode(candidate_arrays["prob__cand"],
                        candidate_arrays["model_period"], float(candidate_arrays["fps"].item()), audit.CONFIG)
                    verify_dead_mask(prediction == "D", adopted_pred == "D")
            # Candidate scoring must preserve the reference cohort/counts exactly.
            baseline_count = control_row["metrics"]["cand"]["reference_spell_count"]
            for decoder in saved:
                if saved[decoder]["reference_spell_count"] != baseline_count:
                    raise ValueError(f"Reference spell count changed for {recipe}/{key}/{decoder}")
            replayed += 1
    return replayed


def verify(result: dict, declaration: dict, replay: dict, seal: dict) -> dict:
    recipes = list(declaration["budget"]["recipes"])
    decoder_arms = [arm for arm in declaration["budget"]["candidate_arms_per_recipe"]
                    if arm != "adopted_dead_original_ha3"]
    _verify_cohort(declaration, result["matches"])
    if result.get("study_id") != STUDY:
        raise ValueError("Wrong evaluation result identity")
    if result.get("declaration_sha256") != sha(DECLARATION):
        raise ValueError("Evaluation result is not bound to the frozen declaration")
    if result.get("evaluation_seal_sha256") != sha(SEAL) or result.get("control_replay_sha256") != sha(REPLAY):
        raise ValueError("Evaluation result seal/replay binding mismatch")
    if (result.get("organizer_reference_access") is not False or
            result.get("public_or_GAR_scoring") is not False or result.get("promotion") is not False):
        raise ValueError("Evaluation result must explicitly exclude public/GAR scoring and promotion")
    if replay.get("passed") is not True or replay.get("organizer_reference_access") is not False:
        raise ValueError("Control replay did not pass or lacks its no-organizer assertion")
    if seal.get("declaration_sha256") != sha(DECLARATION) or seal.get("control_replay_sha256") != sha(REPLAY):
        raise ValueError("Evaluation seal declaration or replay binding mismatch")
    _verify_source_bindings(declaration)
    _verify_control_aggregates(replay)
    fit_facts = _verify_fit_receipts(json.loads(TRAIN_DECLARATION.read_text()))
    if set(seal.get("models", {})) != set(fit_facts):
        raise ValueError("Evaluation seal model inventory differs from fit receipts")
    for key, expected in fit_facts.items():
        saved = seal["models"][key]
        if any(saved.get(field) != expected[field]
               for field in ("path", "sha256", "receipt", "receipt_sha256")):
            raise ValueError(f"Evaluation seal model hash mismatch: {key}")
    if seal.get("control_emissions") != replay.get("emission_cache"):
        raise ValueError("Evaluation seal does not preserve control emission inventory")
    expected_arms = {f"{recipe}/{decoder}" for recipe in recipes for decoder in decoder_arms}
    if set(result.get("arms", ())) != expected_arms:
        raise ValueError("Candidate pipeline inventory differs from frozen 12-arm budget")
    expected_emissions = {f"{recipe}/{key}" for recipe in recipes for key in result["matches"]}
    if set(result.get("candidate_emissions", {})) != expected_emissions:
        raise ValueError("Candidate emission inventory differs from the frozen cohort")
    if set(replay.get("matches", {})) != set(result["matches"]):
        raise ValueError("Control replay and candidate result cohorts differ")
    recomputed = _metric_aggregates(result["matches"], recipes, decoder_arms)
    for view in VIEWS:
        for arm in result["arms"]:
            if arm not in recomputed[view]:
                raise ValueError(f"Undeclared candidate arm: {arm}")
            for metric in METRICS:
                saved = result["aggregate"][view][arm][metric]
                actual = recomputed[view][arm][metric]
                if set(saved["providers"]) != set(actual["providers"]):
                    raise ValueError(f"Provider inventory differs for {view}/{arm}/{metric}")
                if not _close(saved["equal_provider"], actual["equal_provider"]):
                    raise ValueError(f"Aggregate mismatch for {view}/{arm}/{metric}")
                for provider, score in actual["providers"].items():
                    if not _close(saved["providers"][provider], score):
                        raise ValueError(f"Provider aggregate mismatch for {view}/{arm}/{metric}/{provider}")
    adopted = {view: replay["aggregate"][view]["cand"] for view in VIEWS}
    gates, historical = {}, {}
    for arm in result["arms"]:
        recipe, decoder = arm.split("/", 1)
        candidate = {view: {metric: result["aggregate"][view][arm][metric]["equal_provider"]
                            for metric in STRICT_METRICS} for view in VIEWS}
        baseline = {view: {metric: adopted[view][metric]["equal_provider"]
                           for metric in STRICT_METRICS} for view in VIEWS}
        gate = evaluate_gates(candidate, baseline)
        gates[arm] = gate
        historical[arm] = historical_diagnostics(
            result["aggregate"], arm, adopted_arm="cand", adopted_aggregate=replay["aggregate"])
    if result.get("strict_success_gates") is not None:
        for recipe in recipes:
            for decoder in decoder_arms:
                arm_id = f"{recipe}/{decoder}"
                if result["strict_success_gates"][recipe][decoder] != gates[arm_id]:
                    raise ValueError(f"Saved strict gate differs from independent recomputation: {arm_id}")
    replayed = _verify_native_emissions(result, replay, seal)
    return {
        "study_id": STUDY,
        "result_sha256": sha(RESULT),
        "declaration_sha256": sha(DECLARATION),
        "training_declaration_sha256": sha(TRAIN_DECLARATION),
        "cell_count": len(result["matches"]),
        "official_scalar_cells_replayed": replayed,
        "aggregates_recomputed_from_saved_official_cell_metrics": True,
        "aggregation_order": "realizations within match, matches within provider, providers equally",
        "gates_recomputed_with_zero_tolerance": gates,
        "historical_provider_and_v2_guards_reported_separately": historical,
        "organizer_or_GAR_scoring": False,
        "inference_repeated": False,
    }


def _verify_fit_receipts(training_declaration: dict) -> None:
    """Bind all 18 model payloads to their declaration, teacher and held-out fold."""
    from experiments import v2_data_capacity as training

    if training_declaration.get("study_id") != "v2_data_capacity_v1":
        raise ValueError("Wrong training declaration identity")
    arms = training_declaration["arms"]
    folds = training_declaration["folds"]
    seeds = training_declaration["seeds"]
    expected_teachers = training_declaration["input_bindings"]["teachers"]
    plans = training_declaration["cohort"]["provider_folds"]
    if len(arms) != 3 or len(folds) != 3 or len(seeds) != 2:
        raise ValueError("Training declaration must bind three arms, folds, and two seeds")
    fit_facts = {}
    for fold in folds:
        plan = plans[fold]
        train_rows = plan["train"]
        expected_matches = {str(row["match"]) for row in train_rows}
        excluded_provider = fold.removeprefix("leave_out_")
        if any(str(row["source"]).lower() == excluded_provider for row in train_rows):
            raise ValueError(f"Held-out provider appears in training plan: {fold}")
        for arm in arms:
            for seed in seeds:
                model_path = training.fit_path(arm, fold, seed)
                receipt_path = training.receipt_path(arm, fold, seed)
                if not model_path.is_file() or not receipt_path.is_file():
                    raise ValueError(f"Missing declared fit or receipt: {arm}/{fold}/{seed}")
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("study") != "v2_data_capacity_v1":
                    raise ValueError(f"Wrong fit study identity: {arm}/{fold}/{seed}")
                if receipt.get("arm") != arm or receipt.get("fold") != fold or int(receipt.get("seed", -1)) != int(seed):
                    raise ValueError(f"Fit receipt identity mismatch: {arm}/{fold}/{seed}")
                if receipt.get("sha256") != sha(model_path):
                    raise ValueError(f"Fit model hash mismatch: {arm}/{fold}/{seed}")
                if receipt.get("declaration_sha256") != sha(TRAIN_DECLARATION):
                    raise ValueError(f"Fit receipt is not bound to training declaration: {arm}/{fold}/{seed}")
                if receipt.get("teacher_sha256") != expected_teachers[f"{fold}/{seed}"]:
                    raise ValueError(f"Teacher hash mismatch: {arm}/{fold}/{seed}")
                if set(receipt.get("training_matches", ())) != expected_matches:
                    raise ValueError(f"Training match inventory mismatch: {arm}/{fold}/{seed}")
                config = receipt.get("config", {})
                expected_width = 192 if arm == "width192_capacity" else 96
                if (int(receipt.get("updates", -1)) != 2048 or
                        int(config.get("steps", -1)) != 2048 or
                        int(config.get("width", -1)) != expected_width or
                        config.get("train_d") is not True or
                        config.get("ha_target") != "dense"):
                    raise ValueError(f"Fit config differs from frozen budget: {arm}/{fold}/{seed}")
                if receipt.get("heldout_provider") != excluded_provider:
                    raise ValueError(f"Held-out provider receipt mismatch: {arm}/{fold}/{seed}")
                if excluded_provider in " ".join(receipt.get("training_sources", ())).lower():
                    raise ValueError(f"Held-out provider appears in fit receipt: {arm}/{fold}/{seed}")
                if receipt.get("organizer_reference_access") is not False:
                    raise ValueError(f"Organizer reference access is not explicitly false: {arm}/{fold}/{seed}")
                fit_facts[f"{arm}/{fold}/{seed}"] = {
                    "path": str(model_path.relative_to(ROOT)), "sha256": sha(model_path),
                    "receipt": str(receipt_path.relative_to(ROOT)),
                    "receipt_sha256": sha(receipt_path)}
    return fit_facts


def _verify_source_bindings(declaration: dict) -> None:
    provenance = declaration.get("provenance", {})
    for rel, expected in provenance.get("source_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen evaluator source changed: {rel}")
    for key, rel in (("baseline_manifest_sha256", "experiments/baseline_manifest.json"),
                     ("decision_policy_sha256", "experiments/model_decision_policy.json")):
        if key not in provenance or sha(ROOT / rel) != provenance[key]:
            raise ValueError(f"Missing or changed frozen provenance binding: {key}")
    for rel, expected in declaration.get("source_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Training source changed: {rel}")
    provenance = declaration.get("provenance", {})
    training_binding = provenance.get("training_declaration", {})
    if (training_binding.get("path") != str(TRAIN_DECLARATION.relative_to(ROOT)) or
            training_binding.get("sha256") != sha(TRAIN_DECLARATION)):
        raise ValueError("Evaluation declaration is not bound to the training declaration")
    for label, rows in (("archived input", provenance.get("archived_inputs", {}).values()),
                        ("historical control", provenance.get("historical_control_sources", {}).items())):
        items = rows if label == "archived input" else (dict(path=path, sha256=digest) for path, digest in rows)
        for item in items:
            path = Path(item["path"])
            expected = item["sha256"]
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Frozen {label} changed: {path}")
    for item in provenance.get("control_models", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Frozen control model changed: {path}")
    from experiments import v2_spell_target_runtime as runtime
    calibration = runtime.frozen.CALIBRATION
    if sha(calibration) != provenance.get("feed_like_v2_calibration_sha256"):
        raise ValueError("Frozen v2 calibration hash mismatch")


def main() -> dict:
    result = json.loads(RESULT.read_text())
    declaration = json.loads(DECLARATION.read_text())
    replay = json.loads(REPLAY.read_text())
    seal = json.loads(SEAL.read_text())
    receipt = verify(result, declaration, replay, seal)
    with VERIFICATION.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
