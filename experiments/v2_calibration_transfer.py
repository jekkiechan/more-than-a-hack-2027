"""Transfer one preselected decoder calibration to the three v2-trained arms.

This study uses only the frozen ``adop_dt4_dead3_ha1.5_ht1`` calibration
selected by ``v2_calibration_target``. It consumes the seed/phase emissions
already produced and sealed by ``v2_training_eval`` and refuses to infer a
missing cache. The transfer declaration must be frozen before the main
training-evaluation result exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_training_eval as base
from experiments import v2_training_eval_runtime_v2 as eval_runtime
from experiments import v2_calibration_target as calibration
from experiments import v2_spell_target_runtime as upstream


ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_calibration_transfer"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
SEAL = ROOT / f"data/experiments/{STUDY}/transfer_seal.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
SELECTED_CONFIG = "adop_dt4_dead3_ha1.5_ht1"
ARMS = tuple(base.ARMS)
SEEDS = tuple(base.SEEDS)
VIEWS = tuple(base.VIEWS)
METRICS = tuple(base.METRICS)
# Preserve the frozen fold-to-model path correction before any transfer work.
base._fit_path = eval_runtime.corrected_fit_path


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


def _calibration_bindings():
    if not calibration.RESULT.is_file() or not calibration.DECLARATION.is_file():
        raise FileNotFoundError("Completed frozen calibration-target evidence is required")
    result = json.loads(calibration.RESULT.read_text())
    declaration = json.loads(calibration.DECLARATION.read_text())
    if (result.get("study_id") != calibration.STUDY or
            result.get("declaration_sha256") != sha(calibration.DECLARATION) or
            declaration.get("study_id") != calibration.STUDY):
        raise RuntimeError("Calibration result/declaration binding is invalid")
    selected = result.get("selected_candidate")
    if selected != SELECTED_CONFIG:
        raise RuntimeError(f"Unexpected preselected calibration: {selected}")
    arms = {arm["id"]: arm for arm in declaration["arms"]}
    if SELECTED_CONFIG not in arms:
        raise RuntimeError("Selected calibration arm is absent from frozen calibration declaration")
    config = dict(arms[SELECTED_CONFIG]["config"])
    expected = {"dt": 4, "dead": 3, "ha": 1.5, "ht": 1.0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Selected calibration configuration changed: {config}")
    return result, declaration, config


def _source_hashes():
    paths = (
        Path(__file__), Path(base.__file__), Path(calibration.__file__),
        Path(eval_runtime.__file__), Path(base.ROOT / "experiments/v2_training_eval_runtime.py"),
        Path(upstream.__file__), Path(upstream.frozen.__file__),
        Path(upstream.frozen.audit.__file__), Path(upstream.frozen.sn.__file__),
        Path(upstream.frozen.old.__file__), Path(upstream.frozen.audit.decoder.__file__),
        Path(upstream.frozen.CALIBRATION), Path(upstream.frozen.SCORECARD),
        Path(upstream.frozen.BENCHMARK), Path(ROOT / "evaluation/possession_metrics.py"),
    )
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
            for path in paths}


def _runtime_bindings():
    upstream.frozen._roots()
    runtime_decl = upstream.frozen.DECLARATION
    runtime_preflight = upstream.frozen.PREFLIGHT
    if not runtime_decl.is_file() or not runtime_preflight.is_file():
        raise FileNotFoundError("Verified upstream runtime declaration and preflight are required")
    declaration = json.loads(runtime_decl.read_text())
    preflight = json.loads(runtime_preflight.read_text())
    upstream.frozen._verify_frozen_bindings(declaration)
    if (preflight.get("preflight_passed") is not True or
            preflight.get("declaration_sha256") != sha(runtime_decl)):
        raise RuntimeError("Upstream controls-only preflight is not valid")
    return {"study_id": upstream.STUDY, "declaration_sha256": sha(runtime_decl),
            "preflight_sha256": sha(runtime_preflight)}


def _eval_bindings():
    declaration = base._require_declaration()
    _, seal, preflight = base._require_seal()
    correction = eval_runtime._verify()
    if not SEAL.parent.exists():
        SEAL.parent.mkdir(parents=True, exist_ok=True)
    model_facts = {key: {"path": value["path"], "sha256": value["sha256"],
                         "receipt_sha256": value["receipt_sha256"]}
                   for key, value in seal["models"].items()}
    return declaration, seal, preflight, model_facts, correction


def _source_bindings():
    return {"source_sha256": _source_hashes(),
            "baseline_manifest_sha256": sha(ROOT / "experiments/baseline_manifest.json"),
            "decision_policy_sha256": sha(ROOT / "experiments/model_decision_policy.json")}


def freeze() -> dict:
    """Freeze the three-arm transfer before the primary new-model result."""
    if DECLARATION.exists() or SEAL.exists() or RESULT.exists():
        raise RuntimeError("Transfer declaration/seal/result already exists")
    if base.RESULT.exists():
        raise RuntimeError("Transfer declaration must freeze before v2_training_eval results exist")
    calibration_result, calibration_decl, config = _calibration_bindings()
    eval_decl, eval_seal, preflight, models, correction = _eval_bindings()
    runtime = _runtime_bindings()
    declaration = {
        "study_id": STUDY,
        "status": "frozen_three_arm_calibration_transfer_before_v2_training_eval_result",
        "question": "Does the one preselected original-family decoder calibration transfer to the three fixed v2-trained model/decoder arms?",
        "fixed_calibration": {"arm": SELECTED_CONFIG, "config": config,
                              "calibration_result_sha256": sha(calibration.RESULT),
                              "calibration_declaration_sha256": sha(calibration.DECLARATION)},
        "arms": list(ARMS),
        "methods": {
            ARMS[0]: "new full H/A/D emissions; pool two seeds within phase, then equal-average native phases; decode with the one fixed calibration config",
            ARMS[1]: "per seed/phase retain new conditional H/A and replace D with same-seed/fold archived calibrated D; pool seeds within phase, average phases; decode with fixed config",
            ARMS[2]: "per seed/phase average new full and archived calibrated emissions 50/50; pool seeds within phase, average phases; decode with fixed config",
        },
        "cohort": {"matches": [{"provider": p, "match": m}
                                  for p, m, _ in upstream.frozen.audit.MATCHES],
                   "views": ["clean", "feed_like_v2"], "feed_realizations": list(upstream.frozen.REPLICATES),
                   "aggregation": "realizations within match, matches within provider, providers equally",
                   "native_phases": {"25_hz": 5, "10_hz": 2},
                   "component_seeds": list(SEEDS)},
        "budget": {"new_fits": 0, "decoder_configs": 1, "candidate_arms": 3,
                   "candidate_cells": 75, "seed_diagnostic_cells": 150,
                   "total_score_diagnostic_cells": 225, "re_inference": 0},
        "metrics": {"primary": "official possession spell F1; macro F1 breaks exact ties",
                    "diagnostics": list(METRICS),
                    "component_seed_metrics": ["spell_f1", "macro_f1"],
                    "auxiliary": "0.7*spell_f1 + 0.3*macro_f1; descriptive only"},
        "common_readout_guards": {
            "baseline": "currently adopted original_full phase-TTA C0 (`cand`) from upstream preflight",
            "clean_equal_provider_spell_f1_min_delta": -0.005,
            "provider_clean_macro_f1_min_delta": -0.02,
            "provider_clean_d_f1_min_delta": -0.02,
            "feed_like_v2_equal_provider_spell_f1_min_delta": -0.01,
            "posthoc_readout_only": True,
        },
        "provenance": {
            **_source_bindings(),
            "calibration_result_sha256": sha(calibration.RESULT),
            "calibration_declaration_sha256": sha(calibration.DECLARATION),
            "training_evaluation_declaration_sha256": sha(base.DECLARATION),
            "training_evaluation_seal_sha256": sha(base.SEAL),
            "training_evaluation_runtime_correction_sha256": sha(eval_runtime.DECLARATION),
            "training_evaluation_prior_correction_receipt_sha256": sha(eval_runtime.PREVIOUS_CORRECTION),
            "training_evaluation_models": models,
            "training_evaluation_runtime_correction": correction,
            "runtime": runtime,
            "upstream_c0_preflight_sha256": sha(upstream.frozen.PREFLIGHT),
        },
        "gates": {"freeze_before_training_evaluation_result": True,
                  "require_main_evaluation_emission_cache_seal_before_transfer_scores": True,
                  "missing_emission_cache_policy": "refuse; never infer in this study",
                  "organizer_reference_access": False, "public_scoring": False,
                  "promotion": False},
        "limitations": "Same five exposed development matches and synthetic feed_like_v2 views; this is not fresh confirmation.",
    }
    _write_once(DECLARATION, declaration)
    return declaration


def _verify_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the transfer declaration first")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != list(ARMS):
        raise RuntimeError("Transfer declaration identity/arms mismatch")
    calibration_result, _, config = _calibration_bindings()
    if (sha(calibration.RESULT) != declaration["provenance"]["calibration_result_sha256"] or
            sha(calibration.DECLARATION) != declaration["provenance"]["calibration_declaration_sha256"] or
            config != declaration["fixed_calibration"]["config"]):
        raise RuntimeError("Selected calibration result/config changed")
    if _source_bindings() != {k: declaration["provenance"][k]
                              for k in _source_bindings()}:
        raise RuntimeError("Transfer source or active policy/manifest changed")
    _, eval_seal, _, models, correction = _eval_bindings()
    if (sha(base.DECLARATION) != declaration["provenance"]["training_evaluation_declaration_sha256"] or
            sha(base.SEAL) != declaration["provenance"]["training_evaluation_seal_sha256"] or
            sha(eval_runtime.DECLARATION) != declaration["provenance"]["training_evaluation_runtime_correction_sha256"] or
            sha(eval_runtime.PREVIOUS_CORRECTION) != declaration["provenance"]["training_evaluation_prior_correction_receipt_sha256"] or
            models != declaration["provenance"]["training_evaluation_models"]):
        raise RuntimeError("Six-model evaluation seal changed")
    runtime = _runtime_bindings()
    if runtime != declaration["provenance"]["runtime"] or sha(upstream.frozen.PREFLIGHT) != declaration["provenance"]["upstream_c0_preflight_sha256"]:
        raise RuntimeError("Upstream runtime/preflight binding changed")
    return declaration


def _cache_identity(provider, match, view, rep, seed, phase, phase_record, view_data):
    fold = upstream.frozen.audit.FOLD[provider]
    model_fact = base.SEAL.read_text()
    model_fact = json.loads(model_fact)["models"][f"{fold}/{seed}"]
    teacher = upstream.frozen._teacher_path(provider, seed)
    calibrated = upstream.frozen._model_path("adop", provider, seed)
    return {
        "declaration_sha256": sha(base.DECLARATION),
        "model_sha256": model_fact["sha256"],
        "teacher_sha256": sha(teacher),
        "calibrated_model_sha256": sha(calibrated),
        "input_sha256": upstream.frozen._input_sha(view_data),
        "record_sha256": base._record_sha(phase_record),
        "feature_names_sha256": base._array_sha(np.asarray(list(phase_record["names"]), dtype="U")),
        "provider": provider, "match": match, "view": view,
        "replicate": int(rep), "seed": int(seed), "phase": int(phase),
    }


def _cache_seal() -> dict:
    """Verify all complete main-run emissions without inference or scoring."""
    _, eval_seal, _ = base._require_seal()
    if not base.RESULT.is_file():
        raise RuntimeError("Wait until v2_training_eval has completed its frozen run")
    main_result = json.loads(base.RESULT.read_text())
    if (main_result.get("study_id") != base.STUDY or
            main_result.get("declaration_sha256") != sha(base.DECLARATION) or
            main_result.get("evaluation_seal_sha256") != sha(base.SEAL) or
            main_result.get("candidate_arms") != list(ARMS) or
            main_result.get("candidate_cells") != 25):
        raise RuntimeError("Main evaluation result is incomplete or bound to different frozen inputs")
    facts = {}
    for provider, match, _ in upstream.frozen.audit.MATCHES:
        for view_name, reps in VIEWS:
            for rep in reps:
                view_data, _, _, phases, _ = upstream.frozen._prepare_cell(provider, match, view_name, rep)
                for phase, phase_record in enumerate(phases):
                    for seed in SEEDS:
                        path = base._emission_path(provider, match, view_name, rep, seed, phase)
                        if not path.is_file():
                            raise FileNotFoundError(f"Refusing inference for missing main-run emission: {path}")
                        identity = _cache_identity(provider, match, view_name, rep, seed,
                                                  phase, phase_record, view_data)
                        with np.load(path, allow_pickle=False) as saved:
                            actual = {key: str(saved[key].item()) for key in identity}
                            if actual != identity:
                                raise RuntimeError(f"Main-run emission identity mismatch: {path}")
                            new_full, calibrated = saved["new_full"], saved["calibrated"]
                            if (str(saved["new_full_sha256"].item()) != base._array_sha(new_full) or
                                    str(saved["calibrated_sha256"].item()) != base._array_sha(calibrated)):
                                raise RuntimeError(f"Main-run emission array hash mismatch: {path}")
                            for name, values in (("new_full", new_full), ("calibrated", calibrated)):
                                if (values.ndim != 2 or values.shape[1] != 3 or
                                        not np.isfinite(values).all() or (values < 0).any() or
                                        not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)):
                                    raise RuntimeError(f"Invalid {name} emission: {path}")
                        # The archived member must be byte-identical to its upstream C0 emission.
                        old_path = upstream.frozen._cache_path(
                            provider, match, view_name, rep, "adop", phase, seed)
                        with np.load(old_path, allow_pickle=False) as old_saved:
                            if not np.array_equal(old_saved["probabilities"], calibrated):
                                raise RuntimeError(f"Calibrated emission differs from sealed upstream cache: {path}")
                        rel = str(path.relative_to(ROOT))
                        facts[rel] = {"sha256": sha(path), "new_full_sha256": base._array_sha(new_full),
                                      "calibrated_sha256": base._array_sha(calibrated),
                                      "frames": int(len(new_full)), "provider": provider,
                                      "match": match, "view": view_name, "replicate": int(rep),
                                      "seed": int(seed), "phase": int(phase)}
    expected = sum(len(upstream.frozen._prepare_cell(p, m, v, r)[3]) * len(SEEDS)
                   for p, m, _ in upstream.frozen.audit.MATCHES
                   for v, reps in VIEWS for r in reps)
    if len(facts) != expected:
        raise RuntimeError(f"Incomplete new-model emissions: {len(facts)} != {expected}")
    return {"main_evaluation_result_sha256": sha(base.RESULT),
            "main_evaluation_seal_sha256": sha(base.SEAL),
            "emission_count": len(facts), "emissions": facts,
            "main_result_cells": main_result["candidate_cells"],
            "main_result_arms": main_result["candidate_arms"]}


def seal() -> dict:
    declaration = _verify_declaration()
    if SEAL.exists() or RESULT.exists():
        raise RuntimeError("Transfer seal/result already exists")
    cache_facts = _cache_seal()
    seal_value = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
                  "cache_facts": cache_facts,
                  "sealed_before_transfer_scoring": True,
                  "organizer_reference_access": False, "public_scoring": False}
    _write_once(SEAL, seal_value)
    return seal_value


def _require_seal():
    declaration = _verify_declaration()
    if not SEAL.is_file():
        raise RuntimeError("Seal the complete main-run emission cache before transfer scoring")
    seal_value = json.loads(SEAL.read_text())
    if (seal_value.get("study_id") != STUDY or
            seal_value.get("declaration_sha256") != sha(DECLARATION)):
        raise RuntimeError("Transfer cache seal does not bind declaration")
    for rel, fact in seal_value["cache_facts"]["emissions"].items():
        if sha(ROOT / rel) != fact["sha256"]:
            raise RuntimeError(f"Sealed main-run cache changed: {rel}")
    if sha(base.RESULT) != seal_value["cache_facts"]["main_evaluation_result_sha256"]:
        raise RuntimeError("Main evaluation result changed after cache seal")
    return declaration, seal_value


def _read_emission(provider, match, view, rep, seed, phase, phase_record, view_data, seal_value):
    path = base._emission_path(provider, match, view, rep, seed, phase)
    rel = str(path.relative_to(ROOT))
    fact = seal_value["cache_facts"]["emissions"].get(rel)
    if not fact or sha(path) != fact["sha256"]:
        raise RuntimeError(f"Missing or changed sealed emission: {path}")
    identity = _cache_identity(provider, match, view, rep, seed, phase, phase_record, view_data)
    with np.load(path, allow_pickle=False) as saved:
        if {key: str(saved[key].item()) for key in identity} != identity:
            raise RuntimeError(f"Emission cache identity changed: {path}")
        new_full, calibrated = saved["new_full"], saved["calibrated"]
        if (str(saved["new_full_sha256"].item()) != base._array_sha(new_full) or
                str(saved["calibrated_sha256"].item()) != base._array_sha(calibrated)):
            raise RuntimeError(f"Emission cache arrays changed: {path}")
    return new_full, calibrated


def _score_prediction(cell, probabilities, config):
    prediction = upstream.frozen.audit.decode(probabilities, cell["model_period"],
                                              cell["fps"], config)
    return upstream.frozen._score_cell(cell, prediction)


def _score_cell(provider, match, view, rep, config, seal_value):
    view_data, labels, _, phases, period = upstream.frozen._prepare_cell(provider, match, view, rep)
    cell = {"labels": labels, "period": view_data["period"],
            "timestamps": view_data["timestamps"], "fps": float(view_data["fps"]),
            "model_period": period}
    members = {arm: [] for arm in ARMS}
    for phase, phase_record in enumerate(phases):
        for seed in SEEDS:
            new_full, calibrated = _read_emission(
                provider, match, view, rep, seed, phase, phase_record, view_data, seal_value)
            members[ARMS[0]].append((phase, seed, new_full))
            members[ARMS[1]].append((phase, seed,
                base.cross_conditional_ha(new_full, calibrated[:, 2])))
            members[ARMS[2]].append((phase, seed, 0.5 * (new_full + calibrated)))
    scored = {}
    for arm in ARMS:
        phase_means = []
        seed_phase_means = {str(seed): [] for seed in SEEDS}
        for phase in range(len(phases)):
            phase_members = [p for phase_i, _, p in members[arm] if phase_i == phase]
            if len(phase_members) != len(SEEDS):
                raise RuntimeError(f"Incomplete seed/phase ensemble for {arm}/{phase}")
            phase_means.append(np.mean(phase_members, axis=0))
            for seed in SEEDS:
                seed_members = [p for phase_i, seed_i, p in members[arm]
                                if phase_i == phase and seed_i == seed]
                if len(seed_members) != 1:
                    raise RuntimeError(f"Incomplete seed diagnostic for {arm}/{phase}/{seed}")
                seed_phase_means[str(seed)].append(seed_members[0])
        pooled = np.mean(phase_means, axis=0)
        scored[arm] = {"pooled": _score_prediction(cell, pooled, config),
                       "component_seeds": {
                           seed: _score_prediction(cell, np.mean(phases_for_seed, axis=0), config)
                           for seed, phases_for_seed in seed_phase_means.items()}}
    return scored


def _aggregate(cell_scores, use_seed=None):
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for view, reps in VIEWS:
            rows = {key: (row[arm]["pooled"] if use_seed is None
                          else row[arm]["component_seeds"][str(use_seed)])
                    for key, row in cell_scores.items() if key[2] == view and key[3] in reps}
            if len(rows) != len(upstream.frozen.audit.MATCHES) * len(reps):
                raise RuntimeError(f"Incomplete {view} aggregate for {arm}")
            per_view = {}
            providers = {}
            for metric in METRICS if use_seed is None else ("spell_f1", "macro_f1"):
                values = {(provider, match, str(rep)): score[metric]
                          for (provider, match, _, rep), score in rows.items()}
                mean, per_provider = upstream.frozen.audit.equal_provider(values)
                per_view[metric] = mean
                providers[metric] = per_provider
            if use_seed is None:
                per_view["auxiliary_70_30"] = 0.7 * per_view["spell_f1"] + 0.3 * per_view["macro_f1"]
                providers["auxiliary_70_30"] = {p: 0.7 * providers["spell_f1"][p] +
                                                0.3 * providers["macro_f1"][p]
                                                for p in providers["spell_f1"]}
            output[arm][view] = {"equal_provider": per_view, "providers": providers,
                                 "cells": len(rows)}
    return output


def _reused_adopted_c0():
    preflight = json.loads(upstream.frozen.PREFLIGHT.read_text())
    rows = {}
    for key, cell in preflight["controls"].items():
        p, match, view, rep = key.split("__")
        rows[(p, match, view, int(rep))] = cell["cand"]["C0"]
    baseline = {}
    for view, reps in VIEWS:
        selected = {key: row for key, row in rows.items() if key[2] == view and key[3] in reps}
        metrics, providers = {}, {}
        for metric in METRICS:
            mean, per_provider = upstream.frozen.audit.equal_provider({
                (p, m, str(r)): row[metric] for (p, m, _, r), row in selected.items()})
            metrics[metric] = mean
            providers[metric] = per_provider
        baseline[view] = {"equal_provider": metrics, "providers": providers,
                           "cells": len(selected), "source": "upstream C0 preflight; reused"}
    return baseline


def _guardrails(aggregate, baseline):
    rows = {}
    for arm in ARMS:
        checks = {}
        for metric in ("spell_f1",):
            actual = aggregate[arm]["clean"]["equal_provider"][metric]
            ref = baseline["clean"]["equal_provider"][metric]
            minimum = ref - 0.005
            checks[f"clean_equal_provider/{metric}"] = {"value": actual, "baseline": ref,
                "minimum": minimum, "delta": actual-ref, "passed": actual >= minimum}
        for provider in aggregate[arm]["clean"]["providers"]["macro_f1"]:
            for metric in ("macro_f1", "d_f1"):
                actual = aggregate[arm]["clean"]["providers"][metric][provider]
                ref = baseline["clean"]["providers"][metric][provider]
                minimum = ref - 0.02
                checks[f"clean_provider/{provider}/{metric}"] = {"value": actual, "baseline": ref,
                    "minimum": minimum, "delta": actual-ref, "passed": actual >= minimum}
        actual = aggregate[arm]["feed_like_v2"]["equal_provider"]["spell_f1"]
        ref = baseline["feed_like_v2"]["equal_provider"]["spell_f1"]
        minimum = ref - 0.01
        checks["feed_like_v2_equal_provider/spell_f1"] = {"value": actual, "baseline": ref,
            "minimum": minimum, "delta": actual-ref, "passed": actual >= minimum}
        rows[arm] = {"checks": checks, "all_passed": all(v["passed"] for v in checks.values()),
                     "promotion": False}
    return rows


def run() -> dict:
    declaration, seal_value = _require_seal()
    if RESULT.exists():
        raise RuntimeError("Transfer result already exists")
    cell_scores = {}
    for provider, match, _ in upstream.frozen.audit.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                key = (provider, match, view, int(rep))
                cell_scores[key] = _score_cell(provider, match, view, int(rep),
                                               declaration["fixed_calibration"]["config"], seal_value)
                print(f"TRANSFER SCORED {provider}/{match}/{view}/{rep}", flush=True)
    for cell in cell_scores.values():
        counts = {cell[arm]["pooled"]["reference_spell_count"] for arm in ARMS}
        if len(counts) != 1:
            raise RuntimeError("Reference spell count changed across transfer arms")
    aggregate = _aggregate(cell_scores)
    component = {str(seed): _aggregate(cell_scores, use_seed=seed) for seed in SEEDS}
    baseline = _reused_adopted_c0()
    guardrails = _guardrails(aggregate, baseline)
    ranked = sorted(ARMS, key=lambda arm: (aggregate[arm]["feed_like_v2"]["equal_provider"]["spell_f1"],
                                           aggregate[arm]["feed_like_v2"]["equal_provider"]["macro_f1"]),
                   reverse=True)
    out = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
           "transfer_seal_sha256": sha(SEAL),
           "calibration_result_sha256": declaration["provenance"]["calibration_result_sha256"],
           "main_evaluation_result_sha256": seal_value["cache_facts"]["main_evaluation_result_sha256"],
           "selected_config": declaration["fixed_calibration"],
           "candidate_aggregate": aggregate, "component_seed_diagnostics": component,
           "reused_current_adopted_C0": baseline,
           "common_guardrails_vs_current_adopted_C0": guardrails,
           "ranked_by_feed_like_v2_spell_then_macro": ranked,
           "highest_numerical_arm": ranked[0],
           "highest_eligible_arm": next((arm for arm in ranked if guardrails[arm]["all_passed"]), None),
           "candidate_cells": len(cell_scores), "candidate_arms": list(ARMS),
           "cell_scores": {"|".join(map(str, key)): value for key, value in cell_scores.items()},
           "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    _write_once(RESULT, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "seal", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "seal": seal, "run": run}[action]()


if __name__ == "__main__":
    main()
