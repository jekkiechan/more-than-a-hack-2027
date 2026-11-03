"""Frozen development evaluation for the four-view v2-trained full model.

The runner has three explicit gates: ``freeze`` records the fixed arms and
cohort before fitting; ``seal`` verifies all six provider-fold fits and the
upstream controls-only C0 preflight; ``run`` then evaluates the candidates on
the five exposed development matches (clean plus four fixed feed_like_v2
realizations). Organizer labels and public scoring are outside this study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_short_original_diversity as training
from experiments import v2_spell_target_runtime as upstream
from experiments import soccernet_extension as old
from experiments import sn_dense_train as sn


ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_training_eval"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
SEAL = ROOT / f"data/experiments/{STUDY}/evaluation_seal.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
FOLDS = tuple(training.FOLDS)
SEEDS = tuple(training.SEEDS)
ARMS = (
    "new_full_phase_tta",
    "new_conditional_HA_x_archived_calibrated_D",
    "equal_50_50_new_full_plus_archived_calibrated",
)
VIEWS = (("clean", (0,)), ("feed_like_v2", tuple(upstream.frozen.REPLICATES)))
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
           "predicted_spell_count", "reference_spell_count",
           "short_spell_recall_iou_0_5_lt3s")


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


def _fit_path(fold: str, seed: int) -> Path:
    return training.OUT / "models" / f"{fold}_{seed}.joblib"


def _fit_receipt_path(fold: str, seed: int) -> Path:
    return Path(str(_fit_path(fold, seed)) + ".json")


def _archived_cache_key(provider: str, match: str, view_name: str, rep: int,
                        seed: int, phase: int) -> str:
    return f"{provider}__{match}__{view_name}__{rep}__p{phase}__s{seed}"


def _seal_archived_calibrated_emissions() -> dict[str, dict]:
    """Bind and validate cached adop emissions produced by the C0 preflight."""
    frozen = upstream.frozen
    facts = {}
    for provider, match, _ in frozen.audit.MATCHES:
        for view_name, reps in VIEWS:
            for rep in reps:
                view, _, _, phases, _ = frozen._prepare_cell(provider, match, view_name, rep)
                for phase, phase_record in enumerate(phases):
                    for seed in SEEDS:
                        path = frozen._cache_path(provider, match, view_name, rep,
                                                   "adop", phase, seed)
                        if not path.is_file():
                            raise FileNotFoundError(
                                f"C0 preflight emission cache is missing: {path}")
                        # Existing-only assertion: _load_or_infer must validate and reuse
                        # the preflight cache; this stage must never run archived inference.
                        frozen._load_or_infer(view, phase_record, provider, match,
                                              view_name, rep, "adop", phase, seed)
                        key = _archived_cache_key(provider, match, view_name, rep,
                                                  seed, phase)
                        facts[key] = {"path": str(path.relative_to(ROOT)),
                                      "sha256": sha(path)}
    return facts


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__), Path(training.__file__), Path(upstream.__file__),
        Path(upstream.frozen.__file__), Path(upstream.frozen.audit.__file__),
        Path(upstream.frozen.sn.__file__), Path(upstream.frozen.old.__file__),
        Path(upstream.frozen.audit.decoder.__file__),
        Path(upstream.frozen.CALIBRATION), Path(upstream.frozen.SCORECARD),
        Path(upstream.frozen.BENCHMARK), Path(ROOT / "evaluation/possession_metrics.py"),
    )
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
            for path in paths}


def _upstream_bindings() -> dict:
    upstream.frozen._roots()
    if not upstream.frozen.DECLARATION.is_file():
        raise FileNotFoundError("Freeze v2_spell_target_runtime before freezing this evaluation")
    declaration = json.loads(upstream.frozen.DECLARATION.read_text())
    if declaration.get("study_id") != upstream.STUDY:
        raise ValueError("Unexpected upstream v2 runtime declaration")
    upstream.frozen._verify_frozen_bindings(declaration)
    return {
        "study_id": upstream.STUDY,
        "declaration_path": str(upstream.frozen.DECLARATION.relative_to(ROOT)),
        "declaration_sha256": sha(upstream.frozen.DECLARATION),
        "prepared_cache_root": str(upstream.frozen.OUT.relative_to(ROOT)),
    }


def freeze() -> dict:
    """Freeze the three candidate arms, inputs, controls, cohort and sources."""
    upstream.frozen._roots()
    if DECLARATION.exists() or SEAL.exists() or RESULT.exists():
        raise RuntimeError("Evaluation declaration/seal/result already exists; study ID is immutable")
    train_decl = training.DECLARATION
    if not train_decl.is_file():
        raise FileNotFoundError("Freeze v2_short_original_diversity before this evaluation")
    training_decl = training.check()
    if training_decl.get("fits") != 6 or list(training_decl.get("folds", ())) != list(FOLDS) \
            or list(training_decl.get("seeds", ())) != list(SEEDS):
        raise ValueError("Training declaration must bind exactly the six fixed fold/seed fits")
    runtime = _upstream_bindings()
    audit = upstream.frozen.audit
    declaration = {
        "study_id": STUDY,
        "status": "frozen_arms_before_fit_scoring",
        "question": "Do four-view v2-trained full H/A/D fits reach or exceed 0.60 equal-provider official spell F1 on fixed feed_like_v2 development views, and how do conditional-HA transfer and a calibrated-model blend compare?",
        "arms": list(ARMS),
        "candidate_methods": {
            ARMS[0]: "For each held-out-provider fold, ensemble the two new full H/A/D fits within each fixed native sampling phase; equal-mean phases (five at 25 Hz, two at 10 Hz); decode once with frozen f062.",
            ARMS[1]: "For each seed and phase, retain the new model's conditional H/A ratio and replace its D probability with that seed/fold's archived calibrated-D probability; then ensemble seeds, equal-mean phases and decode with frozen f062.",
            ARMS[2]: "For each seed and phase, average new full-model and archived calibrated-model probabilities 50/50; then ensemble seeds, equal-mean phases and decode with frozen f062.",
        },
        "controls": {
            "original_full_phase_tta": "Reused upstream v2_spell_target_runtime controls-only C0 for cand/original_full: equal mean across native phases then f062.",
            "calibrated_phase_tta": "Reused upstream v2_spell_target_runtime controls-only C0 for adop/calibrated D2: equal mean across native phases then f062.",
            "reuse_only": True,
        },
        "cohort": {
            "matches": [{"provider": provider, "match": match}
                        for provider, match, _ in audit.MATCHES],
            "views": ["clean", "feed_like_v2"],
            "feed_realizations": list(upstream.frozen.REPLICATES),
            "v2_seed_rule": "feed_like_v2_benchmark/{rep}/{match} SHA-256-derived seed, exactly as upstream prepared-cell runtime",
            "aggregation": "four v2 realizations within match; matches equally within provider; three provider means equally",
            "heldout_rule": "each candidate fit and both archived controls omit the evaluated provider from all learned components and normalization",
        },
        "budget": {"new_fits": 0, "new_decoder_settings": 0,
                   "candidate_scored_cells": 75, "cells_per_arm": 25,
                   "component_seeds": list(SEEDS), "phases": {"25_hz": 5, "10_hz": 2}},
        "metrics": {"primary": "official possession spell F1; macro F1 breaks exact ties",
                     "target": 0.60, "target_is_exploratory": True,
                     "diagnostics": list(METRICS),
                     "component_seed_diagnostics": {
                         "all_arms": list(ARMS),
                         "metrics": ["spell_f1", "macro_f1"],
                         "aggregation": "save each seed's score in every cell; average realizations within match, matches within provider, then providers equally; report provider values separately",
                         "selection_use": "descriptive only; no seed selection or reselection",
                     },
                     "provider_reporting": "retain provider-level clean and feed_like_v2 spell/macro/D metrics; no adoption decision in runner"},
        "auxiliary_metric": "0.7*spell_f1 + 0.3*macro_f1, descriptive only and never a ranking metric",
        "guardrails": {
            "baseline": "reused calibrated_phase_tta C0 (adop) from the upstream controls-only preflight",
            "clean_equal_provider": "for spell_f1, macro_f1 and d_f1, each candidate arm must be at least baseline - 0.005",
            "clean_provider": "for each provider, macro_f1 and d_f1 must be at least baseline - 0.02",
            "feed_like_v2_equal_provider": "spell_f1 must be at least baseline - 0.01",
            "interpretation": "report each guard; no promotion or adoption decision",
        },
        "gates": {
            "freeze_before_fit": True,
            "require_all_six_fit_receipts_and_model_hashes_before_scoring": True,
            "require_upstream_controls_only_C0_preflight_pass": True,
            "require_upstream_C0_controls_are_reused_and_not_rescored": True,
            "organizer_reference_access": False,
            "public_scoring": False,
        },
        "provenance": {
            "training_declaration": {"path": str(train_decl.relative_to(ROOT)), "sha256": sha(train_decl)},
            "upstream_runtime": runtime,
            "baseline_manifest_sha256": sha(ROOT / "experiments/baseline_manifest.json"),
            "decision_policy_sha256": sha(ROOT / "experiments/model_decision_policy.json"),
            "source_sha256": _source_hashes(),
        },
        "limitations": "Five exposed development matches and synthetic feed_like_v2 views; this cannot establish independent continuous-match generalization.",
        "promotion": False,
    }
    _write_once(DECLARATION, declaration)
    return declaration


def _require_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError(f"Freeze evaluation arms first: {DECLARATION}")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != list(ARMS):
        raise ValueError("Evaluation declaration identity or arm list mismatch")
    if declaration["provenance"]["source_sha256"] != _source_hashes():
        raise RuntimeError("Frozen evaluation source changed")
    if declaration["provenance"]["baseline_manifest_sha256"] != sha(ROOT / "experiments/baseline_manifest.json"):
        raise RuntimeError("Active model manifest changed")
    if declaration["provenance"]["decision_policy_sha256"] != sha(ROOT / "experiments/model_decision_policy.json"):
        raise RuntimeError("Decision policy changed")
    train_decl = training.DECLARATION
    if sha(train_decl) != declaration["provenance"]["training_declaration"]["sha256"]:
        raise RuntimeError("Training declaration differs from the evaluation freeze")
    training.check()
    runtime = _upstream_bindings()
    if runtime != declaration["provenance"]["upstream_runtime"]:
        raise RuntimeError("Upstream v2 runtime declaration changed")
    return declaration


def seal() -> dict:
    """Bind six finished models plus the passing upstream C0 preflight."""
    declaration = _require_declaration()
    if SEAL.exists() or RESULT.exists():
        raise RuntimeError("Evaluation seal/result already exists; study output is immutable")
    expected_training_sha = declaration["provenance"]["training_declaration"]["sha256"]
    models = {}
    for fold in FOLDS:
        for seed in SEEDS:
            key = f"{fold}/{seed}"
            model_path = _fit_path(fold, seed)
            receipt_path = _fit_receipt_path(fold, seed)
            if not model_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(f"Missing sealed fit or receipt: {model_path}")
            receipt = json.loads(receipt_path.read_text())
            expected = {"study": training.STUDY, "fold": fold, "seed": seed,
                        "sha256": sha(model_path),
                        "declaration_sha256": expected_training_sha}
            if any(receipt.get(field) != value for field, value in expected.items()):
                raise RuntimeError(f"Fit receipt mismatch for {key}")
            models[key] = {"path": str(model_path.relative_to(ROOT)),
                           "sha256": expected["sha256"], "receipt_sha256": sha(receipt_path),
                           "receipt": str(receipt_path.relative_to(ROOT))}
    upstream.frozen._roots()
    preflight_path = upstream.frozen.PREFLIGHT
    if not preflight_path.is_file():
        raise FileNotFoundError("Run upstream C0-only preflight before sealing evaluation")
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("study_id") != upstream.STUDY
            or preflight.get("declaration_sha256") != declaration["provenance"]["upstream_runtime"]["declaration_sha256"]
            or preflight.get("preflight_passed") is not True
            or preflight.get("candidate_scores_written") is not False):
        raise RuntimeError("Upstream C0-only preflight did not pass cleanly")
    required_cells = {f"{p}__{m}__{view}__{rep}"
                      for p, m, _ in upstream.frozen.audit.MATCHES
                      for view, reps in VIEWS for rep in reps}
    if set(preflight.get("controls", {})) != required_cells:
        raise RuntimeError("Upstream preflight control-cell inventory is incomplete or unexpected")
    for cell in preflight["controls"].values():
        for family in upstream.frozen.FAMILIES:
            if "C0" not in cell.get(family, {}):
                raise RuntimeError("Upstream preflight is missing a required C0 control")
    archived_emissions = _seal_archived_calibrated_emissions()
    seal_value = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
                  "training_declaration_sha256": expected_training_sha,
                  "models": models,
                  "upstream_preflight": {"path": str(preflight_path.relative_to(ROOT)),
                                         "sha256": sha(preflight_path),
                                         "passed": True, "controls_only": True},
                  "archived_calibrated_emissions": archived_emissions,
                  "control_reuse": ["cand C0 = original_full_phase_tta",
                                    "adop C0 = calibrated_phase_tta"],
                  "sealed_before_candidate_scoring": True,
                  "organizer_reference_access": False, "public_scoring": False}
    _write_once(SEAL, seal_value)
    return seal_value


def _require_seal() -> tuple[dict, dict, dict]:
    declaration = _require_declaration()
    if not SEAL.is_file():
        raise RuntimeError("Seal all six fits and the upstream C0 preflight before scoring")
    seal_value = json.loads(SEAL.read_text())
    if seal_value.get("study_id") != STUDY or seal_value.get("declaration_sha256") != sha(DECLARATION):
        raise RuntimeError("Evaluation seal does not bind the frozen declaration")
    for fold in FOLDS:
        for seed in SEEDS:
            key = f"{fold}/{seed}"
            fact = seal_value["models"].get(key)
            if not fact:
                raise RuntimeError(f"Evaluation model seal missing {key}")
            path = ROOT / fact["path"]
            receipt = ROOT / fact["receipt"]
            if sha(path) != fact["sha256"] or sha(receipt) != fact["receipt_sha256"]:
                raise RuntimeError(f"Sealed fit changed: {key}")
    preflight_path = ROOT / seal_value["upstream_preflight"]["path"]
    if sha(preflight_path) != seal_value["upstream_preflight"]["sha256"]:
        raise RuntimeError("Upstream preflight receipt changed")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("preflight_passed") is not True:
        raise RuntimeError("Upstream C0 preflight is no longer passing")
    for fact in seal_value.get("archived_calibrated_emissions", {}).values():
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Archived calibrated emission changed: {fact['path']}")
    return declaration, seal_value, preflight


def cross_conditional_ha(full_probabilities: np.ndarray,
                         calibrated_d_probabilities: np.ndarray) -> np.ndarray:
    """Transfer a calibrated D probability while preserving new conditional H/A."""
    full = np.asarray(full_probabilities, dtype=np.float64)
    calibrated_d = np.asarray(calibrated_d_probabilities, dtype=np.float64)
    if full.ndim < 2 or full.shape[-1] != 3 or calibrated_d.shape != full.shape[:-1]:
        raise ValueError("Expected (..., 3) full probabilities and matching D probabilities")
    if (not np.isfinite(full).all() or not np.isfinite(calibrated_d).all()
            or (full < 0).any() or (calibrated_d < 0).any() or (calibrated_d > 1).any()):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    ha_mass = full[..., 0] + full[..., 1]
    if (ha_mass <= 0).any():
        raise ValueError("New model must have positive H/A mass at every frame")
    ha_ratio = full[..., :2] / ha_mass[..., None]
    out = np.empty_like(full)
    out[..., :2] = ha_ratio * (1.0 - calibrated_d[..., None])
    out[..., 2] = calibrated_d
    return out


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode())
    if array.dtype.hasobject or array.dtype.kind in "US":
        digest.update(json.dumps(array.tolist(), separators=(",", ":"), default=str).encode())
    else:
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _record_sha(record: dict) -> str:
    return hashlib.sha256("".join(f"{key}:{_array_sha(np.asarray(record[key]))};"
                                   for key in sorted(record)).encode()).hexdigest()


def _emission_path(provider: str, match: str, view_name: str, rep: int,
                   seed: int, phase: int) -> Path:
    view = "clean" if view_name == "clean" else f"v2r{rep}"
    return OUT / "emissions" / f"{provider}__{match}__{view}__s{seed}__p{phase}.npz"


def _native_inputs(view: dict) -> dict:
    return upstream.frozen.tc.inputs_only(view)


def _infer_new(view: dict, phase_record: dict, provider: str, seed: int) -> np.ndarray:
    fold = upstream.frozen.audit.FOLD[provider]
    model_path = _fit_path(fold, seed)
    teacher_path = upstream.frozen._teacher_path(provider, seed)
    model = joblib.load(model_path)
    teacher = joblib.load(teacher_path)
    record = old.inference_record(phase_record)
    record["names"] = list(record["names"])
    probabilities = sn.predict(record, model, teacher)
    return old.upsample(probabilities, record, _native_inputs(view))


def _emissions(view: dict, record: dict, provider: str, match: str,
               view_name: str, rep: int, seed: int, phase: int,
               phase_record: dict, declaration_sha: str, model_sha: str,
               archived_emission_facts: dict) -> tuple[np.ndarray, np.ndarray]:
    path = _emission_path(provider, match, view_name, rep, seed, phase)
    input_sha = upstream.frozen._input_sha(view)
    record_sha = _record_sha(phase_record)
    names_sha = _array_sha(np.asarray(list(phase_record["names"]), dtype="U"))
    teacher_path = upstream.frozen._teacher_path(provider, seed)
    teacher_sha = sha(teacher_path)
    calibrated_path = upstream.frozen._model_path("adop", provider, seed)
    calibrated_sha = sha(calibrated_path)
    identity = {"declaration_sha256": declaration_sha, "model_sha256": model_sha,
                "teacher_sha256": teacher_sha, "calibrated_model_sha256": calibrated_sha,
                "input_sha256": input_sha,
                "record_sha256": record_sha, "feature_names_sha256": names_sha,
                "provider": provider, "match": match, "view": view_name,
                "replicate": int(rep), "seed": int(seed), "phase": int(phase)}
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            actual = {key: str(saved[key].item()) for key in identity}
            if actual != identity:
                raise RuntimeError(f"Candidate emission cache identity mismatch: {path}")
            new_full, calibrated = saved["new_full"], saved["calibrated"]
            if (str(saved["new_full_sha256"].item()) != _array_sha(new_full)
                    or str(saved["calibrated_sha256"].item()) != _array_sha(calibrated)):
                raise RuntimeError(f"Candidate emission cache array hash mismatch: {path}")
            return new_full, calibrated
    new_full = _infer_new(view, phase_record, provider, seed)
    cached_calibrated_path = upstream.frozen._cache_path(
        provider, match, view_name, rep, "adop", phase, seed)
    cache_key = _archived_cache_key(provider, match, view_name, rep, seed, phase)
    cache_fact = archived_emission_facts.get(cache_key)
    if (cache_fact is None or not cached_calibrated_path.is_file()
            or sha(cached_calibrated_path) != cache_fact["sha256"]):
        raise RuntimeError("Archived calibrated emission is absent or differs from its evaluation seal")
    calibrated = upstream.frozen._load_or_infer(
        view, phase_record, provider, match, view_name, rep, "adop", phase, seed)
    if new_full.shape != calibrated.shape or new_full.ndim != 2 or new_full.shape[1] != 3:
        raise ValueError("New and archived probability arrays are not aligned H/A/D Nx3 arrays")
    if not (np.allclose(new_full.sum(axis=1), 1.0, atol=1e-5)
            and np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Invalid probability normalization in candidate emissions")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, new_full=new_full, calibrated=calibrated,
                        new_full_sha256=np.asarray(_array_sha(new_full)),
                        calibrated_sha256=np.asarray(_array_sha(calibrated)),
                        **{key: np.asarray(value) for key, value in identity.items()})
    return new_full, calibrated


def _score_prediction(cell: dict, probability: np.ndarray) -> dict:
    prediction = upstream.frozen.audit.decode(probability, cell["model_period"], cell["fps"])
    return upstream.frozen._score_cell(cell, prediction)


def _score_cell(provider: str, match: str, view_name: str, rep: int,
                declaration: dict, archived_emission_facts: dict) -> dict:
    view, labels, _, phases, period = upstream.frozen._prepare_cell(provider, match, view_name, rep)
    if len(phases) not in (2, 5):
        raise ValueError(f"Expected two/two-or-five native phase records, got {len(phases)}")
    cell = {"labels": labels, "period": view["period"], "timestamps": view["timestamps"],
            "fps": float(view["fps"]), "model_period": period}
    by_arm = {arm: [] for arm in ARMS}
    for phase, phase_record in enumerate(phases):
        for seed in SEEDS:
            fold = upstream.frozen.audit.FOLD[provider]
            model_sha = sha(_fit_path(fold, seed))
            new_full, calibrated = _emissions(
                view, phase_record, provider, match, view_name, rep, seed, phase,
                phase_record, sha(DECLARATION), model_sha, archived_emission_facts)
            by_arm[ARMS[0]].append((phase, seed, new_full))
            by_arm[ARMS[1]].append((phase, seed,
                cross_conditional_ha(new_full, calibrated[:, 2])))
            by_arm[ARMS[2]].append((phase, seed, (new_full + calibrated) / 2.0))
    scored = {}
    for arm in ARMS:
        phase_means = []
        for phase in range(len(phases)):
            members = [prob for p, _, prob in by_arm[arm] if p == phase]
            if len(members) != len(SEEDS):
                raise RuntimeError(f"Incomplete seed ensemble for {arm}, phase {phase}")
            phase_means.append(np.mean(members, axis=0))
        pooled = np.mean(phase_means, axis=0)
        seed_scores = {}
        for seed in SEEDS:
            seed_phase_means = []
            for phase in range(len(phases)):
                member = [prob for p, s, prob in by_arm[arm] if p == phase and s == seed]
                if len(member) != 1:
                    raise RuntimeError(f"Incomplete component seed for {arm}, phase {phase}, seed {seed}")
                seed_phase_means.append(member[0])
            seed_scores[str(seed)] = _score_prediction(cell, np.mean(seed_phase_means, axis=0))
        scored[arm] = {"pooled": _score_prediction(cell, pooled),
                       "component_seeds": seed_scores}
    return scored


def _aggregate(cell_scores: dict) -> dict:
    audit = upstream.frozen.audit
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for view_name, reps in VIEWS:
            rows = {key: row[arm]["pooled"] for key, row in cell_scores.items()
                    if key[2] == view_name and key[3] in reps}
            expected = len(audit.MATCHES) * len(reps)
            if len(rows) != expected:
                raise RuntimeError(f"Incomplete {view_name} cohort for {arm}: {len(rows)} != {expected}")
            metrics = {}
            provider_metrics = {}
            for metric in METRICS:
                values = {(provider, match, str(rep)): score[metric]
                          for (provider, match, view, rep), score in rows.items()}
                mean, per_provider = audit.equal_provider(values)
                metrics[metric] = mean
                provider_metrics[metric] = per_provider
            metrics["auxiliary_70_30"] = 0.7 * metrics["spell_f1"] + 0.3 * metrics["macro_f1"]
            provider_metrics["auxiliary_70_30"] = {
                provider: 0.7 * provider_metrics["spell_f1"][provider]
                + 0.3 * provider_metrics["macro_f1"][provider]
                for provider in provider_metrics["spell_f1"]}
            result[arm][view_name] = {"equal_provider": metrics,
                                      "providers": provider_metrics,
                                      "cells": len(rows)}
    return result


def run() -> dict:
    declaration, seal_value, preflight = _require_seal()
    if RESULT.exists():
        raise RuntimeError("Evaluation result already exists; study scoring is one-shot")
    controls = preflight["controls"]
    cell_scores = {}
    for provider, match, _ in upstream.frozen.audit.MATCHES:
        for view_name, reps in VIEWS:
            for rep in reps:
                key = (provider, match, view_name, int(rep))
                cell_scores[key] = _score_cell(
                    provider, match, view_name, int(rep), declaration,
                    seal_value["archived_calibrated_emissions"])
                print(f"SCORED {provider}/{match}/{view_name}/{rep}", flush=True)
    aggregate = _aggregate(cell_scores)
    component_aggregate = {}
    for arm in ARMS:
        component_aggregate[arm] = {}
        for seed in SEEDS:
            component_aggregate[arm][str(seed)] = {}
            for view_name, reps in VIEWS:
                rows = {key: score[arm]["component_seeds"][str(seed)]
                        for key, score in cell_scores.items()
                        if key[2] == view_name and key[3] in reps}
                provider_metrics = {}
                equal_provider_metrics = {}
                for metric in ("spell_f1", "macro_f1"):
                    values = {(provider, match, str(rep)): score[metric]
                              for (provider, match, view, rep), score in rows.items()}
                    mean, per_provider = upstream.frozen.audit.equal_provider(values)
                    equal_provider_metrics[metric] = mean
                    provider_metrics[metric] = per_provider
                component_aggregate[arm][str(seed)][view_name] = {
                    "equal_provider": equal_provider_metrics,
                    "providers": provider_metrics, "cells": len(rows)}
    control_map = {"original_full_phase_tta": "cand", "calibrated_phase_tta": "adop"}
    control_rows = {}
    for control_name, family in control_map.items():
        control_rows[control_name] = {}
        for view_name, reps in VIEWS:
            rows = {key: controls[f"{key[0]}__{key[1]}__{key[2]}__{key[3]}"][family]["C0"]
                    for key in cell_scores if key[2] == view_name and key[3] in reps}
            if len(rows) != len(upstream.frozen.audit.MATCHES) * len(reps):
                raise RuntimeError(f"Incomplete reused control cohort: {control_name}/{view_name}")
            metrics = {}
            provider_metrics = {}
            for metric in METRICS:
                values = {(provider, match, str(rep)): score[metric]
                          for (provider, match, view, rep), score in rows.items()}
                mean, per_provider = upstream.frozen.audit.equal_provider(values)
                metrics[metric] = mean
                provider_metrics[metric] = per_provider
            metrics["auxiliary_70_30"] = 0.7 * metrics["spell_f1"] + 0.3 * metrics["macro_f1"]
            provider_metrics["auxiliary_70_30"] = {
                provider: 0.7 * provider_metrics["spell_f1"][provider]
                + 0.3 * provider_metrics["macro_f1"][provider]
                for provider in provider_metrics["spell_f1"]}
            control_rows[control_name][view_name] = {
                "equal_provider": metrics, "providers": provider_metrics,
                "cells": len(rows), "source": "upstream C0 preflight receipt; not rescored"}
    comparisons = {}
    for arm in ARMS:
        comparisons[arm] = {}
        for control_name, control_views in control_rows.items():
            comparisons[arm][control_name] = {
                view: {metric: aggregate[arm][view]["equal_provider"][metric]
                       - control_views[view]["equal_provider"][metric]
                       for metric in ("spell_f1", "macro_f1", "d_f1", "auxiliary_70_30")}
                for view, _ in VIEWS
            }
    adopted = control_rows["calibrated_phase_tta"]
    guardrails = {}
    for arm in ARMS:
        clean_actual = aggregate[arm]["clean"]["equal_provider"]
        clean_baseline = adopted["clean"]["equal_provider"]
        v2_actual = aggregate[arm]["feed_like_v2"]["equal_provider"]
        v2_baseline = adopted["feed_like_v2"]["equal_provider"]
        clean_provider_actual = aggregate[arm]["clean"]["providers"]
        clean_provider_baseline = adopted["clean"]["providers"]
        checks = {}
        for metric in ("spell_f1", "macro_f1", "d_f1"):
            threshold = clean_baseline[metric] - 0.005
            checks[f"clean_equal_provider/{metric}"] = {
                "value": clean_actual[metric], "baseline": clean_baseline[metric],
                "minimum": threshold, "passed": clean_actual[metric] >= threshold}
        for provider in clean_provider_actual["macro_f1"]:
            for metric in ("macro_f1", "d_f1"):
                actual = clean_provider_actual[metric][provider]
                baseline_value = clean_provider_baseline[metric][provider]
                threshold = baseline_value - 0.02
                checks[f"clean_provider/{provider}/{metric}"] = {
                    "value": actual, "baseline": baseline_value,
                    "minimum": threshold, "passed": actual >= threshold}
        threshold = v2_baseline["spell_f1"] - 0.01
        checks["feed_like_v2_equal_provider/spell_f1"] = {
            "value": v2_actual["spell_f1"], "baseline": v2_baseline["spell_f1"],
            "minimum": threshold, "passed": v2_actual["spell_f1"] >= threshold}
        guardrails[arm] = {"checks": checks,
                           "all_passed": all(row["passed"] for row in checks.values()),
                           "promotion": False}
    out = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
           "evaluation_seal_sha256": sha(SEAL),
           "upstream_preflight_sha256": seal_value["upstream_preflight"]["sha256"],
           "candidate_aggregate": aggregate,
           "component_seed_diagnostics": component_aggregate,
           "reused_controls": control_rows,
           "delta_vs_reused_controls": comparisons,
           "guardrails_vs_calibrated_phase_tta": guardrails,
           "candidate_cells": len(cell_scores), "candidate_arms": list(ARMS),
           "target_spell_f1": 0.60,
           "target_status": {arm: {view: {"attained": values["equal_provider"]["spell_f1"] >= 0.60}
                                    for view, values in aggregate[arm].items()} for arm in ARMS},
           "cell_scores": {"|".join(map(str, key)): value for key, value in cell_scores.items()},
           "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    _write_once(RESULT, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "seal", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "seal": seal, "run": run}[action]()


if __name__ == "__main__":
    main()
