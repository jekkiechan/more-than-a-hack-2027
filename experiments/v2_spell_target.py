"""Bounded phase-decoder consensus study for feed_like_v2.

The runner compares four fixed aggregation rules for the calibrated dense and
original_full probability families, plus one fixed 50/50 phase-emission blend.
All aggregation functions accept emissions, predictions and time breaks only;
labels enter only after final predictions are produced.

Scoring is intentionally gated on a parent-frozen declaration at
experiments/results/v2_spell_target_declaration.json. This module does not
create or amend that declaration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_like_v2 as v2
from experiments import preparation as prep
from experiments import sn_dense_train as sn
from experiments import soccernet_extension as old
from experiments import stress_components as tc
from experiments import supervised


STUDY = "v2_spell_target"
ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "experiments/results/v2_spell_target_declaration.json"
OUT = ROOT / "data/experiments" / STUDY
RESULT = ROOT / "experiments/results" / f"{STUDY}.json"
PREFLIGHT = ROOT / "experiments/results" / f"{STUDY}_preflight.json"
CALIBRATION = ROOT / "experiments/results/feed_like_v2_calibration.json"
SCORECARD = ROOT / "experiments/results/feed_like_v2_scorecard.json"
BENCHMARK = ROOT / "experiments/results/feed_like_v2_benchmark.json"
REPLICATES = (1, 2, 3, 4)
SEEDS = (2026, 2027)
FAMILIES = ("adop", "cand")
CLASSES = np.asarray(("H", "A", "D"))
ARMS = tuple(f"{family}_{rule}" for family in FAMILIES
             for rule in ("mean_decode", "phase_vote", "confidence_vote", "consensus_gate")) + ("blend_mean_decode",)


def _validate_phase_inputs(probabilities, predictions, fallback):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(predictions)
    f = np.asarray(fallback)
    if (p.ndim != 3 or p.shape[0] not in (2, 5) or p.shape[2] != 3 or p.shape[1] == 0
            or y.shape != p.shape[:2] or f.shape != (p.shape[1],)
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=2), 1.0, atol=1e-5)
            or not np.isin(y, CLASSES).all() or not np.isin(f, CLASSES).all()):
        raise ValueError("Expected five aligned Nx3 probability phases and H/A/D predictions")
    return p, y.astype("U1", copy=False), f.astype("U1", copy=False)


def phase_aggregate(probabilities, predictions, fallback, rule):
    """Aggregate five aligned phase decodes without access to targets.

    `mean_decode` returns the caller-provided exact f062 decode of mean
    probabilities. Ties for the voting rules use that same decode. Confidence
    weights are fixed per-frame inverse normalized 3-class entropy.
    """
    p, y, f = _validate_phase_inputs(probabilities, predictions, fallback)
    if rule == "mean_decode":
        return f.copy()
    if rule not in {"phase_vote", "confidence_vote", "consensus_gate"}:
        raise ValueError(f"Unknown phase aggregation rule: {rule}")
    weights = np.ones(p.shape[:2], dtype=np.float64)
    if rule == "confidence_vote":
        safe = np.clip(p, 1e-12, 1.0)
        weights = 1.0 - (-(safe * np.log(safe)).sum(axis=2) / np.log(3.0))
    votes = np.stack([(y == state) for state in CLASSES], axis=2)
    weighted = (votes * weights[:, :, None]).sum(axis=0)
    chosen = CLASSES[weighted.argmax(axis=1)]
    # `argmax` ties are H-first; explicitly retain the f062 fallback on ties.
    maximum = weighted.max(axis=1)
    ties = (np.isclose(weighted, maximum[:, None], rtol=0, atol=1e-12)).sum(axis=1) > 1
    chosen[ties] = f[ties]
    if rule == "consensus_gate":
        counts = votes.sum(axis=0)
        enough = counts.max(axis=1) >= math.ceil(0.8 * p.shape[0])
        chosen[~enough] = f[~enough]
    return chosen.astype("U1", copy=False)


def _roots():
    """Apply explicit runtime archive overrides without changing frozen helpers."""
    data = Path(os.environ.get("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data")))
    sqd = Path(os.environ.get("MTAH_SQD_DIR", str(ROOT.parent / "mtah-sqd")))
    dt = Path(os.environ.get("MTAH_DT_DIR", str(ROOT.parent / "mtah-dt")))
    audit.DATA = data
    audit.EXP = data / "experiments"
    audit.EVAL_INPUTS = audit.EXP / "seed_gap_context_v1"
    audit.SQD = sqd / "experiments/short_quality_diversity_delivery_v1/artifacts"
    audit.DT = dt
    return data, sqd, dt


def _artifact_fact(manifest_path, root, relative_path):
    """Verify a restored artifact against its private archive manifest entry."""
    manifest_path, root = Path(manifest_path), Path(root)
    manifest = json.loads(manifest_path.read_text())
    facts = [f for f in manifest.get("files", []) if f.get("path") == relative_path]
    if len(facts) != 1:
        raise RuntimeError(f"Expected one manifest fact for {relative_path} in {manifest_path}")
    fact = facts[0]
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if actual != {"bytes": fact["bytes"], "sha256": fact["sha256"]}:
        raise RuntimeError(f"Restored artifact differs from manifest: {path}")
    return {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "path": str(path), "bytes": actual["bytes"], "sha256": actual["sha256"]}


def _source_hashes():
    paths = ("experiments/v2_spell_target.py", "experiments/feed_benchmark_audit_v1.py",
             "experiments/feed_like_v2.py", "experiments/feed_spell_push.py",
             "experiments/decoder_refinement.py", "experiments/decoder_search.py",
             "experiments/soccernet_extension.py", "experiments/sn_dense_train.py",
             "experiments/supervised.py", "experiments/refine_features.py",
             "experiments/learned_temporal.py", "experiments/preparation.py",
             "experiments/stress_components.py", "experiments/tracking_corruption.py",
             "experiments/spell_mbr_run.py", "experiments/metrics.py",
             "experiments/factorized_had.py", "evaluation/possession_metrics.py")
    return {p: _sha256(ROOT / p) for p in paths}


def freeze():
    """Seal the fixed protocol and verified archive inputs; never scores labels."""
    _roots()
    if DECLARATION.exists() or PREFLIGHT.exists() or RESULT.exists():
        raise RuntimeError("Declaration or result already exists; this study ID is immutable")
    sqd_manifest = audit.SQD.parent.parent.parent / "short_quality_diversity_v1_cache_manifest.json"
    dt_manifest = audit.DT / "dense_targeted_followups_cache_manifest_v1.json"
    data_manifest = audit.DATA / "seed_gap_context_cache_manifest.json"
    teacher_manifest = audit.DATA / "objective_alignment_cache_manifest.json"
    # Paths in the private manifests are archive-relative to their respective roots.
    artifact_facts = {}
    for provider in audit.FOLD:
        for seed in SEEDS:
            model = _model_path("adop", provider, seed)
            artifact_facts[f"adop/{provider}/{seed}"] = _artifact_fact(
                dt_manifest, audit.DT, str(model.relative_to(audit.DT)))
            model = _model_path("cand", provider, seed)
            archive_rel = str(model.relative_to(audit.SQD.parent.parent.parent))
            artifact_facts[f"cand/{provider}/{seed}"] = _artifact_fact(
                sqd_manifest, audit.SQD.parent.parent.parent, archive_rel)
            teacher = _teacher_path(provider, seed)
            artifact_facts[f"teacher/{provider}/{seed}"] = _artifact_fact(
                teacher_manifest, audit.DATA, str(teacher.relative_to(audit.DATA)))
    for provider, match, filename in audit.MATCHES:
        clean_rel = f"experiments/seed_gap_context_v1/{filename}_clean_clean.joblib"
        artifact_facts[f"input/{provider}/{match}"] = _artifact_fact(
            data_manifest, audit.DATA, clean_rel)
    calibration_hash = _sha256(CALIBRATION)
    source_hashes = _source_hashes()
    declaration = {
        "study_id": STUDY,
        "status": "frozen_bounded_phase_decoder_consensus_before_preflight_or_candidate_scoring",
        "question": "Does fixed per-phase decoded-state consensus improve spell F1 over the archived five-phase mean-emission f062 decode under feed_like_v2?",
        "mechanism": {
            "decoder": "frozen feed_spell_push f062 on each 5 Hz phase and on fixed phase-mean emissions",
            "C0": "equal arithmetic emission mean over the native 5 Hz phases, then one f062 decode (5 phases at 25 Hz; 2 phases at 10 Hz)",
            "C1": "per-phase f062 decoded-state plurality; exact ties use C0",
            "C2": "per-phase f062 decoded-state vote weighted by 1-normalized three-class entropy; exact ties use C0",
            "C3": "use plurality result only where at least ceil(0.8*P) native phases agree; otherwise C0 (4/5 at 25 Hz, 2/2 at 10 Hz)",
            "blend": "equal arithmetic mean of calibrated and original_full phase-mean emissions, then one f062 decode",
            "learned_parameters": 0,
        },
        "arms": list(ARMS),
        "families": {"adop": "calibrated dense D2", "cand": "original_full"},
        "cohort": {"matches": [{"provider": p, "match": m} for p, m, _ in audit.MATCHES],
                   "views": ["clean", "feed_like_v2"], "feed_realizations": list(REPLICATES),
                   "v2_seed_rule": "exact feed_like_v2_benchmark/{rep}/{match} SHA-256-derived seed",
                   "aggregation": "realizations within match, matches within provider, then providers equally",
                   "excluded_providers": "each archived model family uses its existing full-provider holdout fold"},
        "budget": {"scored_cells": 25, "candidate_arm_cells": 225,
                   "phase_emissions_per_cell": "8 for 10 Hz; 20 for 25 Hz",
                   "phases": {"25_hz": 5, "10_hz": 2}, "component_seeds": list(SEEDS), "extra_fits": 0,
                   "extra_decoder_settings": 0},
        "metrics": {"primary": "official possession spell F1; macro F1 breaks exact ties",
                    "auxiliary_70_30": "0.7*spell_f1 + 0.3*macro_f1; not ranking metric",
                    "diagnostics": ["d_f1", "d_precision", "d_recall", "predicted_spell_count",
                                    "reference_spell_count", "short_spell_recall_iou_0_5_lt3s"],
                    "guards": "record clean provider macro/D-F1 deviations and v2 provider spell deltas against matched C0; no adoption decision in runner"},
        "preflight": {"required_before_candidate_scoring": True,
                    "rule": "C0 pooled clean and v2 official spell/macro reproduce archived per-family aggregates within 1e-8; seal controls before any C1-C3 result is written",
                      "reference_scorecard_sha256": _sha256(SCORECARD),
                      "reference_benchmark_sha256": _sha256(BENCHMARK)},
        "provenance": {"organizer_reference_access": False, "public_scoring": False,
                       "active_manifest": {"path": "experiments/baseline_manifest.json",
                                           "sha256": _sha256(ROOT / "experiments/baseline_manifest.json")},
                       "active_policy": {"path": "experiments/model_decision_policy.json",
                                         "sha256": _sha256(ROOT / "experiments/model_decision_policy.json")},
                       "calibration": {"path": str(CALIBRATION.relative_to(ROOT)),
                                       "sha256": calibration_hash},
                       "private_artifact_facts": artifact_facts,
                       "source_sha256": source_hashes},
    }
    DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n")
    return declaration


def _live_artifact_facts():
    _roots()
    sqd_manifest = audit.SQD.parent.parent.parent / "short_quality_diversity_v1_cache_manifest.json"
    dt_manifest = audit.DT / "dense_targeted_followups_cache_manifest_v1.json"
    data_manifest = audit.DATA / "seed_gap_context_cache_manifest.json"
    teacher_manifest = audit.DATA / "objective_alignment_cache_manifest.json"
    facts = {}
    for provider in audit.FOLD:
        for seed in SEEDS:
            model = _model_path("adop", provider, seed)
            facts[f"adop/{provider}/{seed}"] = _artifact_fact(
                dt_manifest, audit.DT, str(model.relative_to(audit.DT)))
            model = _model_path("cand", provider, seed)
            facts[f"cand/{provider}/{seed}"] = _artifact_fact(
                sqd_manifest, audit.SQD.parent.parent.parent,
                str(model.relative_to(audit.SQD.parent.parent.parent)))
            teacher = _teacher_path(provider, seed)
            facts[f"teacher/{provider}/{seed}"] = _artifact_fact(
                teacher_manifest, audit.DATA, str(teacher.relative_to(audit.DATA)))
    for provider, match, filename in audit.MATCHES:
        facts[f"input/{provider}/{match}"] = _artifact_fact(
            data_manifest, audit.DATA, f"experiments/seed_gap_context_v1/{filename}_clean_clean.joblib")
    return facts


def _verify_frozen_bindings(declaration):
    expected = declaration["provenance"]
    current = {
        "source_sha256": _source_hashes(),
        "active_manifest": {"path": "experiments/baseline_manifest.json",
                            "sha256": _sha256(ROOT / "experiments/baseline_manifest.json")},
        "active_policy": {"path": "experiments/model_decision_policy.json",
                          "sha256": _sha256(ROOT / "experiments/model_decision_policy.json")},
        "calibration_sha256": _sha256(CALIBRATION),
        "private_artifact_facts": _live_artifact_facts(),
    }
    frozen = {"source_sha256": expected["source_sha256"],
              "active_manifest": expected["active_manifest"],
              "active_policy": expected["active_policy"],
              "calibration_sha256": expected["calibration"]["sha256"],
              "private_artifact_facts": expected["private_artifact_facts"]}
    if current != frozen:
        raise RuntimeError("Frozen source, active policy, calibration or restored-artifact binding changed")


def _model_path(family, provider, seed):
    fold = audit.FOLD[provider]
    if family == "adop":
        return audit.DT / "experiments/dense_targeted_followups_v1/artifacts/control_D2/models" / \
            f"control_D2__{fold}__s{seed}.joblib"
    if family == "cand":
        return audit.SQD / f"short_original_diversity_v1/models/leave_out_{fold}_{seed}.joblib"
    raise ValueError(family)


def _teacher_path(provider, seed):
    return audit.DATA / f"experiments/objective_alignment_v1/without_{audit.FOLD[provider]}_combined_{seed}.joblib"


_MODELS = {}
_FILE_HASHES = {}


def _file_hash_cached(path):
    path = str(path)
    if path not in _FILE_HASHES:
        _FILE_HASHES[path] = _sha256(path)
    return _FILE_HASHES[path]


def predict_seed(inputs, record, family, provider, seed):
    """Reproduce the archived one-seed inference path for a named family."""
    record = old.inference_record(record)
    record["names"] = list(record["names"])
    native = tc.inputs_only(inputs)
    model_path = _model_path(family, provider, seed)
    teacher_path = _teacher_path(provider, seed)
    for path in (model_path, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if str(model_path) not in _MODELS:
        _MODELS[str(model_path)] = joblib.load(model_path)
    if str(teacher_path) not in _MODELS:
        _MODELS[str(teacher_path)] = joblib.load(teacher_path)
    model, teacher = _MODELS[str(model_path)], _MODELS[str(teacher_path)]
    return old.upsample(sn.predict(record, model, teacher), record, native)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(provider, match, view, rep, family, phase, seed):
    view_key = "clean" if view == "clean" else f"v2r{rep}"
    return OUT / "emissions" / f"{provider}__{match}__{view_key}__{family}__p{phase}__s{seed}.npz"


def _load_or_infer(inputs, record, provider, match, view, rep, family, phase, seed):
    path = _cache_path(provider, match, view, rep, family, phase, seed)
    model_path, teacher_path = _model_path(family, provider, seed), _teacher_path(provider, seed)
    declaration_sha = _sha256(DECLARATION)
    model_sha, teacher_sha = _file_hash_cached(model_path), _file_hash_cached(teacher_path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            probs = saved["probabilities"]
            if (str(saved["input_sha256"].item()) != _input_sha(inputs)
                    or str(saved["record_sha256"].item()) != _record_sha(record)
                    or str(saved["declaration_sha256"].item()) != declaration_sha
                    or str(saved["model_sha256"].item()) != model_sha
                    or str(saved["teacher_sha256"].item()) != teacher_sha):
                raise RuntimeError(f"Emission cache contract mismatch: {path}")
            return probs
    probs = predict_seed(inputs, record, family, provider, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probs, input_sha256=_input_sha(inputs),
                        record_sha256=_record_sha(record), declaration_sha256=declaration_sha,
                        model_sha256=model_sha, teacher_sha256=teacher_sha)
    return probs


def _array_sha(value):
    x = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{x.dtype.str}:{x.shape}:".encode())
    if x.dtype.hasobject or x.dtype.kind in "US":
        digest.update(json.dumps(x.tolist(), separators=(",", ":"), default=str).encode())
    else:
        digest.update(x.view(np.uint8))
    return digest.hexdigest()


def _input_sha(inputs):
    return hashlib.sha256("".join(f"{k}:{_array_sha(np.asarray(inputs[k]))};"
                                   for k in sorted(inputs)).encode()).hexdigest()


def _record_sha(record):
    return hashlib.sha256("".join(f"{k}:{_array_sha(np.asarray(record[k]))};"
                                   for k in sorted(record)).encode()).hexdigest()


def _phase_records(inputs, base):
    """Use the archived exact 5 Hz feature sampling phases."""
    return audit.phase_records(inputs, base)


def _score_cell(cell, predictions):
    return {k: float(v) if isinstance(v, np.generic) else v
            for k, v in audit.score(cell, predictions).items()}


def _prepare_cell(provider, match, view_name, rep):
    if not CALIBRATION.is_file():
        raise FileNotFoundError(CALIBRATION)
    cached = joblib.load(audit.EVAL_INPUTS / f"{audit.FILE[match]}_clean_clean.joblib")
    inputs = cached["inputs"]
    labels = np.asarray(cached["labels"]).astype(str)
    if view_name == "clean":
        view = inputs
    else:
        calibration = json.loads(CALIBRATION.read_text())["calibration"]
        seed = int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
        view = v2.corrupt(inputs, labels, calibration, seed)
    record = prep.build_record(view, labels, provider, audit.FILE[match])
    return view, labels, record, _phase_records(view, record), supervised.model_periods(view)


def _phase_probabilities(view, provider, match, view_name, rep, phases):
    pooled, per_seed = {}, {}
    for family in FAMILIES:
        for phase, phase_record in enumerate(phases):
            seed_probs = {seed: _load_or_infer(view, phase_record, provider, match, view_name,
                                               rep, family, phase, seed)
                          for seed in SEEDS}
            per_seed[(family, phase)] = seed_probs
            pooled[(family, phase)] = np.mean([seed_probs[seed] for seed in SEEDS], axis=0)
    return pooled, per_seed


def run_cell(provider, match, view_name, rep):
    """Infer/cache all phase emissions then score the frozen nine arms."""
    _roots()
    view, labels, record, phases, period = _prepare_cell(provider, match, view_name, rep)
    cell = dict(labels=labels, period=view["period"], timestamps=view["timestamps"], fps=float(view["fps"]))
    probability, _ = _phase_probabilities(view, provider, match, view_name, rep, phases)
    phase_count = len(phases)
    mean_prob = {family: np.mean([probability[(family, phase)] for phase in range(phase_count)], axis=0)
                 for family in FAMILIES}
    family_fallback = {family: audit.decode(mean_prob[family], period, cell["fps"])
                       for family in FAMILIES}
    out = {}
    for family in FAMILIES:
        phase_probs = np.stack([probability[(family, p)] for p in range(phase_count)])
        phase_predictions = np.stack([audit.decode(phase_probs[p], period, cell["fps"])
                                      for p in range(phase_count)])
        for rule in ("mean_decode", "phase_vote", "confidence_vote", "consensus_gate"):
            pred = phase_aggregate(phase_probs, phase_predictions, family_fallback[family], rule)
            out[f"{family}_{rule}"] = _score_cell(cell, pred)
    blended = (mean_prob["adop"] + mean_prob["cand"]) / 2
    out["blend_mean_decode"] = _score_cell(cell, audit.decode(blended, period, cell["fps"]))
    return out


def _require_declaration():
    if not DECLARATION.is_file():
        raise RuntimeError(f"Scoring blocked until parent freezes {DECLARATION}")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != list(ARMS):
        raise RuntimeError("Frozen declaration does not match the runner's fixed arm list")
    _verify_frozen_bindings(declaration)
    return declaration


def preflight():
    """Score only the two declared C0 controls and require exact aggregate parity."""
    _roots()
    declaration = _require_declaration()
    if PREFLIGHT.exists() or RESULT.exists():
        raise RuntimeError("Preflight/results already exist; this study output is immutable")
    controls = {}
    for provider, match, _ in audit.MATCHES:
        for view_name, reps in (("clean", (0,)), ("feed_like_v2", REPLICATES)):
            for rep in reps:
                key = f"{provider}__{match}__{view_name}__{rep}"
                view, labels, _, phases, period = _prepare_cell(provider, match, view_name, rep)
                pooled, per_seed = _phase_probabilities(view, provider, match, view_name, rep, phases)
                score_cell = dict(labels=labels, period=view["period"], timestamps=view["timestamps"],
                                  fps=float(view["fps"]))
                for family in FAMILIES:
                    phase_prob = np.stack([pooled[(family, p)] for p in range(len(phases))])
                    mean_prob = phase_prob.mean(axis=0)
                    pooled_prediction = audit.decode(mean_prob, period, score_cell["fps"])
                    seed_scores = {}
                    for seed in SEEDS:
                        seed_mean = np.mean([per_seed[(family, p)][seed]
                                             for p in range(len(phases))], axis=0)
                        seed_pred = audit.decode(seed_mean, period, score_cell["fps"])
                        seed_scores[str(seed)] = _score_cell(score_cell, seed_pred)
                    controls.setdefault(key, {})[family] = {
                        "C0": _score_cell(score_cell, pooled_prediction), "component_seeds": seed_scores}
    expected_scorecard = json.loads(SCORECARD.read_text())["arms"]
    expected_benchmark = json.loads(BENCHMARK.read_text())["arms"]
    expected_name = {"adop": "adop_phase_tta", "cand": "cand_phase_tta"}
    checks = {}
    for family in FAMILIES:
        for metric, saved_key in (("spell_f1", "spell"), ("macro_f1", "macro")):
            clean_values = {(k.split("__")[0], k.split("__")[1], "clean"):
                            row[family]["C0"][metric] for k, row in controls.items()
                            if "__clean__" in k}
            v2_values = {(k.split("__")[0], k.split("__")[1], k.rsplit("__", 1)[1]):
                         row[family]["C0"][metric] for k, row in controls.items()
                         if "__feed_like_v2__" in k}
            clean_actual = audit.equal_provider(clean_values)[0]
            v2_actual = audit.equal_provider(v2_values)[0]
            clean_expected = expected_scorecard[expected_name[family]]["clean"][saved_key]
            v2_expected = expected_benchmark[expected_name[family]][metric]
            checks[f"{family}/{metric}/clean"] = {"actual": clean_actual,
                "expected": clean_expected, "abs_error": abs(clean_actual - clean_expected)}
            checks[f"{family}/{metric}/feed_like_v2"] = {"actual": v2_actual,
                "expected": v2_expected, "abs_error": abs(v2_actual - v2_expected)}
    passed = all(check["abs_error"] <= 1e-8 for check in checks.values())
    receipt = {"study_id": STUDY, "declaration_sha256": _sha256(DECLARATION),
               "preflight_passed": passed, "tolerance": 1e-8, "checks": checks,
               "controls_only": True, "candidate_scores_written": False,
               "controls": controls,
               "source_sha256": declaration["provenance"]["source_sha256"]}
    PREFLIGHT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"C0 replay did not match frozen controls: {checks}")
    return receipt


def _summarize(rows):
    """Equalize replicates within match, matches within provider, then providers."""
    summary = {}
    for view in ("clean", "feed_like_v2"):
        summary[view] = {}
        for arm in ARMS:
            values = {}
            for key, cell in rows.items():
                provider, match, cell_view, rep = key.split("__")
                if cell_view != view or arm not in cell:
                    continue
                values.setdefault(provider, {}).setdefault(match, []).append(cell[arm])
            if not values:
                continue
            metrics = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                       "predicted_spell_count", "reference_spell_count",
                       "short_spell_recall_iou_0_5_lt3s")
            per_metric = {}
            for metric in metrics:
                per_provider = {provider: float(np.mean([
                    np.mean([row[metric] for row in match_rows])
                    for match_rows in matches.values()]))
                    for provider, matches in values.items()}
                per_metric[metric] = dict(equal_provider_mean=float(np.mean(list(per_provider.values()))),
                                          providers=per_provider)
            per_metric["auxiliary_70_30"] = dict(
                equal_provider_mean=float(0.7 * per_metric["spell_f1"]["equal_provider_mean"] +
                                          0.3 * per_metric["macro_f1"]["equal_provider_mean"]),
                providers={provider: float(0.7 * per_metric["spell_f1"]["providers"][provider] +
                                           0.3 * per_metric["macro_f1"]["providers"][provider])
                           for provider in per_metric["spell_f1"]["providers"]})
            summary[view][arm] = per_metric
    return summary


def _provider_deltas(summary):
    out = {}
    for arm in ARMS:
        family = arm.split("_", 1)[0] if arm != "blend_mean_decode" else None
        references = ("adop_mean_decode", "cand_mean_decode") if family is None else (f"{family}_mean_decode",)
        out[arm] = {"clean": {}, "feed_like_v2": {}}
        for view, metrics in (("clean", ("macro_f1", "d_f1")),
                              ("feed_like_v2", ("spell_f1",))):
            values = summary.get(view, {}).get(arm, {})
            for reference in references:
                base = summary.get(view, {}).get(reference, {})
                out[arm][view][reference] = {
                    metric: {provider: values.get(metric, {}).get("providers", {}).get(provider, float("nan")) -
                             base.get(metric, {}).get("providers", {}).get(provider, float("nan"))
                             for provider in values.get(metric, {}).get("providers", {})}
                    for metric in metrics}
    return out


def run():
    declaration = _require_declaration()
    if not PREFLIGHT.is_file():
        raise RuntimeError("C0 replay must pass before candidate scoring")
    preflight_receipt = json.loads(PREFLIGHT.read_text())
    if (preflight_receipt.get("preflight_passed") is not True or
            preflight_receipt.get("declaration_sha256") != _sha256(DECLARATION)):
        raise RuntimeError("C0 preflight is failed or bound to a different declaration")
    if RESULT.exists():
        raise RuntimeError("Result already exists; this study output is immutable")
    rows = {}
    jobs = [(provider, match, view, rep)
            for provider, match, _ in audit.MATCHES
            for view, reps in (("clean", (0,)), ("feed_like_v2", REPLICATES))
            for rep in reps]
    for provider, match, view, rep in jobs:
        key = f"{provider}__{match}__{view}__{rep}"
        path = OUT / "scores" / f"{key}.json"
        if path.is_file():
            rows[key] = json.loads(path.read_text())
            if rows[key].get("study_declaration_sha256") != _sha256(DECLARATION):
                raise RuntimeError(f"Saved score cell uses a different declaration: {path}")
            continue
        rows[key] = run_cell(provider, match, view, rep)
        rows[key]["study_declaration_sha256"] = _sha256(DECLARATION)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows[key], indent=1, sort_keys=True) + "\n")
    aggregate = _summarize(rows)
    result = dict(study=STUDY, declaration_sha256=_sha256(DECLARATION),
                  arms=list(ARMS), aggregate=aggregate,
                  provider_guard_deltas_vs_C0=_provider_deltas(aggregate), cells=rows,
                  c0_component_seed_diagnostics=preflight_receipt["controls"],
                  organizer_reference_access=False, public_scoring=False)
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("freeze", "preflight", "run"))
    step = parser.parse_args().step
    {"freeze": freeze, "preflight": preflight, "run": run}[step]()
