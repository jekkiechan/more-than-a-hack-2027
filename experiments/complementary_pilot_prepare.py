"""Prepare native probabilities for the historical original_full phase baseline.

This is a historical-predecessor pilot cache only. It performs inference and
stores probabilities; it does not score candidates or make a model decision.
Inputs are the five exposed original-provider matches in seed_gap_context_v1.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_spell_push, preparation as prep, soccernet_extension as old
from experiments import supervised, tracking_corruption, v2_spell_target as shared

STUDY = "complementary_spell_v1"
OUT = audit.ROOT / "data/experiments" / STUDY / "native_probability"
DATA = old.DATA
CALIBRATION = audit.ROOT / "experiments/results/feed_like_v2_calibration.json"
SEEDS = (2026, 2027)
REPLICATES = (1, 2, 3, 4)
VIEWS = ("clean", "feed_like_v1_r1", "feed_like_v1_r2",
         "feed_like_v2_r1", "feed_like_v2_r2", "feed_like_v2_r3", "feed_like_v2_r4")
HISTORICAL_CONTROL = {"clean": 0.6675555328266306,
                      "feed_like_v1": 0.5315755112553304,
                      "feed_like_v2": 0.5835712076382747}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source(match: str) -> tuple[str, str]:
    provider, _, filename = next(row for row in audit.MATCHES if row[1] == match)
    return provider, filename


def _input_path(filename: str) -> Path:
    return audit.EVAL_INPUTS / f"{filename}_clean_clean.joblib"


def _load_cell(match: str) -> tuple[dict, dict, Path]:
    provider, filename = _source(match)
    path = _input_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Historical seed_gap_context input missing: {path}")
    cached = joblib.load(path)
    labels = np.asarray(cached["labels"]).astype(str)
    inputs = cached["inputs"]
    if labels.shape != (len(inputs["ball"]),):
        raise ValueError(f"Input/label alignment failure: {match}")
    if float(inputs["fps"]) not in (10.0, 25.0):
        raise ValueError(f"Expected 10 or 25 Hz historical source: {match}")
    return inputs, {"labels": labels, "provider": provider, "match": filename}, path


def _make_cell(match: str, view: str) -> tuple[dict, dict]:
    provider, filename = _source(match)
    view_alias = "clean" if view == "clean" else "feed_like_v2"
    rep = int(view.rsplit("r", 1)[1]) if view_alias == "feed_like_v2" else 0
    input_path = _input_path(filename)
    if view.startswith("feed_like_v1_"):
        rep = int(view.rsplit("r", 1)[1])
        view_alias = "feed_like_v1"
        view_path = audit.EVAL_INPUTS / f"{filename}_feed_like_corrupt_{rep}.joblib"
        if not view_path.is_file():
            raise FileNotFoundError(f"Historical v1 seed-gap view missing: {view_path}")
        clean = joblib.load(input_path)
        cached = joblib.load(view_path)
        labels = np.asarray(clean["labels"]).astype(str)
        if not np.array_equal(np.asarray(cached["labels"]).astype(str), labels):
            raise ValueError(f"Historical v1 labels mismatch: {match}, replicate {rep}")
        inputs = cached["inputs"]
        record = prep.build_record(inputs, labels, provider, filename)
        phases = shared._phase_records(inputs, record)
        period = supervised.model_periods(inputs)
    else:
        view_path = None
        inputs, labels, record, phases, period = shared._prepare_cell(provider, match, view_alias, rep)
    phase = []
    for phase_idx, phase_record in enumerate(phases):
        # v2_spell_target's historical import aliases stress_components as tc;
        # the known runtime correction points this inference path at the
        # tracking-only sanitizer. Restore the module binding immediately.
        prior_tc = shared.tc
        shared.tc = tracking_corruption
        try:
            per_seed = {seed: shared.predict_seed(inputs, phase_record, "cand", provider, seed)
                        for seed in SEEDS}
        finally:
            shared.tc = prior_tc
        for seed, prob in per_seed.items():
            if (prob.shape != (len(inputs["ball"]), 3) or not np.isfinite(prob).all()
                    or not np.allclose(prob.sum(axis=1), 1, atol=1e-6, rtol=0)):
                raise ValueError(f"Invalid native probabilities at phase {phase_idx}, seed {seed}")
        phase.append(per_seed)
    phase_means = [np.mean([phase[i][s] for s in SEEDS], axis=0) for i in range(len(phases))]
    native = np.mean(phase_means, axis=0)
    decoded = feed_spell_push.decode(native[None], period, float(inputs["fps"]), audit.CONFIG)
    decoded[np.asarray(inputs["period"]) <= 0] = "X"
    historical_manifest = audit.DATA / "models/joint_gru_sn_dense_original_diversity_phase_ensemble_v1/baseline_manifest.json"
    if not historical_manifest.is_file():
        raise FileNotFoundError(f"Historical predecessor manifest missing: {historical_manifest}")
    historical_manifest_sha = sha(historical_manifest)
    arrays = {"labels": labels, "period": np.asarray(inputs["period"]),
              "timestamps": np.asarray(inputs["timestamps"]), "model_period": period,
              "fps": np.asarray(float(inputs["fps"])), "decoded__historical_f062": decoded,
            "prob__historical_ensemble": native}
    for i in range(len(phases)):
        for s in SEEDS:
            arrays[f"prob__candidate__seed{s}__phase{i}"] = phase[i][s]
    source_files = [input_path, historical_manifest, shared.CALIBRATION]
    if view_path is not None:
        source_files.append(view_path)
    source_hashes = {str(p): sha(p) for p in source_files}
    rep_num = 0 if view == "clean" else rep
    view_kind = "clean" if view == "clean" else ("feed_like_v1" if view_alias == "feed_like_v1" else "feed_like_v2")
    model_hashes = {}
    for seed in SEEDS:
        for kind, path in (("model", shared._model_path("cand", provider, seed)),
                           ("teacher", shared._teacher_path(provider, seed))):
            model_hashes[f"{kind}_{seed}"] = {"path": str(path), "sha256": sha(path)}
    meta = dict(provider=provider, match=match, input_match=filename,
                view=view_kind, rep=rep_num,
                corruption_seed=(int(hashlib.sha256(
                    f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
                    if view_alias == "feed_like_v2" else None), phase_count=len(phases),
                historical_package="original_full_phase_ensemble_v1", historical_predecessor=True,
                model_sha256=model_hashes,
                input_sha256=sha(input_path), calibration_sha256=sha(shared.CALIBRATION),
                model_manifest_sha256=historical_manifest_sha,
                source_sha256=source_hashes,
                historical_view_sha256=sha(view_path) if view_path is not None else None,
                decoder="f062", scoring_performed=False)
    cell_key = f"{provider}|{match}|{view_kind}|{rep_num}"
    meta.update(cell=cell_key, source_study_id="original_full_phase_ensemble_v1")
    for key, value in meta.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            arrays[f"identity__{key}"] = np.asarray(str(value))
    return arrays, meta


def _prepare_cell_job(job: tuple[str, str, str]) -> tuple[str, dict]:
    match, view, out_dir = job
    arrays, identity = _make_cell(match, view)
    dest = Path(out_dir) / f"{match}__{view}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared cell: {dest}")
    temp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, dest)
    rep_key = identity["rep"]
    cell_key = f"{identity['provider']}|{match}|{identity['view']}|{rep_key}"
    fact = {"path": str(dest.relative_to(audit.ROOT)), "sha256": sha(dest),
            "identity": identity, "phase_count": identity["phase_count"],
            "rows": int(len(arrays["labels"]))}
    return cell_key, fact


def prepare(output: Path = OUT, workers: int = 2) -> dict:
    """Create immutable, hash-receipted cell caches after provenance validation."""
    if workers not in (1, 2):
        raise ValueError("Preparation is limited to one or two workers")
    shared._roots()
    # Verify exactly the archived candidate folds, teachers and input views
    # needed here; the unrelated adopted family is outside this preparation.
    data_manifest = audit.DATA / "seed_gap_context_cache_manifest.json"
    sqd_manifest = audit.SQD.parent.parent.parent / "short_quality_diversity_v1_cache_manifest.json"
    teacher_manifest = audit.DATA / "objective_alignment_cache_manifest.json"
    artifact_facts = {}
    for provider in audit.FOLD:
        for seed in SEEDS:
            model = shared._model_path("cand", provider, seed)
            artifact_facts[f"cand/{provider}/{seed}"] = shared._artifact_fact(
                sqd_manifest, audit.SQD.parent.parent.parent,
                str(model.relative_to(audit.SQD.parent.parent.parent)))
            teacher = shared._teacher_path(provider, seed)
            artifact_facts[f"teacher/{provider}/{seed}"] = shared._artifact_fact(
                teacher_manifest, audit.DATA, str(teacher.relative_to(audit.DATA)))
    for provider, match, filename in audit.MATCHES:
        for suffix in ("clean_clean", "feed_like_corrupt_1", "feed_like_corrupt_2"):
            rel = f"experiments/seed_gap_context_v1/{filename}_{suffix}.joblib"
            artifact_facts[f"input_view/{provider}/{match}/{suffix}"] = shared._artifact_fact(
                data_manifest, audit.DATA, rel)
    expected_matches = [m[1] for m in audit.MATCHES]
    cells = {}
    jobs = [(match, view, str(output)) for match in expected_matches for view in VIEWS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_prepare_cell_job, job): job for job in jobs}
        for future in as_completed(futures):
            key, fact = future.result()
            if key in cells:
                raise ValueError(f"Duplicate prepared cell: {key}")
            cells[key] = fact
            print(f"PREPARED {len(cells)}/{len(jobs)} {key}", flush=True)
    receipt = output / "preparation_manifest.json"
    result = dict(study=STUDY, source_study_id="original_full_phase_ensemble_v1",
                  evidence_scope="historical_predecessor_only_no_frontier_efficacy_claim",
                  pooling_order="seed_weighted_mean_within_phase_then_equal_phase_mean",
                  efficacy_claim="none", scoring_performed=False,
                  matches=expected_matches, views=["clean", "feed_like_v1", "feed_like_v2"], seeds=list(SEEDS),
                  calibration_sha256=sha(shared.CALIBRATION), private_artifact_facts=artifact_facts,
                  expected_control_spell_f1=HISTORICAL_CONTROL,
                  fixed_weights={"seed_2026": 0.5, "seed_2027": 0.5,
                                 "phase_weighting": "uniform", "decoder": "f062"},
                  source_hashes={p: sha(audit.ROOT / p) for p in (
                      "experiments/complementary_pilot_prepare.py", "experiments/v2_spell_target.py",
                      "experiments/feed_benchmark_audit_v1.py", "experiments/feed_spell_push.py",
                      "experiments/tracking_corruption.py",
                      "experiments/metrics.py", "evaluation/possession_metrics.py")},
                  cohort={"matches": [[p, m] for p, m, _ in audit.MATCHES],
                          "view_replicates": {"clean": [0], "feed_like_v1": [1, 2],
                                              "feed_like_v2": [1, 2, 3, 4]}},
                  expected_cells=sorted(cells), cells=cells, workers=workers)
    if receipt.exists():
        raise FileExistsError(f"Refusing to overwrite preparation manifest: {receipt}")
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare", choices=("prepare",))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.workers), indent=2))


if __name__ == "__main__":
    main()
