"""Reviewed runtime correction for the frozen v2_spell_target runner.

The predecessor declaration and runner remain unchanged. Its preflight failed
before emission caching because its tracking-corruption import alias was wrong.
This wrapper changes only that alias and binds a new immutable study ID and
output namespace; all inference, arms, cohort, and score logic is delegated to
the frozen predecessor module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import tracking_corruption
from experiments import v2_spell_target as frozen


ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_target_runtime"
frozen.tc = tracking_corruption
frozen.STUDY = STUDY
frozen.DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
frozen.PREFLIGHT = ROOT / f"experiments/results/{STUDY}_preflight.json"
frozen.RESULT = ROOT / f"experiments/results/{STUDY}.json"
frozen.OUT = ROOT / "data/experiments" / STUDY

_base_source_hashes = frozen._source_hashes


def _source_hashes():
    hashes = _base_source_hashes()
    hashes["experiments/v2_spell_target_runtime.py"] = frozen._sha256(Path(__file__))
    return hashes


frozen._source_hashes = _source_hashes

_base_prepare_cell = frozen._prepare_cell


def _prepared_path(provider, match, view_name, rep):
    label = "clean" if view_name == "clean" else f"v2r{rep}"
    return frozen.OUT / "prepared" / f"{provider}__{match}__{label}.joblib"


def _prepare_cell(provider, match, view_name, rep):
    """Reuse declaration-bound deterministic feature preparation across stages."""
    frozen._roots()
    path = _prepared_path(provider, match, view_name, rep)
    receipt_path = path.with_suffix(".receipt.json")
    declaration_sha = frozen._sha256(frozen.DECLARATION)
    if path.is_file() or receipt_path.is_file():
        if not path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"Incomplete prepared-cell cache pair: {path}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("declaration_sha256") != declaration_sha or
                receipt.get("cache_sha256") != frozen._sha256(path)):
            raise RuntimeError(f"Prepared-cell cache receipt mismatch: {path}")
        return joblib.load(path)
    result = _base_prepare_cell(provider, match, view_name, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path, compress=0)
    receipt_path.write_text(json.dumps({"study_id": frozen.STUDY,
                                        "declaration_sha256": declaration_sha,
                                        "cache_sha256": frozen._sha256(path),
                                        "provider": provider, "match": match,
                                        "view": view_name, "replicate": rep},
                                       indent=1, sort_keys=True) + "\n")
    return result


frozen._prepare_cell = _prepare_cell


def _preflight_cell(job):
    provider, match, view_name, rep = job
    view, labels, _, phases, period = frozen._prepare_cell(provider, match, view_name, rep)
    pooled, per_seed = frozen._phase_probabilities(view, provider, match, view_name, rep, phases)
    score_cell = dict(labels=labels, period=view["period"], timestamps=view["timestamps"],
                      fps=float(view["fps"]))
    result = {}
    for family in frozen.FAMILIES:
        phase_prob = np.stack([pooled[(family, p)] for p in range(len(phases))])
        pred = frozen.audit.decode(phase_prob.mean(axis=0), period, score_cell["fps"])
        seed_scores = {}
        for seed in frozen.SEEDS:
            seed_mean = np.mean([per_seed[(family, p)][seed] for p in range(len(phases))], axis=0)
            seed_scores[str(seed)] = frozen._score_cell(
                score_cell, frozen.audit.decode(seed_mean, period, score_cell["fps"]))
        result[family] = {"C0": frozen._score_cell(score_cell, pred),
                          "component_seeds": seed_scores}
    return f"{provider}__{match}__{view_name}__{rep}", result


def preflight():
    """Parallel controls-only replay; no candidate arm scores are produced."""
    frozen._roots()
    declaration = frozen._require_declaration()
    if frozen.PREFLIGHT.exists() or frozen.RESULT.exists():
        raise RuntimeError("Preflight/results already exist; this study output is immutable")
    jobs = [(provider, match, view, rep)
            for provider, match, _ in frozen.audit.MATCHES
            for view, reps in (("clean", (0,)), ("feed_like_v2", frozen.REPLICATES))
            for rep in reps]
    controls = {}
    workers = min(3, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_preflight_cell, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            key, value = future.result()
            controls[key] = value
            print(f"C0 preflight {done}/{len(jobs)}: {key}", flush=True)
    expected_scorecard = json.loads(frozen.SCORECARD.read_text())["arms"]
    expected_benchmark = json.loads(frozen.BENCHMARK.read_text())["arms"]
    expected_name = {"adop": "adop_phase_tta", "cand": "cand_phase_tta"}
    checks = {}
    for family in frozen.FAMILIES:
        for metric, saved_key in (("spell_f1", "spell"), ("macro_f1", "macro")):
            clean_values = {(k.split("__")[0], k.split("__")[1], "clean"):
                            row[family]["C0"][metric] for k, row in controls.items()
                            if "__clean__" in k}
            v2_values = {(k.split("__")[0], k.split("__")[1], k.rsplit("__", 1)[1]):
                         row[family]["C0"][metric] for k, row in controls.items()
                         if "__feed_like_v2__" in k}
            clean_actual = frozen.audit.equal_provider(clean_values)[0]
            v2_actual = frozen.audit.equal_provider(v2_values)[0]
            clean_expected = expected_scorecard[expected_name[family]]["clean"][saved_key]
            v2_expected = expected_benchmark[expected_name[family]][metric]
            checks[f"{family}/{metric}/clean"] = {"actual": clean_actual,
                "expected": clean_expected, "abs_error": abs(clean_actual - clean_expected)}
            checks[f"{family}/{metric}/feed_like_v2"] = {"actual": v2_actual,
                "expected": v2_expected, "abs_error": abs(v2_actual - v2_expected)}
    passed = all(check["abs_error"] <= 1e-8 for check in checks.values())
    receipt = {"study_id": frozen.STUDY,
               "declaration_sha256": frozen._sha256(frozen.DECLARATION),
               "preflight_passed": passed, "tolerance": 1e-8, "checks": checks,
               "controls_only": True, "candidate_scores_written": False,
               "controls": controls,
               "source_sha256": declaration["provenance"]["source_sha256"],
               "workers": workers, "prepared_cache_sha256": {
                   str(path): frozen._sha256(path)
                   for path in sorted((frozen.OUT / "prepared").glob("*.joblib"))}}
    frozen.PREFLIGHT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"C0 replay did not match frozen controls: {checks}")
    return receipt


def freeze():
    return frozen.freeze()


def run():
    return frozen.run()


def phase_aggregate(*args, **kwargs):
    return frozen.phase_aggregate(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("freeze", "preflight", "run"))
    step = parser.parse_args().step
    {"freeze": freeze, "preflight": preflight, "run": run}[step]()
