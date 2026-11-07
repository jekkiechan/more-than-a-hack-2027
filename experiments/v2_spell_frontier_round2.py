"""Stage-1536 repeat/fresh evaluation of the frozen frontier recipe shortlist."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_spell_frontier_eval as r1
from experiments import v2_spell_frontier_train as trainer

ROOT = r1.ROOT
STUDY = r1.STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_round2_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}_round2.json"
OUT = ROOT / "data/experiments" / STUDY / "round2"
ARMS = tuple(trainer.ARMS)
SEEDS = tuple(trainer.SEEDS)
VIEWS, METRICS = r1.VIEWS, r1.METRICS
WORKERS = min(2, max(1, int(os.environ.get("V2_SPELL_FRONTIER_WORKERS", "2"))))
TORCH_THREADS = 1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, default=float)
        f.write("\n")


def _pipeline(recipe, arm):
    frozen_recipe = {k: recipe[k] for k in
                     ("recipe_id", "ce_weight", "calibrated_weight", "adopted_weight", "pooling", "ha_cost")}
    return {"pipeline_id": f"{arm}__{recipe['recipe_id']}", "recipe": frozen_recipe,
            "training_arm": arm, "training_stage": 1536}


def _source_hashes():
    frozen = r1.balanced.runtime.frozen
    paths = (Path(__file__), Path(r1.__file__), Path(trainer.__file__),
             Path(trainer.iterative.__file__), Path(trainer.prior_run.__file__),
             Path(frozen.__file__), Path(frozen.audit.__file__),
             Path(frozen.audit.decoder.__file__), Path(frozen.audit.metrics.__file__),
             Path(frozen.sn.__file__), Path(frozen.sn.cap.__file__), Path(frozen.sn.fh.__file__),
             Path(frozen.sn.lt.__file__), Path(frozen.old.__file__), Path(frozen.supervised.__file__),
             Path(frozen.tc.__file__), Path(frozen.v2.__file__),
             ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py",
             ROOT / "experiments/soccernet_capacity_model.py", ROOT / "evaluation/possession_metrics.py",
             r1.PLAN, r1.DECLARATION, r1.REPLAY, r1.RESULT, trainer.PLAN, trainer.DECL,
             trainer.SEAL, ROOT / "experiments/tests/test_v2_spell_frontier_round2.py")
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def _fit_facts():
    trainer.check(full=True)
    seal = json.loads(trainer.SEAL.read_text())
    if (seal.get("study_id") != STUDY or seal.get("declaration_sha256") != sha(trainer.DECL) or
            seal.get("all_12_complete") is not True or len(seal.get("fits", [])) != 12):
        raise RuntimeError("Frontier stage1536 12-fit seal is incomplete")
    result = {}
    for arm in ARMS:
        for fold in trainer.FOLDS:
            for seed in SEEDS:
                path = trainer.output_path(arm, fold, seed)
                receipt = Path(str(path) + ".json")
                matches = [row for row in seal.get("fits", []) if
                           (row.get("arm"), row.get("fold"), int(row.get("seed", -1))) == (arm, fold, seed)]
                if len(matches) != 1 or not path.is_file() or not receipt.is_file():
                    raise RuntimeError(f"Missing frontier fit: {arm}/{fold}/{seed}")
                row = matches[0]
                if (row != json.loads(receipt.read_text()) or row.get("sha256") != sha(path) or
                        row.get("stage") != 1536 or row.get("arm") != arm or
                        row.get("declaration_sha256") != sha(trainer.DECL)):
                    raise RuntimeError(f"Fit receipt or checkpoint changed: {arm}/{fold}/{seed}")
                result[f"{arm}/{fold}/{seed}"] = {"path": str(path.relative_to(ROOT)),
                    "sha256": sha(path), "receipt": str(receipt.relative_to(ROOT)),
                    "receipt_sha256": sha(receipt)}
    return result


def _shortlist():
    result = json.loads(r1.RESULT.read_text())
    if result.get("declaration_sha256") != sha(r1.DECLARATION):
        raise RuntimeError("Frontier R1 result does not bind its declaration")
    selected = result.get("seed_shortlist", [])
    eligible = result.get("eligible_ranking", [])
    expected = eligible[:2]
    anchor = r1.LEADER_IDS[0]
    if anchor not in expected:
        expected.append(anchor)
    expected = list(dict.fromkeys(expected))[:3]
    if selected != expected:
        raise RuntimeError("Frozen R1 seed shortlist differs from pooled eligibility/ranking")
    recipes = {row["recipe_id"]: row for row in result["recipes"]}
    if any(rid not in recipes for rid in selected):
        raise RuntimeError("Shortlisted recipe is absent from the declared frontier grid")
    return [recipes[rid] for rid in selected]


def _controls():
    dec = r1.require_declaration()
    replay = json.loads(r1.CONTROL_REPLAY.read_text())
    if replay.get("passed") is not True:
        raise RuntimeError("Balanced adopted control replay is unavailable")
    facts = dec["inputs"]["control"]
    expected = r1.expected_cells()
    if set(facts) != expected:
        raise RuntimeError("Frontier R2 requires all 35 frozen matched cells")
    return facts


def freeze():
    if DECLARATION.exists() or RESULT.exists():
        raise RuntimeError("Frontier R2 output exists; preserve study ID")
    if not r1.RESULT.is_file() or not r1.REPLAY.is_file():
        raise RuntimeError("Frontier R1 result and leader replay must finish first")
    recipes = _shortlist()
    fits, controls = _fit_facts(), _controls()
    pipelines = [_pipeline(recipe, arm) for recipe in recipes for arm in ARMS]
    value = {"study_id": STUDY, "round": 2,
        "status": "frozen_before_stage1536_scoring",
        "parent": {"r1_declaration_sha256": sha(r1.DECLARATION), "r1_result_sha256": sha(r1.RESULT),
                   "r1_replay_sha256": sha(r1.REPLAY)},
        "training": {"declaration_sha256": sha(trainer.DECL), "seal_sha256": sha(trainer.SEAL),
                     "fits": fits, "arms": list(ARMS), "stage": 1536},
        "recipes": recipes, "pipelines": pipelines,
        "pipeline_ids": [p["pipeline_id"] for p in pipelines],
        "controls": controls,
        "cohort": {"cells": 35, "seeds": list(SEEDS), "views": [v for v, _ in VIEWS]},
        "aggregation": "paired seed mean within phase, then phase mean; separate per-seed diagnostics",
        "eligibility": "same adopted-based gates as R1; current-selection improvement is final selector logic",
        "source_sha256": _source_hashes(), "organizer_reference_access": False,
        "public_scoring": False, "promotion": False}
    _write_once(DECLARATION, value)
    return value


def require_declaration():
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze frontier R2 before use")
    dec = json.loads(DECLARATION.read_text())
    if dec.get("source_sha256") != _source_hashes():
        raise RuntimeError("Frontier R2 source changed after freeze")
    if dec.get("parent") != {"r1_declaration_sha256": sha(r1.DECLARATION),
        "r1_result_sha256": sha(r1.RESULT), "r1_replay_sha256": sha(r1.REPLAY)}:
        raise RuntimeError("Frontier R1 parent artifacts changed")
    if dec.get("training", {}).get("fits") != _fit_facts() or dec.get("controls") != _controls():
        raise RuntimeError("Frontier stage1536 fits or control cache changed")
    recipes = _shortlist()
    expected = [_pipeline(recipe, arm) for recipe in recipes for arm in ARMS]
    if dec.get("recipes") != recipes or dec.get("pipelines") != expected:
        raise RuntimeError("Frontier pipeline inventory differs from the frozen R1 shortlist")
    return dec


def _load_controls(control_fact):
    return r1.balanced._load_control_arrays(control_fact)


def _cell_inputs(key, fact):
    # The sealed phase record was prepared with labels for scoring, but labels
    # are never passed to inference functions below.
    prepared_path = ROOT / fact["prepared_path"]
    if sha(prepared_path) != fact["prepared_sha256"]:
        raise RuntimeError(f"Prepared phase cache changed: {key}")
    prepared = joblib.load(prepared_path)
    if (prepared.get("identity") != fact["identity"] or
            prepared.get("identity", {}).get("cell") != key):
        raise RuntimeError(f"Prepared phase identity mismatch: {key}")
    control = _load_controls({**fact, "cell": key})
    return prepared, control


def _predict_seed(prepared, provider, arm, seed, fit_fact):
    fold = f"leave_out_{r1.balanced.runtime.frozen.audit.FOLD[provider]}"
    model = joblib.load(ROOT / fit_fact[f"{arm}/{fold}/{seed}"]["path"])
    teacher = joblib.load(r1.balanced.runtime.frozen._teacher_path(provider, seed))
    frozen = r1.balanced.runtime.frozen
    predictions = []
    for record in prepared["phases"]:
        infer = frozen.old.inference_record(record)
        infer["names"] = list(infer["names"])
        sampled = frozen.sn.predict(infer, model, teacher)
        predictions.append(frozen.old.upsample(sampled, infer, frozen.tc.inputs_only(prepared["view"])))
    return predictions


def _pool(phases_by_seed):
    if set(phases_by_seed) != set(SEEDS):
        raise ValueError("Both paired seed phase outputs are required")
    counts = {len(v) for v in phases_by_seed.values()}
    if len(counts) != 1 or not next(iter(counts)):
        raise ValueError("Paired seeds have different/empty phase inventories")
    count = next(iter(counts))
    for phase in range(count):
        if phases_by_seed[SEEDS[0]][phase].shape != phases_by_seed[SEEDS[1]][phase].shape:
            raise ValueError("Paired phase probabilities are misaligned")
    pooled = np.mean([np.mean([phases_by_seed[s][phase] for s in SEEDS], axis=0)
                      for phase in range(count)], axis=0)
    return pooled, {s: np.mean(phases_by_seed[s], axis=0) for s in SEEDS}


def _cell_job(job):
    key, fact, recipes, pipelines, fit_facts = job
    prepared, control = _cell_inputs(key, fact)
    provider = key.split("|", 1)[0]
    adopted_decoded = r1.balanced.runtime.frozen.audit.decode(control["adopted"],
        control["model_period"], control["fps"], r1.balanced.runtime.frozen.audit.CONFIG)
    cell = {name: control[name] for name in ("labels", "period", "timestamps", "fps")}
    rows, seed_rows, probs, predictions, seed_preds = {}, {}, {}, {}, {}
    for arm in ARMS:
        phase_outputs = {seed: _predict_seed(prepared, provider, arm, seed, fit_facts) for seed in SEEDS}
        pooled, per_seed = _pool(phase_outputs)
        for recipe in recipes:
            pid = f"{arm}__{recipe['recipe_id']}"
            mixtures = {"ce": pooled, "calibrated": control["calibrated"], "adopted": control["adopted"]}
            p = r1.balanced.recipe_probability(mixtures, recipe["ce_weight"],
                                               recipe["calibrated_weight"], recipe["pooling"])
            pred = r1.balanced.decode_fixed_d(p, adopted_decoded, control["model_period"], control["fps"], 3.)
            rows[pid] = r1.balanced.controls.audit.score(cell, pred)
            probs[f"pooled__{pid}"] = pooled
            predictions[f"pred__{pid}"] = r1.balanced._encode_prediction(pred)
            for seed in SEEDS:
                seed_components = {"ce": per_seed[seed], "calibrated": control[f"adop_seed{seed}"],
                                   "adopted": control[f"cand_seed{seed}"]}
                seed_p = r1.balanced.recipe_probability(seed_components, recipe["ce_weight"],
                    recipe["calibrated_weight"], recipe["pooling"])
                seed_pred = r1.balanced.decode_fixed_d(seed_p, adopted_decoded,
                    control["model_period"], control["fps"], 3.)
                seed_rows[f"{pid}|{seed}"] = r1.balanced.controls.audit.score(cell, seed_pred)
                seed_preds[f"seedpred__{pid}|{seed}"] = r1.balanced._encode_prediction(seed_pred)
        for seed in SEEDS:
            for phase, value in enumerate(phase_outputs[seed]):
                probs[f"phase__{arm}__{seed}__{phase}"] = value
    return key, rows, seed_rows, probs, predictions, seed_preds


def run_round2():
    dec = require_declaration()
    if RESULT.exists():
        raise RuntimeError("Frontier R2 result already exists")
    recipes, pipelines = dec["recipes"], dec["pipelines"]
    fits = dec["training"]["fits"]
    jobs = [(key, {**fact, "cell": key}, recipes, pipelines, fits)
            for key, fact in sorted(dec["controls"].items())]
    cell_rows, seed_rows, cache = {}, {}, {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs)), initializer=_worker_init) as pool:
        futures = [pool.submit(_cell_job, job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            key, rows, seeds, probs, preds, seed_preds = fut.result()
            cell_rows[key], seed_rows[key] = rows, seeds
            cache[key] = (probs, preds, seed_preds)
            print(f"FRONTIER R2 {n}/{len(jobs)} {key}", flush=True)
    ids = dec["pipeline_ids"]
    if set(cell_rows) != r1.expected_cells() or any(set(rows) != set(ids) for rows in cell_rows.values()):
        raise RuntimeError("R2 pipeline/cell coverage mismatch")
    aggregate = {pid: r1.balanced.aggregate_cell_metrics({k: v[pid] for k, v in cell_rows.items()}) for pid in ids}
    baseline = json.loads(r1.CONTROL_REPLAY.read_text())
    refs = baseline["reference_spell_counts"]
    gates = {}
    for pid in ids:
        cand = {view: {m: aggregate[pid][view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        base = {view: {m: baseline["adopted"][view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        gate = r1.balanced.eligibility(cand, base)
        gate["d_metrics_unchanged"] = all(cand[v][m] == base[v][m] for v, _ in VIEWS
            for m in ("d_f1", "d_precision", "d_recall"))
        gate["reference_spell_counts_invariant"] = all(
            cell_rows[k][pid]["reference_spell_count"] == refs[k] for k in cell_rows)
        gate["beats_current_best"] = {"actual": aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"],
            "minimum_strictly_greater": r1.CURRENT_BEST_V2,
            "passed": aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"] > r1.CURRENT_BEST_V2}
        gate["eligible"] = gate["eligible"] and gate["d_metrics_unchanged"] and gate["reference_spell_counts_invariant"]
        gates[pid] = gate
    seed_aggregate = {}
    for pid in ids:
        seed_aggregate[pid] = {str(seed): r1.balanced.aggregate_cell_metrics({
            key: {m: row[f"{pid}|{seed}"][m] for m in METRICS} for key, row in seed_rows.items()})
            for seed in SEEDS}
    ranking = sorted(ids, key=lambda pid: (
        -aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"],
        -aggregate[pid]["feed_like_v2"]["macro_f1"]["equal_provider"], pid))
    pred_inventory, seed_inventory = {}, {}
    for key in sorted(cache):
        probs, preds, seed_preds = cache[key]
        path = OUT / "predictions" / f"{key.replace('|', '__')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        ident = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "cell": key,
                 "pipeline_order": ids, "seed_order": list(SEEDS)}
        arrays = {**probs, **preds, **seed_preds,
                  **{f"identity__{k}": np.asarray(json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                     for k, v in ident.items()}}
        np.savez_compressed(path, **arrays)
        pred_inventory[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "identity": ident}
    result = {"study_id": STUDY, "round": 2, "training_stage": 1536,
        "declaration_sha256": sha(DECLARATION), "pipelines": pipelines, "pipeline_ids": ids,
        "aggregate": aggregate, "eligibility": gates, "cell_metrics": cell_rows,
        "eligible_ranking": [pid for pid in ranking if gates[pid]["eligible"]],
        "seed_cells": seed_rows, "seed_diagnostics": seed_aggregate,
        "prediction_cache": pred_inventory, "cohort": {"cell_count": len(cell_rows),
        "reference_spell_counts": refs}, "organizer_reference_access": False,
        "public_scoring": False, "promotion": False}
    _write_once(RESULT, result)
    return result


def _worker_init():
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run-round2"))
    action = parser.parse_args().action
    (freeze if action == "freeze" else run_round2)()


if __name__ == "__main__":
    main()
