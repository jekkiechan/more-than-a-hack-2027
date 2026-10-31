"""Post-hoc fixed 50/50 blend of the frozen D2 and short-team ensembles.

This zero-fit follow-up uses the unchanged control and candidate ensemble
probabilities from dense_short_team_v1. It has one fixed blend weight, uses the
unchanged cost-3 decoder, and does not read organizer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments import dense_short_team_eval_v1 as evaluator
from experiments import dense_error_audit_v1 as error_audit
from experiments import dense_short_team_v1 as source_study
from experiments import objective_ensemble as scoring
from experiments import supervised

ROOT = source_study.ROOT
DATA_OUT = ROOT / "data/experiments/dense_short_team_blend_v1"
RESULTS = ROOT / "experiments/results"
STUDY_ID = "dense_short_team_blend_v1"
ARM = "short_team_blend_50_50"
CONTROL_ARM = "control_D2"
CANDIDATE_ARM = "short_team_2x"
BLEND_WEIGHT = .5
DECLARATION = RESULTS / f"{STUDY_ID}_declaration.json"
RESULT = RESULTS / f"{STUDY_ID}_readout.json"
PREDICTIONS = DATA_OUT / "predictions"
SOURCE_PATHS = (
    Path(__file__),
    ROOT / "experiments/tests/test_dense_short_team_blend_v1.py",
    ROOT / "experiments/dense_short_team_eval_v1.py",
    ROOT / "experiments/dense_short_team_v1.py",
    ROOT / "experiments/dense_label_weight_v1.py",
    ROOT / "experiments/dense_error_audit_v1.py",
    ROOT / "experiments/objective_ensemble.py",
    ROOT / "experiments/supervised.py",
    ROOT / "experiments/metrics.py",
    ROOT / "experiments/spell_diagnostics.py",
    ROOT / "experiments/soccernet_extension.py",
    ROOT / "experiments/soccernet_capacity.py",
    ROOT / "evaluation/possession_metrics.py",
    ROOT / "experiments/model_decision_policy.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path: Path, value: dict) -> None:
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite frozen artifact: {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def load_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def prediction_path(item: dict, view: str, replicate: str, arm: str) -> Path:
    fold = "leave_out_" + item["source"]
    filename = f"seedensemble__{item['match']}__{view}__{replicate}.npz"
    if arm == CONTROL_ARM:
        return source_study.CONTROL_OUT / "predictions" / "control_D2" / fold / filename
    if arm == CANDIDATE_ARM:
        return source_study.PREDICTIONS / "short_team_2x" / fold / filename
    raise ValueError(f"Unknown source prediction arm {arm}")


def prior_readout() -> dict:
    path = RESULTS / "dense_short_team_v1_readout.json"
    return json.loads(path.read_text())


def _archive_manifest(readout: dict) -> dict[str, str]:
    files = readout.get("prediction_files")
    if not isinstance(files, dict):
        raise ValueError("Prior dense short-team readout has no prediction file manifest")
    return files


def blend_probabilities(control_probability, candidate_probability) -> np.ndarray:
    """Apply the single declared symmetric 50/50 probability blend."""
    control = np.asarray(control_probability, dtype=np.float64)
    candidate = np.asarray(candidate_probability, dtype=np.float64)
    if control.ndim != 2 or control.shape[1] != 3 or candidate.shape != control.shape:
        raise ValueError("Both source probabilities must have matching (frames, 3) arrays")
    return (control + candidate) * BLEND_WEIGHT


def validate_source_pair(control: dict, candidate: dict, cell: dict) -> None:
    """Check target-independent inference metadata and paired native arrays."""
    evaluator.assert_native_arrays_equal(control, candidate)
    fold = "leave_out_" + cell["source"]
    expected = {"source": cell["source"], "match": cell["match"],
                "view": cell["view"], "replicate": cell["replicate"], "fold": fold}
    for archive, arm in ((control, CONTROL_ARM), (candidate, CANDIDATE_ARM)):
        for name, value in expected.items():
            if name not in archive or str(np.asarray(archive[name]).reshape(()).item()) != str(value):
                raise ValueError(f"Prior {arm} archive has changed identity field {name}")
        ensemble_seed = "ensemble" if arm == CONTROL_ARM else "seedensemble"
        for name, value in (("arm", arm), ("seed", ensemble_seed)):
            if name not in archive or str(np.asarray(archive[name]).reshape(()).item()) != value:
                raise ValueError(f"Prior {arm} archive has changed ensemble identity field {name}")


def declaration() -> dict:
    """Describe the fixed post-hoc blend and bind all prior evidence."""
    prior_path = RESULTS / "dense_short_team_v1_readout.json"
    prior_decl_path = RESULTS / "dense_short_team_v1_declaration.json"
    prior_seal_path = RESULTS / "dense_short_team_v1_evaluation_seal.json"
    prior = prior_readout()
    prior_declaration = json.loads(prior_decl_path.read_text())
    prior_seal = json.loads(prior_seal_path.read_text())
    if (prior_declaration.get("study") != "dense_short_team_v1" or
            prior_seal.get("study") != "dense_short_team_v1" or
            prior_seal.get("declaration_sha256") != sha256(prior_decl_path) or
            prior.get("seal_sha256") != sha256(prior_seal_path)):
        raise ValueError("Prior declaration, evaluation seal, and readout hash chain is invalid")
    prior_hashes = _archive_manifest(prior)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}
    input_hashes = {}
    item_cells = []
    for item in evaluator._selection():
        for view, replicate in evaluator.VIEWS:
            paths = {arm: prediction_path(item, view, replicate, arm)
                     for arm in (CONTROL_ARM, CANDIDATE_ARM)}
            for arm, path in paths.items():
                rel = str(path.relative_to(ROOT))
                if not path.is_file() or prior_hashes.get(rel) != sha256(path):
                    raise ValueError(f"Prior ensemble archive is absent or changed: {rel}")
                input_hashes[rel] = sha256(path)
            control = load_archive(paths[CONTROL_ARM])
            candidate = load_archive(paths[CANDIDATE_ARM])
            validate_source_pair(control, candidate, {"source": item["source"],
                "match": item["match"], "view": view, "replicate": replicate})
            if control["probabilities"].shape != candidate["probabilities"].shape:
                raise ValueError(f"Prior source ensemble shapes differ for {item['match']}/{view}")
            item_cells.append({"source": item["source"], "match": item["match"],
                               "view": view, "replicate": replicate})
    if len(item_cells) != 15 or len({(x["source"], x["match"], x["view"], x["replicate"])
                                    for x in item_cells}) != 15:
        raise ValueError("Prior readout cohort is not exactly 15 paired cells")
    return {
        "schema": 1,
        "study": STUDY_ID,
        "status": "posthoc_fixed_blend_predeclared",
        "question": "Can a fixed equal-probability average of the frozen D2 control ensemble and short-team candidate ensemble retain complementary seed gains while passing the original practical guards?",
        "motivation": {
            "prior_readout_sha256": sha256(prior_path),
            "prior_declaration_sha256": sha256(prior_decl_path),
            "prior_input_seal_sha256": sha256(prior_seal_path),
            "observed_prior_ensemble_deltas": {
                view: prior["summaries"][CANDIDATE_ARM][view]["equal_provider_mean"]["spell_f1"] -
                      prior["summaries"][CONTROL_ARM][view]["equal_provider_mean"]["spell_f1"]
                for view in ("clean", "feed_like")},
            "interpretation": "Post-hoc follow-up motivated by both short-team seed members improving clean spell F1 while their probability ensemble was below control. This is development evidence and is not an independent confirmation.",
        },
        "arms": {CONTROL_ARM: "Frozen D2 ensemble probabilities.",
                 CANDIDATE_ARM: "Frozen short_team_2x ensemble probabilities.",
                 ARM: "Exactly 0.5 * control_D2 probabilities + 0.5 * short_team_2x probabilities, then the unchanged cost-3 decoder."},
        "blend": {"control_probability_weight": BLEND_WEIGHT,
                  "candidate_probability_weight": BLEND_WEIGHT,
                  "decoder": "unchanged shared supervised decoder, switch cost 3",
                  "fit_count": 0, "weight_search": False},
        "cohort": {"cells": item_cells,
                   "description": "The exact 5-match x clean/corrupt_1/corrupt_2 provider-holdout cohort from dense_short_team_v1."},
        "guards": {"clean_spell_f1_delta_vs_control": ">= +0.01",
                   "feed_like_spell_f1_delta_vs_control": ">= -0.01",
                   "each_provider_clean_macro_f1_delta": ">= -0.02",
                   "each_provider_clean_d_f1_delta": ">= -0.02",
                   "ranking": "official spell F1; macro F1 is secondary",
                   "no_promotion": True,
                   "stop_rule": "This is the final follow-up; stop if any declared guard fails."},
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "public_or_organizer_labels_accessed": False,
        "frame_artifacts": "Private only under data/experiments/dense_short_team_blend_v1/predictions/",
    }


def write_declaration() -> dict:
    value = declaration()
    write_once(DECLARATION, value)
    return value


def verify_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the blend declaration before evaluation")
    value = json.loads(DECLARATION.read_text())
    if value.get("study") != STUDY_ID or value.get("status") != "posthoc_fixed_blend_predeclared":
        raise ValueError("Wrong blend declaration")
    fresh = declaration()
    if value != fresh:
        raise ValueError("Blend source or prior prediction hashes changed after declaration")
    return value


def _save_blend(path: Path, probability, prediction, control, candidate, input_hashes,
                cell, fold) -> str:
    payload = {"probabilities": np.asarray(probability, dtype=np.float64),
        "prediction": np.asarray(prediction, dtype="U1"),
        "labels": control["labels"], "period": control["period"],
        "fps": control["fps"], "timestamps": control["timestamps"],
        "ball_observed": control["ball_observed"],
        "source": np.asarray(cell["source"]), "match": np.asarray(cell["match"]),
        "view": np.asarray(cell["view"]), "replicate": np.asarray(cell["replicate"]),
        "arm": np.asarray(ARM), "fold": np.asarray(fold),
        "control_prediction_sha256": np.asarray(input_hashes[CONTROL_ARM]),
        "candidate_prediction_sha256": np.asarray(input_hashes[CANDIDATE_ARM])}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_archive(path)
        for key, expected in payload.items():
            if key not in existing or not np.array_equal(existing[key], expected):
                raise ValueError(f"Refusing to overwrite changed blend archive: {path}")
        return sha256(path)
    fd_tmp = path.with_suffix(path.suffix + ".tmp")
    with fd_tmp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(fd_tmp, 0o600)
    os.replace(fd_tmp, path)
    return sha256(path)


def aggregate_blend_diagnostics(frames) -> dict:
    """Reuse frozen diagnostic code through its declared candidate arm name."""
    paired = frames.copy()
    paired["arm"] = CANDIDATE_ARM
    return evaluator.aggregate_error_cells(paired)[CANDIDATE_ARM]


def aggregate_short_spell_readout(rows: list[dict]) -> dict:
    """Keep complete <=1s and official <3s recalls separate by view/provider."""
    output = {}
    for view in ("clean", "feed_like"):
        provider_values = {}
        for provider in ("metrica", "skillcorner", "sportec"):
            selected = [r for r in rows if r["source"] == provider and r["view"] == view]
            n_refs = sum(r["complete_le_1s_reference_spells"] for r in selected)
            n_matched = sum(r["complete_le_1s_matched_spells"] for r in selected)
            n_under3 = sum(r["short_reference_spells"] for r in selected)
            matched_under3 = sum(r["short_matched_spells"] for r in selected)
            provider_values[provider] = {"complete_le_1s_reference_spells": n_refs,
                "complete_le_1s_matched_spells": n_matched,
                "complete_le_1s_recall": n_matched / n_refs if n_refs else None,
                "under_3s_reference_spells": n_under3,
                "under_3s_matched_spells": matched_under3,
                "under_3s_recall": matched_under3 / n_under3 if n_under3 else None}
        output[view] = {"per_provider": provider_values,
            "equal_provider_mean": {metric: float(np.mean([p[metric] for p in provider_values.values()
                if p[metric] is not None])) if any(p[metric] is not None for p in provider_values.values()) else None
                for metric in ("complete_le_1s_recall", "under_3s_recall")},
            "overall_counts": {key: sum(p[key] for p in provider_values.values())
                for key in ("complete_le_1s_reference_spells", "complete_le_1s_matched_spells",
                            "under_3s_reference_spells", "under_3s_matched_spells")}}
    return output


def evaluate_cell(control: dict, candidate: dict, cell: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the production blend→decode→official-score path for one paired cell."""
    validate_source_pair(control, candidate, cell)
    probability = blend_probabilities(control["probabilities"], candidate["probabilities"])
    labels = np.asarray(control["labels"], dtype="U1")
    fps = float(np.asarray(control["fps"]))
    inputs = {"period": np.asarray(control["period"]),
              "timestamps": np.asarray(control["timestamps"], dtype=float), "fps": fps}
    scoring._verify_probability(probability, len(labels),
                                f"{ARM}/{cell['source']}/{cell['match']}/{cell['view']}")
    prediction = supervised.decode(probability, supervised.model_periods(inputs),
                                   fps, evaluator.prior.lt.COST)
    score_cell = {**cell, "arm": ARM, "member": "ensemble"}
    row = scoring.score_prediction(labels, prediction, inputs, ARM, score_cell,
                                   scoring.digest_array(probability))
    row.update(evaluator.short_spell_metrics(labels, prediction, inputs))
    return probability, prediction, row


def evaluate() -> dict:
    """Evaluate the single fixed blend after its declaration is reviewed."""
    declaration_value = verify_declaration()
    prior = prior_readout()
    prior_hashes = _archive_manifest(prior)
    rows, prediction_hashes, frames = [], {}, []
    cells = evaluator._selection()
    expected_keys = set()
    for item in cells:
        fold = "leave_out_" + item["source"]
        for view, replicate in evaluator.VIEWS:
            key = (item["source"], item["match"], view, replicate)
            expected_keys.add(key)
            archives, file_hashes = {}, {}
            for arm in (CONTROL_ARM, CANDIDATE_ARM):
                path = prediction_path(item, view, replicate, arm)
                rel = str(path.relative_to(ROOT))
                actual_hash = sha256(path)
                if prior_hashes.get(rel) != actual_hash or declaration_value["input_sha256"].get(rel) != actual_hash:
                    raise ValueError(f"Source ensemble archive changed after declaration: {rel}")
                archives[arm] = load_archive(path)
                file_hashes[arm] = actual_hash
            control, candidate = archives[CONTROL_ARM], archives[CANDIDATE_ARM]
            cell = key_to_dict((item["source"], item["match"], view, replicate))
            probability, prediction, row = evaluate_cell(control, candidate, cell)
            row["source_control_prediction_sha256"] = file_hashes[CONTROL_ARM]
            row["source_candidate_prediction_sha256"] = file_hashes[CANDIDATE_ARM]
            rows.append(row)
            output = PREDICTIONS / fold / f"seedblend50__{item['match']}__{view}__{replicate}.npz"
            prediction_hashes[str(output.relative_to(ROOT))] = _save_blend(output, probability,
                prediction, control, candidate, file_hashes, key_to_dict(key), fold)
            frame = error_audit.frame_table(item["source"], item["match"], control["labels"],
                prediction, control["ball_observed"], control["period"], control["timestamps"],
                float(control["fps"]))
            frame["arm"], frame["view"], frame["replicate"] = ARM, view, replicate
            frames.append(frame)
    observed_keys = {(r["source"], r["match"], r["view"], r["replicate"]) for r in rows}
    if len(rows) != 15 or observed_keys != expected_keys:
        raise ValueError("Blend evaluation does not cover exactly the paired 15-cell cohort")
    import pandas as pd
    diagnostics = aggregate_blend_diagnostics(pd.concat(frames, ignore_index=True))
    summaries = {}
    for view in ("clean", "feed_like"):
        current = [r for r in rows if r["view"] == view]
        summaries[view] = evaluator.prior.sc._aggregate_provider(current, ARM, view, "ensemble")
    complete_short = aggregate_short_spell_readout(rows)
    comparison = {view: {"spell_f1_delta_vs_control": summaries[view]["equal_provider_mean"]["spell_f1"] -
        prior["summaries"][CONTROL_ARM][view]["equal_provider_mean"]["spell_f1"],
        "spell_f1_delta_vs_short_team": summaries[view]["equal_provider_mean"]["spell_f1"] -
        prior["summaries"][CANDIDATE_ARM][view]["equal_provider_mean"]["spell_f1"]}
        for view in ("clean", "feed_like")}
    clean_control = prior["summaries"][CONTROL_ARM]["clean"]["per_provider"]
    clean_blend = summaries["clean"]["per_provider"]
    provider_guards = {provider: {
        "macro_f1_delta": clean_blend[provider]["macro_f1"] - clean_control[provider]["macro_f1"],
        "d_f1_delta": clean_blend[provider]["d_f1"] - clean_control[provider]["d_f1"],
        "macro_pass": clean_blend[provider]["macro_f1"] - clean_control[provider]["macro_f1"] >= -.02,
        "d_f1_pass": clean_blend[provider]["d_f1"] - clean_control[provider]["d_f1"] >= -.02}
        for provider in ("metrica", "skillcorner", "sportec")}
    guards = {"clean_spell_gain_at_least_0_01": comparison["clean"]["spell_f1_delta_vs_control"] >= .01,
        "feed_like_no_worse_than_0_01": comparison["feed_like"]["spell_f1_delta_vs_control"] >= -.01,
        "provider_clean_macro_d_guard": provider_guards,
        "all_pass": comparison["clean"]["spell_f1_delta_vs_control"] >= .01 and
            comparison["feed_like"]["spell_f1_delta_vs_control"] >= -.01 and
            all(v["macro_pass"] and v["d_f1_pass"] for v in provider_guards.values())}
    result = {"schema": 1, "study": STUDY_ID,
        "declaration_sha256": sha256(DECLARATION),
        "prior_readout_sha256": sha256(RESULTS / "dense_short_team_v1_readout.json"),
        "source_sha256": declaration_value["source_sha256"],
        "source_prediction_sha256": declaration_value["input_sha256"],
        "blend_prediction_sha256": prediction_hashes, "rows": rows,
        "summaries": summaries, "diagnostics": diagnostics,
        "complete_short_spell_readout": complete_short,
        "comparison": comparison, "guards": guards,
        "public_or_organizer_labels_accessed": False,
        "private_blend_archive_count": len(prediction_hashes)}
    write_once(RESULT, result)
    return result


def key_to_dict(key):
    source, match, view, replicate = key
    return {"source": source, "match": match, "view": view, "replicate": replicate}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "evaluate"))
    args = parser.parse_args(argv)
    result = write_declaration() if args.command == "declare" else evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
