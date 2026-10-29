"""Independent pre-fit source-manifest binding for turnover_decoder_v1.

This supplemental receipt preserves the already frozen model declaration and
runner. It verifies restored TRAIN and evaluation caches against their pinned
private archive manifests before any turnover-head fitting or scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments import objective_ensemble as oe, preparation as prep, turnover_decoder as td


ROOT = td.ROOT
RECEIPT = ROOT / "experiments/results/turnover_decoder_provenance_receipt_v2.json"
OBJECTIVE_ENSEMBLE_RECEIPT = ROOT / "experiments/results/objective_ensemble_artifacts.json"
OBJECTIVE_ALIGNMENT_PREDECLARED = Path("experiments/objective_alignment_v1/predeclared.json")
OBJECTIVE_ALIGNMENT_PREPARED_MANIFEST = Path("experiments/objective_alignment_v1/prepared_training_manifest.json")
SEED_GAP_PREDECLARED = Path("experiments/seed_gap_context_v1/predeclared.json")
TRAIN_ROOT = Path("experiments/objective_alignment_v1")
SEED_GAP_ROOT = Path("experiments/seed_gap_context_v1")
OBJECTIVE_ENSEMBLE_ROOT = Path("experiments/objective_ensemble_v1")


def _local_index(archive_root: Path, filename: str, expected_sha256: str | None = None) -> tuple[dict[str, dict], str]:
    path = archive_root / filename
    if not path.is_file():
        raise td.StudyError(f"Missing private archive manifest: {path}")
    actual = td.sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise td.StudyError(f"Archive manifest hash mismatch: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("files"), list):
        raise td.StudyError(f"Malformed archive manifest: {path}")
    index = {}
    for entry in payload["files"]:
        rel = str(entry.get("path", "")).replace("\\", "/")
        if not rel or rel in index:
            raise td.StudyError(f"Malformed or duplicate archive path: {rel}")
        index[rel] = entry
    return index, actual


def _verify_indexed_file(root: Path, index: dict[str, dict], rel: str) -> dict[str, object]:
    if rel not in index:
        raise td.StudyError(f"File is absent from the independently pinned archive manifest: {rel}")
    path = root.joinpath(*Path(rel).parts)
    if not path.is_file():
        raise td.StudyError(f"Manifest file is missing from the restored workspace: {path}")
    entry = index[rel]
    actual = {"path": rel, "bytes": path.stat().st_size,
              "sha256": td.sha256_file(path)}
    if actual["bytes"] != entry.get("bytes") or actual["sha256"] != entry.get("sha256"):
        raise td.StudyError(f"Restored file does not match private archive manifest: {rel}")
    return actual


def verify(data_dir: Path | None = None) -> dict[str, object]:
    data = td.data_root(data_dir)
    declaration, declaration_sha = td._load_frozen_declaration(data)
    if RECEIPT.exists():
        raise td.StudyError(f"Provenance receipt is immutable once written: {RECEIPT}")
    archive = oe._archive_root()
    objective_index, objective_manifest_sha = _local_index(
        archive, oe.OBJECTIVE_ARCHIVE_MANIFEST, oe.OBJECTIVE_ARCHIVE_MANIFEST_SHA256)
    seed_gap_index, seed_gap_manifest_sha = _local_index(
        archive, oe.SEED_GAP_ARCHIVE_MANIFEST, oe.SEED_GAP_ARCHIVE_MANIFEST_SHA256)

    ensemble_archive = archive / "objective_ensemble_cache_manifest.json"
    ensemble_receipt = json.loads(OBJECTIVE_ENSEMBLE_RECEIPT.read_text())
    ensemble_index, ensemble_manifest_sha = _local_index(
        archive, "objective_ensemble_cache_manifest.json",
        ensemble_receipt.get("manifest_sha256"))
    if (not ensemble_receipt.get("remote_push_complete") or
            not ensemble_receipt.get("fresh_remote_clone_restored_all_files") or
            ensemble_receipt.get("manifest") != ensemble_archive.name or
            ensemble_receipt.get("fresh_restored_source_snapshot_check", {}).get("inputs_verified") is not True):
        raise td.StudyError("Objective-ensemble artifact receipt lacks completed verification")
    if (len(ensemble_index) != ensemble_receipt.get("files") or
            sum(int(entry.get("bytes", -1)) for entry in ensemble_index.values()) != ensemble_receipt.get("bytes")):
        raise td.StudyError("Objective-ensemble artifact receipt file count/bytes differ from its manifest")

    objective_predeclared = json.loads(
        (data / OBJECTIVE_ALIGNMENT_PREDECLARED).read_text())
    if objective_predeclared.get("prepared_training_manifest_sha256") is None:
        raise td.StudyError("Objective-alignment declaration lacks prepared-manifest hash")
    prepared_manifest_hash = td.sha256_file(data / OBJECTIVE_ALIGNMENT_PREPARED_MANIFEST)
    if prepared_manifest_hash != objective_predeclared["prepared_training_manifest_sha256"]:
        raise td.StudyError("Prepared training manifest does not match the objective declaration")
    prepared_manifest = json.loads((data / OBJECTIVE_ALIGNMENT_PREPARED_MANIFEST).read_text())
    if prepared_manifest_hash != objective_predeclared.get("prepared_training_manifest_sha256"):
        raise td.StudyError("Prepared training manifest hash is inconsistent")

    objective_source_fact = _verify_indexed_file(
        data, objective_index, OBJECTIVE_ALIGNMENT_PREDECLARED.as_posix())
    objective_prepared_fact = _verify_indexed_file(
        data, objective_index, OBJECTIVE_ALIGNMENT_PREPARED_MANIFEST.as_posix())
    prep_sha = td.sha256_file(data / OBJECTIVE_ALIGNMENT_PREDECLARED)
    if prep_sha != objective_source_fact["sha256"]:
        raise td.StudyError("Objective-alignment declaration differs from its archived source receipt")

    pooled_plan = prep.split_plan(fold="pooled")
    plan_matches = {item["match"]: item for item in pooled_plan["train"]}
    selection_matches = {(item["source"], item["match"]): item for item in pooled_plan["selection"]}
    train_input_facts = []
    for match, item in sorted(plan_matches.items()):
        relative = item["path"].removeprefix("data/")
        raw_fact = _verify_indexed_file(data, objective_index, relative)
        if raw_fact["sha256"] != item["sha256"]:
            raise td.StudyError(f"Raw TRAIN input differs from evaluation manifest/archive: {match}")
        train_input_facts.append({"match": match, "source": item["source"],
                                  "raw_input": raw_fact})
    if set(plan_matches) != set(objective_predeclared["prepared_training_caches"]):
        raise td.StudyError("Prepared cache manifest does not cover the 13 declared TRAIN matches")
    frozen_train_hashes = {}
    for provider in td.PROVIDERS:
        for match, digest in declaration["cohort"]["provider_folds"][provider]["train_cache_sha256"].items():
            if match in frozen_train_hashes and frozen_train_hashes[match] != digest:
                raise td.StudyError(f"Frozen train-cache hash differs across folds: {match}")
            frozen_train_hashes[match] = digest
    train_facts = []
    for match, item in sorted(plan_matches.items()):
        expected = objective_predeclared["prepared_training_caches"][match]
        cache_rel = expected["cache"]
        receipt_rel = expected["receipt"]
        if cache_rel != f"data/{TRAIN_ROOT.as_posix()}/train_{match}.joblib":
            raise td.StudyError(f"Unexpected prepared cache path for {match}")
        cache_path = data.joinpath(*Path(cache_rel).parts[1:])
        receipt_path = data.joinpath(*Path(receipt_rel).parts[1:])
        cache_fact = _verify_indexed_file(data, objective_index, cache_rel.removeprefix("data/"))
        receipt_fact = _verify_indexed_file(data, objective_index, receipt_rel.removeprefix("data/"))
        receipt = json.loads(receipt_path.read_text())
        outer = __import__("joblib").load(cache_path)
        if (cache_fact["sha256"] != expected["sha256"] or
                cache_fact["bytes"] != expected["bytes"] or
                receipt_fact["sha256"] != expected["receipt_sha256"] or
                item["sha256"] != expected["input_sha256"] or
                receipt.get("sha256") != expected["sha256"] or
                receipt.get("bytes") != expected["bytes"] or
                receipt.get("input_sha256") != item["sha256"] or
                receipt.get("preparation_sha256") != objective_predeclared.get("preparation_declaration_sha256") or
                outer.get("input_sha256") != item["sha256"] or
                outer.get("preparation_sha256") != receipt.get("preparation_sha256") or
                outer.get("match") != match or outer.get("source") != item["source"]):
            raise td.StudyError(f"TRAIN cache/receipt/preparation/input provenance mismatch: {match}")
        if cache_fact["sha256"] != frozen_train_hashes.get(match):
            raise td.StudyError(f"TRAIN cache changed between declaration and archive verification: {match}")
        train_facts.append({"match": match, "source": item["source"],
                            "input_sha256": item["sha256"], "cache": cache_fact,
                            "receipt": receipt_fact})

    checkpoint_facts = []
    for provider in td.PROVIDERS:
        fold = declaration["cohort"]["provider_folds"][provider]
        model_rows = fold["checkpoint_facts"]["models"]
        for seed, item in sorted(model_rows.items()):
            model_rel = item["model"]
            receipt_rel = item["fit_receipt"]
            model_fact = _verify_indexed_file(data, objective_index, model_rel)
            receipt_fact = _verify_indexed_file(data, objective_index, receipt_rel)
            if (model_fact["sha256"] != item["sha256"] or
                    receipt_fact["sha256"] != item["fit_receipt_sha256"]):
                raise td.StudyError(f"Fold checkpoint or fit receipt differs from frozen/archive provenance: {provider}/{seed}")
            checkpoint_facts.append({"provider_excluded": provider, "seed": int(seed),
                                     "model": model_fact, "fit_receipt": receipt_fact})

    seed_predeclared_fact = _verify_indexed_file(
        data, seed_gap_index, SEED_GAP_PREDECLARED.as_posix())
    seed_predeclared = json.loads((data / SEED_GAP_PREDECLARED).read_text())
    if seed_predeclared.get("id") != "seed_gap_context_v1":
        raise td.StudyError("Unexpected seed-gap predeclaration identity")

    evaluation_facts = []
    selection_input_facts = []
    for frozen in declaration["cohort"]["cells"]:
        cell = frozen["cell"]
        selection_item = selection_matches[(cell["source"], cell["match"])]
        raw_rel = selection_item["path"].removeprefix("data/")
        raw_fact = _verify_indexed_file(data, objective_index, raw_rel)
        declared_raw_sha = seed_predeclared.get("inputs_sha256", {}).get(selection_item["path"])
        if raw_fact["sha256"] != selection_item["sha256"] or raw_fact["sha256"] != declared_raw_sha:
            raise td.StudyError(f"Raw selection input differs from manifest/archive: {cell}")
        if not any(item["cell"] == {"source": cell["source"], "match": cell["match"]}
                   for item in selection_input_facts):
            selection_input_facts.append({"cell": {"source": cell["source"], "match": cell["match"]},
                                           "raw_input": raw_fact,
                                           "seed_gap_predeclared_input_sha256": declared_raw_sha})
        stem = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
        eval_joblib = f"{SEED_GAP_ROOT.as_posix()}/{stem}.joblib"
        eval_probability = f"{SEED_GAP_ROOT.as_posix()}/{stem}.npz"
        job_fact = _verify_indexed_file(data, seed_gap_index, eval_joblib)
        probability_fact = _verify_indexed_file(data, seed_gap_index, eval_probability)
        if job_fact["sha256"] != frozen["inputs_labels_record_sha256"]:
            raise td.StudyError(f"Evaluation input cache differs from frozen/archived source: {cell}")
        candidate_rel = f"{OBJECTIVE_ENSEMBLE_ROOT.as_posix()}/candidate_{stem}.npz"
        candidate_fact = _verify_indexed_file(data, ensemble_index, candidate_rel)
        if candidate_fact["sha256"] != frozen["candidate_ensemble_sha256"]:
            raise td.StudyError(f"Candidate ensemble differs from its verified archive: {cell}")
        members = {}
        for seed in td.SEEDS:
            member_rel = (f"{TRAIN_ROOT.as_posix()}/{cell['source']}_{cell['match']}_"
                          f"{cell['view']}_{cell['replicate']}_combined_{seed}.npz")
            member_fact = _verify_indexed_file(data, objective_index, member_rel)
            ensemble_member_fact = _verify_indexed_file(data, ensemble_index, member_rel)
            if (member_fact["sha256"] != frozen["member_probabilities"][str(seed)]["sha256"] or
                    ensemble_member_fact["sha256"] != member_fact["sha256"]):
                raise td.StudyError(f"Evaluation member probability does not match both archive manifests: {cell}/{seed}")
            members[str(seed)] = member_fact
        evaluation_facts.append({"cell": cell, "inputs_labels_record": job_fact,
                                 "seed_gap_probability_archive": probability_fact,
                                 "objective_ensemble_candidate": candidate_fact,
                                 "member_probabilities": members})

    receipt = {
        "study_id": td.STUDY_ID,
        "declaration_sha256": declaration_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "pre_fit_no_turnover_head_fit_or_evaluation_scoring",
        "verifier_source": "experiments/turnover_decoder_provenance.py",
        "verifier_source_sha256": td.sha256_file(Path(__file__)),
        "archive_manifests": {
            "objective_alignment": {"path": oe.OBJECTIVE_ARCHIVE_MANIFEST,
                                    "sha256": objective_manifest_sha,
                                    "pinned_sha256": oe.OBJECTIVE_ARCHIVE_MANIFEST_SHA256,
                                    "files": len(objective_index),
                                    "bytes": sum(int(entry.get("bytes", -1)) for entry in objective_index.values())},
            "seed_gap_evaluation": {"path": oe.SEED_GAP_ARCHIVE_MANIFEST,
                                    "sha256": seed_gap_manifest_sha,
                                    "pinned_sha256": oe.SEED_GAP_ARCHIVE_MANIFEST_SHA256,
                                    "files": len(seed_gap_index),
                                    "bytes": sum(int(entry.get("bytes", -1)) for entry in seed_gap_index.values())},
            "objective_ensemble": {"path": ensemble_archive.name,
                                   "sha256": ensemble_manifest_sha,
                                   "artifact_receipt_sha256": td.sha256_file(OBJECTIVE_ENSEMBLE_RECEIPT),
                                   "remote_restore_claim": ensemble_receipt["fresh_remote_clone_restored_all_files"],
                                   "files": len(ensemble_index), "bytes": ensemble_receipt["bytes"]},
        },
        "objective_alignment_declaration": objective_source_fact,
        "prepared_training_manifest": {**objective_prepared_fact,
                                        "declared_sha256": prepared_manifest_hash,
                                        "cache_count": len(prepared_manifest.get("caches", []))},
        "training_caches": train_facts,
        "training_raw_inputs": train_input_facts,
        "frozen_fold_checkpoints": checkpoint_facts,
        "seed_gap_predeclaration": seed_predeclared_fact,
        "selection_raw_inputs": selection_input_facts,
        "evaluation_cells": evaluation_facts,
        "head_fits": 0, "scores_read": 0, "organizer_data_or_scores_read": False,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return {"receipt": str(RECEIPT.relative_to(ROOT)),
            "receipt_sha256": td.sha256_file(RECEIPT),
            "training_matches_verified": len(train_facts),
            "evaluation_cells_verified": len(evaluation_facts),
            "head_fits": 0, "scores_read": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
