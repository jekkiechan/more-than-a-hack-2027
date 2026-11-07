"""Second, pre-scoring runtime receipt for the frozen v2 training evaluation.

The original evaluation source and declaration are preserved. This wrapper
corrects only the alias mismatch between the evaluator's provider-fold names
and the already sealed `leave_out_*` training artifact names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments import v2_training_eval as frozen


ROOT = frozen.ROOT
STUDY = "v2_training_eval_runtime_v2"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
PREVIOUS_CORRECTION = ROOT / "experiments/results/v2_training_eval_runtime_correction.json"
OBSERVED_FAILURE = (
    "First run stopped before candidate inference/scoring at the first phase: "
    "evaluation resolved data/experiments/v2_short_original_diversity/models/"
    "sportec_2026.joblib, while the sealed fit is named leave_out_sportec_2026.joblib"
)


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  default=str).encode()).hexdigest()


def corrected_fit_path(fold: str, seed: int) -> Path:
    trainer_fold = fold if fold.startswith("leave_out_") else f"leave_out_{fold}"
    if trainer_fold not in frozen.FOLDS or seed not in frozen.SEEDS:
        raise ValueError(f"Unlisted fit fold/seed alias: {fold}/{seed}")
    return frozen.training.OUT / "models" / f"{trainer_fold}_{seed}.joblib"


def declare() -> dict:
    """Write the runtime amendment after fit sealing and before candidate scoring."""
    if DECLARATION.exists():
        raise RuntimeError("Runtime v2 declaration already exists; preserve it")
    if frozen.RESULT.exists():
        raise RuntimeError("A candidate result exists; runtime amendment is too late")
    if not PREVIOUS_CORRECTION.is_file():
        raise FileNotFoundError("Preserve the first bounded correction receipt before v2 amendment")
    base, seal, preflight = frozen._require_seal()
    original_training = json.loads(frozen.training.DECLARATION.read_text())
    aliases, models = {}, {}
    for provider, fold_alias in frozen.upstream.frozen.audit.FOLD.items():
        trainer_fold = f"leave_out_{fold_alias}"
        aliases[provider] = {"runtime_alias": fold_alias, "trainer_fold": trainer_fold}
        for seed in frozen.SEEDS:
            key = f"{trainer_fold}/{seed}"
            fact = seal["models"].get(key)
            path = corrected_fit_path(fold_alias, seed)
            if fact is None or str(path.relative_to(ROOT)) != fact["path"]:
                raise RuntimeError(f"Alias map does not equal the sealed path: {key}")
            if _sha(path) != fact["sha256"]:
                raise RuntimeError(f"Alias target hash differs from the sealed fit: {key}")
            models[key] = {"path": fact["path"], "sha256": fact["sha256"]}
    candidate_cache_dir = frozen.OUT / "emissions"
    candidate_cache_files = sorted(candidate_cache_dir.glob("*.npz")) if candidate_cache_dir.is_dir() else []
    if candidate_cache_files:
        raise RuntimeError("A candidate cache exists; preserve and inspect before resuming")
    value = {
        "study_id": STUDY,
        "status": "runtime_correction_frozen_after_six_fits_before_any_candidate_scoring",
        "original_pre_fit_plan": {
            "evaluation_declaration_sha256": _sha(frozen.DECLARATION),
            "training_declaration_sha256": _sha(frozen.training.DECLARATION),
            "training_plan_sha256": _canonical_sha(original_training["train_plan"]),
            "training_source_sha256": original_training["source_sha256"],
            "original_question": base["question"],
            "arms": base["arms"],
            "cohort": base["cohort"],
            "budget": base["budget"],
        },
        "correction": "Normalize short audit fold aliases to the existing exact leave_out_* model paths; leave an already-prefixed leave_out_* fold unchanged.",
        "observed_failure": OBSERVED_FAILURE,
        "execution_timing": "after all six fits and their hashes were sealed; before any candidate inference, emission caching or scoring",
        "failed_attempt": {"stopped_before_candidate_inference": True,
                           "candidate_emission_cache_files": 0,
                           "candidate_result_written": False},
        "runtime_source_sha256": {
            str(Path(__file__).relative_to(ROOT)): _sha(__file__),
            str(Path(frozen.__file__).relative_to(ROOT)): _sha(frozen.__file__),
        },
        "original_declaration_sha256": _sha(frozen.DECLARATION),
        "original_seal_sha256": _sha(frozen.SEAL),
        "upstream_preflight_sha256": seal["upstream_preflight"]["sha256"],
        "provider_aliases": aliases,
        "sealed_model_paths": models,
        "preserved_prior_correction_receipt": {
            "path": str(PREVIOUS_CORRECTION.relative_to(ROOT)),
            "sha256": _sha(PREVIOUS_CORRECTION),
        },
        "arms_and_protocol": {"arms": base["arms"], "decoder": "f062",
                              "cohort": base["cohort"], "controls": base["controls"],
                              "aggregation": base["cohort"]["aggregation"]},
        "new_fits": 0,
        "new_decoder_settings": 0,
        "organizer_reference_access": False,
        "public_scoring": False,
        "promotion": False,
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    with DECLARATION.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")
    return value


def _verify() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the runtime v2 correction declaration before scoring")
    value = json.loads(DECLARATION.read_text())
    if value.get("study_id") != STUDY:
        raise RuntimeError("Unexpected runtime correction ID")
    if value["runtime_source_sha256"].get(str(Path(__file__).relative_to(ROOT))) != _sha(__file__):
        raise RuntimeError("Runtime correction source changed")
    if value["runtime_source_sha256"].get(str(Path(frozen.__file__).relative_to(ROOT))) != _sha(frozen.__file__):
        raise RuntimeError("Original frozen evaluator source changed")
    if value["original_declaration_sha256"] != _sha(frozen.DECLARATION):
        raise RuntimeError("Original pre-fit evaluation plan changed")
    if value["original_seal_sha256"] != _sha(frozen.SEAL):
        raise RuntimeError("Six-model seal changed")
    if value["preserved_prior_correction_receipt"]["sha256"] != _sha(PREVIOUS_CORRECTION):
        raise RuntimeError("Prior correction receipt changed")
    base, seal, preflight = frozen._require_seal()
    if base["arms"] != value["arms_and_protocol"]["arms"]:
        raise RuntimeError("Runtime correction arm list differs from the pre-fit declaration")
    for key, fact in value["sealed_model_paths"].items():
        if seal["models"][key]["path"] != fact["path"] or _sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Corrected path no longer matches sealed model: {key}")
    if preflight.get("preflight_passed") is not True:
        raise RuntimeError("Upstream controls-only C0 preflight is not passing")
    return value


def run() -> dict:
    _verify()
    frozen._fit_path = corrected_fit_path
    return frozen.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run"))
    action = parser.parse_args().action
    {"declare": declare, "run": run}[action]()


if __name__ == "__main__":
    main()
