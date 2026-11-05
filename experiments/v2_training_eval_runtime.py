"""Bounded execution correction for the frozen v2_training_eval runner.

The frozen runner passes `audit.FOLD` aliases such as `sportec` to its fit
path helper, while the declared trainer stores artifacts as
`leave_out_sportec_{seed}.joblib`. This wrapper records and verifies that
one-to-one path mapping before scoring, then delegates all candidate logic to
the unchanged frozen runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments import v2_training_eval as frozen


ROOT = frozen.ROOT
CORRECTION = ROOT / "experiments/results/v2_training_eval_runtime_correction.json"
OBSERVED_FAILURE = (
    "FileNotFoundError before candidate inference/scoring: the first evaluation cell "
    "resolved data/experiments/v2_short_original_diversity/models/sportec_2026.joblib; "
    "the frozen training runner stores the sealed model at leave_out_sportec_2026.joblib"
)


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corrected_fit_path(fold: str, seed: int) -> Path:
    trainer_fold = fold if fold.startswith("leave_out_") else f"leave_out_{fold}"
    if trainer_fold not in frozen.FOLDS or seed not in frozen.SEEDS:
        raise ValueError(f"Unlisted fit fold/seed alias: {fold}/{seed}")
    return frozen.training.OUT / "models" / f"{trainer_fold}_{seed}.joblib"


def declare_correction() -> dict:
    """Freeze the path-only correction and prove all sealed targets are exact."""
    if CORRECTION.exists():
        raise RuntimeError("Runtime correction receipt already exists; preserve it")
    if frozen.RESULT.exists():
        raise RuntimeError("A candidate result already exists; runtime correction is too late")
    declaration, seal, preflight = frozen._require_seal()
    models = {}
    aliases = {}
    for provider, short_fold in frozen.upstream.frozen.audit.FOLD.items():
        trainer_fold = f"leave_out_{short_fold}"
        aliases[provider] = {"runtime_alias": short_fold, "trainer_fold": trainer_fold}
        for seed in frozen.SEEDS:
            key = f"{trainer_fold}/{seed}"
            fact = seal["models"].get(key)
            path = _corrected_fit_path(short_fold, seed)
            if fact is None or str(path.relative_to(ROOT)) != fact["path"]:
                raise RuntimeError(f"Runtime alias does not resolve to the sealed model: {key}")
            if _sha(path) != fact["sha256"]:
                raise RuntimeError(f"Corrected runtime path hash differs from sealed model: {key}")
            models[key] = {"path": fact["path"], "sha256": fact["sha256"]}
    cache_dir = frozen.OUT / "emissions"
    cache_files = sorted(cache_dir.glob("*.npz")) if cache_dir.is_dir() else []
    if cache_files:
        raise RuntimeError("A failed or partial candidate cache exists; preserve and review it")
    value = {
        "study_id": "v2_training_eval_runtime_correction",
        "status": "frozen_before_candidate_inference_or_scoring",
        "correction": "Map audit fold aliases sportec/metrica/skillcorner to the exact sealed trainer paths leave_out_{provider}_{seed}.joblib.",
        "observed_failure": OBSERVED_FAILURE,
        "scope": "Path lookup only. Frozen training declaration, evaluation declaration, model seal, three arms, cohort, f062 decoding, aggregation and scorer remain bound unchanged.",
        "failed_attempt": {"occurred_before_candidate_inference": True,
                           "candidate_emission_cache_files": 0,
                           "candidate_result_written": False},
        "original_study": {
            "declaration_path": str(frozen.DECLARATION.relative_to(ROOT)),
            "declaration_sha256": _sha(frozen.DECLARATION),
            "runner_path": str(Path(frozen.__file__).relative_to(ROOT)),
            "runner_sha256": _sha(frozen.__file__),
            "seal_path": str(frozen.SEAL.relative_to(ROOT)),
            "seal_sha256": _sha(frozen.SEAL),
            "upstream_preflight_sha256": seal["upstream_preflight"]["sha256"],
            "training_declaration_sha256": seal["training_declaration_sha256"],
        },
        "provider_aliases": aliases,
        "sealed_model_paths": models,
        "arm_list": declaration["arms"],
        "new_fits": 0,
        "new_decoder_settings": 0,
        "organizer_reference_access": False,
        "public_scoring": False,
    }
    CORRECTION.parent.mkdir(parents=True, exist_ok=True)
    with CORRECTION.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")
    return value


def _verify_correction() -> dict:
    if not CORRECTION.is_file():
        raise RuntimeError("Freeze the runtime path correction before scoring")
    correction = json.loads(CORRECTION.read_text())
    if correction.get("study_id") != "v2_training_eval_runtime_correction":
        raise RuntimeError("Unexpected runtime correction receipt")
    if correction["original_study"]["declaration_sha256"] != _sha(frozen.DECLARATION):
        raise RuntimeError("Original evaluation declaration changed")
    if correction["original_study"]["runner_sha256"] != _sha(frozen.__file__):
        raise RuntimeError("Frozen evaluation runner changed")
    if correction["original_study"]["seal_sha256"] != _sha(frozen.SEAL):
        raise RuntimeError("Original evaluation model seal changed")
    declaration, seal, preflight = frozen._require_seal()
    if declaration["arms"] != correction["arm_list"]:
        raise RuntimeError("Corrected runtime arms differ from frozen declaration")
    for key, fact in correction["sealed_model_paths"].items():
        if seal["models"][key]["path"] != fact["path"]:
            raise RuntimeError(f"Corrected model path differs from original seal: {key}")
        if _sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Corrected model hash changed: {key}")
    if preflight.get("preflight_passed") is not True:
        raise RuntimeError("Upstream C0 preflight is not passing")
    return correction


def run() -> dict:
    _verify_correction()
    frozen._fit_path = _corrected_fit_path
    return frozen.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare-correction", "run"))
    action = parser.parse_args().action
    {"declare-correction": declare_correction, "run": run}[action]()


if __name__ == "__main__":
    main()
