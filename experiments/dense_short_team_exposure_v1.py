"""Read-only audit of short-spell exposure under the fixed D2 sampler.

Replays only Python's declared window-selection RNG for the six short-team
fits. It does not instantiate a network, run inference, or read evaluation
scores. Aggregate receipts can be written after the fit receipts exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from experiments import dense_short_team_v1 as study

ROOT = study.ROOT
RESULT = ROOT / "experiments/results/dense_short_team_exposure_v1.json"
STEPS = 2048
BATCH_SIZE = 32
OLD_PER_BATCH = 16
GAR_PER_BATCH = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_draws(old_groups: dict, old_records: list[dict], old_ids: list[tuple],
                   new_groups: dict, new_records: list[dict], new_ids: list[tuple],
                   old_short: list[np.ndarray], new_short: list[np.ndarray], *,
                   seed: int, steps: int = STEPS) -> dict:
    """Mirror train_candidate's Python RNG calls and count sampled core rows."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = random.Random(int(seed))
    exposure = Counter()
    source_counts = {"old": Counter(), "gar": Counter()}
    rows = {"old": Counter(), "gar": Counter()}
    short_rows = {"old": Counter(), "gar": Counter()}
    contains_short = Counter()

    for _ in range(int(steps)):
        for group, count in (("old", OLD_PER_BATCH), ("gar", GAR_PER_BATCH)):
            for _slot in range(count):
                if group == "old":
                    ri, start, size, _full, _alternate = study.cap._draw_old(
                        rng, old_groups, old_records)
                    record, identities, weights = old_records, old_ids, old_short
                else:
                    ri, start, size, _full, _alternate = study.cap._draw_new(
                        rng, new_groups)
                    record, identities, weights = new_records, new_ids, new_short
                    # train_candidate draws the SoccerNet team-swap flag here.
                    rng.random()

                identity = identities[ri]
                key = "/".join(str(part) for part in identity)
                exposure[f"{'old' if group == 'old' else 'new'}/{key}"] += 1
                source_counts[group][key] += 1
                y = np.asarray(record[ri]["y"])[start:start + size]
                w = np.asarray(weights[ri])[start:start + size]
                known_ha = (y == 0) | (y == 1)
                short_ha = known_ha & (w > 1.0)
                rows[group]["known_ha"] += int(known_ha.sum())
                short_rows[group]["eligible_le_1s"] += int(short_ha.sum())
                rows[group]["core_rows"] += int(size)
                if short_ha.any():
                    contains_short[group] += 1

        # Match the trainer's final random operation in every batch; this
        # affects the next batch's sampled windows even though counts ignore
        # the resulting tensor order.
        order = list(range(BATCH_SIZE))
        rng.shuffle(order)

    return {
        "seed": int(seed), "steps": int(steps), "batch_size": BATCH_SIZE,
        "draws": {"old": int(steps * OLD_PER_BATCH), "gar": int(steps * GAR_PER_BATCH)},
        "known_ha_core_rows": {key: int(value["known_ha"]) for key, value in rows.items()},
        "eligible_le_1s_ha_core_rows": {key: int(value["eligible_le_1s"]) for key, value in short_rows.items()},
        "short_containing_draws": {key: int(value) for key, value in contains_short.items()},
        "window_exposures": dict(sorted(exposure.items())),
        "draws_by_source_match_or_game": {
            key: dict(sorted(value.items())) for key, value in source_counts.items()
        },
    }


def synthetic_check() -> dict:
    """Small deterministic sampler contract; it uses no repository data."""
    old_records = [
        {"y": np.array([0, 0, 1, 1, 2, 0], dtype=np.int8)},
        {"y": np.array([2, 1, 1, 2, 0, 0], dtype=np.int8)},
    ]
    new_records = [
        {"y": np.array([0, 1, 2, 0, 0, 1], dtype=np.int8)},
        {"y": np.array([2, 0, 0, 1, 1, 2], dtype=np.int8)},
    ]
    old_ids = [("sportec", "synthetic_a"), ("metrica", "synthetic_b")]
    new_ids = [("gar", "synthetic_game_a"), ("gar", "synthetic_game_b")]
    old_groups = {
        ("sportec", "synthetic_a"): [(0, 0, 3, np.arange(3), False),
                                      (0, 3, 3, np.arange(3, 6), False)],
        ("metrica", "synthetic_b"): [(1, 0, 3, np.arange(3), False),
                                     (1, 3, 3, np.arange(3, 6), False)],
    }
    new_groups = {
        "synthetic_game_a": [(0, 0, 3, np.arange(3), False),
                             (0, 3, 3, np.arange(3, 6), False)],
        "synthetic_game_b": [(1, 0, 3, np.arange(3), False),
                             (1, 3, 3, np.arange(3, 6), False)],
    }
    old_short = [np.array([2, 2, 1, 1, 1, 2], dtype=np.float32),
                 np.array([1, 1, 1, 2, 2, 1], dtype=np.float32)]
    new_short = [np.array([2, 1, 1, 1, 2, 2], dtype=np.float32),
                 np.array([1, 2, 2, 1, 1, 1], dtype=np.float32)]
    first = simulate_draws(old_groups, old_records, old_ids, new_groups, new_records,
                           new_ids, old_short, new_short, seed=2026, steps=19)
    second = simulate_draws(old_groups, old_records, old_ids, new_groups, new_records,
                            new_ids, old_short, new_short, seed=2026, steps=19)
    if first != second:
        raise AssertionError("Sampler replay is not deterministic")
    expected_draws = 19 * 16
    if first["draws"] != {"old": expected_draws, "gar": expected_draws}:
        raise AssertionError("Synthetic draw totals do not match the fixed batch recipe")
    if not all(first["short_containing_draws"][key] > 0 for key in ("old", "gar")):
        raise AssertionError("Synthetic fixture failed to exercise eligible short frames")
    if len(first["window_exposures"]) < 4:
        raise AssertionError("Synthetic fixture did not exercise hierarchical window choices")
    return {"passed": True, "steps": 19, "counts": first}


def _fit_receipt(fold: str, seed: int) -> dict | None:
    path = study.candidate_fit_receipt_path(fold, seed)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def audit_fit(fold: str, seed: int, *, steps: int = STEPS) -> dict:
    """Load one fold's TRAIN records and replay one fit's draw counts."""
    teacher, old_records, new_records = study.load_training(fold, seed)
    names, _median, _scale, _width = study.cap._model_input_width(teacher)
    old_groups, old_ids = study.cap._window_index(old_records, names, old=True)
    new_groups, new_ids = study.cap._window_index(new_records, names, old=False)
    old_short = [study.short_spell_frame_weights(record) for record in old_records]
    new_short = [study.short_spell_frame_weights(record) for record in new_records]
    observed = simulate_draws(old_groups, old_records, old_ids, new_groups, new_records,
                              new_ids, old_short, new_short, seed=seed, steps=steps)
    receipt = _fit_receipt(fold, seed)
    receipt_check = "pending_fit_receipt"
    if receipt is not None:
        if receipt.get("exposures") != observed["window_exposures"]:
            raise ValueError(f"Sampler exposure differs from candidate receipt: {fold}/{seed}")
        receipt_check = "exact_match"

    data_hashes = study._training_input_hashes(fold, seed)
    input_digest = hashlib.sha256(json.dumps(
        data_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_paths = (Path(__file__), ROOT / "experiments/dense_short_team_v1.py",
                    ROOT / "experiments/soccernet_capacity_model.py",
                    ROOT / "experiments/objective_losses.py",
                    ROOT / "experiments/learned_temporal.py")
    return {
        "fold": fold, "heldout_provider": fold.removeprefix("leave_out_"),
        "sampler_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "training_input_file_count": len(data_hashes),
        "training_input_set_sha256": input_digest,
        "fit_receipt_sha256": (sha256(study.candidate_fit_receipt_path(fold, seed))
                                if receipt is not None else None),
        "fit_receipt_exposure_check": receipt_check,
        "exposure": observed,
    }


def run(*, folds: tuple[str, ...] = study.FOLDS,
        seeds: tuple[int, ...] = study.SEEDS, steps: int = STEPS,
        output: Path | None = RESULT) -> dict:
    """Run fold-at-a-time and write a small aggregate-only receipt once."""
    rows = []
    for fold in folds:
        for seed in seeds:
            rows.append(audit_fit(fold, seed, steps=steps))
    result = {
        "schema": 1, "study": "dense_short_team_exposure_v1",
        "purpose": "Read-only deterministic replay of TRAIN window exposure; no fit, inference, or evaluation scoring",
        "sampler": {"steps": steps, "batch_size": BATCH_SIZE,
                    "old_windows_per_batch": OLD_PER_BATCH,
                    "gar_windows_per_batch": GAR_PER_BATCH,
                    "rng": "Python random.Random(seed), same old/GAR draws, GAR swap flag draws, and per-batch shuffle as train_candidate"},
        "folds": list(folds), "seeds": list(seeds), "fits": rows,
        "public_or_organizer_labels_accessed": False,
    }
    if output is not None:
        output = Path(output)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.read_text() != payload:
                raise ValueError(f"Refusing to overwrite read-only audit receipt: {output}")
        else:
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "run"))
    parser.add_argument("--fold", choices=study.FOLDS, action="append")
    parser.add_argument("--seed", type=int, choices=study.SEEDS, action="append")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        value = synthetic_check()
    else:
        value = run(folds=tuple(args.fold or study.FOLDS),
                    seeds=tuple(args.seed or study.SEEDS), steps=args.steps,
                    output=None if args.no_write else RESULT)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
