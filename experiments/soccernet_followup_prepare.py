"""Recover paired raw ball-gap views for the frozen SoccerNet TRAIN segments.

This preparation step consumes only the verified TRAIN source and the sealed
base segment cache. It preserves the historical assembler and feature recipe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import io
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing

import joblib
import numpy as np

from experiments import soccernet_segments as seg
from experiments import soccernet_gap_consistency as gap

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "soccernet_followup_raw_gap_views_v1"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _identity_seed(game_id: str, segment_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{game_id}\0{segment_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _raw_inputs(rows: list[dict], jerseys: list[list[str]], fps: float) -> dict:
    n = len(rows)
    slots = [{value: i for i, value in enumerate(jerseys[0])},
             {value: i + len(jerseys[0]) for i, value in enumerate(jerseys[1])}]
    players = np.full((n, len(jerseys[0]) + len(jerseys[1]), 2), np.nan, dtype=np.float64)
    player_observed = np.zeros(players.shape[:2], dtype=bool)
    ball = np.full((n, 2), np.nan, dtype=np.float64)
    ball_observed = np.zeros(n, dtype=bool)
    for i, row in enumerate(rows):
        for side_i, side in enumerate(("home", "away")):
            for jersey, visibility, x, y in row[side]:
                if jersey is None or visibility != "VISIBLE" or not np.isfinite((x, y)).all():
                    continue
                slot = slots[side_i].get(jersey)
                if slot is not None and not player_observed[i, slot]:
                    players[i, slot] = (x + 52.5, y + 34.0)
                    player_observed[i, slot] = True
        for _, visibility, x, y in row["balls"]:
            if visibility == "VISIBLE" and np.isfinite((x, y)).all():
                ball[i] = (x + 52.5, y + 34.0)
                ball_observed[i] = True
                break
    return {
        "ball": ball,
        "players": players,
        "teams": np.r_[np.zeros(len(jerseys[0]), dtype=np.int8),
                       np.ones(len(jerseys[1]), dtype=np.int8)],
        "period": np.full(n, int(rows[0]["period"]), dtype=np.int32),
        "fps": np.asarray(float(fps)),
        "timestamps": np.asarray([r["timestamp"] for r in rows], dtype=np.float64),
        "pitch": np.asarray([105.0, 68.0]),
        "ball_observed": ball_observed,
        "player_observed": player_observed,
        "tracking_valid": np.ones(n, dtype=bool),
    }


def _read_game_samples(tracking: Path):
    samples, _ = seg._read_train_samples(tracking)
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["game_id"], []).append(sample)
    return {game: sorted(items, key=lambda x: (x["start_frame"], x["sample_id"]))
            for game, items in grouped.items()}


def _capture_raw_segments(game_id: str, samples: list[dict], clips, feature_names: list[str]) -> dict[str, dict]:
    """Capture raw segments through the frozen assembler without extracting features."""
    original = seg._feature_arrays
    raw: dict[str, dict] = {}
    def capture(rows, jerseys, fps):
        sid = f"{game_id}_p{rows[0]['period']}_{rows[0]['frame']}-{rows[-1]['frame']}"
        raw_inputs = _raw_inputs(rows, jerseys, fps)
        raw[sid] = raw_inputs
        n = len(rows)
        zero_stats = {"missing_jersey_entities": 0, "duplicate_jersey_entities": 0,
                      "visible_player_observations": 0, "estimated_player_entities_ignored": 0,
                      "visible_ball_observations": 0, "estimated_ball_entities_ignored": 0,
                      "entity_coords_outside_nominal_pitch": 0}
        # _assemble_game needs only the frame count, schema and masks here. This
        # placeholder is discarded; attach_consistency_view later recomputes both
        # real views and checks clean features against the sealed base record.
        return (np.zeros((n, len(feature_names)), dtype=np.float32), list(feature_names),
                raw_inputs["player_observed"], raw_inputs["ball_observed"], zero_stats)
    seg._feature_arrays = capture
    try:
        seg._assemble_game(game_id, samples, clips)
    finally:
        seg._feature_arrays = original
    return raw


def _raw_segments_for_game(game_id: str, samples: list[dict], tracking: Path,
                           feature_names: list[str]) -> dict[str, dict]:
    shard_names = tuple(sorted({s["shard_name"] for s in samples}))
    archives = {name: tarfile.open(tracking / "train/shards" / name, mode="r:")
                for name in shard_names}
    def read_clip(sample):
        handle = archives[sample["shard_name"]].extractfile(sample["wds_member"])
        if handle is None:
            raise ValueError(f"Cannot read TRAIN member {sample['wds_member']}")
        payload = handle.read()
        import pyarrow.parquet as pq
        rows = pq.read_table(io.BytesIO(payload), columns=list(seg.RAW_COLUMNS)).to_pylist()
        if len(rows) != 144:
            raise ValueError(f"{sample['sample_id']} has {len(rows)} rows; expected 144")
        for index, row in enumerate(rows):
            if row.get("frameNum") is None or int(row["frameNum"]) != sample["start_frame"] + index:
                raise ValueError(f"Frame numbering mismatch in {sample['sample_id']} at row {index}")
        return rows
    try:
        return _capture_raw_segments(game_id, samples, read_clip, feature_names)
    finally:
        for archive in archives.values():
            archive.close()


def _build_game_payload(game_id: str, game_records: list[dict], samples: list[dict],
                        tracking: Path, feature_names: list[str], base_sha256: str,
                        seed: int) -> dict:
    raw_by_segment = _raw_segments_for_game(game_id, samples, tracking, feature_names)
    base_ids = {str(r["segment_id"]) for r in game_records}
    if not base_ids.issubset(raw_by_segment):
        raise ValueError(f"Raw recovery missed base segments for game {game_id}")
    gap_records = {}
    raw_hashes = {}
    for record in game_records:
        segment_id = str(record["segment_id"])
        attached = gap.attach_consistency_view(
            record, raw_by_segment[segment_id],
            seed=_identity_seed(game_id, segment_id, seed))
        gap_records[segment_id] = {key: attached[key] for key in
            ("gap_x", "consistency_mask", "gap_provenance")}
        raw_hashes[segment_id] = attached["gap_provenance"]["raw_input_sha256"]
    return {"game_id": game_id, "base_sha256": base_sha256, "seed": int(seed),
            "records": gap_records, "raw_input_sha256": raw_hashes}


def _stage_one_game(args):
    game_id, base_path, samples, tracking, seed = args
    base_path = Path(base_path)
    base_payload = joblib.load(base_path)
    records = base_payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Malformed atomically written base game payload: {game_id}")
    feature_names = list(records[0]["names"]) if records else []
    payload = _build_game_payload(game_id, records, samples, Path(tracking),
                                  feature_names, _file_hash(base_path), int(seed))
    return game_id, payload


def stage_completed(root: str | Path, base: str | Path, staging: str | Path, *,
                    seed: int = 2027, poll_seconds: int = 10, workers: int = 3, progress=print) -> dict:
    """Incrementally stage maps from atomically written base game files.

    Staged outputs are provisional until `prepare` verifies them against the
    final sealed base manifest. An incomplete base cache is never called valid.
    """
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= 3:
        raise ValueError("staging workers must be in 1..3")
    tracking = seg._tracking_root(root)
    receipt = seg._verify_acquisition(tracking)
    base = Path(base); staging = Path(staging)
    grouped = _read_game_samples(tracking)
    games = sorted(grouped, key=lambda x: int(x) if x.isdigit() else x)
    stage_games = staging / "games"; stage_games.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / "manifest.json"
    stage_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": SCHEMA + "_staging", "seed": int(seed),
        "acquisition_receipt": receipt, "games": games,
        "feature_source_sha256": {p.name: _file_hash(p) for p in
            (Path(__file__), Path(seg.__file__), Path(gap.__file__),
             ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py")},
        "game_files": {}}
    if (stage_manifest.get("schema") != SCHEMA + "_staging" or
            stage_manifest.get("seed") != int(seed) or
            stage_manifest.get("acquisition_receipt") != receipt or
            stage_manifest.get("games") != games or
            stage_manifest.get("feature_source_sha256") != {p.name: _file_hash(p) for p in
                (Path(__file__), Path(seg.__file__), Path(gap.__file__),
                 ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py")}):
        raise ValueError("Staging directory is bound to different source or recipe")
    audits = stage_manifest["game_files"]
    for game_id, audit in list(audits.items()):
        cached_path = stage_games / f"{game_id}.joblib"
        if not cached_path.is_file() or _file_hash(cached_path) != audit.get("sha256"):
            raise ValueError(f"Staged payload hash mismatch: {game_id}")
    remaining = [game for game in games if game not in audits]
    pending = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 3)),
                             mp_context=context) as pool:
        while remaining or pending:
            for game_id in list(remaining):
                base_path = base / "games" / f"{game_id}.joblib"
                if base_path.is_file():
                    future = pool.submit(_stage_one_game, (
                        game_id, str(base_path), grouped[game_id], str(tracking), int(seed)))
                    pending[future] = game_id
                    remaining.remove(game_id)
            if not pending:
                game_id = remaining[0]
                progress(f"waiting for base game {game_id}; {len(audits)}/{len(games)} staged")
                time.sleep(max(1, int(poll_seconds)))
                continue
            completed, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in completed:
                game_id = pending.pop(future)
                _, payload = future.result()
                out = stage_games / f"{game_id}.joblib"
                tmp = out.with_suffix(".joblib.tmp")
                joblib.dump(payload, tmp, compress=0); tmp.replace(out)
                audits[game_id] = {"game_id": game_id, "sha256": _file_hash(out),
                                   "base_sha256": payload["base_sha256"],
                                   "segments": len(payload["records"])}
                stage_manifest["game_files"] = audits
                temp = staging / "manifest.json.tmp"
                temp.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n")
                temp.replace(manifest_path)
                progress(f"staged game {len(audits)}/{len(games)}: {game_id} ({audits[game_id]['segments']} segments)")
    stage_manifest["complete"] = len(audits) == len(games)
    temp = staging / "manifest.json.tmp"
    temp.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n")
    temp.replace(manifest_path)
    return stage_manifest


def _valid_game_payload(path: Path, expected: dict) -> dict | None:
    if not path.is_file():
        return None
    if _file_hash(path) != expected.get("sha256"):
        raise ValueError(f"Existing per-game gap payload hash mismatch: {path.name}")
    payload = joblib.load(path)
    if (payload.get("game_id") != expected.get("game_id") or
            payload.get("base_sha256") != expected.get("base_sha256")):
        raise ValueError(f"Existing per-game gap payload provenance mismatch: {path.name}")
    return payload


def prepare(root: str | Path, base: str | Path, output: str | Path, *, seed: int = 2027,
            staging: str | Path | None = None, progress=print) -> dict:
    """Prepare verified per-game gap maps; safe to resume completed game files."""
    tracking = seg._tracking_root(root)
    receipt = seg._verify_acquisition(tracking)
    base = Path(base); output = Path(output)
    base_manifest = seg._load_manifest(base)
    if base_manifest.get("split_used") != "train" or base_manifest.get("organizer_reference_access") is not False:
        raise ValueError("Base segment manifest must be TRAIN-only")
    grouped = _read_game_samples(tracking)
    games = sorted(grouped, key=lambda x: int(x) if x.isdigit() else x)
    if games != base_manifest.get("games"):
        raise ValueError("Raw TRAIN game cohort differs from the base segment cache")
    by_game: dict[str, list[dict]] = {}
    if staging is None:
        for record in seg.load_records(base):
            match = str(record["match"])
            if not match.startswith("soccernet_"):
                raise ValueError("Unexpected non-SoccerNet base record")
            by_game.setdefault(match.removeprefix("soccernet_"), []).append(record)
    output.mkdir(parents=True, exist_ok=True)
    game_dir = output / "games"; game_dir.mkdir(exist_ok=True)
    module_hashes = {p.name: _file_hash(p) for p in
                     (Path(__file__), Path(seg.__file__), Path(gap.__file__),
                      ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py")}
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    binding = {"schema": SCHEMA, "seed": int(seed), "acquisition_receipt": receipt,
               "base_manifest_sha256": _file_hash(base / "manifest.json"),
               "base_source_sha256": base_manifest["source_sha256"],
               "feature_source_sha256": module_hashes, "games": games}
    if old and any(old.get(k) != v for k, v in binding.items()):
        raise ValueError("Existing gap manifest is bound to different source, base, or recipe")
    audits = dict(old.get("game_files", {})) if old else {}
    for number, game_id in enumerate(games, 1):
        path = game_dir / f"{game_id}.joblib"
        cached = _valid_game_payload(path, audits.get(game_id, {})) if old else None
        if cached is None:
            if staging is not None:
                staged_path = Path(staging) / "games" / f"{game_id}.joblib"
                stage_manifest_path = Path(staging) / "manifest.json"
                if not stage_manifest_path.is_file() or not staged_path.is_file():
                    raise ValueError(f"Missing staged gap payload for game {game_id}")
                stage_manifest = json.loads(stage_manifest_path.read_text())
                if (stage_manifest.get("seed") != int(seed) or
                        stage_manifest.get("acquisition_receipt") != receipt or
                        stage_manifest.get("games") != games or
                        stage_manifest.get("feature_source_sha256") != module_hashes or
                        not stage_manifest.get("complete")):
                    raise ValueError("Staging manifest does not bind the complete TRAIN cohort and recipe")
                stage_audit = stage_manifest.get("game_files", {}).get(game_id)
                staged = _valid_game_payload(staged_path, stage_audit or {})
                if (staged is None or staged.get("seed") != int(seed) or
                        staged.get("base_sha256") != base_manifest["game_audits"][game_id]["sha256"]):
                    raise ValueError(f"Staged map does not match final base cache for game {game_id}")
                base_payload = joblib.load(base / "games" / f"{game_id}.joblib")
                base_ids = {str(record["segment_id"]) for record in base_payload["records"]}
                if set(staged.get("records", {})) != base_ids:
                    raise ValueError(f"Staged segment keys differ from final base game {game_id}")
                payload = staged
            else:
                game_records = by_game.get(game_id, [])
                feature_names = list(game_records[0]["names"]) if game_records else list(base_manifest["feature_names"])
                payload = _build_game_payload(
                    game_id, game_records, grouped[game_id], tracking, feature_names,
                    base_manifest["game_audits"][game_id]["sha256"], int(seed))
            tmp = path.with_suffix(".joblib.tmp")
            joblib.dump(payload, tmp, compress=0); tmp.replace(path)
            audit = {"game_id": game_id, "sha256": _file_hash(path),
                     "base_sha256": payload["base_sha256"],
                     "segments": len(payload["records"])}
            audits[game_id] = audit
        progress(f"prepared game {number}/{len(games)}: {game_id} ({audits[game_id]['segments']} segments)")
        current = dict(binding, complete=(len(audits) == len(games)), game_files=audits)
        temp_manifest = output / "manifest.json.tmp"
        temp_manifest.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        temp_manifest.replace(manifest_path)
    result = json.loads(manifest_path.read_text())
    if not result.get("complete") or set(result.get("game_files", {})) != set(games):
        raise ValueError("Gap-view preparation did not produce the complete TRAIN cohort")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/independent/soccernet_gar")
    parser.add_argument("--base", type=Path, default=ROOT / "data/experiments/soccernet_followup_execution_v1/segments")
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments/soccernet_followup_execution_v1/gaps")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--staging", type=Path, help="Use precomputed provisional staging maps")
    parser.add_argument("--stage-while-base-runs", action="store_true",
                        help="Continuously stage each completed base game until all 45 exist")
    args = parser.parse_args()
    if args.stage_while_base_runs:
        staging_path = args.output.parent / (args.output.name + "_staging")
        staged = stage_completed(args.root, args.base, staging_path, seed=args.seed,
                                 workers=args.workers,
                                 progress=lambda message: print(message, flush=True))
        if not staged.get("complete"):
            raise ValueError("Staging did not cover the complete TRAIN cohort")
        while not (args.base / "manifest.json").is_file():
            print("staging complete; waiting for sealed base manifest", flush=True)
            time.sleep(5)
        args.staging = staging_path
    result = prepare(args.root, args.base, args.output, seed=args.seed, staging=args.staging,
                     progress=lambda message: print(message, flush=True))
    print(json.dumps({"games": len(result["games"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
