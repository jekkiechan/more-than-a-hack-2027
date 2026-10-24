"""Download and normalize open SkillCorner tracking; targets never become features.

Run ``python experiments/datasets.py --download`` to reproduce the fixed split.
All downloads and generated arrays stay in the gitignored data/external directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "external"
SOURCE_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
SOURCE = "https://github.com/SkillCorner/opendata"
MATCH_IDS = (1874553, 1886347, 1899585, 1925299, 1927964)
FIXED_SPLIT = {
    "train": [f"skillcorner_{i}" for i in MATCH_IDS[:3]],
    "validation": [f"skillcorner_{MATCH_IDS[3]}"],
    "test": [f"skillcorner_{MATCH_IDS[4]}"],
    "policy": "Ascending available match IDs; fixed before label/model inspection; whole matches.",
    "seed": None,
    "source_commit": SOURCE_COMMIT,
}


def download_match(match_id: int, root: Path = DEFAULT_ROOT) -> None:
    folder = root / "raw" / str(match_id)
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("match.json", "tracking_extrapolated.jsonl"):
        name = f"{match_id}_{suffix}"
        path = folder / name
        if path.exists() and path.stat().st_size > 1000:
            continue
        # The tracking file is in Git LFS; raw.githubusercontent would return a pointer.
        host = ("media.githubusercontent.com/media" if suffix.endswith("jsonl")
                else "raw.githubusercontent.com")
        url = f"https://{host}/SkillCorner/opendata/{SOURCE_COMMIT}/data/matches/{match_id}/{name}"
        temp = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        temp.replace(path)


def _coordinate(value):
    return np.nan if value is None else float(value)


def _seconds(value):
    if value is None:
        return np.nan
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def convert_match(match_id: int, root: Path = DEFAULT_ROOT) -> dict:
    folder = root / "raw" / str(match_id)
    metadata_path = folder / f"{match_id}_match.json"
    tracking_path = folder / f"{match_id}_tracking_extrapolated.jsonl"
    metadata = json.loads(metadata_path.read_text())
    home_id = metadata["home_team"]["id"]
    away_id = metadata["away_team"]["id"]
    roster = sorted(metadata["players"], key=lambda p: p["id"])
    assert all(p["team_id"] in (home_id, away_id) for p in roster)
    lookup = {p["id"]: i for i, p in enumerate(roster)}
    assert len(lookup) == len(roster)
    teams = np.array([0 if p["team_id"] == home_id else 1 for p in roster], dtype=np.int8)
    pitch = np.array([metadata["pitch_length"], metadata["pitch_width"]], dtype=np.float32)
    with tracking_path.open() as handle:
        n = sum(1 for _ in handle)
    players = np.full((n, len(roster), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    labels = np.full(n, "X", dtype="<U1")
    period = np.zeros(n, dtype=np.int8)
    source_timestamps = np.full(n, np.nan, dtype=np.float64)
    ball_observed = np.zeros(n, dtype=bool)
    player_observed = np.zeros((n, len(roster)), dtype=bool)
    group_counts = Counter()
    with tracking_path.open() as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            if row["frame"] != i:
                raise ValueError(f"Noncontiguous frame {row['frame']} at row {i}")
            period[i] = row["period"] or 0
            source_timestamps[i] = _seconds(row["timestamp"])
            observation = row["ball_data"]
            ball[i] = [_coordinate(observation[axis]) for axis in ("x", "y")]
            ball_observed[i] = observation.get("is_detected") is True
            for player in row["player_data"]:
                slot = lookup[player["player_id"]]
                players[i, slot] = [_coordinate(player[axis]) for axis in ("x", "y")]
                player_observed[i, slot] = player.get("is_detected") is True
            # This target extraction is the ONLY possession access. Never infer D
            # from missing possession: the provider does not publish dead labels.
            group = row["possession"]["group"]
            group_counts[str(group)] += 1
            if group == "home team":
                labels[i] = "H"
            elif group == "away team":
                labels[i] = "A"
            elif group is not None:
                raise ValueError(f"Unknown possession group {group!r}; inspect definition")
    # The source clock is absent on unassigned rows around halves. Preserve those
    # rows (and their X targets), using the verified contiguous 10-Hz frame index
    # as the model clock. Period changes still separate both halves. This clock
    # describes sampled tracking time, not wall-clock elapsed time across halftime.
    timestamps = np.arange(n, dtype=np.float64) / 10.0
    within_half = (period[1:] == period[:-1]) & (period[1:] > 0)
    clock_known = np.isfinite(source_timestamps[1:]) & np.isfinite(source_timestamps[:-1])
    if not np.allclose(np.diff(source_timestamps)[within_half & clock_known], 0.1, atol=1e-6):
        raise ValueError("Source match clock is not 10 Hz within a half")
    # Both ball and players are originally pitch-centred metres. Translate together.
    ball += pitch / 2
    players += pitch / 2
    output = root / "processed" / f"skillcorner_{match_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, ball=ball, players=players, teams=teams, labels=labels,
        period=period, fps=np.array(10.0), timestamps=timestamps, pitch=pitch,
        ball_observed=ball_observed, player_observed=player_observed,
    )
    description = {
        "match_id": f"skillcorner_{match_id}",
        "source": SOURCE,
        "source_commit": SOURCE_COMMIT,
        "licence": "MIT; SkillCorner attribution retained in experiments/DATASETS.md",
        "date": metadata["date_time"],
        "home": metadata["home_team"]["name"],
        "away": metadata["away_team"]["name"],
        "n_frames": n,
        "fps": 10.0,
        "timestamp_definition": "Contiguous source frame index / 10 Hz; period changes delimit halves.",
        "missing_source_clock_rows": int(np.sum(~np.isfinite(source_timestamps))),
        "pitch_metres": pitch.tolist(),
        "label_counts": dict(Counter(labels.tolist())),
        "source_group_counts": dict(group_counts),
        "target_quality": "provider-generated silver labels, not independent human ground truth",
        "target_definition": "home team=H; away team=A; null=X. No D labels available.",
        "input_definition": "Tracking positions and detection flags only; no ownership or ball-status features.",
        "npz": str(output.relative_to(ROOT)),
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (metadata_path, tracking_path)},
    }
    output.with_suffix(".json").write_text(json.dumps(description, indent=2) + "\n")
    return description


def load_match(match_id: str, root: Path = DEFAULT_ROOT) -> dict:
    """Load the numerical contract. Callers must separate labels from input arrays."""
    with np.load(root / "processed" / f"{match_id}.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "split.json").write_text(json.dumps(FIXED_SPLIT, indent=2) + "\n")
    if args.download:
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda mid: download_match(mid, args.root), MATCH_IDS))
    for mid in MATCH_IDS:
        summary = convert_match(mid, args.root)
        print(json.dumps({k: summary[k] for k in ("match_id", "n_frames", "label_counts", "npz")}), flush=True)


if __name__ == "__main__":
    main()
