"""Download and convert the CC BY 4.0 Sportec Open DFL possession benchmark.

BallStatus and BallPossession are read ONLY to form labels. Model inputs contain
coordinates, roster team assignments, observation availability, and time.
The fixed whole-match split is declared below before model development.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ARTICLE_API = "https://api.figshare.com/v2/articles/28196177"
ARTICLE_VERSION_API = ARTICLE_API + "/versions/1"
ARTICLE_DOI = "10.6084/m9.figshare.28196177.v1"
DEFAULT_ROOT = Path("data/external_audit/sportec")
SPLIT = {
    "train": ["J03WMX", "J03WN1", "J03WOH"],
    "validation": ["J03WOY"],
    "test": ["J03WPY"],
    "policy": "Fixed by match ID before model fitting; no random frame splits.",
    "source": ARTICLE_API,
    "license": "CC BY 4.0",
    "label_policy": "BallStatus=0 -> D; live BallPossession 1/2 -> H/A; otherwise X.",
}


def md5_file(path: Path) -> str:
    result = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def initialize(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    (root / "processed").mkdir(exist_ok=True)
    split_path = root / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != SPLIT:
        raise ValueError("Refusing to replace an existing different split")
    split_path.write_text(json.dumps(SPLIT, indent=2) + "\n")
    article_path = root / "article.json"
    if not article_path.exists():
        with urllib.request.urlopen(ARTICLE_VERSION_API, timeout=60) as response:
            article_path.write_bytes(response.read())
    article = json.loads(article_path.read_text())
    if article.get("version") != 1 or article.get("doi") != ARTICLE_DOI:
        raise ValueError("Benchmark requires Figshare dataset version 1")
    return article


def selected_files(article: dict, match: str) -> list[dict]:
    return [item for item in article["files"] if match in item["name"]
            and ("matchinformation" in item["name"] or "positions_raw" in item["name"])]


def download(item: dict, root: Path) -> str:
    target = root / "raw" / item["name"]
    if not target.exists() or target.stat().st_size != item["size"]:
        partial = target.with_suffix(".xml.partial")
        with urllib.request.urlopen(item["download_url"], timeout=120) as response, partial.open("wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
        if partial.stat().st_size != item["size"]:
            raise ValueError(f"Size mismatch: {target.name}")
        partial.replace(target)
    if md5_file(target) != item["computed_md5"]:
        raise ValueError(f"Checksum mismatch: {target.name}")
    return target.name


def convert(match: str, root: Path, article: dict) -> dict:
    files = selected_files(article, match)
    if len(files) != 2:
        raise ValueError(f"Expected metadata and tracking files for {match}")
    for item in files:
        path = root / "raw" / item["name"]
        if path.stat().st_size != item["size"] or md5_file(path) != item["computed_md5"]:
            raise ValueError(f"Input integrity failure: {path}")
    meta_path = root / "raw" / next(item["name"] for item in files if "matchinformation" in item["name"])
    tracking_path = root / "raw" / next(item["name"] for item in files if "positions_raw" in item["name"])
    meta = ET.parse(meta_path).getroot()
    info = meta.find(".//General").attrib
    env = meta.find(".//Environment").attrib
    pitch = np.array([float(env["PitchX"]), float(env["PitchY"])], dtype=np.float32)
    side_ids = {info["HomeTeamId"]: 0, info["GuestTeamId"]: 1}
    roster = []
    for team in meta.findall(".//Teams/Team"):
        for player in team.findall("Players/Player"):
            roster.append((side_ids[team.get("TeamId")], player.get("PersonId")))
    roster.sort()
    if len({person for _, person in roster}) != len(roster):
        raise ValueError("Duplicate roster identities")
    slots = {person: index for index, (_, person) in enumerate(roster)}
    teams = np.array([side for side, _ in roster], dtype=np.int8)
    players_by_period = {1: [], 2: []}
    balls_by_period = {1: [], 2: []}
    period_map = {"firstHalf": 1, "secondHalf": 2}
    context = None
    rows = []
    ball_labels = []
    ball_times = []
    for event, element in ET.iterparse(tracking_path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            attr = element.attrib
            period = period_map.get(attr.get("GameSection"))
            team_id = attr.get("TeamId")
            person_id = attr.get("PersonId")
            context = (period, team_id, person_id)
            rows, ball_labels, ball_times = [], [], []
        elif event == "end" and element.tag == "Frame":
            period, team_id, person_id = context
            if period and (team_id == "BALL" or team_id in side_ids):
                attr = element.attrib
                rows.append((int(attr["N"]), float(attr["X"]), float(attr["Y"])))
                if team_id == "BALL":
                    status, owner = attr.get("BallStatus"), attr.get("BallPossession")
                    label = "D" if status == "0" else (
                        {"1": "H", "2": "A"}.get(owner, "X") if status == "1" else "X")
                    ball_labels.append(label)
                    ball_times.append(datetime.fromisoformat(attr["T"]).timestamp())
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            period, team_id, person_id = context
            if rows:
                values = np.asarray(rows, dtype=np.float64)
                if team_id == "BALL":
                    balls_by_period[period].append((values, np.asarray(ball_times), np.asarray(ball_labels, dtype="U1")))
                elif team_id in side_ids:
                    if person_id not in slots:
                        raise ValueError(f"Player {person_id} absent from roster")
                    if int(teams[slots[person_id]]) != side_ids[team_id]:
                        raise ValueError("Player team disagreement")
                    players_by_period[period].append((slots[person_id], values))
            rows, ball_labels, ball_times = [], [], []
            element.clear()

    periods, balls, people, targets, time_arrays = [], [], [], [], []
    period_reports = []
    for period in (1, 2):
        chunks = balls_by_period[period]
        if not chunks:
            raise ValueError(f"Missing ball frames in period {period}")
        values = np.concatenate([x[0] for x in chunks])
        times = np.concatenate([x[1] for x in chunks])
        labels = np.concatenate([x[2] for x in chunks])
        order = np.argsort(values[:, 0])
        values, times, labels = values[order], times[order], labels[order]
        ids = values[:, 0].astype(np.int64)
        if not np.all(np.diff(ids) == 1):
            raise ValueError(f"Non-consecutive ball frames in period {period}")
        if not np.allclose(np.diff(times), .04, atol=.002):
            raise ValueError("Tracking timestamp steps are not 25 Hz")
        player_array = np.full((len(ids), len(slots), 2), np.nan, dtype=np.float32)
        for slot, player_values in players_by_period[period]:
            player_ids = player_values[:, 0].astype(np.int64)
            indices = np.searchsorted(ids, player_ids)
            keep = (indices < len(ids))
            keep[keep] &= ids[indices[keep]] == player_ids[keep]
            player_array[indices[keep], slot] = player_values[keep, 1:].astype(np.float32) + pitch / 2
        ball = values[:, 1:].astype(np.float32) + pitch / 2
        periods.append(np.full(len(ids), period, dtype=np.int8))
        balls.append(ball)
        people.append(player_array)
        targets.append(labels)
        time_arrays.append(times)
        period_reports.append({"period": period, "frames": len(ids), "source_frame_first": int(ids[0]), "source_frame_last": int(ids[-1])})

    ball = np.concatenate(balls)
    labels = np.concatenate(targets)
    timestamps = np.concatenate(time_arrays)
    timestamps -= timestamps[0]
    output = root / "processed" / f"{match}.npz"
    np.savez_compressed(output, ball=ball, players=np.concatenate(people), teams=teams,
                        labels=labels, period=np.concatenate(periods), fps=np.array(25.0),
                        timestamps=timestamps, pitch=pitch,
                        ball_observed=np.isfinite(ball).all(axis=1))
    report = {"match": match, "title": info["MatchTitle"], "kickoff": info["KickoffTime"],
              "output": str(output), "frames": len(labels), "roster_slots": len(roster),
              "pitch": pitch.tolist(), "periods": period_reports,
              "label_counts": {label: int(np.sum(labels == label)) for label in "HADX"},
              "ball_observed_fraction": float(np.isfinite(ball).all(axis=1).mean()),
              "sources": [{"name": item["name"], "file_id": item["id"], "md5": item["computed_md5"]} for item in files]}
    (root / "processed" / f"{match}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--matches", nargs="+")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    article = initialize(args.root)
    matches = args.matches or [match for role in ("train", "validation", "test") for match in SPLIT[role]]
    if not args.skip_download:
        files = [item for match in matches for item in selected_files(article, match)]
        with ThreadPoolExecutor(args.workers) as pool:
            for name in pool.map(lambda item: download(item, args.root), files):
                print(f"verified {name}", flush=True)
    for match in matches:
        print(json.dumps(convert(match, args.root, article)), flush=True)


if __name__ == "__main__":
    main()
