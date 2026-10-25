"""Convert Metrica Sports' open sample games to the shared NPZ contract with H/A/D targets.

Same provider, frame rate (25 Hz), coordinate convention and Game 3 file format
(FIFA EPTS) as the challenge tracking, and the ball is missing on ~40% of frames,
as in the challenge. Targets are derived from Metrica's human-annotated event
stream only; tracking positions are never used to build a label, so scoring a
tracking model against them is not circular.

Run ``python experiments/metrica_open.py --download``. Everything stays under the
gitignored data/external directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "external"
SOURCE = "https://github.com/metrica-sports/sample-data"
SOURCE_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
FPS = 25.0
PITCH = np.array([105.0, 68.0], dtype=np.float32)
FILES = {
    1: ("Sample_Game_1_RawEventsData.csv", "Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1_RawTrackingData_Away_Team.csv"),
    2: ("Sample_Game_2_RawEventsData.csv", "Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2_RawTrackingData_Away_Team.csv"),
    3: ("Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"),
}

# Events that tell us which team has the ball from their start frame onward.
# BALL LOST and CHALLENGE do not: a loose ball stays with the last team until the
# opponent's RECOVERY (challenge definition: control must change).
POSSESSION_EVENTS = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL OUT"}
# Events that never mark the restart of play after a stoppage.
NON_RESTART = {"CARD", "FAULT RECEIVED", "CHALLENGE", "BALL LOST"}
DEAD_AT_END_SUBTYPES = {"OUT", "GOAL", "OFFSIDE", "HAND BALL", "END HALF"}


def download(game: int, root: Path = DEFAULT_ROOT) -> None:
    folder = root / "raw" / "metrica" / f"Sample_Game_{game}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES[game]:
        path = folder / name
        if path.exists() and path.stat().st_size > 1000:
            continue
        url = (f"https://raw.githubusercontent.com/metrica-sports/sample-data/{SOURCE_COMMIT}"
               f"/data/Sample_Game_{game}/{name}")
        temp = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=300) as response, temp.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        temp.replace(path)


def _events_csv(path):
    events = []
    with open(path) as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            events.append(dict(index=i, team="H" if row["Team"] == "Home" else "A",
                               type=row["Type"], subtypes=set(filter(None, row["Subtype"].split("-"))),
                               period=int(row["Period"]), start=int(row["Start Frame"]),
                               end=int(row["End Frame"])))
    return events


def _events_json(path, home_id):
    events = []
    for i, row in enumerate(json.loads(Path(path).read_text())["data"]):
        sub = row["subtypes"] or []
        sub = [sub] if isinstance(sub, dict) else sub
        events.append(dict(index=i, team="H" if row["team"]["id"] == home_id else "A",
                           type=row["type"]["name"], subtypes={s["name"] for s in sub},
                           period=int(row["period"]), start=int(row["start"]["frame"]),
                           end=int(row["end"]["frame"])))
    return events


def labels_from_events(events, frames, period):
    """Frame labels from events alone. ``frames`` are source frame numbers.

    Live play: the team of the latest possession event. Dead ball: from the
    stoppage trigger (ball out, shot out/goal, offside, foul whistle) to the next
    restart event; a set piece with no recorded trigger is dead from the end of
    the previous event. Before a period's first and after its last event: X.
    """
    labels = np.full(len(frames), "X", dtype="<U1")
    stats = Counter()
    for half in sorted({e["period"] for e in events}):
        evs = sorted((e for e in events if e["period"] == half), key=lambda e: (e["start"], e["index"]))
        # Live/dead change points as (frame, state) where state is "H"/"A"/"D".
        changes, team, dead_from = [], None, None
        for k, e in enumerate(evs):
            if dead_from is not None and e["type"] not in NON_RESTART and e["start"] >= dead_from:
                changes.append((dead_from, "D"))
                dead_from = None
                stats["stoppages"] += 1
            elif e["type"] == "SET PIECE" and k > 0 and dead_from is None:
                previous = max(p["end"] for p in evs[:k] if p["type"] != "CARD")
                if previous < e["start"]:
                    changes.append((previous, "D"))
                    stats["untriggered_set_pieces"] += 1
            if e["type"] in POSSESSION_EVENTS:
                team = e["team"]
                changes.append((e["start"], team))
            if e["type"] == "BALL OUT" or (e["subtypes"] & DEAD_AT_END_SUBTYPES and e["type"] != "RECOVERY"):
                dead_from = e["end"] if dead_from is None else min(dead_from, e["end"])
            elif e["type"] == "FAULT RECEIVED":
                dead_from = e["start"] if dead_from is None else min(dead_from, e["start"])
        first, last = evs[0]["start"], max(e["end"] for e in evs)
        if dead_from is not None:
            last = min(last, dead_from)
        # Stable sort keeps a restart's possession after the D that precedes it.
        changes.sort(key=lambda c: c[0])
        in_half = np.flatnonzero(period == half)
        half_frames = frames[in_half]
        state = np.full(len(in_half), "X", dtype="<U1")
        points = np.array([c[0] for c in changes])
        values = np.array([c[1] for c in changes])
        idx = np.searchsorted(points, half_frames, side="right") - 1
        ok = idx >= 0
        state[ok] = values[idx[ok]]
        state[(half_frames < first) | (half_frames > last)] = "X"
        labels[in_half] = state
    return labels, stats


def _tracking_csv(home_path, away_path):
    def read(path):
        with open(path) as handle:
            rows = list(csv.reader(handle))
        header = rows[2]
        data = np.array([[np.nan if v in ("NaN", "") else float(v) for v in r] for r in rows[3:]],
                        dtype=np.float64)
        player_cols = [i for i, h in enumerate(header) if h.startswith("Player")]
        ball_col = header.index("Ball")
        players = np.stack([data[:, [c, c + 1]] for c in player_cols], axis=1)
        return data[:, 0].astype(int), data[:, 1].astype(int), players, data[:, [ball_col, ball_col + 1]]
    period, frames, home, ball = read(home_path)
    period_a, frames_a, away, ball_a = read(away_path)
    if not (np.array_equal(frames, frames_a) and np.array_equal(period, period_a)):
        raise ValueError("Home and away tracking frames differ")
    if not np.array_equal(np.isnan(ball), np.isnan(ball_a)):
        raise ValueError("Home and away ball tracks differ")
    players = np.concatenate([home, away], axis=1)
    teams = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    return frames, period, players, ball, teams


def _tracking_epts(meta_path, raw_path):
    root = ET.parse(meta_path).getroot()
    home_id = root.find(".//Score").attrib["idLocalTeam"]
    player_team = {p.attrib["id"]: p.attrib["teamId"] for p in root.iter("Player")}
    channel_player = {c.attrib["id"]: c.attrib["playerId"] for c in root.iter("PlayerChannel")}
    roster = sorted(player_team)
    slot = {pid: i for i, pid in enumerate(roster)}
    params = {p.find("Name").text: p.find("Value").text for p in root.iter("ProviderParameter")}
    halves = [(1, int(params["first_half_start"]), int(params["first_half_end"])),
              (2, int(params["second_half_start"]), int(params["second_half_end"]))]
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [channel_player[r.attrib["playerChannelId"]]
                for r in spec.iter("PlayerChannelRef") if r.attrib["playerChannelId"].endswith("_x")]
        specs.append((int(spec.attrib["startFrame"]), int(spec.attrib["endFrame"]), [slot[p] for p in refs]))
    lines = Path(raw_path).read_text().splitlines()
    n = len(lines)
    frames = np.zeros(n, dtype=int)
    players = np.full((n, len(roster), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, line in enumerate(lines):
        frame, body, b = line.split(":")
        frames[i] = int(frame)
        cols = next(s[2] for s in specs if s[0] <= frames[i] <= s[1])
        xy = np.array([[np.nan if v == "NaN" else float(v) for v in p.split(",")] for p in body.split(";")])
        players[i, cols] = xy
        ball[i] = [np.nan if v == "NaN" else float(v) for v in b.split(",")[:2]]
    period = np.zeros(n, dtype=int)
    for half, first, last in halves:
        period[(frames >= first) & (frames <= last)] = half
    teams = np.array([0 if player_team[p] == home_id else 1 for p in roster], dtype=np.int8)
    return frames, period, players, ball, teams, home_id


def convert(game: int, root: Path = DEFAULT_ROOT) -> dict:
    folder = root / "raw" / "metrica" / f"Sample_Game_{game}"
    names = FILES[game]
    if game == 3:
        frames, period, players, ball, teams, home_id = _tracking_epts(folder / names[1], folder / names[2])
        events = _events_json(folder / names[0], home_id)
    else:
        frames, period, players, ball, teams = _tracking_csv(folder / names[1], folder / names[2])
        events = _events_csv(folder / names[0])
    if np.any(np.diff(frames) != 1):
        raise ValueError("Tracking frames are not contiguous")
    # Labels are computed from events only. Tracking is not passed in.
    labels, stats = labels_from_events(events, frames, period)
    players = (players * PITCH).astype(np.float32)
    ball = (ball * PITCH).astype(np.float32)
    output = root / "processed" / f"metrica_{game}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, ball=ball, players=players, teams=teams, labels=labels,
        period=period.astype(np.int8), fps=np.array(FPS), timestamps=np.arange(len(frames)) / FPS,
        pitch=PITCH.astype(np.float32), ball_observed=np.isfinite(ball).all(axis=1),
        player_observed=np.isfinite(players).all(axis=2),
    )
    summary = describe(labels, period, np.isfinite(ball).all(axis=1))
    description = {
        "match_id": f"metrica_{game}", "source": SOURCE, "source_commit": SOURCE_COMMIT,
        "licence": "No licence file; README asks for responsible use and acknowledgement of Metrica Sports.",
        "n_frames": int(len(frames)), "fps": FPS, "pitch_metres": PITCH.tolist(),
        "n_events": len(events), "label_stats": stats, **summary,
        "target_quality": "Derived from Metrica's human-annotated events; silver labels, not the challenge's possession annotation.",
        "target_definition": ("Live: team of the latest PASS/CARRY/SHOT/SET PIECE/RECOVERY/BALL OUT event. "
                              "D: stoppage trigger to next restart event. X: outside first/last event of a period."),
        "input_definition": "Tracking positions and finiteness masks only.",
        "npz": str(output.relative_to(ROOT)),
        "sha256": {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in names},
    }
    output.with_suffix(".json").write_text(json.dumps(description, indent=2) + "\n")
    return description


def describe(labels, period, ball_observed, fps=FPS):
    """Aggregate target statistics: class shares, spell counts/durations, ball visibility by class."""
    graded = labels != "X"
    out = {"label_share": {c: round(float(np.mean(labels[graded] == c)), 4) for c in "HAD"},
           "graded_minutes": round(float(graded.sum() / fps / 60), 1),
           "ball_missing_graded": round(float(1 - ball_observed[graded].mean()), 4),
           "ball_missing_by_label": {c: round(float(1 - ball_observed[labels == c].mean()), 4)
                                     for c in "HAD" if np.any(labels == c)}}
    durations = []
    for half in np.unique(period[period > 0]):
        lab = labels[period == half]
        cuts = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1, len(lab)]
        durations += [(lab[a], (b - a) / fps) for a, b in zip(cuts[:-1], cuts[1:]) if lab[a] in "HA"]
    d = np.array([x[1] for x in durations])
    out["possession_spells"] = int(len(d))
    out["spells_per_graded_minute"] = round(len(d) / max(out["graded_minutes"], 1e-9), 3)
    out["spell_seconds_mean_median"] = [round(float(d.mean()), 2), round(float(np.median(d)), 2)] if len(d) else None
    out["spell_share_under_3s"] = round(float(np.mean(d < 3)), 3) if len(d) else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    for game in args.games:
        if args.download:
            download(game, args.root)
        summary = convert(game, args.root)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("sha256",)}), flush=True)


if __name__ == "__main__":
    main()
