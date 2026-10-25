"""SkillCorner open broadcast tracking with three-state H/A/D targets.

``datasets.py`` keeps SkillCorner as an H/A-only source because its tracking
files publish no ball-status flag. The same pinned repository also ships
``dynamic_events.csv``, whose player-possession events carry
``game_interruption_after`` / ``game_interruption_before`` (throw-in, goal kick,
corner, free kick, goal). Those bound every stoppage, so D can be derived from
events without reading any tracking coordinate:

- D core: from the end of the possession tagged with an interruption *after* to
  the start of the next player possession (the restart's first reception),
  shrunk by ``MARGIN_S`` at both ends. The shrunk margins are X: the ball may
  still be travelling out, or the restart pass may already be in flight.
- Live: SkillCorner's ``possession.group``; frames where it is null inside live
  play inherit the last known team (the challenge rule: a loose or contested ball
  stays with the last team in control). Before the first known team of a live
  stretch: X.

Inputs are the same tracking-only arrays as ``datasets.py``; this module only
changes the target. All ten matches in the pinned repository are converted.
Run ``python experiments/skillcorner_had.py --download``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from experiments import datasets  # noqa: E402
from experiments.metrica_open import describe  # noqa: E402

DEFAULT_ROOT = datasets.DEFAULT_ROOT
COMMIT = datasets.SOURCE_COMMIT
MATCH_IDS = (1874553, 1886347, 1899585, 1925299, 1927964,
             1953632, 1959846, 1986691, 1996435, 1996436, 2006229)
FPS = 10.0
MARGIN_S = 1.0


def download_events(match_id: int, root: Path = DEFAULT_ROOT) -> Path:
    path = root / "raw" / str(match_id) / f"{match_id}_dynamic_events.csv"
    if path.exists() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for host in ("raw.githubusercontent.com", "media.githubusercontent.com/media"):
        url = f"https://{host}/SkillCorner/opendata/{COMMIT}/data/matches/{match_id}/{path.name}"
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read()
        if not body.startswith(b"version https://git-lfs"):
            path.write_bytes(body)
            return path
    raise RuntimeError(f"Could not fetch {path.name}")


def stoppages(events_path, fps=FPS, margin_s=MARGIN_S):
    """(dead_start, dead_end) frame pairs, half-open, from interruption tags only."""
    with open(events_path) as handle:
        rows = [r for r in csv.DictReader(handle) if r["event_type"] == "player_possession"]
    rows.sort(key=lambda r: (int(r["frame_start"]), int(r["frame_end"])))
    margin = int(round(margin_s * fps))
    out, tags = [], []
    for current, following in zip(rows[:-1], rows[1:]):
        if not current["game_interruption_after"] or current["period"] != following["period"]:
            continue
        start, stop = int(current["frame_end"]) + 1 + margin, int(following["frame_start"]) - margin
        tags.append(current["game_interruption_after"])
        if stop > start:
            out.append((int(current["frame_end"]) + 1, start, stop, int(following["frame_start"])))
    return out, tags


def convert(match_id: int, root: Path = DEFAULT_ROOT) -> dict:
    base = datasets.convert_match(match_id, root)
    with np.load(root / "processed" / f"skillcorner_{match_id}.npz", allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    group = data["labels"]          # provider possession.group: H/A/X(null)
    period = data["period"]
    n = len(group)
    dead = np.zeros(n, dtype=bool)
    unsure = np.zeros(n, dtype=bool)
    spans, tags = stoppages(root / "raw" / str(match_id) / f"{match_id}_dynamic_events.csv")
    for outer_start, start, stop, outer_stop in spans:
        dead[start:stop] = True
        unsure[outer_start:start] = True
        unsure[stop:outer_stop] = True
    labels = np.full(n, "X", dtype="<U1")
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        last = "X"
        for i in idx:
            if dead[i]:
                labels[i], last = "D", "X"   # a restart re-establishes the team
            elif unsure[i]:
                labels[i] = "X"
            else:
                if group[i] in ("H", "A"):
                    last = group[i]
                labels[i] = last
    data["labels"] = labels
    output = root / "processed" / f"skillcorner_had_{match_id}.npz"
    np.savez_compressed(output, **data)
    description = {
        "match_id": f"skillcorner_had_{match_id}", "source": datasets.SOURCE, "source_commit": COMMIT,
        "home": base["home"], "away": base["away"], "n_frames": n, "fps": FPS,
        "n_stoppages": len(spans), "stoppage_tags": dict(zip(*np.unique(tags, return_counts=True))) if tags else {},
        **describe(labels, period, data["ball_observed"], FPS),
        "target_definition": (f"D: interruption-tagged possession end to next possession start, minus {MARGIN_S}s "
                              "margins (X). Live: possession.group, null forward-filled within live play."),
        "target_quality": "provider-generated silver labels",
        "npz": str(output.relative_to(ROOT)),
    }
    description["stoppage_tags"] = {k: int(v) for k, v in description["stoppage_tags"].items()}
    output.with_suffix(".json").write_text(json.dumps(description, indent=2) + "\n")
    return description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--matches", type=int, nargs="+", default=list(MATCH_IDS))
    args = parser.parse_args()
    for match_id in args.matches:
        if args.download:
            datasets.download_match(match_id, args.root)
            download_events(match_id, args.root)
        summary = convert(match_id, args.root)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
