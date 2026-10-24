"""Prepare challenge tracking inputs without loading any possession ground truth.

All generated arrays and predictions belong under the gitignored data directory.
The tactical inputs are already on video time: source frame ID minus one.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track
from evaluation.possession_metrics import validate_submission


def load_challenge(output=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(meta_path)
    n = source_frames()
    pitch = np.array(GAMES[GAME]["pitch"], dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    count = 0
    with open(raw_path) as handle:
        for i, line in enumerate(handle):
            fields = line.strip().split(":")
            if len(fields) != 3 or int(fields[0]) != i + 1 or i >= n:
                raise ValueError(f"Unexpected tracking record at row {i + 1}")
            row = np.fromstring(fields[1].replace(";", ","), sep=",", dtype=np.float32)
            b = np.fromstring(fields[2], sep=",", dtype=np.float32)
            if row.size != len(teams) * 3 or b.size != 3:
                raise ValueError(f"Unexpected coordinate count at row {i + 1}")
            players[i] = row.reshape(len(teams), 3)[:, :2] * pitch
            ball[i] = b[:2] * pitch
            count += 1
    if count != n:
        raise ValueError(f"Expected {n} tracking records; found {count}")
    period = np.zeros(n, dtype=np.int16)
    for half, first, last in half_video_blocks(GAME):
        period[first:last + 1] = half
    fps = float(GAMES[GAME]["fps"])
    data = {
        "ball": ball,
        "players": players,
        "teams": teams,
        "period": period,
        "fps": np.array(fps),
        "timestamps": np.arange(n, dtype=float) / fps,
        "pitch": pitch,
        "ball_observed": np.isfinite(ball).all(axis=1),
        "player_observed": np.isfinite(players).all(axis=2),
        # This placeholder cannot train or grade a model. The public GT is never read.
        "labels": np.full(n, "X", dtype="U1"),
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **data)
    return data


def write_submission(predictions, output):
    predictions = np.asarray(predictions).astype(str)
    frame = pd.DataFrame({"frame": np.arange(len(predictions)), "label": predictions})
    error = validate_submission(frame, source_frames())
    if error is not None:
        raise ValueError(error)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/challenge/inputs.npz")
    args = parser.parse_args()
    data = load_challenge(args.out)
    print(f"Prepared {len(data['ball'])} frames, {len(data['teams'])} track slots, "
          f"{data['ball_observed'].sum()} observed ball positions")
    print(f"No ground-truth labels loaded. Wrote {args.out}")
