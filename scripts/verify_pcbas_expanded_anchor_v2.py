"""Independently recompute the expanded PCBAS anchor study from sealed arrays.

This verification reads saved predictions; it never runs inference or opens a
scoring gate. Event targets are sparse actions, not possession spell targets.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = "pcbas_expanded_anchor_v2"
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    declaration_path = RESULTS / f"{STUDY}_declaration.json"
    result_path = RESULTS / f"{STUDY}_result.json"
    declaration = json.loads(declaration_path.read_text())
    result = json.loads(result_path.read_text())
    seal_path = OUT / "prediction_seal.json"
    seal = json.loads(seal_path.read_text())
    assert result["declaration_sha256"] == sha(declaration_path)
    assert result["prediction_seal_sha256"] == sha(seal_path)
    assert seal["declaration_sha256"] == sha(declaration_path)
    assert seal["target_events_read"] is False and seal["input_action_class_read"] is False
    assert set(seal["halves"]) == set(declaration["cohort"]["halves"])
    for group in ("input_sha256", "source_sha256"):
        for path, expected in declaration[group].items():
            assert sha(ROOT / path) == expected, path
    manifest_path = ROOT / "experiments/baseline_manifest.json"
    assert sha(manifest_path) == declaration["baseline_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    for name, expected in declaration["model_sha256"].items():
        assert sha(ROOT / manifest["model_directory"] / name) == expected

    truth = json.loads((ROOT / "data/independent/sn_pcbas_2026/playbyplay_train.json").read_text())
    classes = {1: "drive", 2: "pass", 3: "cross", 5: "shot"}
    totals = {}
    replayed = 0
    for half in declaration["cohort"]["halves"]:
        path = OUT / f"{half}.npz"
        assert sha(path) == seal["halves"][half]["output_sha256"]
        receipt = seal["halves"][half]
        assert sha(OUT / f"{half}.receipt.json") == receipt["receipt_path_sha256"]
        assert receipt["declaration_sha256"] == sha(declaration_path)
        assert receipt["half"] == half and receipt["model_sha256"] == declaration["model_sha256"]
        assert receipt["target_events_read"] is False and receipt["input_action_class_read"] is False
        with np.load(path, allow_pickle=False) as saved:
            frames, predictions, visible = saved["frames"], saved["prediction"], saved["visible_players"]
        assert frames.shape == predictions.shape == visible.shape
        assert np.all(np.diff(frames) > 0)
        assert set(predictions) <= {"H", "A", "D"}
        frame_to_index = {int(f): i for i, f in enumerate(frames)}
        events = truth["events"][half]
        teams_at_frame = {}
        for frame, team, shirt, cls, box, replay in events:
            if replay == 0 and 1 <= cls <= 8:
                teams_at_frame.setdefault(frame, set()).add(team)
        excluded, seen, accepted = Counter(), set(), []
        for frame, team, shirt, cls, box, replay in events:
            if cls not in classes:
                excluded["nonselected_class"] += 1
            elif replay:
                excluded["replay"] += 1
            elif len(teams_at_frame[frame]) != 1:
                excluded["opposing_team_same_frame"] += 1
            elif (frame, team, shirt, cls) in seen:
                excluded["exact_duplicate"] += 1
            else:
                seen.add((frame, team, shirt, cls))
                if frame not in frame_to_index:
                    excluded["no_exact_tracking_frame"] += 1
                else:
                    accepted.append((frame_to_index[frame], team, cls, box))
        assert dict(excluded) == result["excluded"][half], half
        for cls, name in classes.items():
            current = [(idx, team, box) for idx, team, action, box in accepted if action == cls]
            correct = sum(predictions[idx] == ("H" if team == 0 else "A") for idx, team, _ in current)
            cell = dict(n=len(current), correct=int(correct),
                        predicted_state_counts=dict(Counter(str(predictions[idx]) for idx, _, _ in current)),
                        target_direction_team_counts=dict(Counter(str(team) for _, team, _ in current)),
                        actor_bbox_absent=sum(box == 0 for _, _, box in current))
            assert cell == result["half_counts"][half][name], (half, name)
            game = half.rsplit("_H", 1)[0]
            total = totals.setdefault((game, name), dict(n=0, correct=0, dead=0, teams=Counter()))
            total["n"] += len(current)
            total["correct"] += int(correct)
            total["dead"] += cell["predicted_state_counts"].get("D", 0)
            total["teams"].update(cell["target_direction_team_counts"])
            replayed += 1
        for low, high, label in ((0, 5, "0-5"), (6, 11, "6-11"), (12, float("inf"), "12+")):
            group = [(idx, team) for idx, team, _, _ in accepted if low <= visible[idx] <= high]
            cell = result["visibility_by_half"][half][label]
            assert cell["n"] == len(group)
            correct = sum(predictions[idx] == ("H" if team == 0 else "A") for idx, team in group)
            dead = sum(predictions[idx] == "D" for idx, _ in group)
            assert cell["correct"] == correct
            assert cell["accuracy"] == (correct / len(group) if group else None)
            assert cell["predicted_d_rate"] == (dead / len(group) if group else None)

    for name in classes.values():
        cells = [v for (game, cls), v in totals.items() if cls == name]
        n, correct = sum(c["n"] for c in cells), sum(c["correct"] for c in cells)
        s = result["summary"][name]
        assert (s["n"], s["correct"]) == (n, correct)
        assert s["pooled_team_agreement"] == correct / n
        assert abs(s["equal_game_team_agreement"] - np.mean([c["correct"] / c["n"] for c in cells if c["n"]])) < 1e-15

    games = [row["game_id"] for row in declaration["cohort"]["selected"]]
    drive = [totals[(game, "drive")] for game in games]
    accuracies = np.array([c["correct"] / c["n"] for c in drive])
    deltas = accuracies - np.array([max(c["teams"].values()) / c["n"] for c in drive])
    assert result["selected_games"] == games
    assert result["reserve_games"] == [row["game_id"] for row in declaration["cohort"]["reserve"]]
    for row in result["game_results"]:
        for cls, cell in row["classes"].items():
            total = totals[(row["game_id"], cls)]
            assert cell["n"] == total["n"] and cell["correct"] == total["correct"]
            assert cell["accuracy"] == (total["correct"] / total["n"] if total["n"] else None)
            assert cell["predicted_d_rate"] == (total["dead"] / total["n"] if total["n"] else None)
    rng = np.random.default_rng(declaration["uncertainty"]["seed"])
    indices = rng.integers(0, len(games), (declaration["uncertainty"]["replicates"], len(games)))
    bootstrap = result["bootstrap"]
    assert bootstrap["game_count"] == len(games) == 12
    assert bootstrap["replicates"] == declaration["uncertainty"]["replicates"]
    assert bootstrap["seed"] == declaration["uncertainty"]["seed"]
    assert bootstrap["no_frame_or_anchor_bootstrap"] is True
    assert abs(bootstrap["equal_game_drive_accuracy"] - accuracies.mean()) < 1e-15
    assert abs(bootstrap["equal_game_drive_minus_majority_accuracy"] - deltas.mean()) < 1e-15
    np.testing.assert_allclose(bootstrap["equal_game_drive_accuracy_95_interval"],
                               np.quantile(accuracies[indices].mean(axis=1), [.025, .975]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(bootstrap["equal_game_drive_minus_majority_95_interval"],
                               np.quantile(deltas[indices].mean(axis=1), [.025, .975]), rtol=0, atol=1e-15)
    receipt = dict(study=STUDY, result_sha256=sha(result_path), declaration_sha256=sha(declaration_path),
                   prediction_seal_sha256=sha(seal_path), independently_replayed_class_half_cells=replayed,
                   all_counts_exclusions_visibility_and_intervals_match=True,
                   inference_repeated=False, organizer_labels_accessed=False,
                   artifact_state="local verification; no raw-data upload or independent remote restore claimed")
    destination = RESULTS / f"{STUDY}_verification.json"
    with destination.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    verify()
