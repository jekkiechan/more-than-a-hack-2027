"""Render aggregate-only PCBAS match and action-class diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"


def main():
    result = json.loads((RESULTS / "pcbas_expanded_anchor_v2_result.json").read_text())
    rows = result["game_results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("PCBAS: broader broadcast team-attribution check", fontsize=16, fontweight="bold", x=.05, ha="left")
    fig.text(.05, .905, "Frozen submission package · 12 new matches · ball input entirely missing", color="#475569", fontsize=10)
    left, right = axes
    y = np.arange(len(rows))
    left.scatter([r["majority_direction_accuracy"] * 100 for r in rows], y, marker="|", s=130, color="#94a3b8", label="Majority direction reference")
    left.scatter([r["drive_accuracy"] * 100 for r in rows], y, s=40, color="#2563eb", label="Submission Drive agreement", zorder=3)
    left.set_yticks(y, [r["game_id"].replace("game_", "Game ") for r in rows])
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.set_xlabel("Agreement at Drive moments (%)")
    left.grid(axis="x", alpha=.2)
    left.legend(loc="upper left", bbox_to_anchor=(0, -.14), fontsize=8, frameon=False)
    names = ["drive", "pass", "cross", "shot"]
    agreement = [result["summary"][name]["equal_game_team_agreement"] * 100 for name in names]
    dead = []
    for name in names:
        cells = [r["classes"][name] for r in rows]
        dead.append(sum(c["predicted_state_counts"].get("D", 0) for c in cells) / sum(c["n"] for c in cells) * 100)
    positions = np.arange(len(names))
    right.barh(positions - .18, agreement, height=.33, color="#2563eb", label="Equal-game team agreement")
    right.barh(positions + .18, dead, height=.33, color="#dc6b41", label="Pooled predicted D at live anchors")
    right.set_yticks(positions, [n.title() for n in names])
    right.invert_yaxis()
    right.set_xlim(0, 100)
    right.set_xlabel("Percent (different metrics; no shared denominator)")
    right.grid(axis="x", alpha=.2)
    right.legend(loc="upper left", bbox_to_anchor=(0, -.14), fontsize=8, frameon=False)
    for i, (acc, d) in enumerate(zip(agreement, dead)):
        right.text(acc + 1, i - .18, f"{acc:.1f}", va="center", fontsize=8)
        right.text(d + 1, i + .18, f"{d:.1f}", va="center", fontsize=8)
    interval = result["bootstrap"]["equal_game_drive_accuracy_95_interval"]
    fig.text(.05, .045, f"Drive 95% whole-game bootstrap interval: {interval[0]:.1%}–{interval[1]:.1%}. Sparse action anchors; no possession spell F1 or D precision/recall.", fontsize=9, color="#475569")
    for ax in axes:
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=.84, bottom=.24, left=.09, right=.98, wspace=.32)
    destination = RESULTS / "pcbas_expanded_anchor_v2.svg"
    fig.savefig(destination, facecolor=fig.get_facecolor())
    fig.savefig(ROOT / "data/experiments/pcbas_expanded_anchor_v2/aggregate_chart.png", facecolor=fig.get_facecolor(), dpi=140)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
