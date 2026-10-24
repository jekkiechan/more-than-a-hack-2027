"""Plot aggregate external-test scores; no tracking data or frame labels read."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / "results/primary_test.json").read_text())
results = report["matches"]["J03WPY"]
names = ["state_machine", "temporal", "supervised", "robust"]
labels = ["Control rules", "Mixture + HMM", "Boosting + Viterbi", "Gap-augmented\nboosting (selected)"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
x, width = np.arange(len(names)), .35
for ax, metric, title in zip(axes, ["macro_f1", "spell_f1_mean"],
                             ["Per-frame macro F1", "Possession-spell F1"]):
    for offset, condition, color, label in [(-width/2, "clean", "#276596", "Complete ball tracking"),
                                           (width/2, "masked", "#dd8c37", "40.25% ball observations masked")]:
        values = [results[condition][name][metric] for name in names]
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set(title=title, ylim=(0, 1.02), xticks=x, xticklabels=labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18)
axes[0].axhline(.90, color="#444444", linestyle="--", linewidth=.8)
fig.suptitle("Frozen models · held-out Sportec match J03WPY", fontsize=14)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=2, frameon=False)
fig.text(.5, .025, "Provider labels; ball coordinates freeze during dead play. Synthetic gaps do not establish broadcast performance.",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, .16, 1, .94))
fig.savefig(HERE / "results/primary_comparison.png", dpi=170)
