# Possession errors on the frozen external test match

The frozen `ensemble` model scored frame macro-F1 **0.941** and mean possession-spell F1 **0.706** on Sportec `J03WQQ` under the `clean` condition. This is a post-hoc diagnosis of its saved predictions; no model was changed. It describes the remaining errors, not the best attainable accuracy.

At IoU ≥0.5, **281 / 401** true possessions matched (70.1% recall), with **113** unmatched predictions (71.3% precision).

| True possession duration | True spells | Matched at IoU ≥0.5 | Recall | Share of all misses | Share of live time |
|---|---:|---:|---:|---:|---:|
| <1 s | 37 | 8 | 21.6% | 24.2% | 0.6% |
| 1–<3 s | 118 | 62 | 52.5% | 46.7% | 6.8% |
| 3–<10 s | 136 | 106 | 77.9% | 25.0% | 23.6% |
| >=10 s | 110 | 105 | 95.5% | 4.2% | 69.0% |

Possessions shorter than three seconds account for **70.8%** of missed spells but only **7.4%** of live time. This explains how frame accuracy and possession-spell accuracy can diverge so sharply.

Of the **120** missed spells, **40** had no same-team overlap at all and **80** had some correctly attributed frames but no prediction reaching IoU 0.5. **0** additional misses arose only from the one-to-one matching constraint.

The model produced **394** possessions versus **401** in the reference (mean duration **8.55s** versus **8.41s**). **23** predictions overlap multiple true spells of the same team; **17** true spells overlap multiple predictions of their team. Of the missed true spells, **34** touch a merged prediction and **5** are fragmented. These overlap-based counts can intersect and use any positive overlap, including one frame; they should not be added or interpreted as proven causes.

| Matching threshold | Spell F1 | Matched true possessions |
|---|---:|---:|
| IoU ≥0.3 | 0.803 | 319 |
| IoU ≥0.4 | 0.767 | 305 |
| IoU ≥0.5 | 0.707 | 281 |
| IoU ≥0.6 | 0.659 | 262 |
| IoU ≥0.7 | 0.594 | 236 |

**98.2%** of all wrong frames are home/away swaps. The dead-ball F1 is **0.999**, but Sportec's original source freezes ball XY during dead play. That is a strong dataset-specific shortcut; the result does not establish accuracy on the hackathon's noisier, partially missing tracking.

Whole-match home possession share is **44.61%** predicted versus **43.90%** reference (absolute error **0.71 percentage points**). An accurate aggregate can coexist with missing many short possessions.

The existing 1,000-resample official 5th–95th percentile bootstrap intervals are **0.931–0.949** for frame macro-F1 and **0.673–0.737** for mean spell F1. They are conditional on this one match, with a fixed model; they do not measure generalization across matches, leagues or tracking providers.

Matching uses the unchanged official gap-aware scorer, including period/time separators, within-valid-run abstention filling, same-team overlaps and greedy one-to-one IoU matching. Only aggregate statistics are published. No public challenge ground truth was accessed.

Reproduce from the frozen local artifacts:

```bash
python -m experiments.error_analysis
```

Full aggregate results: [results/error_analysis.json](results/error_analysis.json).
