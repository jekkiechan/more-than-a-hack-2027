# Possession errors on the frozen external test match

The frozen `robust` model scored frame macro-F1 **0.930** and mean possession-spell F1 **0.623** on Sportec `J03WPY` under the `clean` condition. This is a post-hoc diagnosis of its saved predictions; no model was changed. It describes the remaining errors, not the best attainable accuracy.

At IoU ≥0.5, **229 / 421** true possessions matched (54.4% recall), with **90** unmatched predictions (71.8% precision).

| True possession duration | True spells | Matched at IoU ≥0.5 | Recall | Share of all misses | Share of live time |
|---|---:|---:|---:|---:|---:|
| <1 s | 57 | 7 | 12.3% | 26.0% | 0.9% |
| 1–<3 s | 118 | 37 | 31.4% | 42.2% | 7.0% |
| 3–<10 s | 149 | 94 | 63.1% | 28.6% | 24.7% |
| >=10 s | 97 | 91 | 93.8% | 3.1% | 67.4% |

Possessions shorter than three seconds account for **68.2%** of missed spells but only **7.9%** of live time. This explains how frame accuracy and possession-spell accuracy can diverge so sharply.

Of the **192** missed spells, **81** had no same-team overlap at all and **111** had some correctly attributed frames but no prediction reaching IoU 0.5. **0** additional misses arose only from the one-to-one matching constraint.

The model produced **319** possessions versus **421** in the reference (mean duration **10.20s** versus **7.74s**). **50** predictions overlap multiple true spells of the same team; **13** true spells overlap multiple predictions of their team. Of the missed true spells, **79** touch a merged prediction and **7** are fragmented. These overlap-based counts can intersect and use any positive overlap, including one frame; they should not be added or interpreted as proven causes.

| Matching threshold | Spell F1 | Matched true possessions |
|---|---:|---:|
| IoU ≥0.3 | 0.741 | 274 |
| IoU ≥0.4 | 0.686 | 254 |
| IoU ≥0.5 | 0.619 | 229 |
| IoU ≥0.6 | 0.562 | 208 |
| IoU ≥0.7 | 0.508 | 188 |

**98.2%** of all wrong frames are home/away swaps. The dead-ball F1 is **0.999**, but Sportec's original source freezes ball XY during dead play. That is a strong dataset-specific shortcut; the result does not establish accuracy on the hackathon's noisier, partially missing tracking.

Whole-match home possession share is **63.83%** predicted versus **62.84%** reference (absolute error **0.99 percentage points**). An accurate aggregate can coexist with missing many short possessions.

The existing 1,000-resample official 5th–95th percentile bootstrap intervals are **0.918–0.939** for frame macro-F1 and **0.584–0.662** for mean spell F1. They are conditional on this one match, with a fixed model; they do not measure generalization across matches, leagues or tracking providers.

Matching uses the unchanged official gap-aware scorer, including period/time separators, within-valid-run abstention filling, same-team overlaps and greedy one-to-one IoU matching. Only aggregate statistics are published. No public challenge ground truth was accessed.

Reproduce from the frozen local artifacts:

```bash
python -m experiments.error_analysis
```

Full aggregate results: [results/error_analysis.json](results/error_analysis.json).
