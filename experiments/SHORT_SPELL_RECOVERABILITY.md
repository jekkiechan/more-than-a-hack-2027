# Short-spell recoverability audit

**Decision:** the current cost-3 path leaves a small, measurable subset of short possessions recoverable from native emissions, but an argmax decoder is a poor overall replacement. Complete under-three-second recall rises from .307 to .384 on clean views and .137 to .196 on feed-like views under the diagnostic raw argmax path, while spell F1 falls from .633 to .561 and .499 to .419. The remaining misses divide between correct evidence that still forms an insufficient interval and spans with no correct-team top-1 emission at all.

This is a read-only audit of the combined-objective two-seed ensemble on the same five previously exposed, full-provider-holdout development matches. It reuses the saved candidate probabilities, saved cost-3 paths, and exact labels/tracking snapshots from the earlier opportunity audit. The unchanged official scorer globally matches all references and predictions at IoU .5 before filtering by duration, censoring, or ball visibility. All 15 saved cells reproduced their paired candidate score rows.

The primary cohort is candidate-path misses among reference spells under three seconds whose observed endpoints do not touch a graded-run boundary. Duration summaries also include graded-edge spells and spells of three seconds or longer. “Visible ball” means a finite observed ball coordinate on every frame of the reference span; “ball gap” means at least one missing frame. Feed-like rows average two fixed corruptions within match; providers are reported separately, with SkillCorner averaging three matches equally.

For each primary miss, the audit assigns one exclusive description:

1. **Cost-3 suppressed a correct emission:** the raw native probability argmax chooses the reference team on at least one frame inside the span, while the saved cost-3 path chooses it on none.
2. **Correct evidence, insufficient interval:** some correct-team argmax or same-team decoded/raw overlap exists, but the globally matched candidate still misses because the interval is too short, displaced, or fragmented.
3. **No local correct-team emission:** there is no correct-team argmax frame and no same-team raw or decoded overlap inside the reference span.
4. **Global assignment conflict:** a cost-3 same-team spell has IoU at least .5, but the unchanged one-to-one matcher assigns it elsewhere. This residual category is reported separately.

These descriptions diagnose saved outputs. In particular, raw argmax is a counterfactual path used only to test emission recoverability; it is not a deployable candidate. No reference-informed repair, oracle prediction, new fit, threshold sweep, organizer-label access, or model selection was performed.

| View | Provider | Complete under-3s references | Misses | Cost-3 suppressed | Correct evidence, interval insufficient | No local correct-team emission | Assignment conflict | Cost-3 recall | Raw-argmax recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | Sportec | 197 | 130 | 15 (12%) | 53 (41%) | 62 (48%) | 0 | .340 | .416 |
| Clean | Metrica | 125 | 61 | 17 (28%) | 30 (49%) | 14 (23%) | 0 | .512 | .632 |
| Clean | SkillCorner | 51.7 | 48.3 | 2.3 (5%) | 14.7 (30%) | 31.3 (65%) | 0 | .069 | .102 |
| Feed-like | Sportec | 197 | 170.5 | 17.5 (10%) | 74.5 (44%) | 78.5 (46%) | 0 | .135 | .193 |
| Feed-like | Metrica | 125 | 95.5 | 15 (16%) | 39.5 (41%) | 41 (43%) | 0 | .236 | .324 |
| Feed-like | SkillCorner | 51.7 | 49.7 | 2.3 (5%) | 15.7 (32%) | 31.7 (64%) | 0 | .039 | .071 |

Counts for SkillCorner and feed-like views are per-match means, so fractional values are expected. Equal-provider mean mechanism shares are 15% / 40% / 45% for clean views and 10% / 39% / 51% for feed-like views, in the column order above. No assignment conflicts occurred in the primary cohort.

The emission measurements support the distinction. Among clean misses classed as having correct evidence but insufficient intervals, mean reference-team probability is .736, mean probability margin over the other teams is +.513, and the best candidate IoU averages .178. For this group the raw argmax covers 91% of reference frames correctly, yet it still cannot form a sufficiently overlapping interval. In the no-local-emission group, mean reference-team probability is .133, the mean margin is −.664, and the nearest same-team raw emission in the same graded run is 8.1 seconds away on average. Feed-like values are .648 / +.355 / .141 for the insufficient-interval group and .214 / −.494 / 9.0 seconds for the no-local-emission group. These are descriptive event averages, not calibrated confidence or a repair-gain estimate.

The raw argmax globally matches more complete short spells, but loses overall spell F1 in both views. Its equal-provider spell F1 is .561 clean and .419 feed-like, versus .633 and .499 for the saved cost-3 path. The marginal short-recall improvement therefore does not establish a decoder change. A turnover-conditioned decoder could target the roughly 10–15% of these misses where cost-3 removes all correct emissions, while improved emissions are needed for the no-local-signal group, which is especially large in SkillCorner. The correct-evidence group also needs interval-aware work; simply lowering persistence would leave many spells too short or misaligned.

| View | Provider | <0.5 s | 0.5–<1 s | 1–<2 s | 2–<3 s | ≥3 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean | Sportec | 17 (17/0) | 21 (21/0) | 56 (56/0) | 36 (36/0) | 60 (60/0) |
| Clean | Metrica | 8 (8/0) | 15 (15/0) | 23 (23/0) | 16 (15/1) | 26 (26/0) |
| Clean | SkillCorner | 19.7 (6.7/13.0) | 29.0 (13.3/15.7) | 33.7 (16.7/17.0) | 24.3 (11.7/12.7) | 81.0 (37.3/43.7) |
| Feed-like | Sportec | 17 (17/0) | 22.5 (22.5/0) | 75 (75/0) | 56 (56/0) | 108.5 (108.5/0) |
| Feed-like | Metrica | 8 (8/0) | 20 (20/0) | 45.5 (45.5/0) | 23 (22/1) | 85.5 (85.0/0.5) |
| Feed-like | SkillCorner | 21.8 (6.7/15.2) | 29.5 (13.3/16.2) | 35.8 (16.7/19.2) | 29.2 (13.0/16.2) | 96.3 (43.2/53.2) |

Each duration cell is **missed total (complete / graded-edge)**, averaged per match. SkillCorner’s many edge-censored spells make its complete-only under-three-second recall substantially lower than the earlier all-spell short-recall figure. These edge misses remain in the official score; their cut endpoints limit timing interpretation.

| View | Provider | Complete under-3s misses with visible ball | Complete under-3s misses containing a ball gap |
| --- | --- | ---: | ---: |
| Clean | Sportec | 130 | 0 |
| Clean | Metrica | 57 | 4 |
| Clean | SkillCorner | 48.0 | 0.3 |
| Feed-like | Sportec | 75.5 | 95.0 |
| Feed-like | Metrica | 34.5 | 61.0 |
| Feed-like | SkillCorner | 24.5 | 25.2 |

Feed-like short misses mostly occur in spans containing at least one ball gap for Sportec and Metrica. This is an association in these fixed corruptions, not evidence that ball visibility or a particular turnover caused the miss. The audit does not cross the mechanism labels with a separately inferred turnover state.

Evidence files are [the frozen v2 declaration](results/short_spell_recoverability_v2_declaration.json), [the aggregate replay](results/short_spell_recoverability_v2.json), and [the preserved aborted preflight receipt](results/short_spell_recoverability_declaration.json). The first preflight stopped on a probability-sum tolerance just below the saved float rounding error and wrote no result; the corrected v2 receipt used a fixed `2e-7` validation tolerance and completed the audit. All matches were previously exposed development data; Metrica and SkillCorner labels are event-derived silver references. This audit is descriptive and does not establish independent transfer or a deployable gain.
