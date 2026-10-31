# Dense control error audit v1

This audit describes the reconstructed `control_D2` two-seed ensemble on the five previously used provider-held-out development matches: one Metrica match, three SkillCorner matches, and one Sportec match. Each fold excludes its evaluation provider. The two seeds' native probabilities were averaged before the frozen cost-3 decoder. All 15 ensemble archives matched the sealed prediction receipt; their labels, ball-observation masks, timestamps, and evaluation-cache hashes matched the saved external inputs. The replay reproduced clean/feed-like spell F1 of 0.647215/0.505811, within the declared 0.000051 tolerance.

The original per-game GAR joblib binaries were unavailable. The regenerated records use the exact pinned raw SoccerNet source and source modules, match the original 45-game record counts, full-frame totals and period blocks, and reproduce the frozen dense-label TRAIN QA aggregates. The reconstructed archives therefore support a detailed descriptive audit, while remaining a reconstruction of the D2 fold control rather than the original fold checkpoint bytes. This is development evidence from previously exposed matches, not independent confirmation of the pooled deployment package.

## Largest measurable opportunities

Wrong-team errors dominate the stress view. The equal-provider rate over all GT-scored H/A/D frames is 5.33% clean, rising to 9.18% and 9.23% in the two feed-like corruptions. The rate is highest on SkillCorner: 10.77% clean, 13.67% in corruption 1, and 12.80% in corruption 2. The most concentrated errors are inside short GT spells (duration ≤1 second): wrong-team rates are 36.71% clean and 43.42–45.17% under corruption; false-D rates are 15.15–17.55%. The next bin is >1 to ≤5 seconds. These duration-bin rates use GT-alive frames within the stated spell-length bin.

The decoder also merges separate same-team GT spells into one predicted spell more often under feed-like corruption: 13.71% and 13.62% of predicted spells, versus 8.40% clean. Same-direction transitions matched within one second have mean per-segment median absolute shifts of 0.222 s clean, 0.295 s in corruption 1, and 0.252 s in corruption 2. Many transitions fall outside that matching window and are counted as unmatched, so these timing summaries apply only to matched edges.

| View | Equal-provider wrong-team rate | False-D rate | Missed-D rate | Raw scored frames | Raw wrong-team frames | Raw false-D frames | Raw missed-D frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 5.33% | 2.24% | 6.08% | 380,457 | 17,610 | 8,709 | 22,060 |
| Feed-like 1 | 9.18% | 2.51% | 6.12% | 380,457 | 32,648 | 9,819 | 22,308 |
| Feed-like 2 | 9.23% | 2.46% | 6.02% | 380,457 | 33,363 | 9,561 | 21,873 |

Provider wrong-team rates show where the stress degradation concentrates:

| Provider | Clean | Feed-like 1 | Feed-like 2 |
| --- | ---: | ---: | ---: |
| Metrica | 1.31% | 5.91% | 6.26% |
| SkillCorner, equal over three matches | 10.77% | 13.67% | 12.80% |
| Sportec | 3.92% | 7.96% | 8.64% |

Ball-missing/observed rates use all GT-scored frames, including D frames; they do not isolate long live-ball gaps. The equal-provider wrong-team rates are 1.44%/5.67% on missing/observed frames clean, 11.91%/6.81% in corruption 1, and 12.45%/6.20% in corruption 2. These are descriptive subsets of the external evaluation cohort and should not be compared with earlier long-gap-only live-frame rates.

## Limits and follow-up

The error categories are descriptive and can overlap; frame counts do not estimate the counterfactual F1 gain from fixing a category. Raw totals are shown separately from realization → match → equal-provider rate summaries, so longer matches do not set the provider-level conclusion. The ensemble has one realization per match. Short-spell frame rates are not spell-level recall. Transition shifts exclude unmatched boundaries beyond one second. No organizer labels or pooled deployment predictions were used.

The clearest next targets are short-spell team attribution, provider robustness under ball-missing corruption, and over-merging at boundaries. Test changes on the same provider-held-out cohort with the frozen controls and report provider-level results beside the aggregate.

Machine-readable counts, provider rates, spell bins, transition summaries, hashes, and reconstruction status: [dense_error_audit_v1.json](results/dense_error_audit_v1.json).
