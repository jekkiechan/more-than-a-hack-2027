# GRU and boosting probability ensembles

Equal-weight D-head ensembles improve historical external development spell F1
over their individual members. GRU + ordered HGB scores **.5746**, versus
**.5649** for GRU and **.5652** for ordered HGB on equal-provider D-head holdouts.
Adding LightGBM reaches **.5750**, just .0004 above the two-model blend.
The organizer five-minute gate was not reopened, and the previous competition
selection remains unchanged.

## Architecture

The members are the existing frozen bidirectional GRU, ordered-feature HGB,
and LightGBM D head at learning rate .02 / 640 trees. This is an ensemble of
**dead-ball probabilities**, not three independent full H/A/D systems.
Conditional team probabilities and Viterbi cost 3 stay identical throughout.

For the 50/50 GRU + HGB pair:

```text
p_dead = 0.5 * p_dead_GRU + 0.5 * p_dead_ordered_HGB
P(H) = (1 - p_dead) * P(H | live)
P(A) = (1 - p_dead) * P(A | live)
P(D) = p_dead
```

Average the probabilities at native frame rate, before combination and sequence
decoding. Averaging hard H/A/D labels would discard confidence and boundary
information. All members already see offline past/future context; the ensemble
does not concatenate matches, bridge time gaps, or refit a temporal model.

The experiment predeclared four equal-weight combinations: each pair and the
three-way average. There was no weight sweep, calibrator, stacking fit,
retraining, or decoder adjustment. Source, probability-cache and external-data
hashes were frozen before the new evaluation. This is a separate external-only
experiment; it does not extend or alter the completed organizer experiment.

## Results

All **40 ensemble match evaluations** completed, alongside 30 reused single-head
scores. Five historical development matches cover three providers. Pooled rows
measure new-match performance within represented providers; provider holdout
excludes the evaluation provider from D-head training only. The H/A teacher
remains pooled. Scores average matches within source, then sources equally.

| D head | Provider-holdout spell F1 | Provider-holdout D F1 | Pooled spell F1 |
| --- | ---: | ---: | ---: |
| GRU | .5649 | .8303 | .5891 |
| Ordered HGB | .5652 | .8223 | .5853 |
| LightGBM .02 | .5613 | .8187 | .5744 |
| GRU + HGB | .5746 | .8392 | **.5952** |
| GRU + LightGBM | .5716 | .8362 | .5949 |
| HGB + LightGBM | .5671 | .8259 | .5864 |
| GRU + HGB + LightGBM | **.5750** | .8387 | .5944 |

By the predeclared primary score, the three-way blend has the highest external
development value. Its margin over GRU + HGB is too small to establish a useful
advantage. The pair has slightly higher pooled spell F1 and D F1. Both are
reasonable confirmation candidates; these results do not select a competition
replacement or demonstrate an organizer score for an ensemble.

The GRU + HGB gain over GRU is **+.00973**, and over ordered HGB **+.00946**.
Three-way gains are **+.01014 / +.00987**. The two-tree-only ensemble gains
less, consistent with overlapping errors between HGB and LightGBM.

| D-head holdout source | GRU | Ordered HGB | GRU + HGB | Three-way |
| --- | ---: | ---: | ---: | ---: |
| Sportec | .5633 | .5652 | .5751 | .5782 |
| Metrica | .6650 | .6741 | .6837 | .6826 |
| SkillCorner | **.4665** | .4562 | .4650 | .4643 |

The average gain does not mean every provider improves over GRU. GRU remains
slightly ahead on SkillCorner, the partial-view source; only 24–29% of D frames
survive its visibility filter. These are historical development results with
silver labels on Metrica/SkillCorner, not fresh confirmation.

## Error diversity and probability quality

Raw D Brier score improves from **.07306** for GRU and **.07641** for HGB to
**.07006** for their blend, using equal-source means. Lower is better. This
supports averaging probabilities even though their errors remain correlated.
The three-way value is .07069, slightly worse than the pair.

Provider-averaged signed probability residual correlations are about .88 for
GRU/HGB, .84 for GRU/LightGBM, and .91 for HGB/LightGBM. Diversity alone does
not determine ensemble value: the less-correlated LightGBM pairing still has
lower spell F1 than the HGB pairing. The report includes every per-match error
overlap, prediction disagreement, Brier score, and correlation.

Paired bootstrap deltas are conditional on these exact sources and matches.
With one Sportec and one Metrica match, those sources add no resampling
variation. Consequently narrow intervals cannot establish unseen-provider
significance. A genuinely new external match remains the next useful check.

## Reproduce

```sh
python -m experiments.temporal_ensemble
python -m unittest experiments.tests.test_temporal_ensemble -v
```

The runner reuses frozen native-rate probability caches from
`data/experiments/learned_temporal`. It fingerprints inputs, resumes partial
progress, and refuses to overwrite the completed report. Per-frame blend outputs
remain ignored under `data/experiments/temporal_ensemble/`.
Aggregate results:
[results/temporal_ensemble_development.json](results/temporal_ensemble_development.json).
No organizer data or organizer predictions are opened by this runner.
