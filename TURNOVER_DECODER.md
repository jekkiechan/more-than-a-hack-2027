# Learned turnover-edge decoder experiment

This bounded experiment tested whether a small event head could lower the
Viterbi transition cost at likely Home↔Away turnovers and recover short spells
that the uniform cost-3 decoder suppresses. The experiment did not pass its
predeclared clean spell-F1 screen. Keep the archived cost-3 decoder as the
baseline; this result does not support adoption.

## Frozen mechanism

One L2-regularized logistic head was trained per held-out provider on that
fold's TRAIN matches. Each head predicts exact directed H→A and A→H changes
between adjacent eligible 5 Hz probability samples. Its features use the
frozen two-seed ensemble probabilities at both endpoints, oriented H/A
log-odds changes, and local tracking evidence such as control/contact
impulses, nearby-player support and ball availability/speed. Features are
orientation-normalized so swapping team labels swaps the directed outputs.

The head used natural event prevalence, with equal provider mass and then
equal match mass. There was no event undersampling or class balancing. The
single predeclared mapping is
`cost = clip(3 − (logit(q) − logit(prior)), 1.5, 4.5)` in the decoder's
10 Hz log-likelihood units. Each 5 Hz edge estimate is held over its
corresponding 0.2 s native interval; the decoder applies its existing `fps/10`
scale. Uncovered rows retain cost 3. Only H↔A edges change. H/A↔D costs remain
`3 × fps/10`, ensemble emissions are unchanged, and there is no minimum spell
duration. Training eligibility checks every native target within the sampled
interval; time blocks come from tracking periods/timestamps, never from X
labels.

This differs from PR #13's directional D/live stoppage/restart head, which
changed only D/live costs, and from the earlier class-balanced H/A boundary
classifier, which targeted a neighborhood around boundaries. The present
head predicts exact natural-prevalence H/A edges and changes only their costs.

## Data and provenance

All three heads were fit and sealed before reading the 15 evaluation cells.
For every held-out-provider fold, the same frozen `without_<provider>` two-seed
backbone pair generated ensemble predictions for all training matches and the
evaluation matches. Training backbone predictions are in-sample for their
training matches; this is disclosed. The excluded evaluation provider is
absent from every learned component. The supplementary provenance receipt
binds the training caches, model checkpoints, evaluation snapshots and raw
selection inputs to their pinned manifests; the independent raw-input check
also matched all 18 TRAIN/selection NPZ files.

The cohort contains five previously exposed development matches. Metrica and
SkillCorner targets are event-derived silver labels, and `feed_like` is
synthetic tracking corruption. These results are internal development
evidence, not an estimate of organizer performance or generalization to new
matches.

| Held-out provider | Directed rows | Positive directed edges | Weighted natural prevalence | Solver iterations |
| --- | ---: | ---: | ---: | ---: |
| Sportec | 260,156 | 1,775 | 0.006913 | 13 |
| Metrica | 283,012 | 2,390 | 0.009482 | 13 |
| SkillCorner | 156,444 | 1,551 | 0.009573 | 11 |

All fits converged. The actual 50/50 seed ensemble trained each head; the
component-seed results below are diagnostics only, not separately fitted
models.

## Results

Metrics are equal-provider means across the 15 cells (feed-like replicates
averaged within match). `False extra` and `merged` are counts per match.

| View / decoder | Spell F1 | Macro F1 | Short-spell recall (<3 s) | False extra | Merged | IoU .3 | IoU .4 | IoU .5 | IoU .6 | IoU .7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean, cost 3 | .633205 | .854226 | .384161 | 125.556 | 30.556 | .723165 | .695803 | .646311 | .587839 | .512906 |
| Clean, learned edges | .633205 | .854219 | .384161 | 125.556 | 30.556 | .723165 | .695803 | .646311 | .587839 | .512906 |
| Feed-like, cost 3 | .498830 | .811387 | .203371 | 137.889 | 43.667 | .610564 | .570382 | .503104 | .438757 | .371340 |
| Feed-like, learned edges | .498996 | .811431 | .204088 | 137.944 | 43.611 | .610703 | .570531 | .503417 | .438938 | .371389 |

Candidate-minus-baseline clean spell F1 was exactly 0.000000, failing the
required +0.01 screen. Clean macro-F1 changed by −0.000009 / 0 / −0.000011 for
Sportec / Metrica / SkillCorner; clean D-F1 and all clean IoUs were unchanged.
The feed-like spell-F1 delta was +0.000166 and short-spell recall changed by
+0.000717. Feed-like IoU deltas from .3 through .7 were +0.000139,
+0.000149, +0.000312, +0.000181 and +0.000049. The feed-like no-worse-than
−0.01 floor and per-provider clean macro/D-F1 guards passed; the primary clean
screen did not.

Per-provider feed-like spell-F1 changes were 0 for Sportec, 0 for Metrica and
+0.000498 for SkillCorner. The latter came with +0.002151 short-spell recall,
+0.166667 false-extra spells and −0.166667 merged references per match. This
small stress-view change does not offset the failed clean screen.

Applying the shared head to each individual frozen seed was also nearly
neutral: seed 2026 feed-like spell-F1 changed by +0.000029 and seed 2027 by
+0.000029; neither seed's short-spell recall changed. The actual ensemble
remains the primary comparison above.

Across all saved native rows, learned directed edge costs stayed near 3 and
did not hit the bounds: clean 1st/50th/99th percentiles were 2.9920/3.0006/
3.0080; feed-like percentiles were 2.9939/3.0008/3.0061. No tuning or
additional arms followed these results.

## Reproduction and artifacts

Run the synthetic contracts with:

```sh
/workspace/venv/bin/python -m unittest discover -s experiments/tests -p test_turnover_decoder.py -v
```

The declaration, fit/provenance/score receipts and source are under
`experiments/`; fitted heads, per-cell predictions and the complete run are in
the local-only `data/experiments/turnover_decoder_v1/` directory. The artifact
receipt lists their hashes. No organizer labels or scores were accessed, and
there was no automatic promotion.
