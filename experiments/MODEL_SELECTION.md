# Future model decisions

Use the competition headline ordering: **spell F1 first, macro F1 to break exact
spell ties**. The official evaluator sorts on those two metrics before its
further transition/latency tie-breaks. It does not combine them into a weighted
competition score. See `evaluation/run_evaluation.py:165` and `CHALLENGE.md`.

Show this additional internal summary beside both component scores:

```text
weighted_summary = 0.70 * spell_F1 + 0.30 * macro_F1
```

The user confirmed that the weighted value is an extra, not the primary ranking.
70/30 is the configurable reporting default, retaining more emphasis on the
competition's main objective. Choose and record any changed weights before a
future experiment; do not optimize weights to favor an already observed model.
All components use the same 0–1 scale.

## Current working candidate

**The adopted working baseline is GRU plus ordered HGB at 50/50 P(D), with the
25% ball-gap-only team-training replacement.** The user explicitly authorized
this robustness tradeoff: external clean spell/macro .5740/.8439 versus previous
.5746/.8449, and corrupted .4315/.7866 versus .4205/.7699. It does not beat the
previous model under the frozen clean spell-first rule. That historical
selection remains intact; the working-baseline promotion is recorded separately.
The manifest and default inference entrypoint are
`experiments/baseline_manifest.json` and
`python -m experiments.current_baseline predict`. Fresh confirmation remains
pending. The previously recorded public diagnostic is .5257/.8588 on only
17 reference possessions and was known before the user's promotion request;
no public check was repeated. See [BASELINE.md](BASELINE.md) for package versions,
the exact tradeoff and recovery of the previous model.

Current paired external comparison (equal provider means; H/A pooled):

| Inputs | Model | Spell F1 | Macro F1 | Weighted summary |
| --- | --- | ---: | ---: | ---: |
| Clean | Previous GRU/HGB | .5746 | .8449 | .6557 |
| Clean | Adopted ball-gap team model | .5740 | .8439 | .6550 |
| Corrupted | Previous GRU/HGB | .4205 | .7699 | .5253 |
| Corrupted | Adopted ball-gap team model | .4315 | .7866 | .5380 |

Historical temporal-D equal-provider holdout scores, with the previous team
backbone fixed (not a new ranking of the adopted team model):

| Model | Spell F1 | Macro F1 | Weighted summary |
| --- | ---: | ---: | ---: |
| GRU + HGB + LightGBM | .5750 | .8450 | .6560 |
| GRU + HGB | .5746 | .8449 | .6557 |
| GRU + LightGBM | .5716 | .8424 | .6528 |
| HGB + LightGBM | .5671 | .8376 | .6483 |
| Ordered HGB | .5652 | .8363 | .6465 |
| GRU | .5649 | .8383 | .6469 |
| LightGBM .02 | .5613 | .8322 | .6426 |
| TCN | .5605 | .8408 | .6446 |
| HGB baseline | .5588 | .8319 | .6408 |
| LightGBM .08 | .5577 | .8316 | .6399 |

Rows above follow spell-first order using unrounded scores. The weighted column
can reorder lower candidates: GRU's weighted value exceeds ordered HGB's,
despite its slightly lower spell F1. This illustrates why the two ranking rules
must remain separate. GRU plus boosting leads under all inspected weightings,
from macro-only through spell-only; no weight was selected to create that lead.

## How future reports should work

`model_decision_policy.json` is the durable policy. Freeze its content/hash with
future experiment configuration. After external evaluation, show both F1s,
the auxiliary combined value, per-provider results, and a complexity note.
Compare exactly the same source/match cohorts, then average matches within each
provider and give providers equal weight. Do not mix organizer public-window
scores into this selection table.

`model_scorecard.py` compares existing external result rows without opening
tracking, labels, or predictions. It rejects unequal cohorts, conflicting
duplicate metrics, and organizer reports. The current snapshot is explicitly
retrospective; older experiment choices retain their original criteria.

```sh
python -m experiments.model_scorecard
# For a separately recorded alternative auxiliary score:
python -m experiments.model_scorecard --spell-weight 0.5 --output /tmp/scorecard_50_50.json
```

The saved snapshot is
[results/model_scorecard_development.json](results/model_scorecard_development.json).
It records component scores, both ranking views, input hashes, and weight
sensitivity. These five historical development matches do not replace a fresh
confirmation set. Provider holdout still applies to the D head only; conditional
H/A remains fixed and pooled.
