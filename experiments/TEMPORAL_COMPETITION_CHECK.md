# Requested competition check of temporal models

Status: frozen; pooled fits running. The user requested “check comp as well”
after reviewing the completed full-provider holdout results. This is a separate
user-requested check; previous public gates and declarations remain closed.

## Fixed comparison

Four candidates are fixed from external development before opening organizer
reference labels:

| Candidate | Pooled deployment recipe | Seed-2026 clean/stress holdout spell F1 |
| --- | --- | ---: |
| Dropout GRU | Player-only D joint GRU with slot-dropout training | .6088 / .3180 |
| Full127 LightGBM | Direct H/A/D, 1,003 features, 127 leaves/depth 10 | .5555 / .4345 |
| Clean leader blend | Geometry63 LightGBM + control GRU, full-vector 50/50 | .6257 / .3718 |
| Robustness blend | Full127 LightGBM + dropout GRU, full-vector 50/50 | .6088 / .4164 |

The deployment seed is fixed at 2026, matching the earlier competition-check
convention. The original holdout report also shows separate-seed means; those
means are not one deployable ensemble. All four recipes failed at least one
external adoption guardrail. A public point estimate cannot erase that failure.

Training uses the same 13 external TRAIN matches as the prior competition GRU:
three Sportec, two Metrica and eight SkillCorner matches. The five external
development matches remain excluded. These pooled deployment fits differ from
the provider-excluded models used for the holdout scores above.

The fixed budget is two tree fits (geometry63 and full127), two eight-epoch GRU
fits (control and dropout), four full-length CSVs and one four-candidate scoring
gate. Trees retain learning rate .01, 640 multiclass rounds, sampling, weights,
regularization and deterministic settings. Two CPU threads are an execution
change only. There is no threshold, blend-weight, decoder or seed sweep.

## Artifact and scoring boundary

`temporal_competition_fit.py` verifies the frozen training/geometry caches and
builds pooled matrices without loading organizer inputs or labels.
`temporal_competition_check.py` has separate freeze, prepare, seal, predict and
check commands. Inference uses the verified tracking-only cache, original
feature engineering and decoder cost 3; native frames outside playing periods
remain X, as in the existing deployment pipeline.

All four models, input/source hashes, native probabilities and 199,319-row CSVs
must be sealed and format-validated before the exclusive scoring marker is
claimed. The marker remains closed even if scoring raises an error. No candidate
can be appended after the fixed set is sealed.

The check reuses saved baseline spell F1 .5257 and previous breakthrough GRU
.6194. Available competition labels cover only 7,500 graded frames and 17
possessions. Full-match spell F1 remains unavailable. Report official spell F1
first, with macro/D diagnostics and separate official bootstrap intervals;
intervals are not paired significance tests. Do not train, tune, select or adopt
from this diagnostic.

The separately requested confidence-abstention investigation stays on external
development data and does not modify this candidate set.

## Results

[Frozen declaration](results/temporal_competition_declaration.json). All 142 preflight tests passed; an additional inference-mask regression brings the focused diagnostic suite to eight passing tests.

Pending the fixed fits and sealed predictions. Output directory:
`data/experiments/temporal_competition_v1`. Aggregate report after the one-time
check: `results/temporal_competition_check.json`. Model/prediction payloads remain
private or under ignored `data/`.
