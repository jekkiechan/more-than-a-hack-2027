# Requested competition check of temporal models

Status: complete; the one-time check ran on 2026-10-27. The user requested “check comp as well”
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

### Completed one-time check

Ran once, after all four 199,319-row CSVs were format-validated and sealed. The
GRU and LightGBM candidates tie on spell F1. The LightGBM brings lower macro F1 and dead-ball recall.

| Candidate | Spell F1 | 90% interval | Macro F1 | D F1 | D precision / recall | Predicted spells | IoU .5 matched / spurious | IoU .7 matched |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- | ---: |
| Saved baseline | .5257 | [.3657, .6743] | .8588 | .8568 | — | 18 | 11 / 7 | — |
| Saved previous joint GRU | .6194 | [.4387, .8000] | .8574 | .8506 | — | 14 | 11 / 3 | — |
| Dropout GRU | .5818 | [.412, .752] | **.8689** | **.8393** | .901 / .786 | 16 | 11 / 5 | 7 |
| Full127 LightGBM | .5371 | [.400, .686] | .7460 | .5807 | 1.000 / .409 | 18 | 11 / 7 | 5 |
| Clean leader blend | .5500 | [.388, .725] | .8132 | .7363 | .953 / .600 | 15 | 10 / 5 | 5 |
| Robustness blend | .5818 | [.436, .739] | .8419 | .8048 | .987 / .679 | 16 | 11 / 5 | 5 |

There are 17 reference possessions and 7,500 graded frames. One possession moves
spell F1 by about .06. Every interval overlaps the others, and they are separate intervals rather than paired tests. Read the
table directionally:

- Adding Full127 to the dropout GRU leaves spell F1 unchanged (.5818 both) and lowers
  macro F1 by .027. D recall falls from .786 to .679, and it matches fewer spells at IoU .7 (5 vs 7).
- Full127 alone misses most stoppages (D recall .409, precision 1.0). This is the same failure
  seen on clean SkillCorner externally.
- These numbers agree with the [stress-component ablation](STRESS_COMPONENT_ABLATION.md):
  under feed-like stress the tree adds no spell value over the GRU.
- The dropout GRU's .5818 versus the previous joint GRU's .6194 differs by less than one
  possession. The control GRU alone was not a declared candidate and was not scored.

The check was resumed in a fresh environment with a [recorded
deviation](results/temporal_competition_resume_deviation.json). The 13 GRU TRAIN caches and
the cache receipts store wall-clock timings and were re-pinned. Their contents are
equivalent: two archived GRU checkpoints refit bit for bit from the rebuilt caches. All
other frozen guards ran unmodified through `temporal_competition_resume.py`. The gate is
now closed; do not repeat or tune from this result. [Aggregate report](results/temporal_competition_check.json).
Models and predictions remain under ignored `data/experiments/temporal_competition_v1`.
