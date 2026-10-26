# Lighter corruption with task-specific groups

This follow-up freezes one 25% replacement dose, instead of searching doses on
the development matches. It tests four combinations of unchanged or augmented
H/A attribution and unchanged or augmented GRU/HGB D heads. The previous
composite-corruption batch improved stress performance but reduced native-input
spell F1; these already used development matches do not provide fresh evidence.

The original training recipe remains the control. Each original clean/coarse
augmentation view has one quarter of its examples replaced by a targeted view.
The resulting mixture is approximately 37.5% clean, 37.5% original coarse
augmentation and 25% targeted corruption. It is not 75% clean training.

| New arm | Team training replacement | D training replacement |
| --- | --- | --- |
| Baseline | None | None |
| `ball_team25` | 25% ball visibility mask | None |
| `player_D25` | None | 25% player/coordinate corruption |
| `paired25` | 25% ball visibility mask | 25% player/coordinate corruption |

Team replacement applies only the previous recipe's sustained ball gaps and
sample-loss mask. All observed coordinates and player tracking remain native.
Original Markov ball masking is retained in the remaining control examples.
Replacement indices are drawn independently of labels; the same choice applies
to direct and team-swapped examples. The number of HGB rows and its 63 leaves,
260 iterations and learning rate .08 remain unchanged.

D replacement reuses the previous recipe's player/coordinate features, which
exclude the ball by construction. The original clean and player-slot-dropout
inputs remain in the other 75% of windows. A selected GRU input window uses one
observation variant throughout its full context; feature streams are not spliced
inside the window. Normalization uses training-only core rows of the selected
mixture. Window count, batch size, eight epochs and optimizer update count match
the original baseline. Ordered HGB uses the same selected core positions, with
all +/-1/3/5-second features computed inside their own input variant before
selection. Its row count and model capacity remain unchanged.

Evaluation reuses exactly the first batch's clean and two composite stress views
for Sportec J03WOY, Metrica game 3 and SkillCorner 1996435/1996436/2006229. Native
grading masks are fixed even after synthetic player loss. There are 90 new
evaluations and 30 checksum-bound, reused baseline evaluations. Match scores are
averaged within providers, then the three providers equally; stress seeds are
averaged within each match first. Provider holdouts apply only to D; team
backbones remain pooled.

The primary external selection rule is highest clean spell F1 in the D-provider
holdout view, with macro F1 for exact ties. The unchanged baseline remains
eligible. Stress metrics and the established 70% spell / 30% macro summary are
reported separately. No model is automatically promoted. This experiment cannot
identify real broadcast transfer or establish significance across providers.

## External results

All 90 new evaluations and 30 reused baseline comparisons completed. The
preferred D-provider-holdout summaries are below; team models remain pooled.
Stress scores average two seeds within a match before equal-provider means.

| Training arm | Clean spell F1 | Clean macro F1 | Corrupted spell F1 | Corrupted macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Unchanged baseline | .5746 | .8449 | .4205 | .7699 |
| Ball-gap team replacement, 25% | .5740 | .8439 | .4315 | .7866 |
| Player/coordinate D replacement, 25% | .5729 | .8443 | .4386 | .7850 |
| Paired targeted replacements | .5718 | .8434 | .4477 | .7997 |

Ball-only team replacement essentially preserves clean performance: the spell
change is -0.06 percentage points, versus -0.88 points for the previous 50%
composite team recipe. Its paired conditional bootstrap interval is
[-0.20, +0.16] points, not evidence of a reliable clean ranking. Corrupted spell
F1 rises by 1.10 points on all three providers. Conditional H/A accuracy under
stress improves from 80.10% to 82.88%.

The paired targeted arm gains 2.72 stress spell points while losing 0.28 clean
points. The earlier composite-both recipe gained 3.59 stress points but lost
1.29 clean points. This follow-up achieves a more moderate tradeoff; it changes
both the heavy-corruption dose and retention of the original augmentation, so
it does not isolate dose alone. The lighter D-only arm also improves over the
previous D-only point estimates in both clean and corrupted conditions.

No new arm beats baseline on the predeclared clean spell-first criterion. The
primary external selection therefore retains baseline. `ball_team25` is the
best new arm on that same criterion and is frozen separately as a reporting-only
competition diagnostic, in response to the user's explicit request. No working
model, decoder cost or augmentation setting changes after the diagnostic.

Pooled results have the same ordering on native inputs: baseline .5952/.8687
spell/macro versus paired .5935/.8669. Under composite stress, those become
.4307/.7810 and .4641/.8226. The [full external report](results/light_corruption_development.json)
retains every match/view and the paired whole-match deltas. Bootstrap intervals
are conditional on these reused matches; singleton providers cannot establish
cross-provider uncertainty. No fresh confirmation was added.

All 63 pre-check synthetic/protocol tests passed, followed by three public-check
guard/selection tests. Actual GRU budgets were verified: pooled 15,280 windows
per epoch and 1,912 updates; holdouts 11,874/1,488, 12,988/1,624 and 5,698/720.
All nine checkpoints pass their hash-bound receipts. The unchanged baseline was
regenerated byte for byte against its archived competition CSV hash.

## Competition score boundary

The saved adopted-baseline public check is .4889 spell / .8568 macro / .8630
dead-ball F1 on 7,500 graded frames and 17 possessions. These figures describe
the old frozen GRU/HGB baseline, not new experimental candidates. The full-match
competition score is unavailable. Public results were already known before
this experiment and do not select a model.

The external experiment runner has no organizer scoring command and preserves
every existing gate. Its external report copies only archived baseline metrics.
The user additionally requested, "try next experiment and also report to me the
comp's score." A separate, single-candidate descriptive check honors that request
after external selection and CSV generation. It has a new exclusive marker;
earlier markers remain intact. If selection retains the exact baseline CSV,
its existing score is reused without rereading the reference labels.

`AGENTS.md` says, "Never use repeated public-window checks as an iteration loop."
This explicit user-requested check cannot trigger fitting, reselection, another
candidate check, or any configuration change. Its result is reported separately
from external development and does not establish full-match performance.

## Reproduce

The previous experiment's prepared training and evaluation feature caches are
required locally, including `pending_*` feature matrices. Their hashes are
recorded before fitting. Restore the previous model archive and external data,
then run `python -m experiments.corruption_experiment prepare` and
`python -m experiments.prepare_corruption_cache` if these caches are absent.
Raw feature caches are not part of the private model archive.

```sh
python -m experiments.light_corruption freeze
python -m experiments.light_corruption prepare
python -m experiments.light_corruption fit-team
for fold in pooled without_sportec without_metrica without_skillcorner; do
  python -m experiments.light_corruption fit-d --fold $fold
done
python -m experiments.light_corruption team-probs
for fold in pooled without_sportec without_metrica without_skillcorner; do
  python -m experiments.light_corruption evaluate --fold $fold
done
python -m experiments.light_corruption report
python -m experiments.light_candidate
python -m unittest discover -s experiments/tests -v
python -m unittest experiments.eval_set.test_protocol -v
```

`prepare` and `team-probs` accept `--source` for independent worker processes.
Each D fold can also run independently. All new model files, feature caches and
per-frame predictions remain under ignored `data/experiments/light_corruption_v1/`.
Reports and checkpoint receipts bind fitted files to the frozen code/configuration.
The final candidate command generates and validates 199,319 labels from the
externally frozen selection without reading or scoring organizer reference labels.
The primary external selection can retain baseline. To answer the user's request
for the new experiment's competition score, the diagnostic uses the best **new**
arm under the same external clean spell-first ordering. This additional
reporting-only declaration is frozen before any new public check and leaves the
primary selection untouched. Only that one new candidate is scored:

```sh
python -m experiments.light_candidate --diagnostic-best-new
python -m experiments.light_public_check freeze --candidate experimental_candidate.csv
python -m experiments.light_public_check check
```
