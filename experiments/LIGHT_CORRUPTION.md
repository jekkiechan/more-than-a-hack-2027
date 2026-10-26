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

## Competition score boundary

The saved adopted-baseline public check is .4889 spell / .8568 macro / .8630
dead-ball F1 on 7,500 graded frames and 17 possessions. These figures describe
the old frozen GRU/HGB baseline, not new experimental candidates. The full-match
competition score is unavailable. Public results were already known before
this experiment and do not select a model.

This runner has no organizer scoring command and preserves every existing gate.
`AGENTS.md` says, "Never use repeated public-window checks as an iteration loop."
After freezing external selection, the report copies only those archived
aggregate competition metrics and explicitly marks new candidates unscored.

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
