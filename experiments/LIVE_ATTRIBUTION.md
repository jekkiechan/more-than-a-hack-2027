# Live-only team attribution with frozen GRU/HGB dead-ball probabilities

Results pending the predeclared external run. This experiment does not promote a
new baseline or access organizer inputs/reference labels.

## Hypothesis and controls

The current conditional H/A distribution comes from a three-class H/A/D HGB,
renormalized over H/A, even though the GRU/ordered-HGB ensemble supplies P(D).
Train a binary H/A model on live training frames so it spends its tree capacity
on the remaining team decision. Test one fixed weighting ablation that raises
the influence of short training possessions, which frame loss underweights.

There are exactly three candidates, with no adaptive search:

- Packaged contextual HGB baseline, unchanged.
- `live_only`: H/A targets only, uniform weights within provider.
- `live_spell_weighted`: the same binary model, with training-frame weight
  `clip(sqrt(5 / spell_duration_seconds), .25, 4)` before provider normalization.

Both new heads use the existing 137 physical/contextual features, 63 leaves,
260 boosting iterations, learning rate .08, minimum leaf size 35, L2 penalty 1,
no early stopping, and seed 17. Binary HGB uses one tree per iteration rather
than the multiclass baseline's three. This holds the per-class settings fixed,
not the total number of trees.

Sampling is unchanged: take the existing roughly 4-Hz graded training rows,
then retain H/A rows. Augmentation stays clean plus the existing Markov ball-gap
view, with team swaps for both; no new corruption recipe is introduced. Weight
durations are computed from training labels only, split at X/D, half changes,
and timestamp discontinuities. Each provider receives equal total training
weight. Both new heads are frozen before any development scoring.

Every candidate receives the exact same packaged 50/50 GRU + ordered HGB P(D)
and unchanged Viterbi cost 3. The supplied D probability is unchanged by the
new team head; joint Viterbi decoding can still alter final D intervals when
team likelihoods change. This tests replacing team attribution in the current
pipeline, rather than retraining the D detector.

An earlier two-stage classifier was tried on Sportec in `SUPERVISED.md`, and
short-spell weighting helped a later Sportec physical-feature model in
`REFINE_WEIGHTED.md`. Those are prior art for the hypothesis, not results for
this pooled binary head combined with the packaged GRU/HGB D detector.

## Evaluation scope

Training uses the established 13 matches: Sportec J03WMX/J03WN1/J03WOH,
Metrica games 1/2, and the first eight SkillCorner H/A/D matches. The five
whole-match development cases are Sportec J03WOY, Metrica game 3, and
SkillCorner 1996435/1996436/2006229. Historical Sportec test matches are excluded.
SkillCorner retains the existing detected-position masks and X filtering below
12 visible players; its interruption labels have ungraded boundary margins.
Training weights describe graded live runs after this filtering, which can
split one longer possession into several visible pieces.

The D checkpoints are **pooled**, so this run is whole-match development with
all providers represented in training. Its baseline must be compared with the
other rows in this report, not the historical .5746 D-provider-holdout score.
None of the five matches is fresh confirmation. Metrica/SkillCorner targets
are event-derived silver labels, and Sportec has a frozen-ball D shortcut.

The primary view uses clean inputs. The supplementary view applies the existing
41% Markov ball-mask recipe, mean gap one second, independently fixed seed
112026, before recomputing features. All team heads receive identical masked
inputs; player-only P(D) remains identical across conditions. This is a ball-gap
sensitivity check, not a realistic broadcast corruption benchmark.

Scoring uses the unchanged official evaluator, with half/time-gap separation.
Average match scores within provider, then average the three providers equally.
Rank by spell F1, using macro F1 only for exact ties; also show the existing
70% spell / 30% macro auxiliary score. Report D F1/precision/recall/boundaries,
conditional live-frame team accuracy, spell counts, and short-spell recall at
IoU .5. Paired whole-match bootstrap intervals condition on the observed sources;
singleton Sportec/Metrica cases contribute no match-sampling uncertainty.
Only one training seed is used; this does not establish seed robustness.

## Reproduce

Restore the external NPZs and frozen model package as described in `BASELINE.md`.
The organizer inputs are unnecessary. Install the experiment requirements and
CPU PyTorch 2.8.0, then run from the repository root:

```sh
python -m experiments.live_attribution \
  --out data/experiments/live_attribution_reproduction \
  --report experiments/results/live_attribution_reproduction.json
python -m unittest discover -s experiments/tests -v
python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
```

Use fresh output paths; frozen reports/specifications are not overwritten.
Provider feature caches are prepared independently in three processes; model
fits each use one CPU thread. The predeclaration records input, checkpoint,
source, and metric hashes and package versions. Models, frame predictions, and
feature caches remain under ignored `data/`; only aggregate results are tracked.
