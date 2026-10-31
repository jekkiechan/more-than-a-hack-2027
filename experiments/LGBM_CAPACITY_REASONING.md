# LightGBM capacity reasoning for full possession classification

**Feature engineering is the primary comparison.** The full H/A/D study uses
one fixed 63-leaf, depth-8 configuration for its first three feature sets, then
tests 127 leaves and depth 10 on the complete feature set. This distinguishes
the value of better possession evidence from the effect of adding tree
capacity. These are bounded working hypotheses, not proven optimal settings.

The completed full H/A/D structural audit below measures all twelve frozen
multiclass models. A separate historical section describes the earlier binary
player-only D models. The user clarified the intended full H/A/D scope after
that first audit completed. Neither structural audit establishes a held-out
performance gain or determines optimal capacity.

## Full H A D feature and capacity design

| Arm | Feature scope | Feature count | Leaves | Maximum depth |
| --- | --- | ---: | ---: | ---: |
| Legacy | Existing joint ball, team and player features | 204 | 63 | 8 |
| Geometry | Legacy plus all-player control, approach and signed player acceleration | 227 | 63 | 8 |
| Full | Geometry plus curated before/after context, persistence and reliable contact anchors | 1003 | 63 | 8 |
| Full wider | Same complete feature schema | Same as Full | 127 | 10 |

The new [geometry module](possession_geometry.py) defines 23 geometry columns
and a separate 14-column reliable-anchor group. Geometry uses the best control
evidence across all players, opponent ambiguity, projected closest approaches
and collective braking/acceleration. Anchors retain observed team evidence and
its age across short observation gaps, without filling ball positions. The
204/227/1003 schema sizes were verified using real cached base-feature names
before freezing; the run declaration records the complete manifest.

All four arms use learning rate 0.01, **640 multiclass boosting iterations**,
minimum leaf support 120, and L2 regularization 10. There are three class trees
per iteration, or 1,920 trees per model. The same training views, sampling,
provider exclusions, seeds and decoder make the first three comparisons about
features. The final comparison changes leaves and depth together; it does not
isolate one of those parameters on its own.

The 640-iteration budget bounds the initial comparison and keeps all arms
matched. It is not a convergence claim. No additional iterations, feature
groups, thresholds or capacity arms are selected after observing these scores
under the same declaration.

### Why this is a reasonable bounded capacity

The 204 baseline columns and the larger temporal schema contain many correlated
views of the same tracking quantities. Before/after summaries add useful
direction and persistence, but do not create hundreds of independent physical
processes. A tree does not need one depth level per input feature. Depth counts
the conditions along one path; it can even split the same feature repeatedly.
Leaves allocate distinct local regimes across the whole tree.

Full possession classification requires more than identifying stopped motion:
it must compare competing teams, distinguish plausible control from a passing
ball, retain evidence through gaps, and recognize stoppage/restart patterns.
The engineered signals make those comparisons explicit, reducing how many
sequential splits are needed to reconstruct them. A 63-leaf/depth-8 tree can
allocate multiple such regimes while limiting long specialized chains. The
127-leaf/depth-10 arm probes whether the same evidence needs finer partitions.
Neither follows mathematically from the feature count.

The training universe contains thirteen matches, but each complete provider
holdout trains on only **five, ten or eleven matches**. Extra frames, clean and
corrupted views, and H/A swaps do not create new independent matches. They
increase coverage of known conditions. Row support and regularization therefore
remain important even when the flattened matrix contains hundreds of thousands
of rows. Capacity should earn its place through matched whole-provider spell
F1 and stress behavior, not training loss or the raw number of frames.

The full-model sampler retains run edges at 5 Hz and thins interiors to 1.25 Hz,
with weights preserving each original known class's 5 Hz time mass. Thus the
120-row leaf minimum has no fixed conversion to seconds in this study. The
48-second frame-equivalent illustration in the binary audit below applies only
to that older uniformly sampled 2.5 Hz matrix.

The binary audit below suggests two practical checks: distinguish a leaf cap
from a depth cap, and inspect match coverage instead of treating 120 correlated
rows as 120 independent examples. It does not show that 63 or 127 leaves are
optimal for the full multiclass problem.

## Actual full H A D tree structure

The read-only [multiclass audit](joint_lgbm_capacity_audit.py) inspected all
twelve frozen checkpoints and their training histories. It verified model and
receipt hashes, declared parameters, feature manifests, class order, excluded
provider training cohorts, and the 640-iteration/1,920-tree budget. It read no
evaluation scores or labels and made no predictions. The
[complete measurements](results/joint_lgbm_capacity_audit.json) include separate
H, A and D tree summaries.

**Every class tree reaches its configured maximum depth:** 8 for the first
three arms and 10 for the wider arm. The depth cap therefore affects actual
tree shape, including the unchanged legacy feature set. Reaching a cap does
not establish that loosening it would improve generalization.

| Held-out provider | Arm | Trees reaching leaf cap | Row-count-weighted leaf depth | Median rows per leaf | Leaves with at most 150 rows |
| --- | --- | ---: | ---: | ---: | ---: |
| Sportec | Legacy 63 | 99.48% | 6.99 | 1,233 | 5.09% |
| Sportec | Geometry 63 | 99.79% | 7.00 | 1,247 | 4.98% |
| Sportec | Full 63 | 99.43% | 7.07 | 1,086 | 6.12% |
| Sportec | Full 127 | 97.81% | 8.62 | 498 | 11.36% |
| Metrica | Legacy 63 | 99.84% | 6.79 | 1,402 | 3.33% |
| Metrica | Geometry 63 | 100.00% | 6.79 | 1,408 | 3.20% |
| Metrica | Full 63 | 99.95% | 6.80 | 1,376 | 2.97% |
| Metrica | Full 127 | 99.90% | 8.17 | 608 | 7.16% |
| SkillCorner | Legacy 63 | 92.97% | 6.52 | 583 | 13.40% |
| SkillCorner | Geometry 63 | 95.21% | 6.48 | 576 | 13.23% |
| SkillCorner | Full 63 | 91.82% | 6.71 | 514 | 14.99% |
| SkillCorner | Full 127 | 65.42% | 8.25 | 253 | 24.32% |

This makes the width/depth tradeoff concrete. The wider models use deeper paths
for substantial training-row mass: mean count-weighted depth rises from
6.71–7.07 to 8.17–8.62 on the complete features. Their median leaf support
roughly halves. The number of columns alone would not have revealed either
behavior. Geometry adds 23 features with almost unchanged depth and support;
the temporal expansion supplies different evidence at the same 63-leaf cap.

The fold trained on just five matches, holding out SkillCorner, needs special
care. Only 21.56% of its wider **D** trees reach 127 leaves, versus 86.88% of H
trees and 87.81% of A trees, although all reach depth 10. This is evidence that
class-specific tree growth is constrained before the leaf budget is exhausted;
it does not identify which combination of depth, split gain, regularization
and support is responsible. A blanket claim that all classes need more leaves
would be unsupported.

Leaf counts are correlated fitted-row counts, not independent matches, unique
frames, or seconds. The audit does not route training matches through these
multiclass trees, so the exact binary selected-tree match-coverage results below
must not be transferred to them. Actual dumped counts can fall below the
configured `min_child_samples=120`: this occurs for 1.29–12.50% of leaves across
the twelve models. Treat that parameter as a fitting constraint with
LightGBM's count approximations, not a guarantee that every dumped leaf has
120 distinct observations. The reported weighted depth uses `leaf_count`;
the JSON also supplies Hessian-weighted depth using `leaf_weight`. Neither is
an exact depth average under the original class-time sample weights.

### Training-loss progression, without selecting a new budget

| Held-out provider | Legacy 63 final loss | Geometry 63 final loss | Full 63 final loss | Full 127 final loss | Full 63 decrease over final 64 rounds | Full 127 decrease over final 64 rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sportec | 0.2566 | 0.2556 | 0.2184 | 0.1762 | 3.61% | 5.14% |
| Metrica | 0.2609 | 0.2612 | 0.2257 | 0.1856 | 3.31% | 4.69% |
| SkillCorner | 0.1423 | 0.1413 | 0.1144 | 0.0893 | 5.50% | 7.30% |

These are weighted **training multiclass log losses**, not spell F1. Comparisons
are meaningful between arms within a fold; folds contain different training
populations. Every history decreases on every round, with its minimum at round
640. Thus the fixed budget is visibly not a demonstrated training-loss plateau.
The full feature expansion improves training fit at matched capacity in every
fold; increasing capacity improves it further. Neither observation establishes
held-out spell quality, robustness, or the benefit of extra boosting rounds.
The existing declaration remains unchanged. A future budget comparison would
need its own prospective design rather than extending a selected arm after
reading the current evaluation.

The practical conclusion is a justified comparison, not an optimum: 63/depth 8
provides a substantial shared feature-engineering baseline, while 127/depth 10
tests more detailed partitions at visibly lower support. Matched full-pipeline
spell F1 and corruption behavior must decide whether that extra detail helps.

## Scope of the earlier binary audit

All twelve original D-head models were inspected from their frozen checkpoints.
No model was fitted, no external prediction was scored, no organizer labels
were opened, and no declaration or model source was changed. The unfinished
larger binary-model extension is outside this audit.

## What the binary models actually learned

These models use either 67 existing player-motion/shape features or 625
temporally expanded features. The latter are not 625 independent physical
measurements: they expand 18 primitive player signals with existing rolling
summaries, observation indicators, and before/after context at 1, 3 and 5 seconds.
Correlated temporal derivatives increase the number of candidate splits much
more than the number of independent information sources.

| Configuration | Leaf cap | Depth cap | Trees | Median maximum depth per tree across folds | Deepest observed tree across folds | Training-count-weighted mean leaf depth across folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw slow | 31 | Unlimited | 640 | 10–11 | 19–21 | 5.41–6.14 |
| Context slow | 31 | Unlimited | 640 | 10–11 | 16–21 | 5.94–6.35 |
| Context shallow | 15 | 5 | 480 | 5 | 5 | 4.39–4.45 |
| Context regularized | 31 | 7 | 640 | 7 | 7 | 5.85–5.95 |

Every 31-leaf tree reaches its leaf cap. Between 99.79% and 100% of shallow
trees reach 15 leaves. Every shallow tree reaches depth 5; between 98.91% and
99.38% of regularized trees reach depth 7. A small minority of the latter stop
at depth 6, so a median or maximum of 7 must not be read as every tree reaching 7.

**31 leaves does not mean depth 5.** A balanced tree could use approximately
five levels, but LightGBM grows the leaf offering the greatest gain. It can
allocate a long chain to one part of the feature space. The unlimited-depth
models actually reach depth 16–21 while retaining only 31 leaves. Their typical
training observation follows about six splits; the deepest paths affect smaller
regions. A depth cap constrains these chains independently of the leaf cap.

Reaching a cap shows that the optimizer found additional allowable split gain
until the configured limit. It does not prove underfitting or establish that
more leaves generalize. Boosting can also spend capacity fitting correlated
noise or provider-specific artifacts.

## Row support is usually much larger than the minimum

The regularized configuration uses `min_child_samples=120`, `max_depth=7`,
`num_leaves=31`, L1 0.2, L2 10, feature fraction 0.7, row fraction 0.8,
bagging frequency 1, 127 bins, and learning rate 0.02 over 640 trees.

| Held-out provider | Training matches | Training rows including both views | Leaf row count P10 | Leaf row count median | Leaves with at most 150 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metrica | 11 | 207,964 | 440 | 2,250 | 1.31% |
| SkillCorner | 5 | 141,392 | 296 | 1,481 | 2.50% |
| Sportec | 10 | 179,452 | 375 | 1,964 | 1.36% |

At 2.5 Hz, 120 sampled rows correspond to 48 seconds of frame-equivalent
observations. They need not be consecutive, unique seconds or independent
decisions. Clean and dropout variants reuse the same football moments, and
nearby rows are strongly correlated. Bagging also changes the rows used by
each tree. The effective number of independent examples cannot be obtained by
dividing a frame count by the sampling rate.

Only 1.31–2.50% of regularized leaves lie within 25% of the configured row floor.
In this binary experiment, the leaf and depth limits therefore bind much more
often than a tiny leaf population. Increasing `min_child_samples` is not a
substitute for checking match diversity or held-out performance. Conversely,
large leaf row counts do not prove robust support across different matches.

The observed fraction of leaves below the nominal 120-row floor is 0.65–1.20%.
LightGBM documents this constraint as a Hessian-based approximation; such leaves
are possible with weighted boosting. `min_child_weight` is a Hessian constraint,
not a second literal count of independent observations.

## Exact match coverage for selected trees

The follow-up pass routes **all eligible training rows**, including both clean
and dropout views, through the first 16 and last 16 trees of each regularized
model. Each provider fold therefore contributes 992 inspected leaves. Feature
context is built before selecting the original training stride, and processing
streams one match/view at a time. Models, declarations and caches are checked
against their receipts.

| Held-out provider | Training matches | Leaves reached by every training match | Leaves reached by at most two matches | Leaves reached by only one match |
| --- | ---: | ---: | ---: | ---: |
| Metrica | 11 | 82.06% | 0% | 0% |
| SkillCorner | 5 | 92.84% | 4.84% | 0.91% |
| Sportec | 10 | 78.93% | 0.20% | 0% |

Every inspected leaf is reached by at least one training row. Most are reached
by every available training match. The few narrow-support leaves are concentrated
in the fold with only five training matches. This does not support a blanket
claim that the existing 31-leaf binary models are merely one-match lookup tables.

These are exact **routing** counts for the selected trees, including rows that
may have been out of bag when an individual tree was fitted. They are not exact
per-tree in-bag match counts. Only the first and last 16 trees are inspected for
match coverage; middle-tree coverage is unmeasured. Distinct matches are a more
useful support check than adjacent frames, but even broad training-match coverage
does not establish performance on a new provider. Raw routing frequencies are
also not the provider-balanced loss weights used in fitting.

## What carries over to full H A D model design

The original question needs a full three-class model using the existing joint
ball, team and player feature union, followed by temporal expansion. Its
capacity must account for H/A attribution and contested-ball persistence as
well as dead play. The binary player-only audit cannot decide how many leaves
those additional interactions require.

The following reasoning still applies:

1. **Choose leaves and depth separately.** Leaves allocate separate local
   regimes; depth bounds the sequence of conditions along one path. More
   columns alone do not justify a deeper tree. A moderate depth cap can allow
   more parallel regimes without the 16–21-level chains observed here.
2. **Treat 31, 63 and 127 leaves as testable capacities, not formulas derived
   from feature count.** A 63-leaf/depth-8 model is a bounded wider hypothesis;
   127/depth-10 is a higher-capacity probe. Their suitability for the full model
   must be measured on matched full-pipeline provider holdouts. This audit did
   not fit or evaluate either full H/A/D configuration.
3. **Account for multiclass boosting explicitly.** LightGBM normally builds
   three trees per boosting iteration for H/A/D. A 640-iteration three-class
   model therefore has 1,920 trees; it is not directly comparable in tree count
   or compute with the 640-tree binary models measured above. Support and
   gradients also differ by class.
4. **Keep correlated observations in mind.** Leaf support, row subsampling,
   feature subsampling, L1/L2 and histogram resolution regulate different
   behavior. None converts duplicated frames into independent examples.
   Family-level temporal correlation means a 0.7 feature fraction still gives
   the tree many alternative versions of the same physical cue.
5. **Separate architecture evidence from shrinkage evidence.** Halving the
   learning rate and doubling iterations keeps their product fixed, but does
   not make two ensembles equivalent. A wider/deeper model with that change is
   a combined hypothesis. Compare it honestly rather than attributing any
   improvement solely to width.
6. **Select the final decoded H/A/D sequence on spell F1 and robustness.**
   Training split gains, leaf occupancy and binary loss cannot choose the
   capacity that preserves true turnovers and stoppages. Full probability
   blending with the GRU must be evaluated as a complete pipeline.

No adopted baseline, frozen binary model, or model-selection decision changed
because of this historical structural audit. The full H/A/D structure is now
measured separately above; its matched predictive evaluation remains separate
from both audits.

## Reproducible evidence

- [All twelve full H/A/D model structures and training histories](results/joint_lgbm_capacity_audit.json).
- [Multiclass structure audit](joint_lgbm_capacity_audit.py).
- [All twelve model structures and preliminary bounded routing sample](results/lgbm_capacity_audit.json).
- [Exact full-row routing for the three regularized models](results/lgbm_capacity_full_support.json).
- [Structure and bounded-support audit](lgbm_capacity_audit.py).
- [Full-row selected-tree follow-up](lgbm_capacity_full_support.py).

All three audit scripts are read-only with respect to model artifacts and
refuse to overwrite completed audit outputs. The binary selected-tree helper assumes
a binary booster; a multiclass audit needs explicit class-tree indexing before
reuse. The new multiclass structure audit handles H/A/D indexing explicitly;
it does not perform match-routing analysis. No script reports predictive
performance.
