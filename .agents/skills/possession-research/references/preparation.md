# Reusable external feature preparation

1. Restore verified private arrays before feature generation. A split-plan
   preview reads the manifest only and needs no private arrays. Missing cache
   access is not a reason to download organizer
   labels, rebuild external labels, or retrain a model.
2. Use [preparation.py](../../../../experiments/preparation.py) and the existing
   [split manifest](../../../../experiments/eval_set/manifest.json). Do not invent
   a random frame/window split. Preview the exact training and selection list:

   ```sh
   python -m experiments.preparation plan --fold leave_out_skillcorner
   ```

   Canonical folds are `pooled` and `leave_out_<provider>` (not the older
   checkpoint filename convention `without_<provider>`). The shared planner
   validates groups, paths, match identity, excluded historical tests, and full
   provider exclusion. These are reused development matches, never fresh tests.
3. Prepare unnormalized records once, then reuse the same cache across folds:

   ```sh
   python -m experiments.preparation prepare --fold leave_out_skillcorner
   ```

   Defaults: data at `MTAH_DATA_DIR` or repository `data/`; cache under that data
   directory's `experiments/prepared_v1/`. `--data-dir` relocates the manifest's
   `data/` prefix. Keep arrays, receipts and generated plans in ignored/private
   storage. Never commit them to this public repository.
4. Use `clean`, `legacy`, `feed_like`, or `composite` explicitly. For either stress
   view pass `--calibration <frozen-calibration.json>`, containing the actual
   calibration object (`ball_missing_fraction`, `gap_edges_seconds`, and
   `gap_bin_probability`), not an entire report. Record its provenance in the
   study declaration. Calibration is not estimated by this command. Seeds are
   derived from match identity and the explicit seed, independent of iteration
   order. Clean caches ignore the unused seed and can be shared across seed runs.
   This is a new recipe: old per-runner augmentation seeds are not parity
   claims. `feed_like` omits sampling and regional blanking; keep `composite`
   separately named. Corruption always precedes feature extraction.
5. Import `split_plan`, `recipe`, `prepare_match`, and `load_record` in a new
   runner. `prepare_match` returns a content-addressed cache directory;
   `load_record` verifies its receipt and payload before loading without pickle.
   Cache keys include input SHA, match/group/provider, view/calibration/seed,
   feature source hashes, and dependency versions. A cache hit verifies the input
   too. Changed inputs fail; changed recipes create separate entries. Partial or
   tampered entries fail rather than silently rebuilding.

   ```python
   from experiments import preparation as prep

   plan = prep.split_plan(fold='leave_out_skillcorner')
   config = prep.recipe('clean', seed=2026)
   training = [prep.load_record(prep.prepare_match(item, data_dir, cache_dir, config))
               for item in plan['train']]
   # Fit normalization and EVERY learned H/A/D component on training only.
   # Build selection records separately from plan['selection'].
   ```

6. Records use the existing joint-GRU base schema: `x`, `y`, `names`, `indices`,
   `period`, `native_n`, `fps`, `source`, `match`. They contain the union of
   refine_features team features and player-only D features at 5 Hz. `y=-1`
   preserves X; `period` identifies half/time breaks for existing sequence
   chunking. Additional native-grid arrays retain graded labels, original
   periods/timestamps and observation masks for audits. SkillCorner grading is
   fixed before corruption; do not recompute it from a corrupted view. No
   fitted normalization, model, team swap, training-row weighting, geometry/
   anchor expansion, or full LightGBM design is cached. Apply such declared
   transforms after loading; keep player-only D inputs restricted to their
   schema. Do not treat the joint feature union as an acceptable ball-aware D
   head. An architecture needing different features gets a new versioned recipe
   and parity contracts, not an edited frozen runner.
7. Freeze the emitted plan hash and recipe in the new study declaration.
   The CLI stores prepared plans at `<cache>/plans/<sha256>.json`. Exclude the
   provider from normalization and all learned components, not just D. Selection
   arrays may be prepared in advance but must not fit transformations. Keep
   derived views/windows with their source match. Read
   [possession-experiment](experiment.md) before any fitting
   or scoring; this skill does neither.
8. Run the synthetic preparation contracts and required repository suite. Check
   native grading, masks, unknown targets, team symmetry, discontinuities, input
   seals and cache tampering. Existing frozen runners and declarations stay
   unchanged; migrate a future runner when its declared recipe matches this API.

Focused synthetic check (no private arrays, fitting or organizer scoring):

```sh
python -m unittest experiments.tests.test_preparation -v
```

Tests: [test_preparation.py](../../../../experiments/tests/test_preparation.py).
Deliver the actual plan/counts, excluded provider and per-component training-only
normalization rule; for preparation, include the recipe/plan hash and cache receipt.
