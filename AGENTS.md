# Agent handoff: football possession challenge

Read this before changing possession models or evaluating them. The full methods,
provenance, results, and reproduction commands are in
[experiments/REFINEMENT.md](experiments/REFINEMENT.md). Competition definitions and
restrictions are authoritative in [CHALLENGE.md](CHALLENGE.md),
[EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), and
[RULES.md](RULES.md).

## Repository and task

- Upstream: `metrica-sports/more-than-a-hack-2027`; working fork:
  `jekkiechan/more-than-a-hack-2027`. The completed experiment series is on
  `possession/refinement` and merged into the fork's `main`. Preserve the
  `upstream` remote. Earlier stages are on `possession/research` and
  `possession/experiments`.
- Label each of 199,319 tactical-video frames at 25 Hz: `H` home/Barcelona in
  live possession, `A` away/Athletic Club in live possession, `D` dead ball, or
  `X` abstention/outside a half. Loose or contested ball remains with the last
  controlling team until control changes. The submission is a reproducible
  `frame,label` CSV for frames 0–199318.
- Ranking is **possession-spell F1**, with frame macro F1 as first tie-break.
  A spell is one continuous live H or A interval. The evaluator greedily
  matches predicted and reference spells of the same team one-to-one by
  temporal intersection-over-union, computes F1 at IoU 0.3, 0.4, 0.5, 0.6,
  and 0.7, then averages those five F1s. Frame macro F1 averages H/A/D F1.
  High frame F1 can coexist with low spell F1 when short possessions or
  turnover boundaries are missed. Use the official evaluator, rather than
  inventing a substitute metric.

## Data and non-negotiable evaluation boundaries

- Organizer match assets are local under gitignored `data/`. Do not commit or
  redistribute these assets, frame-level tracking exports, video, or other
  derivatives that reconstruct the data. Check `MTAH_DATA_DIR` if assets are
  outside the repository. The full-match reference labels are private.
- The public five-minute ground truth contains only **7,500 graded frames and
  17 possession spells**. Under `RULES.md`, it is for pipeline checks only:
  **do not train, tune, select a candidate, set a threshold, or claim
  full-match performance from it**. Do not hand-label or manually correct any
  part of this match, or use other human-produced event/possession data about
  it. Algorithmic, label-free improvements to its tracking are allowed.
- External data from other matches can be used if the organizers can access
  the method/data and the technical document explains them. The existing
  external benchmark is Sportec Open DFL (CC BY 4.0), with whole-match splits:
  train `J03WMX/J03WN1/J03WOH`, validation `J03WOY`, fresh local-model
  confirmation `J03WQQ`. Earlier tests `J03WPY/J03WR9` must not be recycled
  for selection. `J03WQQ` was supplied as validation for the pretrained
  PathCRF checkpoint, so it is **not** an independent PathCRF test.
- Sportec freezes ball coordinates during dead play. This makes D unusually
  easy and means its clean tracking scores do not transfer directly to the
  organizer's noisy, occluded broadcast-derived tracking. The organizer ball
  is missing in roughly 41% of graded frames. Synthetic missing-ball tests
  probe sensitivity but do not reproduce real broadcast occlusions.

## What has been tried and what the numbers mean

- The original gap-augmented boosting + Viterbi model scored **0.9298 frame
  macro F1 / 0.6232 spell F1** on held-out Sportec `J03WPY`. The later
  validation-selected refinement—physical ball/player-relative features,
  short-spell training weights, a larger model, a fixed 50/50 ensemble, and
  fixed team/dead transition costs—scored **0.9408 / 0.7059** on `J03WQQ`,
  versus **0.9303 / 0.6654** for the starting model on that same match.
- On the organizer's **public five-minute pipeline check**, the externally
  selected refinement scored **0.6487 macro / 0.4300 spell**. The predeclared
  PathCRF H/A + original robust D hybrid scored **0.6795 / 0.5067**: the
  highest *observed* spell score at that stage on this small window.
  It is not a validated winner or a basis to switch the final candidate under
  the rules. No full-match accuracy has been measured. The user's target of
  0.90 on both metrics has **not** been demonstrated on competition data.
- A predeclared player-only stoppage diagnostic scored **0.7874 macro /
  0.3429 spell** on the public window; its D-class F1 was **0.7537**, versus
  **0.3392** for the selected refinement. That suggests the selected model's
  Sportec-derived dead-ball cue transfers poorly. The diagnostic detects D
  better on this window but does not solve spell attribution, and its result
  cannot be used to select a replacement on these public labels.
- PathCRF is an open-source, pretrained player-path model with H/A output, not
  a native D detector. Its original implementation required a generic,
  tracking-only roster adapter for fragmented organizer tracks. The adapter
  was debugged using unlabeled geometry and synthetic cases before public
  scoring; it interpolates 7.75% of player-slot frames, with missing stretches
  as long as 145 seconds. See [experiments/CHALLENGE_TRANSFER.md](experiments/CHALLENGE_TRANSFER.md)
  and [experiments/PATHCRF.md](experiments/PATHCRF.md).
- The external headroom analysis found that short spells remain hard (final
  recall at IoU 0.5: 52.5% for 1–3 s, 21.6% for subsecond spells). No intrinsic
  ceiling near 0.71 was established. Further Sportec-only tuning shows
  diminishing returns and cannot establish broadcast transfer. The most useful
  next evidence is an **independent labeled broadcast match** with real
  missing-ball and stoppage behavior, evaluated at both frame and spell level.
  Better ball/player tracking and stoppage evidence are likely valuable.

## Current research directions

- Future reports follow `experiments/model_decision_policy.json`: official
  spell-first ranking with macro F1 as exact-tie break; show an additional
  70% spell / 30% macro summary, never use it as the competition ranking.
  Keep both components visible and freeze weights before future experiments.
  The current user-authorized working package is `gru_hgb_ballgap25_v1`: the
  existing 50/50 GRU/ordered-HGB D heads plus the HGB team model trained with
  25% ball-gap-only replacement. The user requested this robustness tradeoff:
  clean external spell/macro .5740/.8439 versus previous .5746/.8449;
  corrupted .4315/.7866 versus .4205/.7699. It is not a new clean-score winner.
  `experiments/baseline_manifest.json` binds the new package and explicit user
  authorization; the old manifest/policy are under `experiments/baselines/`.
  The historical frozen clean selection is unchanged. Run
  `python -m experiments.current_baseline predict`; it now writes
  `data/experiments/current_baseline/gru_hgb_ballgap25.csv` without GT and
  reproduces the tested diagnostic byte for byte. Fresh confirmation is pending.
  Three-way GRU/HGB/LightGBM remains numerically highest on the current
  development metrics, by a tiny margin; do not claim the simpler pair wins
  numerically or overwrite prior frozen competition selection history. Details and
  the saved scorecard are in `experiments/MODEL_SELECTION.md`.

- [experiments/RESEARCH_LOG.md](experiments/RESEARCH_LOG.md) records the completed
  contextual-model/D-blend comparison and the later measured temporal work.
  The older contextual implementation stays on `possession/robust-pipeline`
  in draft PR #8 and is not part of the current baseline promotion. TCN and
  BiGRU have now been evaluated; LSTM remains untested.
- The current package's existing public diagnostic is .5257 spell/.8588 macro/
  .8568 D F1 on 7,500 graded frames and 17 reference possessions. It was known
  before the later explicit user-requested working-baseline promotion; it is
  descriptive evidence, not independent confirmation. The previous GRU/HGB
  check remains .4889/.8568 and the historical PathCRF hybrid .5067/.6795.
  No full-match competition score is available. See `experiments/BASELINE.md`,
  `results/light_corruption_public_check.json` and the preserved older reports.
  Promotion did not rerun scoring or reopen any exclusive gate.
- The prioritized plan is [experiments/DIRECTIONS.md](experiments/DIRECTIONS.md):
  representative data and validation, then targeted D/context comparisons and
  preprocessing. Broad LightGBM hyperparameter tuning comes after those.
- New labelled sources and the per-source dead-ball shortcuts are in
  [experiments/OPEN_DATA.md](experiments/OPEN_DATA.md): Metrica's sample games
  and SkillCorner with D labels.
- Do not select models on Sportec J03WOY alone; use the multi-source suite
  (`multisource_eval.py`).
- The bounded temporal-D experiment is [experiments/LEARNED_TEMPORAL.md](experiments/LEARNED_TEMPORAL.md).
  It holds conditional H/A and decoder fixed. Historical provider-holdout D
  development spell F1: HGB baseline .5588, ordered HGB .5652, BiGRU .5649,
  TCN .5605. TCN improves D F1 but has no clear overall spell advantage over
  boosting. LightGBM D heads .08/160 vs .02/640 trees score .5577/.5613.
  This does not repeat the earlier full-model LightGBM .02/1500-tree result.
  These folds hold out providers from D training only; H/A remains pooled.
  `experiments/eval_set/manifest.json` records hashes and prior data exposure;
  none of these matches is fresh confirmation. SkillCorner X margins make D
  boundary timing ungradable. Organizer tracking was recovered via the exact
  Drive folder link in the registration email (keyword search missed it).
  The one-time public check is complete: BiGRU .4889 spell/.8617 macro, TCN
  .4541/.8506, ordered HGB .4205/.8653. Previous refinement was .4300/.6487;
  previous PathCRF hybrid remains higher in spell F1 at .5067. No model was
  selected or tuned on these public results. Details and the animated ChatGPT
  Page are linked in `experiments/TEMPORAL_PUBLIC_CHECK.md`; the public gate
  must not be reset or used for iteration.

## Artifacts and reproducibility

- The lighter, task-specific corruption follow-up is
  `experiments/LIGHT_CORRUPTION.md` and `light_corruption.py`. It retains 75%
  of the original clean/coarse recipe and replaces 25% of rows/windows with
  ball visibility gaps for H/A or player/coordinate corruption for D. Budgets,
  grading masks and evaluation views are fixed; 90 new evaluations plus 30
  reused baseline comparisons and 66 tests completed. Clean/stress spell is
  .5740/.4315 for ball-only teams and .5718/.4477 for both replacements,
  versus baseline .5746/.4205. The old baseline won the frozen external clean
  spell-first ordering; that record is preserved. The user subsequently adopted
  ball-only teams as the current working robustness baseline. At the user's
  explicit request, the best new external
  arm (ball-only teams) received one separately frozen public diagnostic:
  .5257 spell/.8588 macro/.8568 D F1, 18 predicted versus 17 reference spells.
  This exceeds the previous baseline's .4889 public spell point estimate but is
  not independent selection evidence; the 90% spell interval is [.3657,.6743] and full-match
  performance is unknown. Do not repeat the check, reset its gate or tune from
  it. `light_candidate.py` exports the selected/diagnostic CSVs without GT;
  `light_public_check.py` has the closed exclusive gate. Reports/receipts are
  in `experiments/results/light_corruption_*.json`; nine checkpoints and the
  closed gate are archived privately on `mtah-data` branch
  `experiments/light-corruption`, with `light_corruption_cache_manifest.json`.

- The completed composite-corruption ablation is
  `experiments/REALISTIC_CORRUPTION.md`, implemented by `tracking_corruption.py`
  and `corruption_experiment.py`. It froze aggregate unlabeled organizer gap
  statistics, kept native grading masks fixed, and evaluated four training arms
  on clean plus two corrupted views of the same five development matches.
  All 120 evaluations and 59 synthetic/protocol tests completed. D-provider-held-out
  baseline spell/macro is .5746/.8449 clean and .4205/.7699 corrupted;
  augmenting both heads gives .5617/.8399 and .4564/.7993. Stress robustness
  improves on every provider, but clean spell F1 falls. Retain GRU/HGB as the
  adopted baseline. This is reused development data and shared-family synthetic
  stress, not new broadcast confirmation. Results/receipts are tracked under
  `experiments/results/realistic_corruption_*.json`; all 15 checkpoints and
  frozen reports are archived in private `mtah-data`, restored using
  `corruption_cache_manifest.json`. No public gate was reopened. Next, isolate
  corruption groups and lower training weight before evaluating a new recipe.

- The PathCRF retention test is `experiments/PATHCRF_ENSEMBLE.md`. It uses
  Metrica game 3 and the three SkillCorner evaluation matches, excluding
  Sportec J03WOY because PathCRF trained on it. With GRU/HGB P(D) fixed,
  baseline spell/macro is .5744/.8346 on that two-provider cohort; 50/50
  HGB/PathCRF team blending falls to .5414/.8213 and PathCRF teams alone to
  .4993/.8090. Keep PathCRF as an archived comparator, not a default component.
  No organizer recheck was performed. Runtime is 13–15 minutes per match on
  one CPU thread; SkillCorner needs 50–60% player-slot imputation. Baseline
  models and provenance are persisted in private mtah-data under
  `models/gru_hgb_v1/`; restore with `competition_cache.py fetch --manifest
  baseline_cache_manifest.json --data-dir <code-repo>/data`.

- The external-only probability-ensemble follow-up is
  `experiments/TEMPORAL_ENSEMBLE.md`. Fixed 50/50 GRU + ordered HGB scores
  .5746 provider-holdout D spell F1; the equal three-way blend adds little
  (.5750). GRU alone is .5649 and ordered HGB .5652. GRU remains slightly
  stronger on SkillCorner alone. New combinations were evaluated only on
  historical external development matches; no organizer gate was reopened,
  model retrained, or original competition selection changed. Results are in
  `experiments/results/temporal_ensemble_development.json`.

- Start at [experiments/REFINEMENT.md](experiments/REFINEMENT.md), then read
  [experiments/README.md](experiments/README.md) and
  [docs/WORKPLAN.md](docs/WORKPLAN.md) for the earlier baseline and task setup.
  Tracked aggregate results and frozen configuration are in
  `experiments/results/refinement_*.json`; detailed subexperiments are linked
  from `REFINEMENT.md`. Do not present validation or test results as fresh
  independent trials after reading them.
- `experiments/refinement.py` enforces the frozen local-model protocol and
  generates/checks full-length organizer candidates. The selected, validated
  **but not organizer-submitted** CSV is local and ignored at
  `data/experiments/refinement/competition/selected_refinement.csv`.
  The PathCRF transfer code is in `experiments/challenge_pathcrf.py`.
- Run meaningful tests after changes:
  `python -m unittest discover -s experiments/tests -v` and the official
  `python evaluation/validate_submission.py <candidate.csv>`. Follow the
  reproduction sequence in `experiments/REFINEMENT.md` in a fresh checkout;
  frozen manifests deliberately refuse repeated held-out scoring. Never use
  repeated public-window checks as an iteration loop.
- Work on a branch and open a PR to `main`. The CI workflow runs synthetic
  tests and the submission validator; a separate workflow merges passing PRs
  from branches in this fork when their tested commit is still current. See
  [docs/CI.md](docs/CI.md) for the protection setup and scope. Do not use
  organizer labels or assets in CI.
