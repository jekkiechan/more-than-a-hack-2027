# Research directions

The current implementation pass uses the existing 18 external matches (13 train,
5 validation) to test preprocessing, control-context features, probability
ensembling and separate team/dead decoding costs. See [PIPELINE.md](PIPELINE.md)
and `pipeline_eval.py` for the bounded leave-one-provider-out protocol. It does
not depend on acquiring another dataset; additional broadcast data remains useful
for independent confirmation and subsequent work.

Ordered by expected payoff. Most of the gain is expected to come from more and
better data, better validation and better preprocessing. Modelling waits until
those are in place, so its gains can be measured on data that resembles the
challenge.

## 1. More data, closer to the challenge (in progress)

Done ([OPEN_DATA.md](OPEN_DATA.md)):
- Metrica's three sample games: same provider, 25 Hz, one in FIFA EPTS format.
- SkillCorner widened to 11 matches with D labels.

Next:
- **PFF FC / Gradient Sports FIFA World Cup 2022.** 64 matches of
  broadcast-derived tracking with event data. Access is through a free sign-up
  form on the [Gradient Sports blog](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset),
  which a team member has to request. At 6× the current labelled match count,
  it is the largest available jump. Its game/possession events should give H/A/D
  the way Metrica's and SkillCorner's do. Check its D shortcut (camera cuts,
  frozen ball) before use, as in `OPEN_DATA.md`.
- **Challenge-like degradation of clean sources.** Sportec and Metrica have
  near-complete player tracks; the challenge has 46 fragmenting tracklets with
  identity switches, and the ball missing mostly during play. Simulate tracklet
  breaks, slot swaps and in-play ball loss, then calibrate them to aggregates
  measurable on the challenge tracking without labels: tracklet lengths,
  objects per frame and ball gap lengths.
- **SoccerTrack v2** ([arXiv 2508.01802](https://arxiv.org/abs/2508.01802)):
  10 full panoramic 4K university matches with pitch coordinates and
  ball-action labels. Possession and dead-ball labels would have to be derived
  from the action spotting; worth a look for the panoramic camera.

## 2. Better validation

- Select on the multi-source suite (`multisource_eval.py`: Sportec J03WOY,
  Metrica game 3, 3 SkillCorner matches), weighted towards sources unseen in
  training. Do not select on J03WOY alone: its frozen ball makes D nearly
  free, and the model it prefers collapses on other providers
  (`OPEN_DATA.md`).
- Report per-source spread, and seed variance for any stochastic model.
- Keep the public five minutes as a pipeline check only (RULES.md).

## 3. Challenge-specific preprocessing (needs the assets in the environment)

All label-free, so they are allowed on the challenge match:
- **Second ball source:** align the broadcast feed's tracking to the tactical
  timeline by matching player positions in pitch coordinates, then use its
  ball where the tactical ball is missing (broadcast sees the ball on 50.5% of
  its frames, on a different set of moments).
- **Smart Tagging as evidence:** SET PIECES intervals mark restarts, which end
  a D stretch; Team tags give a weak H/A prior.
- **Homography quality:** use `reprojection_error` and `valid` as observation
  confidence.
- **Unused ATD fields:** player speed and the ball's third value.
- **Tracklet stitching** for all models, not only the PathCRF adapter.

## 4. Modelling: LightGBM hyperparameter tuning (after 1–2)

`REFINE_LGBM.md` shows that tuned LightGBM beats the HGB baseline (+0.017 clean,
+0.013 masked spell F1 on J03WOY), while default LightGBM does not. So tuning,
not the library, carries the gain, and a real search should find more. It
waits for more data, so it optimises against the right target:

- **Objective:** mean spell F1 on the multi-source suite, not J03WOY. Evaluate
  the decoder switch cost jointly with the booster settings.
- **Search:** a fixed-budget Bayesian search (e.g. Optuna TPE) over
  num_leaves, min_child_samples, learning rate × n_estimators, subsample,
  colsample_bytree, reg_lambda and reg_alpha. Also search the short-spell
  weight beta and the share of each data source.
- **Guarding against overfitting the validation set:** re-fit the top few
  configs with 3 seeds, choose by the seed mean, and confirm once on a frozen
  held-out set never used during the search.
- **Compute:** about 10 minutes per fit on 4 cores at the current size. With
  PFF data, cache features once per source and sample rows per trial.

## 5. Post-processing

- Decoder: spell-length priors fitted on the external labels (Metrica: median
  live spell ~5 s, about 30% under 3 s), and team-specific D→H/A restart costs.
- Abstention (X) during long ball gaps has no value for spell F1, which fills
  X with the last label (EVALUATION.md). Leave it out unless macro F1 needs it.
