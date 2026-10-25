# Research directions

Representative data and validation remain priorities. The existing 18 matches
(13 training, five validation) already support bounded feature/model comparisons
while better data is sought. [RESEARCH_LOG.md](RESEARCH_LOG.md) records the
completed context/D-blend experiment and the proposed follow-ups. Implementation
stays on `possession/robust-pipeline`; this plan does not promote a new submission
candidate. TCN/GRU/LSTM experiments have not yet been run.

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
- The completed provider-held-out context comparison uses reused development
  matches and one seed. Its small ranking gap needs fresh confirmation. Compare
  the previous submission pipeline and new candidates under the same external
  protocol before claiming a replacement baseline.

## 3. Better preprocessing

Tactical ATD tracking and metadata are restored locally. The additional broadcast,
Smart Tagging and homography assets are still needed for the corresponding work.

All label-free, so they are allowed on the challenge match:
- **Second ball source:** align the broadcast feed's tracking to the tactical
  timeline by matching player positions in pitch coordinates, then use its
  ball where the tactical ball is missing (broadcast sees the ball on 50.5% of
  its frames, on a different set of moments).
- **Smart Tagging as evidence:** SET PIECES intervals provide soft restart
  context; they are not exact dead-onset/restart boundaries. Team tags can
  provide a weak H/A prior. Do not turn tagging windows directly into D labels.
- **Homography quality:** use `reprojection_error` and `valid` as observation
  confidence.
- **Unused ATD fields:** player speed and the ball's third value.
- **Tracklet stitching** for all models, not only the PathCRF adapter.

## 4. Modelling: isolate D, then test learned temporal context

The completed contextual classifier improves aggregate external spell F1 from
0.5259 to 0.5512 against its matched baseline. The best D-blended alternative
scores 0.5455 spell / 0.8398 macro versus 0.8147 macro for context, and improves
SkillCorner D recognition. It also changes H/A attribution, so it does not
isolate the D contribution. See [RESEARCH_LOG.md](RESEARCH_LOG.md) for limits.

Proposed comparisons, not yet run:

- Keep contextual conditional H/A and decoder settings fixed; vary only the
  blend of contextual and player-only D probabilities on a predeclared grid.
- Add before/after activity differences, deceleration, restart acceleration and
  duration to the existing temporal summaries.
- Compare those features with a small TCN and bidirectional GRU/LSTM D head.
  Full-match offline inputs allow both past and future tracking. Use observation
  masks, windows that respect time breaks and whole-match/provider splits.
- Measure official spell F1 alongside D precision/recall and boundary timing.
  Decoder changes and soft duration priors should be separate comparisons.

### Later: LightGBM hyperparameter tuning

`REFINE_LGBM.md` shows that tuned LightGBM beats the HGB baseline (+0.017 clean,
+0.013 masked spell F1 on J03WOY), while default LightGBM does not. So tuning,
not the library, carries the gain, and a real search should find more. It
should follow the targeted D/temporal comparisons and stronger transfer evidence:

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
