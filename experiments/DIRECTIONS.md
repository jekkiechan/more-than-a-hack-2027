# Research directions

The current development head is [GRU + ordered HGB](BASELINE.md), with fixed
contextual HGB team attribution and Viterbi decoding. BiGRU, TCN, ordered
boosting and fixed ensembles have been evaluated; LSTM remains untested.
Representative data and validation remain priorities. The existing 18 matches
(13 training, five development evaluation matches) support bounded comparisons,
but none is fresh confirmation. [RESEARCH_LOG.md](RESEARCH_LOG.md) preserves
the earlier context/D-blend work, whose implementation remains separately in
draft PR #8. No candidate has been submitted to the organizers.

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

### Next bounded experiment: realistic corruption training

Compare the unchanged GRU/HGB head with retrained copies using the same model
capacity and decoding. Existing training already includes synthetic ball gaps
for H/A and fixed player-slot dropout for D; the next experiment should test
whether more realistic corruption improves transfer rather than merely adding
another random mask.

- Estimate ball/player gap durations, co-occurring visibility losses and track
  lengths from unlabeled organizer tracking. Keep camera-error severity within
  plausible ranges; motion jumps alone do not identify true measurement error.
- Add variable-duration ball gaps, temporary player/region dropout, correlated
  coordinate drift and occasional reacquisition jumps. Test track fragmentation
  separately where it changes identity-dependent features.
- Corrupt raw observations before recomputing motion and contextual features;
  propagate observation masks and preserve half/time boundaries and targets.
  Mix clean and corrupted training examples. Never derive masks from possession
  labels or copy source-specific missingness-to-D rules.
- Predeclare a small number of corruption families and severity ranges. Compare
  clean external inputs and independently seeded corrupted views on identical
  whole-match/provider splits, with multiple training seeds. Hold out providers
  from the full trainable pipeline in new transfer evaluations, not only D.
- Rank by official spell F1, macro F1 for exact ties; show D precision/recall,
  boundary timing, short-spell recall, per-source spread and the auxiliary
  70/30 score. Seek a fresh labeled broadcast match before claiming transfer.
- Do not use the public organizer labels to choose corruption settings or
  repeat the completed public checks. No improvement from this proposed recipe
  has been measured, and it is not part of the current packaged head.

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
  protocol before claiming a replacement baseline. The current temporal folds
  hold out providers from D training only; fixed H/A attribution remains pooled.

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

## 4. Modelling: completed temporal comparisons and remaining questions

The completed contextual classifier improves aggregate external spell F1 from
0.5259 to 0.5512 against its matched baseline. The best D-blended alternative
scores 0.5455 spell / 0.8398 macro versus 0.8147 macro for context, and improves
SkillCorner D recognition. It also changes H/A attribution, so it does not
isolate the D contribution. See [RESEARCH_LOG.md](RESEARCH_LOG.md) for limits.

The later [temporal comparison](LEARNED_TEMPORAL.md) held pooled conditional
H/A and decoding fixed while comparing ordered player features, TCN, BiGRU and
boosting D heads. [Fixed ensembles](TEMPORAL_ENSEMBLE.md) improved external
D-provider-holdout spell/macro from .5652/.8363 for ordered HGB to .5746/.8449
for GRU/HGB. The three-way GRU/HGB/LightGBM blend is only slightly higher.
The user adopted the simpler pair as the current development head.

The [PathCRF retention test](PATHCRF_ENSEMBLE.md) holds GRU/HGB D fixed on
Metrica/SkillCorner: HGB team attribution scores .5744/.8346, the fixed team
blend .5414/.8213, and PathCRF teams .4993/.8090. Keep PathCRF archived rather
than in the default pipeline. The historical organizer hybrid's higher public
spell point estimate does not select a new integration.

Next, prioritize corruption training and fresh broadcast validation above.
LSTM, decoder changes and soft duration priors remain separate future comparisons.

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
