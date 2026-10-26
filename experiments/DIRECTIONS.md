# Research directions

The current development head is [GRU + ordered HGB](BASELINE.md), with fixed
contextual HGB team attribution and Viterbi decoding. BiGRU, TCN, ordered
boosting and fixed ensembles have been evaluated; LSTM remains untested.
Representative data and validation remain priorities. The existing 18 matches
(13 training, five development evaluation matches) support bounded comparisons,
but none is fresh confirmation. [RESEARCH_LOG.md](RESEARCH_LOG.md) preserves
the earlier context/D-blend work, whose implementation remains separately in
draft PR #8. No candidate has been submitted to the organizers.

Current priority order, from the 2026-10-26 review below: (1) a development
view that reproduces the challenge's real defects, so preprocessing can be
measured at all; (2) recovering ball positions during tactical ball gaps,
starting with broadcast-feed fusion; (3) gap-level and de-duplicated tracking
features; (4) H/A boundary timing in decoding. Model-family and hyperparameter
work comes after these.

## Why preprocessing has not moved the scores (2026-10-26 review)

This review reads only existing code and recorded aggregate results; no new
fitting or scoring was run. The current package applies **no** input
preprocessing at inference: `current_baseline.py` passes raw tracking to the
feature extractors. The observation repair in draft PR #8 (isolated spike
removal, player gaps up to 0.2 s) scored .5251 versus .5259 spell F1 for its
matched baseline, and was never merged. Corruption training traded clean for
stress performance. Five reasons explain why neither shows a gain:

1. **Development data lacks the defects being repaired.** Sportec has complete
   ball XY (frozen during D), Metrica blanks the ball almost only during D, and
   SkillCorner's gaps come from broadcast cuts. None has the challenge's
   live-play ball loss, 46 fragmenting tracklets, or up to 19 visible tracks
   per team. Repairing clean inputs gives a near-zero delta by construction.
2. **Degraded views do not exercise the repairs.** They come from the same
   synthetic families used in training augmentation (long gaps, slot swaps),
   while the repairs target isolated spikes and subsecond gaps.
3. **The repairs work at the wrong scale.** Organizer ball gaps have a median
   of 10.7 s and cover about 41% of graded frames; the repairs bridge at most
   0.5 s (ball) and 0.2 s (players). HGB already handles NaN natively and the
   feature code already rejects >13 m/s player and >65 m/s ball speeds. During
   long gaps every ball-relative feature is NaN and H/A rests on player motion
   and contact memory.
4. **Synthetic gaps are placed independently of game state.** Real occlusion
   likely concentrates on aerial balls, duels and crowded areas, i.e. around
   turnovers. A renewal-process mask teaches "gaps are random", which is why
   corruption training improves only its own stress family.
5. **The public window cannot resolve the effect.** The .4889 to .5257 public
   change is one fewer spurious possession (8 to 7) out of 17 reference
   possessions; the 90% spell interval is [.3657, .6743]. Changes below about
   0.15 spell F1 are not detectable there, and it must not be used for tuning.

The recorded public diagnostics also locate the remaining error: alive/dead F1
is .968 and reference-live team accuracy 86–87%, but only 2 of 20 H/A/D
transitions fall within ±0.5 s (5 of 20 within ±1 s). Externally, 98% of wrong
frames are H/A swaps and oracle-perfect H/A gives .996 spell F1 on J03WOY
([HEADROOM.md](HEADROOM.md)). The bottleneck is team attribution during ball
gaps and turnover timing, not dead-ball detection or subsecond repair.

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
  frozen ball) before use, as in `OPEN_DATA.md`. **Status (2026-10-26):** access
  was requested by email; no response yet. Do not plan around it before the
  November 17 deadline; the challenge-like view below is the fallback.
- **Challenge-like development view (highest priority).** Sportec and Metrica
  have near-complete player tracks; the challenge has 46 fragmenting tracklets
  with identity switches, and the ball missing mostly during play. Without a
  view that reproduces these, preprocessing cannot be measured (see the review
  above). Build it from unlabeled organizer tracking only:
  - Fit a ball-missingness model on observable state, not labels: ball speed
    and direction before the gap, pitch zone and far-side position, player
    density near the last ball position, and gap duration. Apply it to live play
    in Metrica and Sportec so gaps fall where the challenge loses the ball.
  - Reproduce tracklet length distributions, duplicate detections (same-team
    tracks within about 1 m) and objects per frame, then calibrate against the
    organizer's aggregate statistics.
  - Score each preprocessing change on the quantity it targets as well as the
    official metric: H/A accuracy inside ball gaps, transition F1 at ±0.5 s
    and ±1 s, and short-spell recall. Use at least three seeds.
- **SoccerTrack v2** ([arXiv 2508.01802](https://arxiv.org/abs/2508.01802)):
  10 full panoramic 4K university matches with pitch coordinates and
  ball-action labels. Possession and dead-ball labels would have to be derived
  from the action spotting; worth a look for the panoramic camera.

### Corruption training: completed, and the next step

The [composite](REALISTIC_CORRUPTION.md) and [25% targeted](LIGHT_CORRUPTION.md)
batches are complete; the ball-gap team arm is the current package. Both place
gaps by a renewal process that is independent of game state, so their stress
gains are measured on the same synthetic family. The next corruption
experiment should reuse the state-conditioned missingness model above for
training, and evaluate on the challenge-like view rather than on another
draw from the training family.

- Corrupt raw observations before recomputing motion and contextual features;
  propagate observation masks and preserve half/time boundaries and targets.
  Condition masks on observable tracking state only. Never derive masks from
  possession labels or copy source-specific missingness-to-D rules.
- Hold out providers from the full trainable pipeline in new transfer
  evaluations, not only D. Rank by official spell F1, macro F1 for exact ties;
  show D precision/recall, boundary timing, short-spell recall, per-source
  spread and the auxiliary 70/30 score.
- Do not use the public organizer labels to choose corruption settings or
  repeat the completed public checks.

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

All label-free, so they are allowed on the challenge match. Ordered by expected
value; each needs the challenge-like view in section 1 to be measured.

1. **Second ball source.** Align the broadcast feed's tracking to the tactical
   timeline by matching player positions in pitch coordinates, then use its
   ball where the tactical ball is missing (broadcast sees the ball on 50.5% of
   its frames, on a different set of moments). First measure, without labels,
   what share of tactical ball-missing graded frames receive a broadcast ball
   after alignment. Ball absence removes every ball-relative feature, so this
   is the largest likely lever if coverage is material.
2. **Gap-level features instead of interpolation.** For each ball gap: start
   and end positions, implied mean ball speed across the gap (high values
   indicate a pass or aerial ball), nearest team and distance at each end, gap
   duration and position within the gap. Current features carry only last/next
   contact; short interpolation adds little.
3. **De-duplication and tracklet stitching for all models**, not only the
   PathCRF adapter. Merge same-team detections within about 1 m before
   counting. Duplicates inflate within-radius counts and support fractions,
   and slot swaps turn derived speeds into NaN.
4. **Unused ATD fields:** player speed and the ball's third value. If the third
   value is height, it flags aerial balls, which should not change possession
   and may explain many gaps.
5. **Smart Tagging as evidence:** 602 of 638 phases carry a team tag, a strong
   H/A prior during gaps; SET PIECES intervals provide soft restart context.
   They are not exact boundaries; do not turn tagging windows directly into
   labels. External sources have no Smart Tagging, so its weight cannot be
   validated: use a fixed, predeclared prior or simulate noisy phase tags from
   external event data.
6. **Homography quality:** use `reprojection_error` and `valid` as observation
   confidence. Lower priority: the tactical homography is 100% usable.

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

Next, prioritize the challenge-like view and ball recovery above. LSTM,
decoder changes and soft duration priors remain separate future comparisons.

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

- Boundary refinement: after Viterbi, move each H↔A boundary to the frame of
  strongest contact-switch evidence within ±1 s. Only 2 of 20 public-window
  transitions fall within ±0.5 s, and spell IoU at 0.6–0.7 is sensitive to
  boundary error on short spells. Fit the window on external data only.
- Decoder: spell-length priors fitted on the external labels (Metrica: median
  live spell ~5 s, about 30% under 3 s), and team-specific D→H/A restart costs.
- Abstention (X) during long ball gaps has no value for spell F1, which fills
  X with the last label (EVALUATION.md). Leave it out unless macro F1 needs it.
