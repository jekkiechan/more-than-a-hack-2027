# Frozen-model transfer to the competition tracking

This experiment runs the existing pretrained PathCRF checkpoint on the supplied automatic tracking. It does **not** read the public possession annotations, tune against them, or infer any frame-specific possession rules. The checkpoint, its original feature/decode adapter, and the external-data-trained robust model remain unchanged. The new code only converts fragmented tracking into the roster format PathCRF requires.

## Candidates declared before competition scoring

1. `frozen_robust_had`: the already selected gap-augmented gradient boosting model, applied to original tracking.
2. `pathcrf_ha`: the unchanged PathCRF player-path model and source-team mapping, after the geometry-only roster adapter below. It has no dead/alive head.
3. `pathcrf_team_robust_dead`: PathCRF H/A with D wherever the frozen robust model predicts D. This is explicitly a hybrid, not a pretrained all-in-one H/A/D model.

All 199,319 video frames receive an output. Published half boundaries identify 158,857 half frames; the remaining 40,462 frames receive X. Half boundaries are input metadata, not possession or dead-ball annotations. The macro source-head output is saved only as a diagnostic and is not a competing candidate.

Any later public-window score is solely a pipeline check under `RULES.md`. It cannot select these variants, change the tracking adapter or checkpoint, tune thresholds, or establish full-match accuracy. Improvements must be selected on external matches or through label-free input correctness checks.

## Why the original roster adapter needs a transfer layer

The challenge contains 46 tracklets, with 24 assigned to home and 22 to away. Their metadata names are `Track 1` through `Track 46`; the shirt-number fields are track indices, not true player identities. They cannot be treated as a forty-six-player roster. Between first and last observations, as many as 21 home and 16 away tracklets overlap. Visible counts occasionally exceed eleven, reaching 19 home and 15 away in the second half. The strict external-match adapter correctly rejects that representation.

The new `challenge_pathcrf.py` creates eleven approximate player trajectories per team independently in each half:

- Reserve a goalkeeper trajectory using the largest product of distance of mean X from pitch centre and observed-frame count. This uses only coordinates and observation presence, not names, jerseys, events, or labels.
- Initialize ten field-player slots at the first frame with ten observed field tracks. If there are extras, prefer tracks observed more often in that half; order slots geometrically.
- Associate observations forward and backward from that anchor using Hungarian assignment. Preserve continuously visible, geometrically consistent tracklets before associating new fragments. A tracklet can belong to only one slot.
- A motion gate uses a generic 12 m/s player-speed limit and 2 m positional tolerance. Predictions extrapolate for at most 0.5 seconds, with a 1 m identity-continuity preference. Unmatched slots have an age-independent cost, so stale trajectories cannot force reassignment of currently visible players.
- Retain missing observations, then interpolate each approximate trajectory within its half, holding the first/last observed position at the ends. This is an explicit model-input assumption, not a measurement of an occluded player.

The original PathCRF adapter then computes the same eight features, rotates periods to static home/away orientation, downsamples to 5 Hz, and runs the original edge head and dynamic CRF in 10-second windows at a 5-second stride. No source model code or weights are modified.

## Geometry checks and limitations

Synthetic checks recover fragmented identities and missing intervals and are invariant to track-column permutation. On the first sixty seconds of the external training match J03WMX, the adapter preserves every frame's player-coordinate set exactly, with no discarded or imputed observations. No possession labels were loaded for these checks. The code asserts the original adapter, checkpoint, and robust-model hashes against their pre-existing frozen manifests before inference.

On competition input, 5.5–9.4% of player-slot frames require interpolation, depending on team and half. Between 3,282 and 12,178 observations per team/half are discarded when eleven slots cannot accommodate all candidates or the motion gate rejects them. The longest missing slot interval is **145 seconds**. This is a substantial transfer limitation: a complete-roster model receives invented linear paths during those intervals, and track identity is not guaranteed to represent the same human throughout a half. Goalkeeper trajectories retain the coordinate-derived sides rather than drifting into field-player slots.

These automatic tracks differ markedly from the complete professional tracking used to train the checkpoint. A strong score on Sportec does not establish competition accuracy. The robust model's dead-ball predictions have a separate risk: its training provider freezes ball coordinates during stoppages, while the challenge has broadcast occlusion and homography noise. Neither PathCRF nor the hybrid removes that domain mismatch.

## Reproduction and artifacts

With the original PathCRF dependencies and checkpoint installed as described in [PATHCRF.md](PATHCRF.md):

```bash
python -m experiments.challenge_pathcrf
```

`--prepare-only` performs roster reconstruction and its input audits without neural inference. The script loads an explicit tracking-only allowlist from `data/challenge/inputs.npz`; it never deserializes even that file's placeholder labels.

All trajectories, per-frame predictions, and submission CSVs remain ignored under `data/experiments/challenge_pathcrf/`. `frozen_config.json` records the adapter, input, checkpoint, and robust-model hashes, fixed assignment settings, and predeclared candidates before prediction. `geometry_audit.json` records imputation and discarded-observation counts. `inference.json` records runtime, coverage, and output hashes. The three submission CSV names match the candidate names above.

The complete run produced all three 199,319-frame CSVs in **600.5 seconds** after roster preparation, using at most two PyTorch CPU threads. PathCRF used 635 windows per half, taking 287.5 and 291.8 seconds. Validation confirmed full coverage, X on exactly the 40,462 outside-half frames, and that the hybrid's D mask exactly matches the frozen robust prediction. The adapter SHA-256 remained `4813a0bfa3d5e4b4c20de8885de526c53ccf72a57d50a7eb5ecb746faf917778` throughout inference. No competition possession score was computed by this experiment.
