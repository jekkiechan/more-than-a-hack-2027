# Label-free target adaptation (target_adaptation_v1)

Question: do label-free adaptations to the target feed's own inputs improve
full-provider-holdout spell F1 under `feed_like` stress without hurting clean?
Answer: no. Neither arm meets the pre-declared rule; no competition CSV was made.

Runner `experiments/target_adaptation.py`; aggregates `results/target_adaptation.json`;
declaration (sealed before A1/A2 scoring) and all private outputs under
`data/experiments/target_adaptation_v1/` (git-ignored). Organizer labels never read.

## Where normalization lives
Each checkpoint stores `median` and `scale` (204 each; train-only nanmedian/nanstd of
raw features), applied by `learned_temporal.normalize` as `clip((x-median)/scale,-10,10)`
plus 204 finite-value mask channels. The network (BiGRU 24 + Linear) has no
normalization layers, so replacing `median`/`scale` is exactly the
standardize-then-restore transform.

## Arms
- A0 `none`: unchanged checkpoints, two-seed probability mean, Viterbi cost 3.
- A1 `target_norm`: `median`/`scale` from the target cell's own (corrupted) features,
  nanmedian/nanstd over finite entries, pooled with the home/away-swapped copy as
  training pooled them. Median (not mean) because that is the checkpoint's statistic.
  Features whose finite target values are all in {0,1} keep checkpoint stats; scale
  floored at 0.1 x train scale; mask channels untouched.
- A2 `self_train`: per seed, Adam lr 1e-4, 2 epochs, batch 64, grad-clip 1, torch seed =
  checkpoint seed. Target windows (10 s core / 5 s context, original + swapped) keep
  only frames where the A0 ensemble max probability >= 0.90 (hard argmax pseudo-label,
  plain factorized NLL). Each epoch adds an equal number of randomly drawn labelled
  windows from the fold's own TRAIN matches (their cached gap-alternate choice, plain
  NLL, not the combined objective). Checkpoint normalization kept. Seeds fine-tuned
  separately, probabilities averaged, then decoded.

## Parity
A0 reproduces saved `objective_ensemble_rows.json` bit-for-bit (max abs diff 0.0, equal
probability SHA, equal spell counts) on J03WOY clean and feed_like corrupt_1; all 15
A0 cells give equal-provider .6332 clean / .4988 feed-like as saved. Cached
seed_gap_context feed_like features equal freshly corrupted ones (checked on J03WOY corrupt_1).

## Results (equal-provider, two-seed ensemble; replicates -> match -> provider -> equal)
| arm | clean spell | feed-like spell | clean macro | feed-like macro | clean D-F1 | feed-like D-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 none | .6332 | .4988 | .8542 | .8114 | .8491 | .8446 |
| A1 target_norm | .5388 | .4207 | .7690 | .7273 | .7469 | .7383 |
| A2 self_train | .6283 | .4960 | .8527 | .8125 | .8473 | .8475 |

Decision (feed-like >= +.01, clean >= -.005, every provider >= -.02): A1 fails (-.078 /
-.094, worst provider -.101); A2 fails (-.003 / -.0049, worst provider -.017 = Metrica
clean). Per-provider, IoUs, spell counts and per-seed numbers are in the JSON.
A target-statistic shift is not benign for this network: several features move by
>1 sigma_train between providers and a clipped 0.1 sigma floor can inflate values.
