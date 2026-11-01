# Promotion: dense ensemble with D-logit temperature 4

Promoted 1 November 2026 by the user's explicit request, “i think we should promote this?”. The active package is `joint_gru_sn_dense_calibrated_ensemble_v1`. The baseline CLI now selects frozen `feed_spell_push_v1` arm `f062` by default. The same four checkpoints, features, normalization, two seeds and native interpolation are reused; only the postprocessing changes.

Normalize each native member probability array, average the two equally, divide D log odds by 4 while preserving conditional H/A, then use the original cost-3 decoder. The inference module calls the exact selected frozen decoder; it does not reopen a study runner or scoring gate.

## Saved evidence supporting adoption

| Metric | Previous dense package | Promoted package |
| --- | ---: | ---: |
| Clean spell F1 | 0.647215 | 0.658625 |
| Synthetic feed-like spell F1 | 0.505811 | 0.518527 |
| Clean macro F1 | 0.857677 | 0.871419 |
| Feed-like macro F1 | 0.815535 | 0.823343 |
| Public spell F1 | 0.630303 | 0.630303 |
| Public macro F1 | 0.851387 | 0.864181 |
| Public D-F1 | 0.843424 | 0.873502 |

Both component seeds and every provider improve external spell F1 in both views. The clean +0.01 improvement screen and all declared provider guards pass. Metrica feed-like macro and D-F1 decline slightly, as preserved in the original report. The [five-round study](FEED_SPELL_PUSH.md) owns provider/seed details, short-spell diagnostics, the 522 configurations and separate public diagnostic. Promotion reads those saved aggregates; it performs no fitting, candidate search, metric calculation or organizer reference access.

The user knew the public diagnostic before authorizing promotion. That diagnostic was computed after external selection, and public labels were not used for model or parameter selection. The public spell tie covers only 17 reference possessions. Full-match organizer performance and independent dense confirmation remain unknown. The five external matches are repeatedly exposed development data and feed-like defects are synthetic.

## Versioning and replay

The predecessor [manifest](baselines/joint_gru_sn_dense_ensemble_v1.json) and [decision policy](baselines/model_decision_policy_joint_gru_sn_dense_ensemble_v1.json) are preserved byte for byte. Its checkpoint hashes are exactly the promoted package's hashes; no weights are refitted or replaced. The historical failed screen on the earlier dense-training promotion remains unchanged. The new policy's passed clean screen describes this decoder promotion, not that historical training study.

The new architecture kind is `sn_dense_calibrated_ensemble`, with fixed `postprocessing` metadata. The shared baseline entrypoint dispatches this kind through the frozen f062 path and keeps the original dispatch for older manifests. Historical studies that sealed the old shared entrypoint and active metadata must use their original source/metadata (pre-promotion repository commit `466ca1e`); their hashes and seals have not been rewritten. A copy of the old entrypoint is retained in the private package's `previous_source/` directory.

Fresh feature-only inference through `experiments.current_baseline predict --manifest <staged manifest>` produces all **199,319 consecutive rows**. The official format validator passes. Its SHA-256 is exactly the sealed f062 CSV hash:

`7d0e55b84e6080c5ffc5a9d47868e7ff30b397352102ed52f7d179486604b8e5`

The replay accesses neither organizer reference labels nor public scoring. All earlier gates remain closed. The new default output is `data/experiments/current_baseline/joint_gru_sn_dense_calibrated_ensemble.csv`, so it preserves the predecessor CSV. [Promotion receipt](results/baseline_calibrated_dense_promotion.json) binds the manifest, policy, checkpoint hashes, saved evidence and validation receipt.

## Restore and default inference

Use the private delivery branch `research/calibrated-dense-baseline-v1` until its delivery is merged. The versioned package and source-cache manifests identify this package even when generic private defaults are older:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_calibrated_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/current_baseline/calibrated_dense_replay.csv
python evaluation/validate_submission.py data/experiments/current_baseline/calibrated_dense_replay.csv
```

The cache includes the four checkpoints and matching metadata, not organizer reference labels. Restore raw organizer inputs using their existing authorized competition cache if missing. The matching public code is required; the private source snapshot supports an isolated old-version restore if current shared libraries later change. Never overlay a frozen study just to bypass a source-hash check.

## Delivery checks

Focused inference and promotion contracts pass: frozen decoder parity, package validation, predecessor dispatch, current-CLI routing and inactive-period masking, policy coherence and immutable archives. All **525 experiment tests**, **26 integrity/gate tests**, the file-backed fit contract, documentation links and CSV format validation pass. Private remote restore is recorded separately in delivery receipts when complete. Research-site updates change adoption status and pipeline descriptions while retaining the original five-round evidence.
