# Promotion: original-possession diversity ensemble

On 1 November 2026, after Astra consultation, the user explicitly authorized promotion: “for original_full what is the public score? i do agree that we should promote”. The active package is `joint_gru_sn_dense_original_diversity_ensemble_v1`, version 9. It uses the exact already-sealed `original_full` pooled checkpoints from the [fixed public diagnostic](SHORT_QUALITY_PUBLIC_V1.md). No models are fitted during promotion, no organizer references are accessed, and no scoring gate is reopened.

## Decision and matched evidence

| Metric | Previous adopted calibrated dense | Promoted original diversity | PR #68 temporal H/A stack |
| --- | ---: | ---: | ---: |
| Clean spell F1 | .658625 | .662590 | .661814 |
| Feed-like spell F1 | .518527 | .528180 | .525872 |
| Clean macro F1 | .871419 | .870244 | .869859 |
| Feed-like macro F1 | .823343 | .825370 | .824854 |
| Feed D-F1 | .859048 | .860203 | .857084 |
| Feed short-spell recall, H/A <3s at IoU .5 | .198820 | .220039 | .205198 |
| Feed auxiliary 70/30 | .609972 | .617337 | .615566 |

These are the same five exposed provider-held-out matches, native grading, original synthetic corruption replicas, equal-provider aggregation and temperature-4/cost-3 decoding. Rank unrounded spell F1, macro for exact ties; the fixed 70/30 summary is auxiliary. [Original-diversity results](results/short_original_diversity_v1_aggregate.json) and the [stacking report](SMART_STACKING_V1.md) own full provider/seed diagnostics.

Original diversity gains +.003966 clean and +.009652 feed-like spell against the predecessor and passes the declared feed-first/provider guards. Both component seeds improve aggregate feed spell against the previous research leader. It also exceeds the four-recipe stack on both aggregate spell metrics, short recall and feed D-F1, while using a single two-seed recipe with existing pooled artifacts.

Provider variation remains: original diversity feed spell is .616220 Metrica, .450872 SkillCorner and .517448 Sportec. SkillCorner declines .003720 against the predecessor, within the guard. The stack does better on Metrica and SkillCorner but loses .016618 to original diversity on Sportec. Adoption is a working decision, not evidence of uniform or population-level superiority.

**The historical +.01 clean-improvement screen fails and remains failed.** The user explicitly adopts an exception based on the matched gains, passing robustness guards and simpler verified deployment package. The .55 feed target remains unmet; it was a stretch target. All frozen historical results, declarations, source snapshots, gates and prior failed decisions are preserved. Independent dense/full-match organizer confirmation remains unknown.

The [PR #67 audit](FEED_BENCHMARK_AUDIT_V1.md) adds post-hoc sensitivity evidence: original diversity beats the predecessor in all 12 realization comparisons across three gap regimes. Those views reuse exposed matches; challenge-matched rates use the old model's predicted D as a proxy and development truth to place simulated gaps. They are not independent confirmation. Phase averaging's .667556/.531576 result remains a separate post-hoc proposal, costs roughly 5× inference, and is **not** installed with this promotion. Learned stacking is also not installed.

## Known public diagnostic

| Public metric | Predecessor | Promoted recipe |
| --- | ---: | ---: |
| Spell F1 | .630303 | .611765 |
| Macro F1 | .864181 | .851742 |
| D-F1 | .873502 | .862420 |

These are already-saved results on 7,500 graded frames and 17 previously exposed reference possessions. The public spell delta is −.018538; both CSVs match 12 references at IoU .5, with 16 predecessor predictions and 17 candidate predictions. The result was known during the recommendation and authorization. It is disclosed as a pipeline diagnostic and does not select or tune the model. Promotion does not reread the reference or recompute any public metric.

## Exact deployment and replay

Two width-96 conditional H/A GRUs and fine-tuned D paths reuse the paired combined-objective teachers' frozen normalizers. The sealed fits use seeds 2026/2027, 2,048 updates, 13 original TRAIN matches and 45 GAR TRAIN games, raw-feed training views and `OriginalDiversitySampler`. Model byte hashes, teacher pairing, normalizers and frozen fit receipts are verified before staging.

Inference imports shared feature/model libraries. The baseline dispatcher selects `sn_dense_original_diversity_ensemble`; historical manifests retain their original routes. Native full H/A/D probability vectors are averaged **before** normalization, exactly as in the sealed diagnostic. The mean then enters the frozen temperature-4/cost-3 decoder. This differs slightly from the predecessor's normalize-each-member order, so the promoted path preserves the actual original-full construction.

Fresh feature-only inference reproduces the sealed candidate CSV byte for byte: **199,319 consecutive rows**, SHA-256 `7fe3da71b0cbd66bee046597611aac016fd714ddfaec6ba924dc400d7466bd1f`. The official submission-format validator passes. Its receipt records no target-label access or organizer scoring. All frame outputs remain ignored/private.

The predecessor [manifest](baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json) and [policy](baselines/model_decision_policy_joint_gru_sn_dense_calibrated_ensemble_v1.json) remain byte-identical to their original versions. The old dispatcher is archived under the private package's `previous_source/current_baseline.py`; the entire original diagnostic source set is verified against its frozen declaration before promotion. The new manifest hashes the current inference source rather than rewriting an old seal.

The [promotion receipt](results/baseline_original_full_promotion.json) binds authorization, the screen exception, model hashes, exact CSV replay, active metadata and the preserved predecessor. Re-running the promotion is idempotent if those active hashes still agree.

## Restore and delivery

The exact original diagnostic archive was restored and all 22 entries passed byte-size and SHA-256 checks. Promotion additionally packages four verified checkpoints and matching versioned metadata/source in private delivery branch `research/promote-original-full-v1`. The versioned manifests are authoritative while private main or generic defaults are older:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_original_diversity_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/current_baseline/original_full_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/original_full_restored.csv
```

Restore organizer inputs through their existing input cache if missing. A later incompatible shared-code checkout requires an isolated matching source snapshot; never overwrite a frozen study to bypass a hash check. Source-only and weight-cache manifests are separate. No organizer ground truth or raw SoccerNet shards are added by this promotion.

Verification, public/private PRs, independent remote restoration and the restricted research-site adoption update are recorded in subsequent delivery receipts. Local replay success does not by itself establish remote availability or publication.

## Completed local verification

All **575 experiment tests**, **26 integrity/gate tests**, exact file-backed fit parity, documentation links and bundled submission format pass. The first full run lacked pinned pyarrow and found that the generic active-baseline contract did not recognize the new architecture. The dependency was installed and a test-only branch now checks the new package, teacher reuse, failed-screen exception and disabled TTA. The original test is preserved privately. This test is outside the manifest-bound source set. Runtime source, weights, manifest, policy and exact replay output are unchanged by this test amendment. [Verification receipt](results/baseline_original_full_verification.json).

The source map includes the predecessor's active manifest/policy as captured before activation. Their hashes resolve through `previous_manifest` / `previous_policy`, not through the newly active metadata (which cannot hash itself). The private source snapshot preserves those predecessor bytes. For an isolated snapshot replay, explicitly pass `--manifest experiments/baselines/joint_gru_sn_dense_original_diversity_ensemble_v1.json`; do not use the snapshot's historical generic default. The versioned weight cache contains the promoted manifest/policy and receipt.

## Completed private restore and site publication

Private [delivery PR #17](https://github.com/jekkiechan/mtah-data/pull/17) is merged. An independent remote clone with a separate LFS store restores all **213 entries / 6,417,817 bytes** into an empty destination with exact sizes and hashes. Seven weight/metadata entries and 206 source/provenance entries pass; generic and versioned defaults agree, and restored manifest/policy/receipt hashes are coherent. [Remote verification](results/baseline_original_full_remote_verification.json). The source snapshot is an overlay for an isolated public checkout with the pinned dependencies, not a standalone installation.

The existing restricted [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/experiments.html#original-full-promotion) is published as **version 22**, source `532ad9ba5561bb0021e85206d194f040868ba9a1`. Overview KPIs, the eight-row adoption chart/timeline, Experiments, Architecture and Method identify original diversity as current and preserve the public decline, failed clean screen and provider tradeoffs. PR #67's audit and phase-TTA proposal are separate from installed inference. Desktop/mobile checks pass on all four affected pages, all three overview metric modes and all eight stacking view/provider combinations. Custom access revision 2 and the two existing external viewers are unchanged. [Publication receipt](results/baseline_original_full_site_publication.json).
