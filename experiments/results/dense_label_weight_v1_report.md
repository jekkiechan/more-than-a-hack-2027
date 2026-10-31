# Dense silver label weighting: targeted follow-up

## Question and design

This study tested whether reducing dense SoccerNet silver-label weight near uncertain touch and inferred transitions improves full-provider transfer. The finite comparison was the reconstructed D2 control and a single 0.25-weight arm, each fit for 2,048 updates on three leave-one-provider-out folds and seeds 2026/2027. Each fit mixed 16 original windows and 16 SoccerNet windows per batch. Only TRAIN SoccerNet labels and provenance produced weights; no GAR VALID/TEST labels or organizer labels were read.

The ambiguity mask marked native frames within ±round(0.5 × native FPS) of a same-period known-label transition when either side had touch, propagated, or held-uncontrolled provenance. Unknown labels remained masked. All frames outside these bands retained weight 1.0. Candidate H/A and D losses use a per-window weighted mean over valid labels. Transition edge weights use the same mask inside the existing stable/change group-balanced loss; that normalization can cancel a uniform scale within a group, so the transition term reweights edge locations rather than reducing its group-level mass.

## Control replay

The reconstructed control gate passed. Expected clean/feed-like equal-provider spell F1 was 0.647215/0.505811; observed was 0.647215/0.505811. The declared tolerance was 0.000051. Candidate fits were permitted only after this receipt passed.

## External development results

Official metrics are paired over the same five previously exposed selection matches. For each provider/match/view, the two seeds' native-frame probabilities are averaged first and the averaged probabilities are decoded once. Match scores are then averaged within provider, followed by an equal-provider mean. Candidate and control use the exact restored fold teachers and the declared D2 training recipe; original trained D2 fold checkpoints were unavailable.

| Arm | View | Spell F1 | Macro F1 | D-F1 | Short-spell recall | Δ spell vs control |
|---|---|---:|---:|---:|---:|---:|
| control_D2 | clean | 0.647215 | 0.857677 | 0.850177 | 0.402475 | +0.000000 |
| control_D2 | feed_like | 0.505811 | 0.815535 | 0.847721 | 0.217097 | +0.000000 |
| weight_025 | clean | 0.647915 | 0.858315 | 0.850939 | 0.402264 | +0.000701 |
| weight_025 | feed_like | 0.505227 | 0.815450 | 0.847699 | 0.214699 | -0.000583 |

### Fixed decision screens

The fixed upgrade screen requires at least +0.01 clean equal-provider ensemble spell F1, no worse than -0.01 feed-like spell F1, and per-provider clean macro-F1 and D-F1 deltas of at least -0.02. The clean delta was +0.000701, so the candidate is a **no-go** on the primary screen. The feed-like delta was -0.000583 and every provider's clean macro-F1 and D-F1 guard passed.

### Per-provider spell F1

| Arm | View | Metrica | SkillCorner | Sportec |
|---|---|---:|---:|---:|
| control_D2 | clean | 0.804959 | 0.486934 | 0.649751 |
| control_D2 | feed_like | 0.603767 | 0.441088 | 0.472577 |
| weight_025 | clean | 0.804959 | 0.490152 | 0.648635 |
| weight_025 | feed_like | 0.600139 | 0.442437 | 0.473106 |

### Provider guard deltas

| Provider | Clean macro-F1 Δ | Clean D-F1 Δ | Both guards |
|---|---:|---:|---|
| Metrica | -0.000036 | +0.000032 | Pass |
| SkillCorner | +0.001530 | +0.001831 | Pass |
| Sportec | +0.000421 | +0.000423 | Pass |

### Seed spread and short spells

| View | Arm | Seed 2026 equal-provider spell F1 | Seed 2027 | Seed range |
|---|---|---:|---:|---:|
| clean | control_D2 | 0.635521 | 0.637245 | 0.001724 |
| clean | weight_025 | 0.636885 | 0.637515 | 0.000630 |
| feed_like | control_D2 | 0.501847 | 0.495759 | 0.006088 |
| feed_like | weight_025 | 0.502248 | 0.496456 | 0.005792 |

Equal-provider short-spell recall for ground-truth alive spells shorter than 3 seconds was 0.402475 for control and 0.402264 for weight_025 on clean inputs (Δ -0.000211), and 0.217097 versus 0.214699 under feed-like corruption (Δ -0.002398). The seed values are reported separately in the [decision receipt](dense_label_weight_v1_decision.json); the ensemble result comes from averaging native probabilities before decoding and is not the mean of the two seed-level F1 scores. A supplementary [member-wise provider guard diagnostic](dense_label_weight_v1_member_provider_guard_diagnostic.json) confirms that all 18 provider × seed × clean-metric checks exceed the fixed -0.02 macro-F1 and D-F1 floors; it does not change the predeclared screen or score artifacts.

### Weight coverage

Across 11249 dense records and 5,063,339 native TRAIN frames, 9,705 transitions qualified. The mask covered 297,140 native frames (5.87%); 48,825 known sampled frames (6.59%) received reduced weight, including 38,869 H/A and 9,956 D targets.

The experiment used a reconstructed TRAIN record set from the pinned raw revision and frozen preparer sources. All 45 regenerated game joblib hashes differ from the historical PR47 values; the original manifest hash is recovered exactly when those 45 historical hashes are substituted into the otherwise identical reconstructed manifest. The reconstructed records match all 273 shared TRAIN QA values, including labels, provenance, anchors, touch/tackle and spell aggregates. Exact feature-array byte identity to unavailable historical game objects is not established. See the [preparation reconstruction receipt](dense_label_weight_v1_preparation_reconstruction.json) for per-game hashes, raw restore hashes, environment versions and this limitation.

## Interpretation and limits

This is a single predeclared weighting value on previously used development matches, with no independent broadcast gold set. It supports a decision about this specific loss reweighting under the declared full-provider protocol; it does not establish whether SoccerNet silver labels are reliable in general. The candidate missed the fixed clean upgrade screen despite passing feed and provider guards, so it is a no-go. No model promotion occurred.

Declaration SHA-256: `4db01fc554ee8628445c3f82371cb3af1a14426339710c5c731194ff93a551f3`. Control replay receipt: `experiments/results/dense_label_weight_v1_control_replay.json`. Paired score readout: `experiments/results/dense_label_weight_v1_readout.json`. Fixed-policy checks and artifact hashes: [decision receipt](dense_label_weight_v1_decision.json).
