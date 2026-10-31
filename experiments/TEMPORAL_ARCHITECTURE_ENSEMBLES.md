# Temporal architecture probability ensembles

Status: completed and independently verified (2026-10-28).

Retain the adopted two-seed GRU. None of the eleven pooled architecture
combinations improves its clean spell F1, and none passes the declared screen.
The closest blend is GRU+TCN: clean .6262 versus .6265 for GRU, feed-like .4832
versus .4874. GRU+BiLSTM+TCN nearly matches feed-like performance (.48745 versus
.48742) but loses clean spell (.6249). Adding models has not demonstrated useful
incremental value under these equal-weight recipes. This does not rule out other
predeclared training or blending schemes on independent data.

The [declaration](results/temporal_architecture_ensembles_declaration.json)
was sealed before scoring; SHA-256 `f7ea999e27a2e08be1904ccb23c624802bcffda39f4fa01176f525d8dcfe42b8`.
Implementation snapshot: local working-branch commit `ce0a159`.
The [aggregate report](results/temporal_architecture_ensembles_development.json)
contains all 900 cells. No model was refitted or promoted.

The user requested ensembles of the completed BiLSTM, small Transformer and TCN
challengers and the GRU. The question is whether architecture diversity adds
value beyond the now-adopted two-seed GRU probability ensemble. The reference
is main commit `54f9e50`, package `joint_gru_seed_ensemble_v1`, policy version 4.
Its saved full-provider-holdout clean/feed-like spell F1 is .6265/.4874.
The earlier architecture study's .6132/.4798 GRU figures are averages of
separate seed scores, not this actual ensemble.

## Fixed comparison

Use all eleven architecture subsets of size two, three or four from GRU,
BiLSTM, Transformer and TCN. Weights are equal across architectures. Evaluate
each subset in three modes: seed 2026 only, seed 2027 only, and both seeds.
The both-seed mode first averages each architecture's two probability vectors,
then averages the architecture vectors. It therefore gives each of the 2k
members equal weight for a k-architecture subset. Never average decoded labels.

Include GRU-only controls in the same three modes. This gives 36 recipes,
25 match/view cells per recipe and exactly 900 evaluated cells. No additional
fits, weight sweep, calibration, decoder changes or additional subsets are
included. Component results are already known; this is a bounded follow-up on
reused development data, not fresh confirmation.

Every recipe averages native H/A/D probabilities before the unchanged cost-3
Viterbi decoder. All components come from the original full-provider holdouts:
the held-out provider is excluded from training and normalization. The five
matches and raw views are unchanged: Sportec J03WOY, Metrica game 3 and
SkillCorner 1996435/1996436/2006229; clean, two feed-like and two composite
replicates. Preserve native unknown labels, masks, period and time breaks.
No organizer labels or organizer predictions are used.

The primary screen compares each both-seed architecture ensemble with the
actual two-seed GRU control: clean spell F1 must improve by at least .01;
feed-like spell may fall by at most .01; and each provider's clean D-F1 and
macro-F1 may fall by at most .02. Rank eligible recipes by unrounded clean
spell F1, using macro-F1 only for exact ties. The 70/30 summary is auxiliary.
Same-seed modes describe seed sensitivity; they do not add an eligibility
rule. Composite stress remains diagnostic only. No automatic adoption follows.

Average stress replicates within match, matches within provider, then providers
equally. Report both-seed results as actual prediction ensembles, and report
same-seed outcomes separately. Include all five IoU thresholds, provider D
precision/recall, short spells, fragmentation and boundary diagnostics.

## Verification and delivery

Freeze all recipes, cell identities, source/reference/runtime hashes, the 200
member probability artifacts, 25 evaluation caches and original model provenance
before scoring. Bind exact snapshots of the current manifest, policy and PR #22
reference reports. Replay every ensemble from its declared member arrays and
verify the official scores independently. Single-seed GRU controls must reproduce
the original study; pooled GRU controls must reproduce current reference
predictions and metrics. Archived runtime probability hashes are descriptive;
new ensemble probabilities must match their frozen-input arithmetic exactly.

The runner and independent verifier live on the implementation working branch.
Checkpoints and frame arrays remain ignored local artifacts; only documentation
and aggregate evidence are candidates for main. No baseline promotion or
organizer scoring is authorized by this experiment.

## Results

The following tables summarize the completed report. These are actual
probability ensembles, not averages of separately scored models. Single-seed
modes are retained as diagnostics; only both-seed ensembles are evaluated for
eligibility against the adopted both-seed GRU. All five development matches
were previously exposed. There is no independent confirmation or claim of
statistical superiority. No organizer gate was opened.


Completed score cells: 900 across 36 fixed recipes. Ranking uses unrounded equal-provider clean spell F1; macro F1 breaks exact ties. Single-seed tables are diagnostic, and composite corruption is diagnostic-only.

### Primary pooled comparison

The reference row is the adopted pooled two-seed GRU. Candidates are ranked by unrounded equal-provider clean spell F1, with macro F1 used only to break exact ties. The 70/30 value is auxiliary; composite corruption is diagnostic.

| Rank | Candidate | Clean spell | Clean macro | Clean 70/30 | Feed-like spell | Composite spell | Clean Δ | Feed-like Δ | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRU reference | Adopted pooled GRU | 0.6265 | 0.8546 | 0.6949 | 0.4874 | 0.3198 | — | — | Reference |
| 1 | slot_dropout_gru + tcn | 0.6262 | 0.8559 | 0.6951 | 0.4832 | 0.3376 | -0.0003 | -0.0042 | Fail: clean |
| 2 | slot_dropout_gru + bilstm + tcn | 0.6249 | 0.8564 | 0.6944 | 0.4875 | 0.3323 | -0.0016 | +0.0000 | Fail: clean |
| 3 | bilstm + tcn | 0.6191 | 0.8529 | 0.6892 | 0.4831 | 0.3372 | -0.0075 | -0.0043 | Fail: clean |
| 4 | slot_dropout_gru + bilstm | 0.6178 | 0.8536 | 0.6885 | 0.4857 | 0.3164 | -0.0087 | -0.0018 | Fail: clean |
| 5 | slot_dropout_gru + bilstm + transformer + tcn | 0.6085 | 0.8571 | 0.6831 | 0.4811 | 0.3741 | -0.0181 | -0.0063 | Fail: clean |
| 6 | slot_dropout_gru + bilstm + transformer | 0.6052 | 0.8553 | 0.6802 | 0.4794 | 0.3687 | -0.0214 | -0.0080 | Fail: clean |
| 7 | slot_dropout_gru + transformer + tcn | 0.6033 | 0.8568 | 0.6793 | 0.4806 | 0.3904 | -0.0232 | -0.0068 | Fail: clean |
| 8 | slot_dropout_gru + transformer | 0.5978 | 0.8544 | 0.6748 | 0.4733 | 0.3935 | -0.0288 | -0.0141 | Fail: clean, feed-like |
| 9 | bilstm + transformer + tcn | 0.5976 | 0.8546 | 0.6747 | 0.4709 | 0.3952 | -0.0289 | -0.0165 | Fail: clean, feed-like |
| 10 | transformer + tcn | 0.5832 | 0.8517 | 0.6638 | 0.4670 | 0.4080 | -0.0433 | -0.0204 | Fail: clean, feed-like |
| 11 | bilstm + transformer | 0.5825 | 0.8503 | 0.6628 | 0.4699 | 0.3930 | -0.0440 | -0.0175 | Fail: clean, feed-like |

Eligibility requires clean spell Δ ≥ +0.0100, feed-like spell Δ ≥ -0.0100, and for every provider clean D-F1 and macro deltas ≥ -0.0200 and -0.0200, respectively.

### Same-seed diagnostics

These tables show same-seed architecture probability combinations. They are diagnostics and do not use the pooled eligibility screen.

#### Seed 2026

| Rank | Combination | Clean spell | Clean macro | Clean 70/30 | Feed-like spell | Composite spell |
| --- | --- | --- | --- | --- | --- | --- |
| GRU reference | Single-seed GRU control | 0.6088 | 0.8434 | 0.6792 | 0.4742 | 0.3180 |
| 1 | slot_dropout_gru + bilstm + tcn | 0.6226 | 0.8518 | 0.6914 | 0.4815 | 0.3392 |
| 2 | slot_dropout_gru + tcn | 0.6217 | 0.8516 | 0.6907 | 0.4787 | 0.3259 |
| 3 | bilstm + tcn | 0.6174 | 0.8514 | 0.6876 | 0.4825 | 0.3401 |
| 4 | slot_dropout_gru + bilstm | 0.6127 | 0.8472 | 0.6831 | 0.4768 | 0.3244 |
| 5 | slot_dropout_gru + bilstm + transformer + tcn | 0.6082 | 0.8533 | 0.6817 | 0.4819 | 0.3784 |
| 6 | slot_dropout_gru + transformer + tcn | 0.6032 | 0.8530 | 0.6782 | 0.4767 | 0.3869 |
| 7 | slot_dropout_gru + bilstm + transformer | 0.6011 | 0.8508 | 0.6760 | 0.4725 | 0.3793 |
| 8 | bilstm + transformer + tcn | 0.6004 | 0.8534 | 0.6763 | 0.4762 | 0.3987 |
| 9 | transformer + tcn | 0.5882 | 0.8485 | 0.6663 | 0.4718 | 0.4042 |
| 10 | bilstm + transformer | 0.5837 | 0.8450 | 0.6621 | 0.4637 | 0.3999 |
| 11 | slot_dropout_gru + transformer | 0.5818 | 0.8449 | 0.6607 | 0.4710 | 0.3982 |

#### Seed 2027

| Rank | Combination | Clean spell | Clean macro | Clean 70/30 | Feed-like spell | Composite spell |
| --- | --- | --- | --- | --- | --- | --- |
| GRU reference | Single-seed GRU control | 0.6176 | 0.8543 | 0.6886 | 0.4853 | 0.3101 |
| 1 | slot_dropout_gru + tcn | 0.6175 | 0.8578 | 0.6896 | 0.4891 | 0.3425 |
| 2 | slot_dropout_gru + bilstm + tcn | 0.6170 | 0.8548 | 0.6883 | 0.4871 | 0.3210 |
| 3 | slot_dropout_gru + bilstm | 0.6129 | 0.8519 | 0.6846 | 0.4816 | 0.2990 |
| 4 | bilstm + tcn | 0.6087 | 0.8485 | 0.6807 | 0.4748 | 0.3228 |
| 5 | slot_dropout_gru + bilstm + transformer + tcn | 0.6037 | 0.8558 | 0.6793 | 0.4782 | 0.3601 |
| 6 | slot_dropout_gru + bilstm + transformer | 0.6021 | 0.8559 | 0.6783 | 0.4743 | 0.3478 |
| 7 | slot_dropout_gru + transformer + tcn | 0.5968 | 0.8568 | 0.6748 | 0.4764 | 0.3875 |
| 8 | slot_dropout_gru + transformer | 0.5921 | 0.8540 | 0.6706 | 0.4745 | 0.3845 |
| 9 | bilstm + transformer + tcn | 0.5891 | 0.8523 | 0.6680 | 0.4719 | 0.3778 |
| 10 | bilstm + transformer | 0.5756 | 0.8488 | 0.6576 | 0.4612 | 0.3627 |
| 11 | transformer + tcn | 0.5661 | 0.8500 | 0.6513 | 0.4581 | 0.4005 |

### Primary numerical leader diagnostics

Numerical leader: `pooled_2026_2027__slot_dropout_gru_tcn`. Eligibility: Fail: clean. No pooled candidate passes every eligibility guardrail.

The closest blend improves Metrica but loses on Sportec and SkillCorner.
Higher macro-F1 does not override the primary spell-F1 ordering.

| Provider | GRU clean spell | GRU+TCN clean spell | Delta |
| --- | ---: | ---: | ---: |
| Sportec | 0.6409 | 0.6229 | -0.0180 |
| Metrica | 0.7722 | 0.7928 | +0.0206 |
| Skillcorner | 0.4665 | 0.4629 | -0.0036 |

#### Clean D and macro metrics by held-out provider

| Provider | D precision | D recall | D-F1 | Macro F1 |
| --- | --- | --- | --- | --- |
| Metrica | 0.9186 | 0.8753 | 0.8964 | 0.9189 |
| SkillCorner | 0.8754 | 0.6508 | 0.7457 | 0.7745 |
| Sportec | 0.9359 | 0.8982 | 0.9167 | 0.8742 |

#### Spell F1 at every IoU threshold

| View | IoU 0.3 | IoU 0.4 | IoU 0.5 | IoU 0.6 | IoU 0.7 | Mean spell F1 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.7186 | 0.6886 | 0.6379 | 0.5802 | 0.5056 | 0.6262 |
| feed_like | 0.5918 | 0.5506 | 0.4859 | 0.4275 | 0.3601 | 0.4832 |
| composite | 0.4553 | 0.3968 | 0.3301 | 0.2763 | 0.2294 | 0.3376 |

#### Short-spell recall and fragmentation

| View | Recall <3s | Fragmented reference spells / match |
| --- | --- | --- |
| clean | 0.3702 | 8.5556 |
| feed_like | 0.1815 | 7.3889 |
| composite | 0.0988 | 21.1667 |

#### D-boundary diagnostics

For each saved score row, the renderer independently selects the available tolerance nearest the nominal 0.5s or 1.0s target. The selected tolerance range is shown; no single global tolerance key is applied across matches.

| View | Nominal | Selected tolerance range | Precision | Recall | F1 | Mean absolute latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| clean | ~0.5s | 0.480–0.500s | 0.1812 | 0.1497 | 0.1639 | 0.206 |
| clean | ~1.0s | 1.000s | 0.3418 | 0.2803 | 0.3079 | 0.457 |
| feed_like | ~0.5s | 0.480–0.500s | 0.1757 | 0.1476 | 0.1603 | 0.238 |
| feed_like | ~1.0s | 1.000s | 0.3197 | 0.2669 | 0.2908 | 0.461 |
| composite | ~0.5s | 0.480–0.500s | 0.0344 | 0.0443 | 0.0387 | 0.247 |
| composite | ~1.0s | 1.000s | 0.0742 | 0.0961 | 0.0836 | 0.546 |

## Next action and availability

Keep `joint_gru_seed_ensemble_v1` as the working package. The evidence gap remains
an independent labeled broadcast match; another weight search on these five
exposed matches would be development tuning, not confirmation. Implementation
and model-derived frame arrays remain on the local working branch/ignored
experiment directory. Documentation and aggregate evidence are published
separately through a docs-only PR; no code merge or baseline promotion is part
of this study.

## Verification and artifact record

Independent verification passed all 900 exact member-mean probability checks,
900 cost-3 decode replays, 900 official metric and spell/boundary diagnostic
replays, 25 native views and 200 original member artifacts. It reproduced all
50 original single-GRU controls and all 15 pooled-GRU clean/feed-like reference
cells, then checked 108 summaries, eleven pooled eligibility screens and ranking.
The runner also required exact prediction/metric parity on all 45 current-main
single-seed and pooled reference rows. All 175 existing contract tests and nine
new ensemble tests passed; the independent verifier's synthetic self-tests pass.

The [verification receipt](results/temporal_architecture_ensembles_verification.json)
and [local artifact inventory](results/temporal_architecture_ensembles_artifacts.json)
bind the outputs and code. The [operational record](results/temporal_architecture_ensembles_operational_events.json)
documents verifier-only schema corrections and the interrupted audit attempt;
only the completed full replay counts as verification. No frozen runner,
member input, scored prediction or selection rule changed after scoring began.
The aggregate report is an exact copy of the ignored run report; its SHA-256 is
`3b865794190db8942c4e280da121989875eb3b530f897aeb2f0e40bba5a9e04e`.
Frame predictions and caches remain local and were not uploaded.
