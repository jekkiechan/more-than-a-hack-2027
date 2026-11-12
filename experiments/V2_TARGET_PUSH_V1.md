# v2_target_push_v1: pushing v2 spell F1 above 0.60 from the snap_m0.0 baseline

Status: development study on exposed matches; no candidate adopted. Aggregate evidence is in `experiments/results/v2_target_push_v1/`; frame-level predictions, caches and fitted models stay in ignored `data/experiments/v2_target_push_v1/`.

## Question and setup
Can v2 spell F1 exceed 0.60 (control `snap_m0.0`: clean 0.681508, v1 0.534731, v2 0.598690) without regressing clean spell, v1 spell, macro, D-F1 or short-spell recall (repo retention guards, `ball_exit_dmask_v1.guards`)? Cohort: Metrica metrica_3, SkillCorner 1996435/1996436/2006229, Sportec J03WOY x clean + feed_like_v1 (2) + feed_like_v2 (4) = 35 cells; realizations within match, matches within provider, providers equally. No organizer/public labels, no GAR. The frontier control replays exactly (0.6701365499557103/0.532128306931943/0.5936187198896373) and snap_m0.0 to 6 decimals (macro .876204/.827310/.854915). Harness: `experiments/v2_target_push_harness.py` (full 35-cell evaluation about 4 s). Code: `experiments/v2_target_push_{harness,diagnostics,candidates,refiner,refiner_round2}.py`.

## Phase-1 diagnostics (vs snap_m0.0)
- Oracle ladder, v2 spell: oracle D mask 0.685; oracle D starts 0.611, ends 0.611, both 0.622; remove false D runs 0.608; add missed 0.616; both 0.625; oracle H/A inside predicted live spans 0.853. H/A errors, not D, dominate the ceiling.
- Clean to v2 drop (0.0828): Shapley D 0.017 (21%), H/A 0.066 (79%).
- Unmatched v2 reference spells: wrong team 43%, merged with neighbour 32%, covered by D 21%; boundary misalignment about 1%. False positives: 45% sit on truly dead frames, mostly SkillCorner spells under 3 s.
- D-run edge timing (pred minus true, v2, snap mask): SkillCorner starts +1.3 s late and ends 1.7 s early; Metrica ends +0.8 s late; Sportec starts -0.5 s early.
- Snap delay edits: 31% good on SkillCorner vs 97-100% on Sportec/Metrica; best label-free descriptor (ball speed before the start) AUC about 0.71 inverted on SkillCorner; an oracle gate adds only +0.0016 v2.
- The CE and predecessor P(D) are bitwise identical (only calibrated differs). `decoded__predecessor` is the f062 decode (dead-logit temperature 4, H/A temperature 1, 3-state Viterbi, switch cost 3) of the pooled predecessor probabilities, on all 35 cells; it is not a threshold.

## Families A-C (all under snap_m0.0; none eligible)
### A: re-decode D (dt x dead switch cost; P(D) source). H/A switch cost stays 3; P(D) blends keep the predecessor conditional H/A.
| arm | spell clean/v1/v2 | Δ v2 | Δ v2 Met/SC/Spo | short recall c/v1/v2 | failed guards |
|---|---|---:|---|---|---|
| A_dt3_c2 | 0.6838/0.5350/0.5977 | -0.0010 | +0.0024/-0.0043/-0.0011 | 0.475/0.241/0.325 | feed_like_v2/spell_f1, feed_like_v2/macro_f1 |
| A_dt3_c3 | 0.6833/0.5373/0.5984 | -0.0002 | +0.0031/-0.0024/-0.0014 | 0.468/0.241/0.319 | feed_like_v2/spell_f1 |
| A_dt3_c4 | 0.6822/0.5366/0.5987 | -0.0000 | +0.0035/-0.0016/-0.0019 | 0.465/0.240/0.316 | feed_like_v2/spell_f1 |
| A_dt4_c2 | 0.6797/0.5356/0.5985 | -0.0002 | +0.0018/-0.0023/-0.0002 | 0.451/0.236/0.312 | clean/spell_f1, clean/short, feed_like_v2/spell_f1 |
| A_dt4_c4 | 0.6795/0.5347/0.5994 | +0.0007 | +0.0024/+0.0021/-0.0024 | 0.447/0.228/0.306 | clean/spell_f1, clean/short, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v2/short |
| A_dt5_c2 | 0.6765/0.5343/0.5962 | -0.0025 | -0.0038/-0.0029/-0.0007 | 0.433/0.225/0.295 | clean/spell_f1, clean/short, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short, feed_like_v2/macro_f1 |
| A_dt5_c3 | 0.6743/0.5311/0.5976 | -0.0011 | -0.0016/+0.0003/-0.0020 | 0.420/0.214/0.294 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short |
| A_dt5_c4 | 0.6755/0.5303/0.5982 | -0.0005 | +0.0000/+0.0016/-0.0030 | 0.421/0.208/0.293 | clean/spell_f1, clean/short, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short |
| A_dt6_c2 | 0.6682/0.5255/0.5939 | -0.0048 | -0.0058/-0.0033/-0.0052 | 0.405/0.205/0.281 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short, feed_like_v2/macro_f1 |
| A_dt6_c3 | 0.6714/0.5267/0.5957 | -0.0030 | -0.0051/+0.0008/-0.0047 | 0.409/0.202/0.280 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short, feed_like_v2/macro_f1 |
| A_dt6_c4 | 0.6712/0.5245/0.5969 | -0.0017 | -0.0039/+0.0033/-0.0045 | 0.408/0.198/0.281 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short |
| A_dt4_c3 | 0.6815/0.5347/0.5987 | +0.0000 | +0.0000/+0.0000/+0.0000 | 0.452/0.230/0.308 | feed_like_v2/spell_f1 |
| A_pD_avg_dt4_c3 | 0.6810/0.5331/0.5983 | -0.0003 | +0.0003/-0.0006/-0.0008 | 0.458/0.228/0.306 | clean/spell_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v2/spell_f1, feed_like_v2/short |
| A_pD_cal_dt4_c3 | 0.6812/0.5348/0.5982 | -0.0005 | -0.0016/-0.0007/+0.0008 | 0.460/0.229/0.308 | clean/spell_f1, clean/macro_f1, feed_like_v1/short, feed_like_v2/spell_f1 |

### B: H/A cost under the snap mask
| arm | spell clean/v1/v2 | Δ v2 | Δ v2 Met/SC/Spo | short recall c/v1/v2 | failed guards |
|---|---|---:|---|---|---|
| B_ha2.0 | 0.6795/0.5338/0.5937 | -0.0050 | -0.0121/-0.0031/+0.0003 | 0.457/0.239/0.314 | clean/spell_f1, feed_like_v1/spell_f1, feed_like_v2/spell_f1 |
| B_ha2.5 | 0.6806/0.5346/0.5971 | -0.0016 | -0.0044/-0.0008/+0.0005 | 0.454/0.238/0.313 | clean/spell_f1, feed_like_v1/spell_f1, feed_like_v2/spell_f1 |
| B_ha3.5 | 0.6806/0.5347/0.5986 | -0.0001 | +0.0013/+0.0004/-0.0021 | 0.448/0.226/0.298 | clean/spell_f1, clean/short, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v2/spell_f1, feed_like_v2/short |
| B_ha4.0 | 0.6786/0.5343/0.5970 | -0.0017 | +0.0020/+0.0002/-0.0073 | 0.438/0.222/0.291 | clean/spell_f1, clean/short, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v2/spell_f1, feed_like_v2/short |

### C: nearest-player evidence added to home log-odds (r m x lambda)
Sanity (clean, ball observed, nearest player within 1 m, H/A frames): nearest team equals label 0.956 Metrica, 0.907 Sportec, 0.856 SkillCorner.
| arm | spell clean/v1/v2 | Δ v2 | Δ v2 Met/SC/Spo | short recall c/v1/v2 | failed guards |
|---|---|---:|---|---|---|
| C_r1.0_l0.5 | 0.6814/0.5346/0.5985 | -0.0001 | -0.0006/-0.0001/+0.0003 | 0.451/0.233/0.312 | clean/spell_f1, clean/short, feed_like_v1/spell_f1, feed_like_v2/spell_f1 |
| C_r1.0_l1.0 | 0.6806/0.5356/0.5960 | -0.0027 | -0.0025/-0.0025/-0.0032 | 0.455/0.241/0.313 | clean/spell_f1, feed_like_v2/spell_f1 |
| C_r1.0_l2.0 | 0.6774/0.5329/0.5918 | -0.0069 | -0.0108/-0.0031/-0.0068 | 0.467/0.242/0.316 | clean/spell_f1, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v2/spell_f1, feed_like_v2/macro_f1 |
| C_r1.5_l0.5 | 0.6804/0.5362/0.5978 | -0.0009 | -0.0017/-0.0002/-0.0008 | 0.453/0.237/0.310 | clean/spell_f1, feed_like_v2/spell_f1 |
| C_r1.5_l1.0 | 0.6766/0.5357/0.5951 | -0.0036 | -0.0052/-0.0007/-0.0049 | 0.446/0.239/0.311 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v2/spell_f1 |
| C_r1.5_l2.0 | 0.6669/0.5289/0.5836 | -0.0151 | -0.0278/-0.0034/-0.0142 | 0.457/0.240/0.312 | clean/spell_f1, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/macro_f1 |
| C_r2.5_l0.5 | 0.6794/0.5347/0.5948 | -0.0039 | -0.0072/-0.0005/-0.0039 | 0.450/0.229/0.308 | clean/spell_f1, clean/short, feed_like_v1/short, feed_like_v2/spell_f1 |
| C_r2.5_l1.0 | 0.6742/0.5314/0.5897 | -0.0090 | -0.0169/+0.0001/-0.0102 | 0.445/0.223/0.308 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v2/spell_f1, feed_like_v2/macro_f1 |
| C_r2.5_l2.0 | 0.6547/0.5216/0.5659 | -0.0328 | -0.0545/-0.0103/-0.0335 | 0.453/0.227/0.304 | clean/spell_f1, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short, feed_like_v2/macro_f1 |

## Family D, round 1: provider-held-out LightGBM P(D) refiner
Leave-one-provider-out; train rows clean + feed_like_v2 of the other two providers (every 5th frame). v1 not in training.
| arm | spell clean/v1/v2 | Δ v2 | Δ v2 Met/SC/Spo | short recall c/v1/v2 | failed guards |
|---|---|---:|---|---|---|
| D1_refined | 0.7014/0.5056/0.6158 | +0.0171 | +0.0415/-0.0044/+0.0142 | 0.512/0.255/0.345 | feed_like_v1/spell_f1, feed_like_v1/macro_f1, clean/macro_f1/SkillCorner, clean/d_f1/SkillCorner |
| D2_half_refined | 0.6941/0.5252/0.6162 | +0.0175 | +0.0294/+0.0045/+0.0187 | 0.469/0.234/0.322 | feed_like_v1/spell_f1, feed_like_v1/macro_f1 |
| D3_refined_nosnap | 0.7003/0.5049/0.6148 | +0.0161 | +0.0421/-0.0051/+0.0114 | 0.513/0.255/0.342 | feed_like_v1/spell_f1, feed_like_v1/macro_f1, clean/macro_f1/SkillCorner, clean/d_f1/SkillCorner |

Held-out AUC refined vs predecessor fell below the predecessor on v1 for SkillCorner (0.841 vs 0.925) and Sportec (0.913 vs 0.970); `ball_speed` dominated importance (54-67% gain).

## Family D, round 2 (`D_round2`): v1 rows in training and shortcut ablations
Training rows: clean + v1 (2) + v2 (4) of the training providers. Pipeline: f062 decode of w*refined + (1-w)*predecessor P(D), then snap_m0.0, then the frontier H/A decode. Feature sets: full; nomiss (no observed flag, since/until observed or next-observation features; ball speed and beyond-boundary distance forward-filled); nospeed; nomiss_nospeed; noball.
| arm | spell c/v1/v2 | Δ spell c/v1/v2 | macro c/v1/v2 | D-F1 c/v1/v2 | short c/v1/v2 | failed guards |
|---|---|---|---|---|---|---|
| R2_full_w1.0 | 0.7003/0.5333/0.6157 | +0.0188/-0.0015/+0.0170 | 0.8875/0.8109/0.8731 | 0.8854/0.8256/0.8964 | 0.514/0.253/0.342 | feed_like_v1/spell_f1, feed_like_v1/macro_f1, clean/d_f1/SkillCorner |
| R2_full_w0.5 | 0.6932/0.5392/0.6135 | +0.0117/+0.0045/+0.0148 | 0.8899/0.8223/0.8712 | 0.8973/0.8491/0.8962 | 0.466/0.235/0.317 | feed_like_v1/macro_f1 |
| R2_full_w0.25 | 0.6899/0.5373/0.6076 | +0.0084/+0.0025/+0.0089 | 0.8841/0.8252/0.8652 | 0.8886/0.8555/0.8869 | 0.463/0.232/0.307 | feed_like_v1/macro_f1, feed_like_v2/short |
| R2_nomiss_w0.5 | 0.6845/0.5368/0.6036 | +0.0030/+0.0021/+0.0049 | 0.8846/0.8291/0.8612 | 0.8921/0.8646/0.8812 | 0.445/0.229/0.300 | clean/short, feed_like_v1/short, feed_like_v2/short |
| R2_nospeed_w0.5 | 0.6894/0.5343/0.6133 | +0.0079/-0.0004/+0.0147 | 0.8883/0.8213/0.8723 | 0.8970/0.8487/0.8979 | 0.457/0.231/0.315 | feed_like_v1/spell_f1, feed_like_v1/macro_f1 |
| R2_nomiss_nospeed_w0.5 | 0.6841/0.5351/0.6049 | +0.0025/+0.0003/+0.0062 | 0.8843/0.8265/0.8615 | 0.8918/0.8604/0.8815 | 0.436/0.225/0.298 | clean/short, feed_like_v1/short, feed_like_v2/short |
| R2_noball_w0.5 | 0.6737/0.5275/0.5953 | -0.0078/-0.0073/-0.0034 | 0.8743/0.8207/0.8524 | 0.8753/0.8508/0.8666 | 0.421/0.207/0.284 | clean/spell_f1, clean/short, clean/macro_f1, feed_like_v1/spell_f1, feed_like_v1/short, feed_like_v1/macro_f1, feed_like_v2/spell_f1, feed_like_v2/short, feed_like_v2/macro_f1 |

Per-provider spell deltas vs snap_m0.0, by view:
| arm | clean Met/SC/Spo | v1 Met/SC/Spo | v2 Met/SC/Spo |
|---|---|---|---|
| R2_full_w1.0 | +0.0365/-0.0195/+0.0394 | +0.0259/-0.0321/+0.0019 | +0.0380/-0.0022/+0.0152 |
| R2_full_w0.5 | +0.0261/-0.0044/+0.0134 | +0.0141/-0.0085/+0.0078 | +0.0239/+0.0037/+0.0167 |
| R2_full_w0.25 | +0.0101/+0.0004/+0.0146 | +0.0055/-0.0060/+0.0081 | +0.0122/+0.0038/+0.0107 |
| R2_nomiss_w0.5 | +0.0125/-0.0003/-0.0033 | +0.0085/-0.0004/-0.0019 | +0.0098/+0.0029/+0.0020 |
| R2_nospeed_w0.5 | +0.0123/-0.0007/+0.0119 | +0.0010/-0.0101/+0.0079 | +0.0194/+0.0058/+0.0188 |
| R2_nomiss_nospeed_w0.5 | +0.0007/+0.0015/+0.0054 | -0.0004/+0.0013/+0.0001 | +0.0083/+0.0046/+0.0058 |
| R2_noball_w0.5 | -0.0139/-0.0021/-0.0075 | -0.0100/+0.0008/-0.0126 | -0.0065/+0.0029/-0.0065 |

Best arm by fewest failed guards then v2: `R2_full_w0.5`; v2 cells improved 17, declined 3 of 20 (declines are SkillCorner 1996435/1996436 realizations). Eligible arms: none.
Held-out AUC/logloss per fold and view, and top-12 importances, are in `results_D_round2.json`.

### Ablation reading
Gain over snap_m0.0 on v2 / clean spell: full w0.5 +0.0148 / +0.0117; nospeed +0.0147 / +0.0079; nomiss +0.0049 / +0.0030; nomiss_nospeed +0.0062 / +0.0025; noball -0.0034 / -0.0078. Removing ball speed barely changes v2, but removing the missingness features cuts about two thirds of the v2 gain and 75% of the clean gain, and the remainder (about +0.005 to +0.006 v2) fails the short-recall guards. Removing all ball features erases the gain and regresses below the control. The nomiss held-out logloss improves on SkillCorner (0.27 vs 0.30) and keeps AUC above the predecessor on every fold and view, so a missingness-free refiner is real but weaker.

## Limitations
- All five matches are exposed development data; the LOPO folds still use these matches as training/test, and design choices (arms, features) were made after seeing earlier results.
- feed_like_v2 is model- and label-conditioned (ball removal rates placed with development labels), so v2 gains from features that encode ball missingness may not transfer to the challenge feed; the ablations test this but cannot settle it.
- No pooled five-match deployment fit was run; no GAR VALID/TEST or public-label check. SkillCorner silver labels start dead ball before the ball leaves the pitch, which limits any ball-exit style gain there.
- Guard outcomes are against the unrounded snap_m0.0 control and use the repo guard set (clean D-F1 per-provider included).
