# v2_target_push_v1 phase-1 diagnostics (development cohort, vs snap_m0.0)

Spell F1 aggregation: realizations within match, matches within provider, providers equally. All labels used only for diagnostics.

## Control replay

| | clean | v1 | v2 |
|---|---:|---:|---:|
| frontier | 0.670137 | 0.532128 | 0.593619 |
| snap_m0.0 | 0.681508 | 0.534731 | 0.598690 |

## a. Oracle ladder (spell F1; one thing replaced by truth on the snap_m0.0 prediction)

| rung | clean | v1 | v2 | v2 Met | v2 SC | v2 Spo | v2 short recall | v2 pred spells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| snap_control | 0.6815 | 0.5347 | 0.5987 | 0.7147 | 0.4787 | 0.6026 | 0.308 | 337 |
| i_oracle_D_mask | 0.7665 | 0.6187 | 0.6848 | 0.8033 | 0.5739 | 0.6773 | 0.415 | 334 |
| ii_oracle_D_starts | 0.6935 | 0.5447 | 0.6107 | 0.7240 | 0.4830 | 0.6252 | 0.340 | 354 |
| iii_oracle_D_ends | 0.6958 | 0.5440 | 0.6114 | 0.7338 | 0.4797 | 0.6207 | 0.330 | 349 |
| viii_oracle_D_starts_and_ends | 0.7071 | 0.5531 | 0.6223 | 0.7426 | 0.4839 | 0.6405 | 0.362 | 366 |
| iv_remove_false_D_runs | 0.6913 | 0.5459 | 0.6080 | 0.7285 | 0.4871 | 0.6083 | 0.316 | 340 |
| v_add_missed_D_runs | 0.6976 | 0.5495 | 0.6158 | 0.7348 | 0.5053 | 0.6075 | 0.321 | 327 |
| iv_v_both | 0.7074 | 0.5608 | 0.6252 | 0.7485 | 0.5139 | 0.6131 | 0.329 | 329 |
| vi_oracle_HA_in_live | 0.8776 | 0.8368 | 0.8527 | 0.8790 | 0.8335 | 0.8457 | 0.777 | 411 |
| vii_a_one_true_team_per_live_span | 0.2630 | 0.2683 | 0.2612 | 0.1983 | 0.3890 | 0.1964 | 0.068 | 152 |
| vii_b_true_team_boundaries_in_live | 0.8776 | 0.8390 | 0.8537 | 0.8790 | 0.8384 | 0.8436 | 0.763 | 404 |

Rung definitions: ii/iii/viii move each matched predicted D run's start / end / both to the true edge of the true D run it overlaps most (first/last predicted run touching that true run); iv removes predicted D runs overlapping no true D run; v adds true D runs overlapping no predicted D run; vi uses the true H/A label wherever the prediction is live and truth is H/A (D mask unchanged); vii_a gives each predicted live span one team (the true majority team inside it: no internal switches); vii_b uses true team boundaries inside predicted live spans (truth-D frames take the neighbouring true team).

## b. D-run timing (pred - true, seconds; matched = greedy one-to-one IoU >= 0.2)

### base mask

| view | provider | start err median [IQR] | end err median [IQR] |
|---|---|---|---|
| clean | all | +0.28 [-1.06,+1.36] p10/90 -2.8/+2.7 (n=263) | -0.18 [-1.45,+0.21] p10/90 -2.9/+2.0 (n=232) |
| clean | Metrica | -0.02 [-1.22,+0.61] p10/90 -3.6/+1.5 (n=76) | +0.32 [-0.08,+1.30] p10/90 -0.5/+3.5 (n=75) |
| clean | SkillCorner | +1.50 [+0.60,+2.75] p10/90 +0.3/+3.9 (n=95) | -1.70 [-2.50,-1.00] p10/90 -4.3/-0.4 (n=67) |
| clean | Sportec | -1.02 [-1.80,-0.24] p10/90 -4.4/+0.6 (n=92) | -0.02 [-0.75,+0.16] p10/90 -2.6/+1.8 (n=90) |
| feed_like_v2 | all | +0.20 [-1.16,+1.28] p10/90 -3.6/+2.7 (n=1034) | -0.28 [-1.60,+0.96] p10/90 -3.5/+3.3 (n=917) |
| feed_like_v2 | Metrica | -0.16 [-1.52,+0.81] p10/90 -5.1/+1.9 (n=292) | +0.78 [+0.08,+2.24] p10/90 -0.7/+4.8 (n=288) |
| feed_like_v2 | SkillCorner | +1.30 [+0.60,+2.60] p10/90 +0.2/+3.7 (n=370) | -1.70 [-2.60,-0.90] p10/90 -4.4/-0.4 (n=265) |
| feed_like_v2 | Sportec | -1.04 [-2.54,-0.23] p10/90 -6.0/+0.4 (n=372) | -0.20 [-1.24,+1.08] p10/90 -3.6/+4.2 (n=364) |

False D runs (pred run overlapping no true D run) / missed D runs (true run overlapping no pred run): count (total seconds), summed over cells of the view.

| view | provider | cells | false <1s | 1-3s | 3-10s | >10s | missed <1s | 1-3s | 3-10s | >10s |
|---|---|---:|---|---|---|---|---|---|---|---|
| clean | all | 5 | 51 (17) | 62 (121) | 38 (194) | 1 (12) | 43 (18) | 49 (93) | 50 (250) | 2 (27) |
| clean | Metrica | 1 | 2 (1) | 0 (0) | 5 (30) | 0 (0) | 0 (0) | 4 (10) | 11 (64) | 0 (0) |
| clean | SkillCorner | 3 | 49 (17) | 61 (118) | 29 (136) | 0 (0) | 42 (18) | 43 (78) | 35 (165) | 2 (27) |
| clean | Sportec | 1 | 0 (0) | 1 (3) | 4 (28) | 1 (12) | 1 (0) | 2 (6) | 4 (21) | 0 (0) |
| feed_like_v2 | all | 20 | 217 (84) | 229 (448) | 155 (754) | 11 (136) | 166 (69) | 178 (333) | 177 (899) | 7 (98) |
| feed_like_v2 | Metrica | 4 | 8 (2) | 5 (10) | 17 (99) | 5 (65) | 0 (0) | 15 (35) | 49 (292) | 0 (0) |
| feed_like_v2 | SkillCorner | 12 | 209 (82) | 218 (423) | 124 (579) | 1 (12) | 162 (69) | 157 (281) | 112 (522) | 7 (98) |
| feed_like_v2 | Sportec | 4 | 0 (0) | 6 (15) | 14 (76) | 5 (59) | 4 (0) | 6 (16) | 16 (85) | 0 (0) |

### snap mask

| view | provider | start err median [IQR] | end err median [IQR] |
|---|---|---|---|
| clean | all | -0.04 [-0.16,+1.28] p10/90 -1.1/+3.0 (n=288) | -0.18 [-1.45,+0.21] p10/90 -2.9/+2.0 (n=232) |
| clean | Metrica | -0.08 [-0.16,+0.00] p10/90 -0.8/+0.6 (n=76) | +0.32 [-0.08,+1.30] p10/90 -0.5/+3.5 (n=75) |
| clean | SkillCorner | +1.50 [+0.57,+2.73] p10/90 +0.2/+4.0 (n=120) | -1.70 [-2.50,-1.00] p10/90 -4.3/-0.4 (n=67) |
| clean | Sportec | -0.20 [-1.05,-0.12] p10/90 -4.4/-0.1 (n=92) | -0.02 [-0.75,+0.16] p10/90 -2.6/+1.8 (n=90) |
| feed_like_v2 | all | +0.00 [-0.44,+1.20] p10/90 -2.5/+2.9 (n=1090) | -0.28 [-1.60,+0.96] p10/90 -3.5/+3.4 (n=919) |
| feed_like_v2 | Metrica | -0.08 [-0.60,+0.08] p10/90 -4.8/+1.4 (n=293) | +0.80 [+0.08,+2.28] p10/90 -0.7/+4.8 (n=289) |
| feed_like_v2 | SkillCorner | +1.30 [+0.50,+2.60] p10/90 +0.2/+3.7 (n=425) | -1.70 [-2.68,-0.90] p10/90 -4.4/-0.5 (n=266) |
| feed_like_v2 | Sportec | -0.46 [-1.73,-0.12] p10/90 -6.0/+0.0 (n=372) | -0.20 [-1.24,+1.08] p10/90 -3.6/+4.2 (n=364) |

False D runs (pred run overlapping no true D run) / missed D runs (true run overlapping no pred run): count (total seconds), summed over cells of the view.

| view | provider | cells | false <1s | 1-3s | 3-10s | >10s | missed <1s | 1-3s | 3-10s | >10s |
|---|---|---:|---|---|---|---|---|---|---|---|
| clean | all | 5 | 36 (11) | 55 (104) | 37 (187) | 1 (12) | 44 (19) | 49 (93) | 49 (245) | 2 (27) |
| clean | Metrica | 1 | 2 (1) | 0 (0) | 5 (30) | 0 (0) | 0 (0) | 4 (10) | 11 (64) | 0 (0) |
| clean | SkillCorner | 3 | 34 (10) | 54 (101) | 28 (130) | 0 (0) | 43 (19) | 43 (78) | 34 (160) | 2 (27) |
| clean | Sportec | 1 | 0 (0) | 1 (3) | 4 (26) | 1 (12) | 1 (0) | 2 (6) | 4 (21) | 0 (0) |
| feed_like_v2 | all | 20 | 179 (68) | 217 (421) | 150 (732) | 11 (135) | 174 (74) | 180 (336) | 176 (894) | 7 (98) |
| feed_like_v2 | Metrica | 4 | 8 (2) | 7 (16) | 15 (92) | 5 (63) | 0 (0) | 15 (35) | 49 (292) | 0 (0) |
| feed_like_v2 | SkillCorner | 12 | 171 (65) | 204 (390) | 121 (564) | 1 (12) | 170 (74) | 159 (284) | 111 (517) | 7 (98) |
| feed_like_v2 | Sportec | 4 | 0 (0) | 6 (15) | 14 (77) | 5 (59) | 4 (0) | 6 (16) | 16 (85) | 0 (0) |

## c. Unmatched (IoU 0.5) spell causes, snap_m0.0, counts by duration bucket [<1s, 1-3s, 3-10s, >10s]

### clean / all (5 cells)

Reference spells [n, matched]: <1s [274, 78], 1-3s [509, 214], 3-10s [727, 468], >10s [453, 394]  
Predicted spells [n, matched]: <1s [244, 83], 1-3s [464, 202], 3-10s [695, 469], >10s [459, 400]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| wrong_team | 122 | 127 | 95 | 8 | 352 |
| merged_with_neighbour | 39 | 97 | 103 | 25 | 264 |
| covered_by_D | 35 | 71 | 51 | 6 | 163 |
| fragmented_by_team_switch | 0 | 0 | 6 | 17 | 23 |
| boundary_misalignment_only | 0 | 0 | 3 | 2 | 5 |
| fragmented_by_D_gap | 0 | 0 | 1 | 1 | 2 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| true_dead | 112 | 151 | 87 | 4 | 354 |
| wrong_team | 16 | 65 | 67 | 13 | 161 |
| fragment_of_longer_true_spell | 32 | 44 | 54 | 9 | 139 |
| spans_multiple_true_spells | 0 | 0 | 16 | 33 | 49 |
| extends_beyond_true_spell | 1 | 2 | 2 | 0 | 5 |

### clean / Metrica (1 cells)

Reference spells [n, matched]: <1s [31, 18], 1-3s [95, 70], 3-10s [129, 112], >10s [127, 126]  
Predicted spells [n, matched]: <1s [19, 19], 1-3s [88, 71], 3-10s [117, 109], >10s [133, 127]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| covered_by_D | 5 | 14 | 13 | 0 | 32 |
| merged_with_neighbour | 2 | 9 | 3 | 0 | 14 |
| wrong_team | 6 | 2 | 1 | 0 | 9 |
| fragmented_by_team_switch | 0 | 0 | 0 | 1 | 1 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| fragment_of_longer_true_spell | 0 | 6 | 5 | 4 | 15 |
| wrong_team | 0 | 9 | 1 | 0 | 10 |
| true_dead | 0 | 1 | 2 | 1 | 4 |
| extends_beyond_true_spell | 0 | 1 | 0 | 0 | 1 |
| spans_multiple_true_spells | 0 | 0 | 0 | 1 | 1 |

### clean / SkillCorner (3 cells)

Reference spells [n, matched]: <1s [202, 56], 1-3s [258, 74], 3-10s [428, 237], >10s [237, 184]  
Predicted spells [n, matched]: <1s [215, 58], 1-3s [301, 68], 3-10s [426, 235], >10s [238, 190]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| wrong_team | 101 | 96 | 85 | 7 | 289 |
| merged_with_neighbour | 26 | 60 | 78 | 25 | 189 |
| covered_by_D | 19 | 28 | 19 | 3 | 69 |
| fragmented_by_team_switch | 0 | 0 | 6 | 15 | 21 |
| boundary_misalignment_only | 0 | 0 | 2 | 2 | 4 |
| fragmented_by_D_gap | 0 | 0 | 1 | 1 | 2 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| true_dead | 112 | 150 | 78 | 2 | 342 |
| wrong_team | 15 | 51 | 61 | 11 | 138 |
| fragment_of_longer_true_spell | 30 | 31 | 41 | 5 | 107 |
| spans_multiple_true_spells | 0 | 0 | 10 | 30 | 40 |
| extends_beyond_true_spell | 0 | 1 | 1 | 0 | 2 |

### clean / Sportec (1 cells)

Reference spells [n, matched]: <1s [41, 4], 1-3s [156, 70], 3-10s [170, 119], >10s [89, 84]  
Predicted spells [n, matched]: <1s [10, 6], 1-3s [75, 63], 3-10s [152, 125], >10s [88, 83]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| covered_by_D | 11 | 29 | 19 | 3 | 62 |
| merged_with_neighbour | 11 | 28 | 22 | 0 | 61 |
| wrong_team | 15 | 29 | 9 | 1 | 54 |
| fragmented_by_team_switch | 0 | 0 | 0 | 1 | 1 |
| boundary_misalignment_only | 0 | 0 | 1 | 0 | 1 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| fragment_of_longer_true_spell | 2 | 7 | 8 | 0 | 17 |
| wrong_team | 1 | 5 | 5 | 2 | 13 |
| true_dead | 0 | 0 | 7 | 1 | 8 |
| spans_multiple_true_spells | 0 | 0 | 6 | 2 | 8 |
| extends_beyond_true_spell | 1 | 0 | 1 | 0 | 2 |

### feed_like_v2 / all (20 cells)

Reference spells [n, matched]: <1s [1096, 252], 1-3s [2036, 654], 3-10s [2908, 1689], >10s [1812, 1542]  
Predicted spells [n, matched]: <1s [877, 262], 1-3s [1684, 624], 3-10s [2664, 1693], >10s [1850, 1558]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| wrong_team | 490 | 593 | 455 | 45 | 1583 |
| merged_with_neighbour | 163 | 445 | 469 | 112 | 1189 |
| covered_by_D | 191 | 340 | 233 | 26 | 790 |
| fragmented_by_team_switch | 0 | 0 | 28 | 74 | 102 |
| boundary_misalignment_only | 0 | 4 | 32 | 9 | 45 |
| fragmented_by_D_gap | 0 | 0 | 2 | 4 | 6 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| true_dead | 450 | 534 | 306 | 24 | 1314 |
| wrong_team | 75 | 312 | 319 | 65 | 771 |
| fragment_of_longer_true_spell | 90 | 203 | 248 | 44 | 585 |
| spans_multiple_true_spells | 0 | 0 | 78 | 154 | 232 |
| extends_beyond_true_spell | 0 | 10 | 19 | 4 | 33 |
| boundary_misalignment_only | 0 | 1 | 1 | 1 | 3 |

### feed_like_v2 / Metrica (4 cells)

Reference spells [n, matched]: <1s [124, 28], 1-3s [380, 169], 3-10s [516, 367], >10s [508, 476]  
Predicted spells [n, matched]: <1s [42, 28], 1-3s [282, 163], 3-10s [460, 357], >10s [544, 492]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| covered_by_D | 40 | 87 | 60 | 2 | 189 |
| wrong_team | 40 | 53 | 43 | 4 | 140 |
| merged_with_neighbour | 16 | 69 | 33 | 9 | 127 |
| fragmented_by_team_switch | 0 | 0 | 6 | 17 | 23 |
| boundary_misalignment_only | 0 | 2 | 7 | 0 | 9 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| fragment_of_longer_true_spell | 6 | 43 | 45 | 29 | 123 |
| wrong_team | 8 | 68 | 34 | 2 | 112 |
| true_dead | 0 | 4 | 15 | 7 | 26 |
| spans_multiple_true_spells | 0 | 0 | 7 | 12 | 19 |
| extends_beyond_true_spell | 0 | 4 | 2 | 2 | 8 |

### feed_like_v2 / SkillCorner (12 cells)

Reference spells [n, matched]: <1s [808, 214], 1-3s [1032, 293], 3-10s [1712, 926], >10s [948, 738]  
Predicted spells [n, matched]: <1s [816, 223], 1-3s [1142, 272], 3-10s [1646, 928], >10s [943, 748]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| wrong_team | 408 | 374 | 344 | 33 | 1159 |
| merged_with_neighbour | 109 | 255 | 326 | 99 | 789 |
| covered_by_D | 77 | 109 | 79 | 12 | 277 |
| fragmented_by_team_switch | 0 | 0 | 20 | 53 | 73 |
| boundary_misalignment_only | 0 | 1 | 15 | 9 | 25 |
| fragmented_by_D_gap | 0 | 0 | 2 | 4 | 6 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| true_dead | 450 | 527 | 264 | 7 | 1248 |
| wrong_team | 61 | 209 | 240 | 49 | 559 |
| fragment_of_longer_true_spell | 82 | 130 | 169 | 15 | 396 |
| spans_multiple_true_spells | 0 | 0 | 39 | 123 | 162 |
| extends_beyond_true_spell | 0 | 4 | 5 | 0 | 9 |
| boundary_misalignment_only | 0 | 0 | 1 | 1 | 2 |

### feed_like_v2 / Sportec (4 cells)

Reference spells [n, matched]: <1s [164, 10], 1-3s [624, 192], 3-10s [680, 396], >10s [356, 328]  
Predicted spells [n, matched]: <1s [19, 11], 1-3s [260, 189], 3-10s [558, 408], >10s [363, 318]

False-negative reference spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| covered_by_D | 74 | 144 | 94 | 12 | 324 |
| wrong_team | 42 | 166 | 68 | 8 | 284 |
| merged_with_neighbour | 38 | 121 | 110 | 4 | 273 |
| boundary_misalignment_only | 0 | 1 | 10 | 0 | 11 |
| fragmented_by_team_switch | 0 | 0 | 2 | 4 | 6 |

False-positive predicted spells

| cause | <1s | 1-3s | 3-10s | >10s | total |
|---|---:|---:|---:|---:|---:|
| wrong_team | 6 | 35 | 45 | 14 | 100 |
| fragment_of_longer_true_spell | 2 | 30 | 34 | 0 | 66 |
| spans_multiple_true_spells | 0 | 0 | 32 | 19 | 51 |
| true_dead | 0 | 3 | 27 | 10 | 40 |
| extends_beyond_true_spell | 0 | 2 | 12 | 2 | 16 |
| boundary_misalignment_only | 0 | 1 | 0 | 0 | 1 |

## d. Delay-edit descriptors (label-free) vs fraction of re-opened frames truly live

An edit = one predicted D run whose start the delay rule moved (frames [orig start, first outside observation)). good = >=50% of its graded re-opened frames are truly H/A. AUC = P(descriptor higher for a good edit than a bad one); below 0.5 means lower values are good. Pooled counts multiply the same events across clean/v1/v2 realizations.

### all_views

| provider | edits | frames re-opened | good edits | frame-weighted live | AUC delay_s | AUC speed_edit | AUC speed_pre | AUC dist_at_start | AUC monotone_toward | AUC net_approach_m | AUC approach_rate_mps | AUC frac_ball_obs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 692 | 16548 | 0.604 | 0.747 | 0.4251 | 0.5608 | 0.3847 | 0.5049 | 0.4795 | 0.5101 | 0.5613 | 0.5992 |
| Metrica | 113 | 3889 | 1.000 | 1.000 | - | - | - | - | - | - | - | - |
| SkillCorner | 389 | 6588 | 0.311 | 0.291 | 0.5097 | 0.4002 | 0.2881 | 0.4388 | 0.5036 | 0.4404 | 0.4098 | 0.5841 |
| Sportec | 190 | 6071 | 0.968 | 0.949 | 0.2246 | 0.9955 | 0.7352 | 0.5497 | 0.4810 | 0.6295 | 0.7699 | 0.5439 |

### v2_only

| provider | edits | frames re-opened | good edits | frame-weighted live | AUC delay_s | AUC speed_edit | AUC speed_pre | AUC dist_at_start | AUC monotone_toward | AUC net_approach_m | AUC approach_rate_mps | AUC frac_ball_obs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 367 | 9004 | 0.632 | 0.783 | 0.4389 | 0.5479 | 0.3956 | 0.5132 | 0.4865 | 0.5210 | 0.5600 | 0.6074 |
| Metrica | 62 | 2189 | 1.000 | 1.000 | - | - | - | - | - | - | - | - |
| SkillCorner | 201 | 3415 | 0.338 | 0.309 | 0.5072 | 0.3793 | 0.2923 | 0.4317 | 0.5154 | 0.4356 | 0.4042 | 0.5891 |
| Sportec | 104 | 3400 | 0.981 | 0.987 | 0.4657 | 0.9902 | 0.7843 | 0.8971 | 0.4755 | 0.9069 | 0.9657 | 0.4314 |

Quantiles (all_views, p10/p25/median/p75/p90) of descriptors for good vs bad edits:

### SkillCorner

| descriptor | good (n=121) | bad |
|---|---|---|
| delay_s | 0.50/0.90/1.40/2.40/3.60 | 0.30/0.60/1.50/2.42/3.50 |
| speed_edit | 2.13/2.74/5.13/8.31/10.78 | 3.09/4.83/6.19/8.45/10.75 |
| speed_pre | 2.53/3.39/4.77/7.23/9.26 | 3.04/5.16/8.39/12.50/16.13 |
| dist_at_start | 0.76/1.31/2.38/5.44/8.12 | 0.50/1.56/3.68/7.03/10.00 |
| monotone_toward | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 |
| net_approach_m | 0.88/1.38/2.27/5.61/12.14 | 0.74/1.60/3.35/6.65/10.17 |
| approach_rate_mps | 0.53/0.99/1.85/4.67/7.19 | 0.64/1.56/3.05/5.41/7.11 |
| frac_ball_obs | 0.79/1.00/1.00/1.00/1.00 | 0.17/0.77/1.00/1.00/1.00 |

### Sportec

| descriptor | good (n=184) | bad |
|---|---|---|
| delay_s | 0.20/0.40/0.84/1.56/2.83 | 0.94/1.15/2.32/3.73/4.20 |
| speed_edit | 2.51/3.91/7.61/13.27/16.43 | 0.18/0.22/0.34/0.58/0.83 |
| speed_pre | 2.15/3.95/5.68/10.58/18.40 | 1.78/4.13/4.21/4.32/4.73 |
| dist_at_start | 0.41/1.32/3.03/7.07/14.35 | 0.18/0.39/0.87/23.71/26.32 |
| monotone_toward | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 |
| net_approach_m | 0.54/1.39/3.30/7.61/15.31 | 0.03/0.13/0.64/18.02/25.90 |
| approach_rate_mps | 1.31/1.93/3.85/8.26/12.98 | 0.03/0.07/0.55/4.69/6.16 |
| frac_ball_obs | 1.00/1.00/1.00/1.00/1.00 | 0.59/1.00/1.00/1.00/1.00 |

### all

| descriptor | good (n=417) | bad |
|---|---|---|
| delay_s | 0.24/0.56/1.04/1.92/3.31 | 0.30/0.60/1.50/2.50/3.50 |
| speed_edit | 2.37/3.75/7.36/10.84/14.43 | 2.87/4.71/6.16/8.41/10.69 |
| speed_pre | 2.06/3.58/5.76/9.39/16.38 | 2.95/5.08/8.19/12.28/15.82 |
| dist_at_start | 0.72/1.49/3.21/6.36/13.74 | 0.46/1.47/3.68/7.14/10.05 |
| monotone_toward | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 |
| net_approach_m | 0.81/1.60/3.43/6.75/13.93 | 0.70/1.57/3.28/6.75/10.18 |
| approach_rate_mps | 0.99/1.71/3.74/7.26/11.69 | 0.57/1.53/3.04/5.43/7.07 |
| frac_ball_obs | 1.00/1.00/1.00/1.00/1.00 | 0.17/0.77/1.00/1.00/1.00 |

Oracle gate (apply delay only to edits with >=50% truly-live re-opened frames, then advance; an upper bound, uses labels): spell F1 clean/v1/v2 = 0.683897 / 0.536364 / 0.600309 (snap_m0.0: 0.681508 / 0.534731 / 0.598690); v2 providers {'Sportec': 0.6028, 'Metrica': 0.7147, 'SkillCorner': 0.4835}

## e. P(D) across families and the origin of decoded__predecessor

Max abs diff of pooled P(D) (prob[:,2]) per view: {"clean": {"ce-cal": 0.6086751639842987, "ce-pred": 0.0, "cal-pred": 0.6086751639842987}, "feed_like_v1": {"ce-cal": 0.5978519598394632, "ce-pred": 0.0, "cal-pred": 0.5978519598394632}, "feed_like_v2": {"ce-cal": 0.560328318476677, "ce-pred": 0.0, "cal-pred": 0.560328318476677}}  
Mean abs diff: {"clean": {"ce-cal": 0.0179, "ce-pred": 0.0, "cal-pred": 0.0179}, "feed_like_v1": {"ce-cal": 0.01774, "ce-pred": 0.0, "cal-pred": 0.01774}, "feed_like_v2": {"ce-cal": 0.01799, "ce-pred": 0.0, "cal-pred": 0.01799}}

- f062_decode_of_pooled_predecessor_equals_decoded__predecessor_D_mask_all_cells: true
- agreement_of_mask_with_pD_gt_0.5: {"min": 0.6571333775713337, "mean": 0.8112754141039062}
- best_single_threshold_agreement: {"min": 0.7618447246184472, "mean": 0.8626057387971684, "thresholds_range": [0.2, 0.35000000000000003]}
- pD_mean_in_mask_vs_out: [0.5686382789389293, 0.0486212166310462]
- runs_mask_vs_thresholded_0.5: [3763, 6503]

decoded__predecessor == feed_benchmark_audit_v1.decode(pooled predecessor P, model_period, fps) with CONFIG f062 = feed_spell_push.configurations()[62]: family calibration dt=4, ht=1, ha=3, dead=3. feed_spell_push.decode applies a dead-logit temperature (logit(P_D)/4) and H/A odds temperature 1, renormalises, then decoder_search.decode(family matrix) runs a 3-state Viterbi on log-probabilities with switch costs [[0,3,3],[3,0,3],[3,3,0]]*(fps/10) per block of model_period.

## f. clean -> v2 (and v1) drop: D mask vs H/A

| combination (spell F1) | v1 | v2 |
|---|---:|---:|
| D_v2__HA_v2 (v2 snap) | 0.5347 | 0.5987 |
| D_clean__HA_v2 | 0.5430 | 0.6095 |
| D_v2__HA_clean | 0.6474 | 0.6578 |
| D_clean__HA_clean | 0.6815 | 0.6815 |

v2: clean 0.6815 -> 0.5987, drop 0.0828. D-mask effect 0.0109 (holding v-view H/A) or 0.0237 (holding clean H/A); H/A effect 0.0720 (given clean D) or 0.0591 (given v D). Shapley: D 0.0173, H/A 0.0655.

v1: clean 0.6815 -> 0.5347, drop 0.1468. D-mask effect 0.0082 (holding v-view H/A) or 0.0341 (holding clean H/A); H/A effect 0.1386 (given clean D) or 0.1127 (given v D). Shapley: D 0.0212, H/A 0.1256.

v2 by provider (Shapley): {"Metrica": {"clean": 0.8704, "view": 0.7147, "drop": 0.1557, "shapley_D": 0.0335, "shapley_HA": 0.1222}, "SkillCorner": {"clean": 0.4761, "view": 0.4787, "drop": -0.0027, "shapley_D": -0.0041, "shapley_HA": 0.0015}, "Sportec": {"clean": 0.6981, "view": 0.6026, "drop": 0.0954, "shapley_D": 0.0225, "shapley_HA": 0.073}}

## g. Extra: short (<3 s) unmatched predicted spells sitting on truly-dead frames

- clean/all: n=263, flanked by predicted D on both sides 0.000, on at least one side 0.593, mean pooled P(D) in span 0.231, ball observed 0.91, median duration 1.20s
- clean/Metrica: n=1, flanked by predicted D on both sides 0.000, on at least one side 1.000, mean pooled P(D) in span 0.189, ball observed 0.23, median duration 2.64s
- clean/SkillCorner: n=262, flanked by predicted D on both sides 0.000, on at least one side 0.592, mean pooled P(D) in span 0.231, ball observed 0.92, median duration 1.15s
- feed_like_v2/all: n=984, flanked by predicted D on both sides 0.004, on at least one side 0.598, mean pooled P(D) in span 0.186, ball observed 0.62, median duration 1.10s
- feed_like_v2/Metrica: n=4, flanked by predicted D on both sides 0.000, on at least one side 0.500, mean pooled P(D) in span 0.039, ball observed 0.22, median duration 1.72s
- feed_like_v2/SkillCorner: n=977, flanked by predicted D on both sides 0.004, on at least one side 0.597, mean pooled P(D) in span 0.187, ball observed 0.62, median duration 1.10s
- feed_like_v2/Sportec: n=3, flanked by predicted D on both sides 0.000, on at least one side 1.000, mean pooled P(D) in span 0.060, ball observed 0.54, median duration 2.40s
