# Possession-rate structure diagnostic (possession_rate_v1)

Label-free. No model, decoder, gate or STATUS change. No competition ground truth read, no public-window statistic, no scoring of the competition match. Provider labels are event/provider-derived silver labels whose definitions differ somewhat from the competition's, so the truth band is an approximate prior.

Code: `experiments/possession_rate.py` (`structure_stats`), tests `experiments/tests/test_possession_rate.py`, aggregates `experiments/results/possession_rate.json`. The pre-written rule is `data/experiments/possession_rate_v1/declaration.json` (git-ignored; written before any competition statistic was computed).

## Rule outcome
- Truth band (18 TRAIN+eval matches with H/A/D labels, match-level possessions per active minute): 3.44 (metrica_2) to 8.62 (skillcorner_had_1899585). Sportec 4.39-5.30, Metrica 3.44-4.02, SkillCorner 6.41-8.62. Adding the 3 spent Sportec confirmation matches leaves the band unchanged.
- Competition possessions/min: adopted 2.436, previous 2.351 (both below 3.44), so the pre-declared rule returns MERGING for both. Under-3-s share .194/.185 is below the feed_like holdout max .498, so FRAGMENTING does not fire.
- Closest holdout condition (RMS z-distance): feed_like (1.42 adopted, 1.53 previous) over clean (1.90, 2.04).
- Gap-driven switching: not flagged. Competition ball-missing/visible possessions-per-minute ratio is .41 (previous .47), versus .69 mean on feed_like holdout and .66 on clean.

## Interpretation caveat
The band min is set by Metrica and the band pools two very different annotation regimes (SkillCorner 10 Hz, about 7.4/min, many short D). On the two 25 Hz providers the model under-segments under feed_like (predicted/true possessions .62 Sportec, .72 Metrica; adopted clean .77/.89). Applying .62-.72 to the 25 Hz truth range 3.4-5.3 implies about 2.1-3.8 predicted per minute, which contains 2.44. The flag is therefore weak evidence of merging, not of fragmenting.

## Switch-cost table (descriptive only)
Saved feed_like holdout probabilities decoded at several costs. Holdout spellF1 changes by at most .013 across costs 2-8 and peaks at the adopted cost 3; the competition possessions/min falls monotonically as the cost rises, so a higher cost would worsen the merging direction. A lower cost (2) moves the competition toward the band but lowers holdout spellF1 by .003. Any decoder change must be chosen on holdout data (feed_like or a more realistic stress view), never on the competition public window. Verdict: not worth it for fragmentation; a declared study of minimum-gap or ball-missing-aware merging is worth it only if merging is shown on a more realistic stress view.
