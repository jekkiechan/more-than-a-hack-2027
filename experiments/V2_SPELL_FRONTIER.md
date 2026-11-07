# V2 spell frontier: mixtures and fresh training windows

Study: `v2_spell_frontier_v1` · 7 November 2026 · development running.

The user requested further iteration before the final public spell-F1 diagnostic.
This separate follow-up extends [spell balancing](V2_SPELL_BALANCED.md), whose
frozen candidate reaches 0.591174 v2 spell F1 while retaining all three spell
means and short-spell recall above adopted. Its full reference-free public CSV
is sealed but remains unscored. Every prior declaration, result and gate is
preserved.

## Question and fixed design

The best retained mixture used calibrated weight 0.25, the edge of the earlier
grid. Thirty fixed stage-1024 mixtures extend calibrated weights to 0.75,
with nonnegative adopted residual, arithmetic/log pooling and H/A cost 3.
Both prior leader mixtures replay before new scoring. The two best eligible
mixtures plus the current arithmetic anchor form a deduplicated shortlist.

A paired continuation compares repeated versus fresh TRAIN-window draws:
twelve provider-excluded fits, two seeds, 512 further updates at learning rate
0.0001, warm-started from exactly paired stage-1024 checkpoints. Both use the
same loss, optimizer reset, batch size 32, 50% GAR, four fixed v2 TRAIN views,
and Torch/dropout seeds. The fresh arm offsets the original distinct-possession
sampler and trainer draw RNG by 104729; actual window receipts verify changed
exposure. D parameters, normalization and teacher identity remain fixed.
The shortlist is evaluated with both new stage-1536 arms (at most six pipelines).

The original TRAIN pool has 13 matches; each development fold excludes its
whole provider from all learned components and normalization. GAR contributes
45 TRAIN games with generated silver targets. No GAR VALID or TEST evaluation
is part of this declaration. The five exposed development matches produce 35
cells: clean, two historical v1 realizations and four fixed v2 realizations.
Replicates are averaged within match, matches within provider, then providers
equally. Seeds are pooled within native phase before averaging phases.

## Selection and evidence limits

Eligibility remains fixed against adopted: strictly higher spell F1 in all
three views, macro F1 at least adopted minus 0.001 per view, short-spell recall
at least adopted, exact adopted D predictions and unchanged reference counts.
Rank eligible candidates by v2 spell F1, macro for exact ties, then pipeline ID.
A new candidate replaces the balanced selection only with strictly higher v2
spell F1; otherwise retain it. Seed diagnostics never rerank candidates.
The aspiration remains **v2 spell F1 above 0.60**. No automatic adoption follows.

Independent audits rescore saved predictions and verify hashes, identities,
provider reduction, gates, rankings and shortlisted seed/ensemble decoding.
Round two also verifies native phase pooling. Selection precedes pooled TRAIN
reproduction, full-match prediction, sealing and one final public score across
both studies. The old balanced public gate stays unused. Public labels cannot
choose mixtures, seeds, repairs or thresholds.

These are repeatedly exposed development matches and model-conditioned
synthetic views. Additional training draws do not create independent evaluation
evidence or new gold labels. The public window has only 7,500 graded frames and
17 possessions and is a reused pipeline diagnostic.

## Evidence

- [Frozen parent plan](results/v2_spell_frontier_v1_plan.json)
- [Previous development selection](results/v2_spell_balanced_v1_selection.json)
- [PR #86](https://github.com/jekkiechan/more-than-a-hack-2027/pull/86)
