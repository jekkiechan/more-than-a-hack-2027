# Turnover supervision mix follow-up v1

## Question

Can a fixed blend of the fresh-control and hard-turnover conditional H/A experts retain complementary turnover supervision while preserving the existing frontier's guarded behavior?

## Design

This is a post-result, three-point descriptive follow-up to the frozen turnover-supervision evaluation. It uses only the already saved 35 native development cells and their saved pooled CE probabilities. For each cell, the CE family is the arithmetic blend `(1 - α) * fresh_control + α * hard_turnover` at `α ∈ {0.25, 0.50, 0.75}`. It then applies the existing approved conditional family mixture: CE 0.5, calibrated 0.375, predecessor 0.125, followed by the unchanged fixed-D decoder at HA cost 3. No new model inference, fitting, labels, cells, or corruption views are introduced.

The three fractions are fixed before this follow-up's candidate metrics are computed. There is no finer sweep or automatic promotion. Fresh control is the matched control; uniform turnover is omitted because the prior diagnostic did not justify carrying it forward. The analysis will also emit the same three blends for each saved CE seed as descriptive diagnostics, without using them for selection.

## Endpoint replay and checks

Before creating blends, independently replay the official scores and exact D/X masks for all three original pooled candidate arms against the sealed evaluation result. Bind the follow-up declaration to the turnover training/evaluation declarations, original independent readout, baseline native manifest, and turnover-specific diagnostic declarations. Check all 35 cell and prediction artifact hashes before reading candidate arrays.

For each blend, preserve the saved native metadata and verify exact D/X masks. Independently rescore with the shared official scorer, recompute equal-provider aggregates, and replay the current baseline guards. Retention requires strict feed-like-v2 spell-F1 improvement, no clean or feed-like-v1 spell-F1 regression, no short-spell-recall regression in any view, macro-F1 within 0.001 of baseline in every view, and clean provider macro-F1/D-F1 within 0.02 of baseline. Results are descriptive only; this continuation does not create new confirmation or change the adopted model.

## Limits

This reuses five exposed development matches, silver GAR training, and the same saved views. It cannot establish independent-match generalization. Turnover event counts repeat across cells. The mixtures do not add supervision or alter D/X behavior; they only change the conditional H/A family contribution.
