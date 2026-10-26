# Temporal models on the organizer public five minutes

The frozen bidirectional GRU improves spell F1 from the previous selected
refinement's **.4300 to .4889**. The TCN scores **.4541**. Both remain below
the previous PathCRF H/A + robust D hybrid's **.5067**. The externally selected
ordered HGB scores **.4205**, despite substantially better frame and D scores.

The [ChatGPT Page with interactive architecture animations](https://chatgpt.com/space/page_c9f856e9f2b88191b9af86da8c58a059)
explains the shared pipeline, GRU, TCN, ordered boosting, observation masks,
overlapping windows, and results. Its two HTML animations were checked locally
at desktop and mobile widths, including controls and JavaScript errors; the
saved Page embeds and attachments were read back successfully. Live Page-viewer
rendering was not available for inspection.

## Fixed comparison

All six externally fitted heads were declared and their full-length predictions
generated and hashed before the public check. The external `ordered_hgb`
selection, model weights, D threshold, H/A backbone, and Viterbi cost remained
unchanged. A single exclusive check marker prevents accidental repeated public
evaluation or candidate regeneration.

The unchanged official scorer graded **7,500 frames and 17 possessions**.
All new files contain 199,319 rows and passed the official submission validator.
The full-match reference labels remain private; these are public-window scores.

| Method | Spell F1 | Frame macro F1 | D F1 | Predicted public spells |
| --- | ---: | ---: | ---: | ---: |
| Previous selected refinement | .4300 | .6487 | .3392 | Historical report |
| Previous PathCRF hybrid | .5067 | .6795 | .3732 | Historical report |
| New matched HGB baseline | .4000 | .8062 | .7597 | 22 |
| Ordered HGB selected externally | .4205 | .8653 | .8855 | 22 |
| Bidirectional GRU | .4889 | .8617 | .8737 | 19 |
| TCN | .4541 | .8506 | .8496 | 20 |
| LightGBM .08 / 160 trees | .4103 | .8079 | .7635 | 22 |
| LightGBM .02 / 640 trees | .4000 | .8077 | .7626 | 22 |

New variants share exactly the same conditional H/A probabilities and decoder.
Against this matched baseline, GRU gains **.0889 spell F1**, TCN **.0541**, and
ordered HGB **.0205**. The old refinement used a different H/A model, so its
comparison does not isolate a temporal-D effect. The historical scores were
read from `refinement_competition.json`, after checking the same public-GT
SHA-256; old prediction files were unavailable and were not regenerated.

The lower-rate LightGBM head does not improve on .08 in this public check.
The neural results are promising diagnostics, especially for D, but the window
is too small to establish full-match improvement or select a replacement.

## Inputs and artifacts

Organizer inputs were retrieved through the registration email's Drive folder
invitation, after ordinary Drive keyword searches had missed the shared folder.
Only tactical ATD XML and TXT were downloaded; no video was needed. Input
conversion produced 199,319 frames at 25 Hz with 46 track slots and no graded
target labels. Source hashes and download provenance remain under ignored
`data/tactical/`.

Aggregate scores and frozen prediction/model hashes:
[results/learned_temporal_public_check.json](results/learned_temporal_public_check.json).
Inputs, full-frame CSVs, and check guards stay under ignored `data/`.
The Page contains only aggregate results, source code, and synthetic teaching
animations; no organizer tracking or frame-level prediction timeline was uploaded.

The executed command sequence was:

```sh
python -m experiments.temporal_challenge generate --all-frozen-heads
python -m experiments.temporal_challenge check
```

Do not reset the public gate. Follow-up modeling and calibration must use
external matches; a new independent confirmation match is still needed.
