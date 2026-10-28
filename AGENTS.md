# Agent guide: possession research

## Load only what the task needs

- Start with [current status and next work](docs/research/STATUS.md).
- Working in `experiments/`? Read [experiments/AGENTS.md](experiments/AGENTS.md)
  first: vocabulary, study lineage, settled directions and code map.
- Find prior evidence in the [experiment index](docs/research/EXPERIMENTS.md).
  Read the relevant report, not every historical handoff.
- For restore/inference, use [BASELINE.md](experiments/BASELINE.md).
  The active package is bound by `experiments/baseline_manifest.json`;
  ranking/reporting is defined by `experiments/model_decision_policy.json`.
- For a new experiment, read [possession-experiment](.agents/skills/possession-experiment/SKILL.md).
- For feature preparation or split planning in new runners, start with
  [`experiments/preparation.py`](experiments/preparation.py) and read
  [possession-preparation](.agents/skills/possession-preparation/SKILL.md).
  Reuse its validated whole-match/provider folds and verified unnormalized
  caches when the feature recipe matches. Fit normalization and every learned
  component on the training fold only; declare and test a new recipe for
  unsupported features. Preserve frozen study runners.
- For score comparisons or adoption advice, read
  [possession-readout](.agents/skills/possession-readout/SKILL.md).
  [Documentation ownership](docs/research/README.md) explains what to update.

## Before answering model-choice or current-results questions

This applies to chat advice as well as code work. Historical reports and saved
Pages can contain valid scores but superseded recommendations.

1. Check current `main`: read STATUS, BASELINE, the active baseline manifest
   and model decision policy before describing the working model or guardrails.
   The manifest identifies the adopted package; the policy defines the active
   evaluation rule. Historical report verdicts do not override later promotion.
   If current sources conflict, inspect the promotion record and recent commits
   and disclose any unresolved discrepancy.
2. Inspect recent merged and open PRs relevant to the question, including their
   current heads and saved reports. STATUS and this guide are snapshots and may
   lag ongoing work. Distinguish adopted, completed experimental, running and
   proposed work; never call a method untested solely because an older summary
   says so. Saved Pages are discovery aids, not authoritative current state.
3. Name the exact stress view with every robustness claim. In particular,
   `feed_like` and `composite` are different evaluations; compare models on the
   same view and component holdouts. Check the current policy and component
   ablation. Preserve composite failures as diagnostics and disclose a
   post-result guardrail change rather than claiming the old failure vanished.
4. Distinguish model family, training recipe and pipeline. A failed earlier GRU
   recipe is not evidence that the current dropout GRU fails. A D-only GRU/TCN
   comparison does not establish the best joint H/A/D architecture. Describe
   “best supported current choice” separately from superiority demonstrated by
   a matched architecture comparison.
5. Check ensemble status in the latest evidence. Averaging separate seed scores
   is not ensembling predictions; a measured probability ensemble may exist on
   a newer branch while the adopted package remains single-seed. Cite the
   relevant report/PR and its evaluation scope when making recommendations.

## Constraints that always apply

- [CHALLENGE](CHALLENGE.md), [EVALUATION](EVALUATION.md),
  [SUBMISSION](SUBMISSION.md), and [RULES](RULES.md) define the task:
  199,319 consecutive `frame,label` rows at 25 Hz, H/A/D/X. Loose or contested
  ball stays with the last controlling team. Use the official evaluator.
- Organizer public labels are pipeline checks only (7,500 frames, 17 possessions).
  Never train, tune, select candidates, thresholds, repairs or seeds on them.
  Do not manually label/correct this match or use human-produced event/possession
  data about it. Label-free tracking improvements are allowed.
- Read saved diagnostics; never reset a closed scoring gate or rewrite a frozen
  declaration. A fresh directory does not make inspected data fresh evidence.
- Split by whole match/provider. Identify which learned components are held out;
  D-only holdout with pooled H/A is not full-pipeline holdout. Audit pretrained
  checkpoint exposure and source-specific label/missingness shortcuts.
- Keep organizer assets, reconstructive derivatives, weights and frame outputs
  out of this public repo. Use ignored `data/`, `MTAH_DATA_DIR`, or the authorized
  private cache. Publish aggregate evidence. Restore verified artifacts before
  retraining; preserve native masks, unknown targets and match/half/time breaks.

## Delivery

After a promising experiment completes, update the [experiment progress Page](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9) with dated scores, visualizations, evaluation protocol and adoption status.

Work on a branch in `jekkiechan/more-than-a-hack-2027`; PRs target the fork's
`main`. Preserve `upstream` as `metrica-sports/more-than-a-hack-2027`.
Publish an authorized PR without a separate confirmation step. Documentation
maintenance does not promote draft models. Existing [CI/auto-merge](docs/CI.md) apply.

For code changes, run `python -m unittest discover -s experiments/tests -v`
with pinned dependencies (including CPU PyTorch for neural contracts).
Validate generated CSVs with `python evaluation/validate_submission.py <candidate.csv>`.
For docs-only changes, check links, evidence and frozen-file preservation;
CI runs the synthetic suite and bundled format check. Do not fit or score
organizer labels to verify documentation.
