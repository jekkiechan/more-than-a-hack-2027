# Agent guide: possession research

## Load only what the task needs

- Start with [current status and next work](docs/research/STATUS.md).
- Find prior evidence in the [experiment index](docs/research/EXPERIMENTS.md).
  Read the relevant report, not every historical handoff.
- For restore/inference, use [BASELINE.md](experiments/BASELINE.md).
  The active package is bound by `experiments/baseline_manifest.json`;
  ranking/reporting is defined by `experiments/model_decision_policy.json`.
- For a new experiment, read [possession-experiment](.agents/skills/possession-experiment/SKILL.md).
  For score comparisons or adoption advice, read
  [possession-readout](.agents/skills/possession-readout/SKILL.md).
  [Documentation ownership](docs/research/README.md) explains what to update.

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
