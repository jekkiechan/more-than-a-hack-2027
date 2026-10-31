# Agent guide: possession research

## Find what the task needs

- [STATUS](docs/research/STATUS.md): current adoption, priorities and blockers.
- [BASELINE](experiments/BASELINE.md): package evidence and restore/inference.
  The active manifest identifies the adopted package; the policy defines ranking
  and guards. Historical reports and saved Pages may describe older packages.
- [Research skill](.agents/skills/possession-research/SKILL.md): choose restore,
  preparation, experiment or readout; load only the applicable reference.
- [Code map and vocabulary](experiments/README.md): locate shared implementation.
- [Experiment index](docs/research/EXPERIMENTS.md): search by mechanism/study ID
  for prior evidence; read the relevant section/report, not the whole index.
- [Documentation map](docs/research/README.md): ownership and maintenance checks.

## Constraints that always apply

- [CHALLENGE](CHALLENGE.md), [EVALUATION](EVALUATION.md),
  [SUBMISSION](SUBMISSION.md), and [RULES](RULES.md) define the task:
  199,319 consecutive `frame,label` rows at 25 Hz, H/A/D/X. Loose or contested
  ball stays with the last controlling team. Use the official evaluator.
- Organizer public labels are pipeline checks only (7,500 frames, 17 possessions).
  Never train, tune or select candidates, repairs, thresholds or seeds on them.
  No manual labels/corrections or human-produced event/possession data about
  this match. Label-free tracking improvements are allowed.
- Preserve frozen runners, declarations, manifests, policies, receipts, hashes,
  results and hash-bound Markdown (including `results/sn_dense_v1_progress.md`).
  Never reset a closed scoring gate. A new directory is not fresh evidence.
- Split by whole match/provider and exclude it from every learned component for
  full-pipeline holdout. Audit checkpoint exposure and source-specific shortcuts.
- Keep organizer assets, reconstructive derivatives, weights and frame outputs
  in ignored `data/`, `MTAH_DATA_DIR`, or authorized private storage. Publish
  aggregate evidence. Restore verified artifacts before retraining; preserve
  native masks, unknown targets and match/half/time breaks.
- Import shared libraries; a new study gets a new runner/ID. Do not copy or edit
  a frozen runner to start another sweep. Check prior outcomes first.

## Delivery and checks

Work on a branch in `jekkiechan/more-than-a-hack-2027`; PRs target the fork's
`main`. Preserve `upstream` as `metrica-sports/more-than-a-hack-2027` when configured.
Publish an authorized PR without another confirmation. Existing
[CI/auto-merge](docs/CI.md) applies. Docs maintenance never promotes a model.
After a promising experiment, update the
[progress Page](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9)
with dated scores, protocol, visualizations and adoption status.

Run commands from the repo root. Full suites need pinned experiment/SoccerNet
dependencies and CPU PyTorch; activate the repo `.venv` when present (`python`
examples assume that environment), or use its interpreter explicitly. System
Python may support the focused planner tests while lacking full-suite dependencies.
Focused preparation check:
`python -m unittest experiments.tests.test_preparation -v`.
For code changes: `python -m unittest discover -s experiments/tests -v`, plus
applicable checks in [CI](docs/CI.md). CSV format:
`python evaluation/validate_submission.py <candidate.csv>`.
For docs-only changes, run the documentation check in the documentation map and
verify current-state claims/frozen preservation. Never score organizer labels
just to verify documentation.
