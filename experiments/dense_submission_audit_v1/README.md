# Dense submission hardening audit (2026-10-31)

Audit-only checks of the adopted `joint_gru_sn_dense_ensemble_v1` package and PR #48 reproduction path. This directory does not alter the frozen baseline, its policy, manifests, receipts, or published reports. No organizer reference labels or scores were read.

Run the focused contract probes with:

```sh
.venv/bin/python -m unittest experiments.dense_submission_audit_v1.test_contracts -v
```

Full restore/inference replay details and hashes are in `report.json`. Generated private inputs and CSVs remain under ignored `data/submission_audit/`.
