# Restore the right artifact

1. Identify what is missing: raw provider data, prepared arrays, pooled deployment
   weights, provider-held-out research weights, or matching source. Use the
   package guide and relevant receipt to record package/study ID and hashes.
   Dataset access does not recover
   locally trained checkpoints; pooled weights do not replace fold models.
2. Inspect existing ignored data and the private `mtah-data` checkout first. Check
   its branch/commit and versioned manifest; if delivery is on an open PR, private
   `main` and a generic cache default may still select an older package. Use the
   exact versioned manifest from the documented delivery branch. Do not change
   unrelated working files or claim the delivery merged without checking.
3. For SoccerNet raw inputs, use the authorized
   [Hugging Face source](../../../../experiments/SOCCERNET_DATA_ACCESS.md):
   `OpenSportsLab/SoccerNet-GAR`, `tracking-full` branch, declaration-pinned
   revision and file hashes. Remote filenames are `train/...`; the local path
   includes `tracking-full/`. Use configured `HF_TOKEN` through the normal
   client, without printing it or starting a new login. Read the study's split
   policy before acquiring inputs: valid/test label QA in one study is not
   training permission for another. Ask for an archive only after checking the
   known source and reporting the actual access failure.
4. Fetch only the required manifest's artifacts into ignored/private storage.
   Verify SHA-256, expected counts, model/teacher pairing and matching source.
   An LFS pointer is not a checkpoint. Keep old source snapshots intact; use an
   isolated matching checkout/overlay if required by the package metadata.
5. Distinguish exact restore from a reconstructed comparator. Matching scores
   alone does not prove original checkpoint bytes. If an artifact is unavailable,
   report the missing identity and access blocker; do not silently retrain or
   treat absence as a model failure. Any authorized reconstruction is declared
   separately and labelled explicitly in its report.
6. Record local verification, remote upload and independent remote restore as
   separate states. Use a new receipt for a new acquisition, preserving original
   sealed receipts and their timestamps/hashes. Link a later delivery receipt
   rather than editing an earlier local-only record. Deployment availability
   says nothing about completeness of the research archive.
7. Restore/inference does not reopen evaluation gates. For a pipeline replay,
   use a fresh output path and format validation without organizer reference
   access. Read saved scores to answer reporting questions. Follow the
   [experiment reference](experiment.md) before new fitting or
   scoring, and the [readout reference](readout.md) for comparisons.

Evidence: [guidance provenance](../../../../docs/research/LESSONS.md), especially the
SoccerNet source correction, reference reconstruction and later deployment restore.

Deliver the exact artifact identity, versioned restore/inference commands,
verification performed and any access blocker. Keep raw-data availability,
checkpoint availability and reconstruction status separate.
