# SoccerNet-GAR data access

The user confirmed on 30 October 2026 that SoccerNet data should be recovered
from Hugging Face. Check this source before asking for a local dataset archive.
The [verified restore receipt](results/soccernet_huggingface_restore.json)
records all seven TRAIN files recovered into this workspace on that date.

- Dataset: [OpenSportsLab/SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR).
- Tracking branch: [`tracking-full`](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR/tree/tracking-full).
- Baseline revision: `32c840210b7d08d06976e7f9457dfff79fb2da99`.
- Training inputs: `train/metadata.parquet`, `train/shard_manifest.parquet`, and
  `train/shards/shard-000000.tar` through `shard-000004.tar`.
- Exact hashes: `raw_source_sha256.file_sha256` in the frozen
  [capacity declaration](results/soccernet_capacity_declaration.json).
- Authentication: configured `HF_TOKEN`; use the normal Hugging Face client
  and authorized dataset access. Never print the token or commit credentials.
- Local destination: ignored `data/independent/soccernet_gar/tracking-full/`.
  Remote filenames omit the local `tracking-full/` prefix because it is the
  branch name. Download only TRAIN files for the declared training study.

For example, after installing `huggingface_hub`, the standard client can fetch
an individual file without an interactive login:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="OpenSportsLab/SoccerNet-GAR",
    repo_type="dataset",
    revision="32c840210b7d08d06976e7f9457dfff79fb2da99",
    filename="train/metadata.parquet",
    local_dir="data/independent/soccernet_gar/tracking-full",
)
```

Verify every downloaded file's SHA-256 against the frozen declaration before
preparing records. The full seven-file TRAIN source is 5,037,010,355 bytes.
Create a fresh local acquisition receipt with dataset, branch, revision, and
per-file path/size/hash fields. A new download timestamp changes that receipt's
hash; do not claim it reproduces the original acquisition receipt byte-for-byte
or rewrite historical declarations to accept it.

Hugging Face supplies the raw dataset. Our trained width-96 heads, merged
feature caches, saved predictions and experiment inventory are separate
artifacts. Missing local model artifacts do not mean the dataset is unavailable.
Restore those artifacts separately or explicitly document a reproducibility
rebuild; do not silently substitute retraining for a verified checkpoint restore.

The original promoted deployment weights are now restorable from the private
cache using the [baseline restore instructions](BASELINE.md#private-cache-restore)
and [verified remote receipt](results/baseline_soccernet_capacity_remote_artifacts.json).
This package is separate from the follow-up study's reconstructed reference.
