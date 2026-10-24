"""Record the audited pretrained adapter before inference or label scoring."""

import hashlib
import json
from pathlib import Path
import subprocess

from experiments import pathcrf_eval as adapter

CHECKPOINT_SHA = "55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9"
ADAPTER_SHA = "26a6a192d555fdf6ebea0b2d174fac5c670775ea9782f92b5f20d338a2df09d5"


if __name__ == "__main__":
    checkpoint_sha = hashlib.sha256((adapter.UPSTREAM / adapter.CHECKPOINT).read_bytes()).hexdigest()
    adapter_sha = hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "-C", str(adapter.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if (checkpoint_sha, adapter_sha, revision) != (CHECKPOINT_SHA, ADAPTER_SHA, adapter.COMMIT):
        raise RuntimeError("Source or checkpoint differs from the audited, evaluated version")
    config = {
        "checkpoint_commit": adapter.COMMIT, "checkpoint": adapter.CHECKPOINT,
        "checkpoint_sha256": checkpoint_sha, "adapter_sha256": adapter_sha,
        "test_match": "J03WR9", "test_labels_read": False,
        "threads": 2, "batch_size": 4, "window_seconds": 10,
        "stride_seconds": 5, "target_fps": 5,
        "primary_mapping": "originalCRF edge_source team; outsidecarryforward, leadingoutside macro_source team",
        "macro_output": "diagnostic only, not candidate for selection",
        "native_dead_head": False, "uses_true_live_boundaries": False,
        "audit": "independent benchmark_audit review and sixcontract tests passed beforeinference",
    }
    output = adapter.ROOT / "data/experiments/pathcrf/frozen_config.json"
    if output.exists():
        if json.loads(output.read_text()) != config:
            raise RuntimeError("Refusing to replace an existing different experiment")
        print("Existing frozen PathCRF configuration verified")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(config, indent=2) + "\n")
        print("PathCRF frozen before target access:", output)
