"""Publish aggregate scores, omitting per-transition frame locations."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/experiments/benchmark"
OUTPUT = ROOT / "experiments/results"


def aggregates(value):
    if isinstance(value, dict):
        return {key: aggregates(item) for key, item in value.items()
                if key not in ("pairs", "missed", "spurious")}
    if isinstance(value, list):
        return [aggregates(item) for item in value]
    return value


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source, target in (("test.json", "primary_test.json"),
                           ("frozen_manifest.json", "primary_manifest.json")):
        raw = (SOURCE / source).read_bytes()
        result = aggregates(json.loads(raw))
        result["export"] = {
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "note": "Per-transition frame locations omitted; all aggregate metrics preserved.",
        }
        (OUTPUT / target).write_text(json.dumps(result, indent=2) + "\n")
