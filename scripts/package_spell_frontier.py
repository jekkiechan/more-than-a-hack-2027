"""Build and verify a portable, local-only spell-frontier model archive."""
from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "joint_gru_sn_dense_spell_frontier_ensemble_v1"
MODEL_DIR = ROOT / "data/models" / PACKAGE
MANIFEST = MODEL_DIR / "baseline_manifest.json"
POLICY = MODEL_DIR / "model_decision_policy.json"
PROMOTION = MODEL_DIR / "promotion_receipt.json"
ACTIVE_PROMOTION = ROOT / "experiments/results/baseline_spell_frontier_promotion.json"
ACTIVE_MANIFEST = ROOT / "experiments/baseline_manifest.json"
ACTIVE_POLICY = ROOT / "experiments/model_decision_policy.json"
SELECTION = ROOT / "experiments/results/v2_spell_frontier_v1_selection.json"
PUBLIC_RESULT = ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"
GAR_RESULT = ROOT / "experiments/results/spell_frontier_gar_v1.json"
RECEIPT = ROOT / "experiments/results/baseline_spell_frontier_local_artifacts.json"
ARCHIVE = ROOT / "data/models" / f"{PACKAGE}_local.tar.gz"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_inventory():
    # Include tracked and new Python sources, without data or generated outputs.
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    paths = [ROOT / line for line in result.stdout.splitlines()]
    return sorted(path for path in paths if path.is_file())


def _files():
    if not ACTIVE_PROMOTION.is_file() or not PROMOTION.is_file():
        raise FileNotFoundError("Activation receipt is required before local packaging")
    promotion = json.loads(PROMOTION.read_text())
    active = json.loads(ACTIVE_PROMOTION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if (promotion.get("package_version") != PACKAGE or
            active.get("package_version") != PACKAGE or
            promotion.get("baseline_manifest_sha256") != sha(MANIFEST.read_bytes())):
        raise ValueError("Promotion receipt does not bind the active frontier package")
    if (ACTIVE_PROMOTION.read_bytes() != PROMOTION.read_bytes() or
            ACTIVE_MANIFEST.read_bytes() != MANIFEST.read_bytes() or
            ACTIVE_POLICY.read_bytes() != POLICY.read_bytes() or
            promotion.get("policy_sha256") != sha(POLICY.read_bytes()) or
            promotion.get("baseline_manifest_sha256") != sha(ACTIVE_MANIFEST.read_bytes())):
        raise ValueError("Active manifest/policy bytes differ from the promoted package")
    if manifest.get("package_version") != PACKAGE:
        raise ValueError("Unexpected package manifest")
    for rel, digest in manifest["source_sha256"].items():
        if sha((ROOT / rel).read_bytes()) != digest:
            raise ValueError("Promoted source changed: " + rel)
    for path, expected in (
            (SELECTION, manifest['evidence']['selection_sha256']),
            (PUBLIC_RESULT, manifest['organizer_public_diagnostic']['report_sha256']),
            (GAR_RESULT, promotion['gar_report_sha256']),
            (ROOT / 'experiments/current_baseline.py', promotion['dispatcher_sha256'])):
        if sha(path.read_bytes()) != expected:
            raise ValueError('Package provenance changed: ' + str(path))
    gar_audit = json.loads((ROOT / 'experiments/results/spell_frontier_gar_v1_verification.json').read_text())
    if not gar_audit['passed'] or gar_audit['result_sha256'] != sha(GAR_RESULT.read_bytes()):
        raise ValueError('GAR audit does not bind the packaged result')
    files = {MANIFEST: "data/models/" + PACKAGE + "/baseline_manifest.json",
             POLICY: "data/models/" + PACKAGE + "/model_decision_policy.json",
             PROMOTION: "data/models/" + PACKAGE + "/promotion_receipt.json",
             ACTIVE_MANIFEST: "experiments/baseline_manifest.json",
             ACTIVE_POLICY: "experiments/model_decision_policy.json",
             ACTIVE_PROMOTION: "experiments/results/baseline_spell_frontier_promotion.json",
             SELECTION: "experiments/results/v2_spell_frontier_v1_selection.json",
             PUBLIC_RESULT: "experiments/results/v2_spell_frontier_v1_public_result.json",
             GAR_RESULT: "experiments/results/spell_frontier_gar_v1.json",
             ROOT / "experiments/results/spell_frontier_gar_v1_verification.json":
                 "experiments/results/spell_frontier_gar_v1_verification.json"}
    requirements = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*requirements*"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    for rel in requirements.stdout.splitlines():
        path = ROOT / rel
        if path.is_file():
            files[path] = rel
    model_hashes = manifest.get("models_sha256", {})
    if len(model_hashes) != 10:
        raise ValueError("Expected exactly ten deployed models")
    for name, digest in model_hashes.items():
        path = MODEL_DIR / name
        if sha(path.read_bytes()) != digest:
            raise ValueError(f"Model hash mismatch: {name}")
        files[path] = f"data/models/{PACKAGE}/{name}"
    for path in _source_inventory():
        files[path] = path.relative_to(ROOT).as_posix()
    return files


def package():
    if RECEIPT.exists() or ARCHIVE.exists():
        raise FileExistsError("Refusing to overwrite local package artifacts")
    files = _files()
    entries = []
    with ARCHIVE.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path, member in sorted(files.items(), key=lambda item: item[1]):
            data = path.read_bytes()
            info = tarfile.TarInfo(member)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            import io
            tar.addfile(info, io.BytesIO(data))
            entries.append({"path": member, "size": len(data), "sha256": sha(data)})
    receipt = {"package_version": PACKAGE, "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha(ARCHIVE.read_bytes()), "member_count": len(entries),
        "members": entries, "local_only": True, "uploaded": False,
        "organizer_inputs_or_outputs_included": False,
        "fresh_empty_directory_extraction_verified": True,
        "isolated_restore_import_and_ten_model_load_verified": True,
        "source_sha256": sha(Path(__file__).read_bytes())}
    verify(ARCHIVE, receipt, extract=True, deployment_smoke=True)
    with RECEIPT.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify(archive=ARCHIVE, receipt=None, *, extract=False, deployment_smoke=False):
    if receipt is None:
        receipt = json.loads(RECEIPT.read_text())
    if sha(Path(archive).read_bytes()) != receipt["archive_sha256"]:
        raise ValueError("Archive hash differs from receipt")
    expected = {row["path"]: row for row in receipt["members"]}
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        if names != set(expected):
            raise ValueError("Archive member inventory differs from receipt")
        for name, row in expected.items():
            data = tar.extractfile(name).read()
            if len(data) != row["size"] or sha(data) != row["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {name}")
        if extract:
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp)
                tar.extractall(target, filter="data")
                for name, row in expected.items():
                    extracted = target / name
                    data = extracted.read_bytes()
                    if len(data) != row["size"] or sha(data) != row["sha256"]:
                        raise ValueError(f"Fresh extraction mismatch: {name}")
                if deployment_smoke:
                    env = dict(os.environ, PYTHONPATH=str(target),
                               MTAH_DATA_DIR=str(target / 'unused-data-cache'),
                               MTAH_DT_DIR=str(target / 'unused-dt-cache'),
                               MTAH_SQD_DIR=str(target / 'unused-sqd-cache'),
                               MTAH_RF_ARCHIVE_DIR=str(target / 'unused-rf-cache'))
                    subprocess.run([sys.executable, '-c',
                        "import json; from pathlib import Path; "
                        "from experiments.current_baseline import load_models; "
                        "from experiments.spell_frontier_baseline import validate; "
                        "validate(load_models(),json.loads(Path('experiments/baseline_manifest.json').read_text()))"],
                        cwd=target, env=env, check=True, capture_output=True, text=True)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "verify"))
    args = parser.parse_args()
    result = package() if args.action == "package" else verify()
    print(json.dumps({"passed": True, "archive": str(ARCHIVE.relative_to(ROOT)),
                      "member_count": result.get("member_count") if isinstance(result, dict) else None}))
