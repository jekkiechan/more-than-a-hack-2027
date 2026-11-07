import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts import package_spell_frontier as packager


class SpellFrontierPackageTests(unittest.TestCase):
    def test_archive_hash_inventory_and_member_hashes(self):
        import hashlib
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "fixture.tar.gz"
            data = b"portable model fixture"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("models/test/weights.bin")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            digest = hashlib.sha256(data).hexdigest()
            receipt = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "members": [{"path": "models/test/weights.bin", "size": len(data),
                             "sha256": digest}]}
            self.assertTrue(packager.verify(archive, receipt, extract=True))
            receipt["members"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "member hash"):
                packager.verify(archive, receipt)

    def test_package_requires_activation_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.json"
            with mock.patch.object(packager, "ACTIVE_PROMOTION", missing), \
                 mock.patch.object(packager, "PROMOTION", missing):
                with self.assertRaises(FileNotFoundError):
                    packager._files()


if __name__ == "__main__":
    unittest.main()
