"""Offline contracts for public checkpoint installation."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from submission import download_weights as download


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ReleaseInstallation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / 'models'
        manifest = {'package_version': 'fixture', 'model_directory': 'unused',
                    'models_sha256': {'a.joblib': sha(b'checkpoint-a'), 'b.joblib': sha(b'checkpoint-b')}}
        data = json.dumps(manifest).encode()
        (self.root / 'model_manifest.json').write_bytes(data)
        self.payload = {'a.joblib': b'checkpoint-a', 'b.joblib': b'checkpoint-b', 'model_manifest.json': data}
        self.patch = patch.object(download, 'HERE', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.build()

    def build(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, content in self.payload.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        data = stream.getvalue()
        self.archive = self.root / 'weights.tar.gz'
        self.archive.write_bytes(data)
        release = {'package_version': 'fixture', 'archive_bytes': len(data),
                   'archive_sha256': sha(data), 'files': {n: sha(b) for n, b in self.payload.items()}}
        (self.root / 'weights.json').write_text(json.dumps(release))

    def run_install(self, *extra):
        with patch('sys.argv', ['download_weights', '--models-dir', str(self.target),
                                '--archive', str(self.archive), *extra]):
            download.main()

    def test_install_and_offline_idempotent_verification(self):
        self.run_install()
        self.archive.unlink()
        self.run_install('--verify-only')
        self.assertEqual({p.name for p in self.target.iterdir()}, {'a.joblib', 'b.joblib'})
        self.assertEqual((self.target / 'a.joblib').read_bytes(), b'checkpoint-a')

    def test_conflict_stops_before_any_other_model_is_installed(self):
        self.target.mkdir()
        (self.target / 'a.joblib').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'Existing model differs'):
            self.run_install()
        self.assertFalse((self.target / 'b.joblib').exists())

    def test_corrupt_download_never_creates_target(self):
        data = bytearray(self.archive.read_bytes())
        data[-1] ^= 1
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Archive size or SHA-256 mismatch'):
            self.run_install()
        self.assertFalse(self.target.exists())

    def test_path_traversal_rejected_even_with_matching_outer_checksum(self):
        self.payload['../escape'] = b'escape'
        self.build()
        with self.assertRaisesRegex(ValueError, 'allowlisted regular files'):
            self.run_install()
        self.assertFalse((self.root / 'escape').exists())
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
