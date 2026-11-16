"""Download and verify the frozen submission weights (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded archive')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'model_manifest.json').read_text())
    release = json.loads((HERE / 'weights.json').read_text())
    target = args.models_dir or ROOT / manifest['model_directory']
    expected = manifest['models_sha256']
    if release['package_version'] != manifest['package_version']:
        raise ValueError('Release and model package differ')
    if any(release['files'].get(n) != h for n, h in expected.items()):
        raise ValueError('Release and model hashes differ')
    if digest((HERE / 'model_manifest.json').read_bytes()) != release['files']['model_manifest.json']:
        raise ValueError('Frozen manifest changed')
    # Check all conflicts before downloading or writing. Never replace other weights.
    for name, sha in expected.items():
        path = target / name
        if path.exists() and digest(path.read_bytes()) != sha:
            raise ValueError(f'Existing model differs: {path}')
    if all((target / n).is_file() for n in expected):
        print(f'Verified all {len(expected)} model roles: {target}')
        return
    if args.verify_only:
        raise FileNotFoundError('Missing model roles; run without --verify-only')
    if args.archive:
        data = args.archive.read_bytes()
    else:
        print(f'Downloading {release["url"]}', flush=True)
        request = urllib.request.Request(release['url'], headers={'User-Agent': 'mtah-submission-reproduction'})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(release['archive_bytes'] + 1)
    if len(data) != release['archive_bytes'] or digest(data) != release['archive_sha256']:
        raise ValueError('Archive size or SHA-256 mismatch')
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(release['files']) or {m.name for m in members} != set(release['files']):
            raise ValueError('Unexpected archive contents')
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError('Only allowlisted regular files are permitted')
            content = archive.extractfile(member).read()
            if digest(content) != release['files'][member.name]:
                raise ValueError(f'Archive member hash mismatch: {member.name}')
            payload[member.name] = content
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as temporary:
        for name in expected:
            staged = Path(temporary) / name
            staged.write_bytes(payload[name])
        for name in expected:
            staged = Path(temporary) / name
            if not (target / name).exists():
                staged.replace(target / name)
    print(f'Installed and verified all {len(expected)} model roles: {target}')


if __name__ == '__main__':
    main()
