"""Package new research checkpoints and receipts locally, without remote upload."""
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STUDY='v2_spell_frontier_v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    result=ROOT/f'experiments/results/{STUDY}_artifacts.json'
    archive=ROOT/f'data/experiments/{STUDY}/research_models.tar.gz'
    if result.exists() or archive.exists():raise FileExistsError('Preserve artifact receipt')
    paths=[]
    for study in ('v2_spell_balanced_v1',STUDY):
        paths+=sorted((ROOT/f'data/experiments/{study}/train/models').rglob('*.joblib'))
        paths+=sorted((ROOT/f'data/experiments/{study}/train/models').rglob('*.joblib.json'))
    if not paths:raise ValueError('No research artifacts')
    facts={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        if set(tar.getnames())!=set(facts):raise ValueError('Archive inventory mismatch')
        for name,fact in facts.items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=fact['sha256']:raise ValueError(name)
    value=dict(study_id=STUDY,archive_path=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),
               archive_bytes=archive.stat().st_size,files=facts,verified=True,availability='local only',
               new_remote_upload=False,organizer_reference_access=False)
    with result.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:value[k] for k in ('archive_path','archive_sha256','archive_bytes','verified','availability')}))
if __name__=='__main__':main()
