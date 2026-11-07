"""Verify immutable selection and completed audit bindings before explicit adoption."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import promote_spell_frontier as promotion
from experiments.eval_set.protocol import sha256

OUT=ROOT/'experiments/results/baseline_spell_frontier_pre_activation.json'


def read(path):return json.loads(Path(path).read_text())


def check():
    manifest=promotion.stage()
    replay_path=ROOT/'experiments/results/baseline_spell_frontier_adapter_replay.json'
    replay=read(replay_path)
    if sha256(promotion.STAGED)!=replay['manifest_sha256'] or not replay['exact_replay']:
        raise ValueError('Staged manifest differs from the exactly replayed package')
    if sha256(promotion.SELECTION)!=manifest['evidence']['selection_sha256']:
        raise ValueError('Frozen selection changed')
    if read(promotion.SELECTION)['pipeline_id']!='base__ce0.5_cal0.375_ad0.125_arithmetic_ha3':
        raise ValueError('Only the user-approved selection may be promoted')
    if sha256(promotion.PUBLIC)!=manifest['organizer_public_diagnostic']['report_sha256']:
        raise ValueError('Saved public report changed')
    if sha256(promotion.SEAL)!='9f35ef18d8c892c86b68db5edc8909356f1f18f51990fbf1263dbae4c2fffc77':
        raise ValueError('Public prediction seal changed')
    if sha256(promotion.GATE)!='6e38e4bb52d6dbb470a8cc8f7c6f19c3a9356151b83c9a8385ffb5460bbb852e':
        raise ValueError('Closed public gate changed')
    seal=read(promotion.SEAL)
    for rel,digest in seal['development_result_bindings'].items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Frozen development evidence changed: '+rel)
    for rel,digest in manifest['source_sha256'].items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Staged inference source changed: '+rel)
    if promotion.ACTIVE.read_bytes()!=promotion.PREVIOUS.read_bytes() or promotion.POLICY.read_bytes()!=promotion.PREVIOUS_POLICY.read_bytes():
        raise ValueError('Predecessor changed before activation')
    paths=[promotion.STAGED,promotion.SELECTION,promotion.PUBLIC,promotion.SEAL,promotion.GATE,replay_path]
    for stem in ('v2_extra_data_v1','v2_extra_data_mix_v1','spell_frontier_gar_v1'):
        result_path=ROOT/f'experiments/results/{stem}.json'
        audit_path=ROOT/f'experiments/results/{stem}_verification.json'
        audit=read(audit_path)
        if not audit['passed'] or audit['result_sha256']!=sha256(result_path):
            raise ValueError('Missing or stale independent audit: '+stem)
        paths.extend((result_path,audit_path))
    execution_path=ROOT/'experiments/results/spell_frontier_gar_v1_execution_verification.json'
    execution=read(execution_path)
    if not execution['passed'] or execution['package_version']!=promotion.PACKAGE:
        raise ValueError('GAR execution did not verify this exact package')
    for rel,digest in execution['bindings_sha256'].items():
        if sha256(ROOT/rel)!=digest:raise ValueError('GAR execution binding changed: '+rel)
    paths.extend((execution_path,Path(__file__)))
    value=dict(package_version=promotion.PACKAGE,passed=True,
        source_revision_before_activation=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        active_predecessor_sha256=sha256(promotion.ACTIVE),
        final_normal_cli_replay_required=True,organizer_reference_access=False,public_scoring=False,
        authorization='Existing explicit user approval of the frozen frontier package; follow-up experiments do not alter that choice.')
    with OUT.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    return {'passed':True,'package_version':promotion.PACKAGE}

if __name__=='__main__':print(check())
