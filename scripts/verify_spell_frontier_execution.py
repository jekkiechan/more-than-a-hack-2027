"""Bind GAR runtime amendments, actual seeds and final audit to the staged package."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import spell_frontier_gar_capture_runtime as runtime
study=runtime.study


def verify():
    runtime.check()
    declaration=study.check_declaration()
    report=study.read(study.RESULT)
    audit_path=ROOT/'experiments/results/spell_frontier_gar_v1_verification.json'
    audit=study.read(audit_path)
    if not audit['passed'] or audit['result_sha256']!=study.digest(study.RESULT):
        raise ValueError('Final GAR result has not passed independent audit')
    erratum_path=ROOT/'experiments/results/spell_frontier_gar_v1_seed_description_erratum.json'
    erratum=study.read(erratum_path)
    if erratum['declaration_sha256']!=study.digest(study.DECLARATION):raise ValueError('Seed erratum declaration changed')
    for rel,digest in erratum['source_sha256'].items():
        if study.digest(ROOT/rel)!=digest:raise ValueError('Seed path source changed')
    cells=0;game_hashes={}
    for split,games in declaration['game_ids'].items():
        for game in games:
            path=study.GAMES/f'{split}__{game}.json';row=study.read(path)
            if row['corruption_seeds']!=erratum['expected_seeds'][f'{split}/{game}']:
                raise ValueError('Actual corruption seed changed')
            if row['declaration_sha256']!=study.digest(study.DECLARATION):raise ValueError('Game declaration changed')
            game_hashes[f'{split}/{game}']=study.digest(path)
            cells+=len(row['views'])
    if cells!=95 or len(game_hashes)!=19:raise ValueError('Incomplete GAR cohort')
    bindings=[study.RESULT,study.DECLARATION,audit_path,erratum_path,runtime.AMENDMENT,
              runtime.fix.AMENDMENT,runtime.execution.AMENDMENT,runtime.execution.BINDING,
              study.MANIFEST,Path(__file__)]
    receipt=dict(study=study.STUDY,passed=True,game_view_cells=cells,exact_seed_pairs=76,
        package_version=study.read(study.MANIFEST)['package_version'],
        bindings_sha256={str(p.relative_to(ROOT)):study.digest(p) for p in bindings},
        game_sidecar_sha256=game_hashes,organizer_reference_access=False,public_scoring=False)
    study.write_once(ROOT/'experiments/results/spell_frontier_gar_v1_execution_verification.json',receipt)
    return {'passed':True,'game_view_cells':cells,'exact_seed_pairs':76}

if __name__=='__main__':print(verify())
