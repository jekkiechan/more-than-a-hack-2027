"""Keep frame-level boundary diagnostics private; publish aggregate evidence only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import lgbm_spell_retention_v1 as study

if __name__ == '__main__':
    public_result = study.RESULT
    if public_result.exists():
        raise FileExistsError(public_result)
    amendment = study.RES / f'{study.STUDY}_output_policy.json'
    study.write(amendment, {
        'study': study.STUDY,
        'declaration_sha256': study.sha(study.DECL),
        'entrypoint_sha256': study.sha(Path(__file__)),
        'timing': 'Before candidate evaluation; no candidate scores inspected',
        'reason': 'Per-run frame indices are reconstructive outputs and must remain private under AGENTS.md',
        'change': 'Redirect frozen runner result to ignored storage; summarize diagnostic lists as counts in public evidence',
        'unchanged': ['source hashes', 'models', 'arms', 'decoding', 'metrics', 'guards', 'ranking', 'fit budget'],
    })
    study.RESULT = study.OUT / 'result_private.json'
    study.evaluate()
    result = study.read(study.RESULT)
    result['topology_diagnostics'] = {
        key: {name: {field: len(value) if isinstance(value, list) else value
                    for field, value in diag.items()}
              for name, diag in arms.items()}
        for key, arms in result['topology_diagnostics'].items()
    }
    result['private_result_sha256'] = study.sha(study.RESULT)
    result['output_policy_sha256'] = study.sha(amendment)
    study.write(public_result, result)
