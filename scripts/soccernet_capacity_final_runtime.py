"""Match each baseline parity row to the same archived ensemble member.

The frozen final evaluation compares all three baseline member rows to its
ensemble row. This adapter fixes only that historical lookup key. The existing
strict numeric/IoU comparisons, inference, metrics and choices are unchanged.
"""
from __future__ import annotations
import ast,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import soccernet_capacity_runtime as reader
study=reader.study


def install():
    reader.install()
    text=Path(study.__file__).read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_final')
    original=ast.get_source_segment(text,node)
    before='replicate, "baseline", "ensemble"'
    after='replicate, "baseline", str(seed_name)'
    if original.count(before)!=1:raise ValueError('Expected exactly one final baseline parity key')
    executed=original.replace(before,after)
    value=dict(study_id=study.STUDY_ID,
        declaration_sha256=study.sha256(study.OUT/'predeclared.json'),
        frozen_driver_sha256=study.sha256(Path(study.__file__)),
        runtime_source_sha256=study.sha256(Path(__file__)),
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        executed_function_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        correction='Historical baseline parity lookup uses the same member (2026, 2027, ensemble), rather than comparing individual seeds to the ensemble.',
        weights_changed=False,probabilities_changed=False,metrics_changed=False,
        numeric_tolerances_changed=False,selection_changed=False,
        correction_before_final_comparison_results=True,organizer_reference_access=False)
    study.write_once(study.OUT/'final_parity_key_amendment.json',value)
    path=study.OUT/'executed_final_eval_function.py'
    if path.exists():
        if path.read_text()!=executed+'\n':raise ValueError('Executed final evaluator source changed')
    else:path.write_text(executed+'\n')
    exec(compile(executed,'<soccernet-final-parity-key-amendment>','exec'),study.__dict__)


def main():
    install();study.main()

if __name__=='__main__':main()
