"""Validate float32 categorical sums at the standard 1e-6 tolerance.

The original predictor's 2e-7 guard can reject sums that differ by two float32
rounding units. This wrapper changes only that final validation tolerance. It
preserves the exact returned arrays, original P(D), weights, and loss budgets.
"""
from __future__ import annotations
import ast,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import soccernet_extension as study, soccernet_finetune as ft


def amend():
    study.check()
    source=Path(ft.__file__).read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    original=ast.get_source_segment(source,node)
    if original.count('atol=2e-7')!=1:raise ValueError('Expected one categorical sum guard')
    executed=original.replace('atol=2e-7','atol=1e-6')
    spec=dict(study_id='soccernet_extension_v1',declaration_sha256=study.sha(study.OUT/'predeclared.json'),
        frozen_predictor_sha256=study.sha(ft.__file__),runtime_source_sha256=study.sha(__file__),
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        executed_function_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        correction='Categorical probability-sum assertion atol2e-7 to1e-6 for float32 rounding; exact probability values unchanged.',
        model_changes=False,probability_changes=False,decoder_changes=False,metric_changes=False,
        selection_threshold_changes=False,organizer_reference_access=False,
        comparison_outcomes_not_inspected_before_correction=True)
    study.write_once(study.OUT/'probability_runtime_amendment.json',spec)
    path=study.OUT/'executed_predict_function.py'
    if path.exists():
        if path.read_text()!=executed+'\n':raise ValueError('Executed predictor source changed')
    else:path.write_text(executed+'\n')
    exec(compile(executed,'<soccernet-probability-validation-amendment>','exec'),ft.__dict__)
