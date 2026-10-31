"""Validate both confirmation seal tables from the frozen capacity runner.

The frozen confirmation writer emits coarse_models and selected_models, while
its generic reader expects models. This adapter validates the unchanged tables
and returns their union to the reader's callers. It changes no model, data,
probability, metric, selection rule, or serialized seal.
"""
from __future__ import annotations
from datetime import datetime,timezone
import inspect
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import soccernet_capacity as study
ORIGINAL_CHECK=study._check_seal


def check_confirmation_seal(name):
    if name!='checkpoints_confirm_frozen.json':return ORIGINAL_CHECK(name)
    value=study.read_json(study.OUT/name)
    if value.get('declaration_sha256')!=study.sha256(study.OUT/'predeclared.json'):
        raise ValueError('Confirmation declaration seal changed')
    if value.get('coarse_seal_sha256')!=study.sha256(study.OUT/'coarse_checkpoints_frozen.json'):
        raise ValueError('Confirmation coarse seal changed')
    tables=[value.get('coarse_models'),value.get('selected_models')]
    if any(not isinstance(table,dict) or not table for table in tables):
        raise ValueError('Confirmation seal requires both original model tables')
    merged={}
    for table in tables:
        for key,expected in table.items():
            if key in merged and merged[key]!=expected:raise ValueError('Conflicting model seals')
            path=study._resolve_key(key)
            if not path.is_file() or study.sha256(path)!=expected:
                raise ValueError('Confirmation sealed model changed: '+key)
            merged[key]=expected
    return dict(value,models=merged)


def install():
    study.check()
    path=study.OUT/'confirmation_reader_amendment.json'
    value=dict(study_id=study.STUDY_ID,
        declaration_sha256=study.sha256(study.OUT/'predeclared.json'),
        frozen_driver_sha256=study.sha256(Path(study.__file__)),
        runtime_source_sha256=study.sha256(Path(__file__)),
        correction='Reader validates unchanged coarse_models and selected_models tables from the frozen confirmation writer; exposes their verified union as models. Stored seals unchanged.',
        model_changes=False,data_changes=False,probability_changes=False,
        decoder_changes=False,metric_changes=False,selection_changes=False,
        correction_identified_before_capacity_scores_inspected=True,
        organizer_reference_access=False)
    study.write_once(path,value)
    study._check_seal=check_confirmation_seal


def main():
    install()
    if len(sys.argv)>1 and sys.argv[1] in ('predict-public','score-public'):
        from experiments import soccernet_capacity_public as public
        return public.predict() if sys.argv[1]=='predict-public' else public.score()
    study.main()

if __name__=='__main__':main()
