"""Fixed, user-requested competition diagnostic after external development.

Training/inference commands never open organizer reference labels. The separate
check command consumes one exclusive gate after every prediction is sealed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np

from experiments import (challenge, current_baseline, corruption_experiment as ce,
                         dead_transfer as dt, factorized_had, joint_lgbm,
                         joint_lgbm_experiment as study, learned_temporal as lt,
                         light_public_check, live_gap_experiment as old,
                         possession_geometry, supervised)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/temporal_competition_v1'
PUBLIC = ROOT/'experiments/results/temporal_competition_check.json'
INPUT = ROOT/'data/challenge/inputs.npz'
INPUT_SHA = '7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'
MODELS = ('tree_geometry_63', 'tree_full_127', 'gru_control_2026', 'gru_slot_dropout_2026')
RECIPES = {
    'dropout_gru': {'gru_slot_dropout_2026': 1.0},
    'full127_lgbm': {'tree_full_127': 1.0},
    'clean_leader_blend': {'tree_geometry_63': 0.5, 'gru_control_2026': 0.5},
    'robust_blend': {'tree_full_127': 0.5, 'gru_slot_dropout_2026': 0.5},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprints():
    from experiments import temporal_competition_fit as fitting
    paths = (Path(__file__), Path(fitting.__file__), Path(challenge.__file__),
             Path(current_baseline.__file__), Path(light_public_check.__file__),
             ROOT/'evaluation/possession_metrics.py', ROOT/'games.py')
    return dict(study.fingerprints(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths})


def verify_files(files, root=ROOT):
    for name, expected in files.items():
        if sha256(root/name) != expected:
            raise ValueError('Frozen file changed: '+name)


def require_open(out):
    if (out/'public_check_started.json').exists() or (out/'report.json').exists() or PUBLIC.exists():
        raise ValueError('Competition diagnostic gate is closed; do not repeat or refit')


def check_spec(out):
    spec = json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprints():
        raise ValueError('Frozen diagnostic source changed')
    verify_files(spec['files_sha256'])
    if spec['recipes'] != RECIPES or spec['models'] != list(MODELS):
        raise ValueError('Frozen candidate set changed')
    return spec


def freeze(out):
    from experiments import temporal_competition_fit as fitting
    require_open(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predeclared.json').exists():
        return check_spec(out)
    study.check(study.DEFAULT)
    if sha256(INPUT) != INPUT_SHA:
        raise ValueError('Competition tracking cache differs')
    paths = [INPUT, ROOT/'experiments/baseline_manifest.json',
             ROOT/'experiments/model_decision_policy.json',
             ROOT/'experiments/results/current_baseline_public_check.json',
             ROOT/'experiments/results/light_corruption_public_check.json',
             ROOT/'experiments/results/joint_gru_public_check.json',
             ROOT/'experiments/results/joint_lgbm_development.json',
             ROOT/'experiments/results/gru_slot_robustness_development.json']
    spec = dict(created_utc=now(), user_request='check comp as well',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256=fingerprints(), files_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
                pooled_fit=fitting.declaration(), models=list(MODELS), recipes=RECIPES,
                seed=2026, decoder_switch_cost=3, bootstrap_draws=1000,
                primary='Official spell_f1_mean; macro F1 exact-tie secondary; 70/30 auxiliary only',
                scope='Four fixed diagnostic candidates from completed external development; no organizer-based training, tuning, selection or adoption. Historical public scores already known.',
                deployment='Pooled fits on the same13 external training matches; five development matches remain excluded. Competition model is not a held-out-provider checkpoint or fold ensemble.',
                selection='Full127 standalone; dropoutGRU seed2026; geometry63/control2026 clean-holdout leader; full127/dropout2026 robustness blend. Fixed50/50 full H/A/D probabilities. Seed2026 matches prior competition deployment convention, not public performance.',
                budget='Two pooled640round tree fits and two pooled8epoch GRU fits; four full199319frame predictions, one exclusive four-candidate public check. No extra arms or adaptive changes.',
                prior_gates='All existing study declarations, gates and results remain immutable. This separately requested diagnostic has its own exclusive gate.',
                full_match_score=None, baseline_replaced=False)
    ce.write_json(out/'predeclared.json', spec, exclusive=True)
    print('FROZEN four-candidate competition diagnostic', flush=True)
    return spec


def prepare(out):
    require_open(out); check_spec(out)
    target=out/'competition_features.joblib'
    if target.exists():
        ce.load_model(out,target)
        return
    started=time.monotonic()
    inputs=current_baseline.load_inputs(INPUT)
    if len(inputs['ball']) != 199319:
        raise ValueError('Unexpected competition frame clock')
    placeholder=np.full(199319,'X',dtype='U1')
    record=old.feature_view(inputs,placeholder,'competition','BAR-ATH')['joint']
    extra,names=possession_geometry.features(inputs,include_anchors=True)
    tree=study.augment(record,dict(x=extra[record['indices']],names=names,indices=record['indices']))
    ce.freeze_model(out,target,dict(gru=record,tree=tree,period=supervised.model_periods(inputs),
                    native_in_play=np.asarray(inputs['period'])>0,fps=float(inputs['fps'])),
                    dict(input_sha256=sha256(INPUT),seconds=time.monotonic()-started,reference_labels_read=False))
    print('FEATURES PREPARED without organizer labels',flush=True)


def seal_models(out):
    require_open(out); spec=check_spec(out)
    files={}
    plan=spec['pooled_fit']
    training=[m for ids in plan['training_matches'].values() for m in ids]
    for name in MODELS:
        path=out/(name+'.joblib')
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['sha256'] != sha256(path) or receipt['predeclared_sha256'] != sha256(out/'predeclared.json'):
            raise ValueError('Model provenance differs')
        if (receipt['training_matches'] != training or
                receipt['excluded_evaluation_matches'] != plan['excluded_evaluation_matches'] or
                receipt['seed'] != 2026 or receipt['deployment'] != 'pooled_external_train_only'):
            raise ValueError('Pooled model training cohort or seed differs')
        if name.startswith('tree_'):
            if receipt['config'] != name.removeprefix('tree_') or receipt['boosting_rounds'] != 640:
                raise ValueError('Tree configuration differs')
        elif receipt['arm'] != name.removeprefix('gru_').removesuffix('_2026') or not receipt['player_only']:
            raise ValueError('GRU configuration differs')
        files[str(path.relative_to(ROOT))]=sha256(path)
        files[str(path.with_suffix('.json').relative_to(ROOT))]=sha256(path.with_suffix('.json'))
    target=out/'checkpoints_frozen.json'
    if target.exists():
        if json.loads(target.read_text())['files_sha256'] != files:
            raise ValueError('Sealed model set changed')
    else:
        ce.write_json(target,dict(sealed_utc=now(),files_sha256=files),exclusive=True)


def combine(probabilities, recipe):
    if not recipe:
        raise ValueError('Empty probability blend')
    values=[]
    shape=None
    for name,weight in recipe.items():
        if not np.isfinite(weight) or weight<0:
            raise ValueError('Blend weights must be finite and nonnegative')
        p=np.asarray(probabilities[name])
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):
            raise ValueError('Invalid full H/A/D probability vector')
        if shape is not None and p.shape != shape:
            raise ValueError('Probability frame clocks differ')
        shape=p.shape
        values.append(weight*p)
    if not np.isclose(sum(recipe.values()),1):
        raise ValueError('Blend weights must sum to one')
    return sum(values)


def predict(out):
    require_open(out); check_spec(out); seal_models(out)
    if (out/'prediction_seal.json').exists():
        verify_files(json.loads((out/'prediction_seal.json').read_text())['files_sha256'])
        return
    cache=ce.load_model(out,out/'competition_features.joblib')
    probabilities={}
    for name in MODELS:
        fitted=ce.load_model(out,out/(name+'.joblib'))
        record=cache['tree'] if name.startswith('tree_') else cache['gru']
        sampled=joint_lgbm.predict(record,fitted) if name.startswith('tree_') else factorized_had.predict(record,fitted)
        probabilities[name]=np.column_stack([lt.upsample(sampled[:,k],record,cache['period']) for k in range(3)])
        del fitted
    artifacts={}
    for name,recipe in RECIPES.items():
        probability=combine(probabilities,recipe)
        labels=supervised.decode(probability,cache['period'],cache['fps'],3)
        labels[~cache['native_in_play']]='X'
        csv=out/(name+'.csv'); npz=out/(name+'.npz')
        if csv.exists() or npz.exists():
            raise ValueError('Unsealed prediction already exists; inspect interruption, do not overwrite')
        challenge.write_submission(labels,csv)
        with npz.open('xb') as handle:
            np.savez_compressed(handle,probability=probability,prediction=labels)
        artifacts[name]=dict(csv=csv.name,csv_sha256=sha256(csv),probability_file=npz.name,
                             probability_file_sha256=sha256(npz),frames=len(labels),recipe=recipe)
    paths=[out/'predeclared.json',out/'checkpoints_frozen.json',out/'competition_features.joblib',out/'competition_features.json']
    paths.extend(out/(name+suffix) for name in RECIPES for suffix in ('.csv','.npz'))
    model_files=json.loads((out/'checkpoints_frozen.json').read_text())['files_sha256']
    ce.write_json(out/'prediction_seal.json',dict(sealed_utc=now(),candidates=artifacts,
                  files_sha256=dict(model_files,**{str(p.relative_to(ROOT)):sha256(p) for p in paths}),
                  reference_labels_read=False,full_length_csvs_validated=True),exclusive=True)
    print('ALL FOUR CSVs VALIDATED AND SEALED; no organizer labels read',flush=True)


def start_gate(out, spec, seal):
    require_open(out)
    if set(seal['candidates']) != set(RECIPES):
        raise ValueError('Incomplete fixed candidate set')
    expected=[out/'predeclared.json',out/'checkpoints_frozen.json',
              out/'competition_features.joblib',out/'competition_features.json']
    expected.extend(out/(name+suffix) for name in MODELS for suffix in ('.joblib','.json'))
    expected.extend(out/(name+suffix) for name in RECIPES for suffix in ('.csv','.npz'))
    if set(seal['files_sha256']) != {str(p.relative_to(ROOT)) for p in expected}:
        raise ValueError('Prediction seal artifact coverage differs')
    if seal.get('reference_labels_read') is not False or seal.get('full_length_csvs_validated') is not True:
        raise ValueError('Missing pre-reference validation')
    for name,recipe in RECIPES.items():
        csv=out/(name+'.csv');npz=out/(name+'.npz')
        expected_candidate=dict(csv=csv.name,csv_sha256=seal['files_sha256'][str(csv.relative_to(ROOT))],
                                probability_file=npz.name,probability_file_sha256=seal['files_sha256'][str(npz.relative_to(ROOT))],
                                frames=199319,recipe=recipe)
        if seal['candidates'][name] != expected_candidate:
            raise ValueError('Candidate metadata differs from fixed recipe and artifacts')
    verify_files(seal['files_sha256'])
    checkpoints=json.loads((out/'checkpoints_frozen.json').read_text())['files_sha256']
    expected_models={str((out/(name+suffix)).relative_to(ROOT)) for name in MODELS for suffix in ('.joblib','.json')}
    if set(checkpoints) != expected_models or any(checkpoints[p] != seal['files_sha256'][p] for p in checkpoints):
        raise ValueError('Checkpoint seal coverage differs')
    ce.write_json(out/'public_check_started.json',dict(started_utc=now(),
                  predeclared_sha256=sha256(out/'predeclared.json'),
                  prediction_seal_sha256=sha256(out/'prediction_seal.json'),
                  candidates=list(spec['recipes'])),exclusive=True)


def score(out):
    require_open(out);spec=check_spec(out)
    seal=json.loads((out/'prediction_seal.json').read_text())
    start_gate(out,spec,seal)
    # The exclusive gate is claimed before opening any organizer reference.
    import pandas as pd
    from evaluation import possession_metrics as pm
    prior=json.loads((ROOT/'experiments/results/current_baseline_public_check.json').read_text())
    gt_path=ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    if sha256(gt_path) != prior['organizer_gt_sha256']:
        raise ValueError('Organizer reference changed from historical check')
    gt=pd.read_csv(gt_path)
    if len(gt) != 199319:
        raise ValueError('Unexpected organizer reference frame count')
    codes=pm.label_codes(gt.label.to_numpy())
    results={}
    baseline=json.loads((ROOT/'experiments/results/light_corruption_public_check.json').read_text())['score']
    previous=json.loads((ROOT/'experiments/results/joint_gru_public_check.json').read_text())['score']
    for name in RECIPES:
        predicted=pd.read_csv(out/seal['candidates'][name]['csv'])
        error=pm.validate_submission(predicted,len(gt))
        if error:
            raise ValueError(error)
        metrics=pm.evaluate(codes,pm.label_codes(predicted.label.to_numpy()),bootstrap_n=spec['bootstrap_draws'])
        results[name]={key:metrics[key] for key in light_public_check.SCORE_KEYS}
        results[name]['weighted_70_30']=.7*metrics['spell_f1_mean']+.3*metrics['macro_f1']
        results[name]['spell_delta_vs_baseline']=metrics['spell_f1_mean']-baseline['spell_f1_mean']
        results[name]['spell_delta_vs_previous_gru']=metrics['spell_f1_mean']-previous['spell_f1_mean']
    report=dict(completed_utc=now(),scope=spec['scope'],declaration=spec,prediction_seal=seal,
                scores=results,archived_baseline=baseline,archived_previous_gru=previous,
                organizer_gt_sha256=prior['organizer_gt_sha256'],baseline_scores_reused=True,
                organizer_labels_used_for_training=False,organizer_reference_labels_read=True,
                full_match_score=None,external_selection_changed=False,baseline_replaced=False,
                bootstrap_interval='Official separate90% intervals; not paired significance or full-match confidence')
    ce.write_json(out/'report.json',report,exclusive=True)
    ce.write_json(PUBLIC,report,exclusive=True)
    print(json.dumps({name:round(value['spell_f1_mean'],6) for name,value in results.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','prepare','seal','predict','check'))
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args()
    {'freeze':freeze,'prepare':prepare,'seal':seal_models,'predict':predict,'check':score}[args.action](args.out)


if __name__=='__main__':
    main()
