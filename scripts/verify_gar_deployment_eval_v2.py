"""Replay official scalar metrics from saved external predictions; no inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import gar_deployment_eval_v2 as study
from experiments import gar_deployment_eval_v2_execution as execution
from experiments import metrics


def verify():
    declaration = execution.check()
    report = study.read(study.RESULT)
    cells, max_error = 0, 0.
    iou_rows, class_rows = {}, {}
    for split in study.SPLITS:
        observed = report['splits'][split]
        if (observed['games'],observed['records']) != study.previous.EXPECTED[split]:
            raise ValueError('Prepared cohort count differs from the frozen GAR inventory')
        iou_rows[split] = {}
        class_rows[split] = {}
        for game in declaration['match_ids'][split]:
            path = study.OUT/'games'/f'{split}__{game}.json'
            receipt = study.read(path)
            native_path = path.with_suffix('.npz')
            if study.digest(native_path) != receipt['prediction_sha256']:
                raise ValueError('Native output hash mismatch')
            if receipt['execution_amendment_sha256'] != study.digest(execution.AMENDMENT):
                raise ValueError('Execution amendment mismatch')
            with np.load(native_path,allow_pickle=False) as data:
                labels,period,stamps,fps = data['labels'].astype(str),data['period'],data['timestamps'],float(data['fps'])
                for view in study.VIEWS:
                    for arm in study.MANIFESTS:
                        for mode in study.MODES:
                            pred = data[f'{view}__{arm}__{mode}'].astype(str)
                            official = metrics.evaluate_tracking(labels,pred,period=period,timestamps=stamps,fps=fps)
                            saved = receipt['scores'][view][arm][mode]
                            actual = dict(spell_f1=official['spell_f1_mean'],macro_f1=official['macro_f1'],
                                d_f1=official['per_class']['D']['f1'],d_precision=official['per_class']['D']['precision'],
                                d_recall=official['per_class']['D']['recall'],graded_frames=official['n_graded'])
                            for metric,value in actual.items():
                                error = abs(float(value)-float(saved[metric]))
                                max_error = max(max_error,error)
                                if error > 1e-12:
                                    raise ValueError(f'Official metric replay failed: {split}/{game}/{view}/{arm}/{mode}/{metric}')
                            key = f'{arm}/{mode}/{"clean" if view == "clean" else "v2"}'
                            iou_rows[split].setdefault(key,{}).setdefault(game,[]).append({
                                str(iou):float(x['f1']) for iou,x in official['spell_f1'].items()})
                            class_rows[split].setdefault(key,{}).setdefault(game,[]).append({
                                f'{label}/{name}':float(value) for label,row in official['per_class'].items()
                                for name,value in row.items() if isinstance(value,(int,float,np.number))})
                            cells += 1
            print('VERIFIED',split,game,flush=True)
        # Verify result aggregation independently using saved per-game scalar cells.
        for view,subviews in [('clean',('clean',)),('v2',study.VIEWS[1:])]:
            for arm in study.MANIFESTS:
                for mode in study.MODES:
                    rows = [study.read(study.OUT/'games'/f'{split}__{g}.json') for g in declaration['match_ids'][split]]
                    for metric in ('spell_f1','macro_f1','d_f1'):
                        value = sum(sum(r['scores'][v][arm][mode][metric] for v in subviews)/len(subviews) for r in rows)/len(rows)
                        expected = report['splits'][split]['scores'][view][arm][mode]['mean'][metric]
                        if abs(value-expected)>1e-12:
                            raise ValueError('Equal-game aggregation failed')
    iou_means = {}
    for split,rows in iou_rows.items():
        iou_means[split] = {key:{iou:float(np.mean([np.mean([x[iou] for x in reps]) for reps in games.values()]))
            for iou in next(iter(games.values()))[0]} for key,games in rows.items()}
    class_means={}
    for split,rows in class_rows.items():
        class_means[split]={key:{name:float(np.mean([np.mean([x[name] for x in reps]) for reps in games.values()]))
            for name in next(iter(games.values()))[0]} for key,games in rows.items()}
    result = dict(study=study.STUDY,verification='official metric replay and independent equal-game aggregation',
        cells_verified=cells,max_scalar_error=max_error,report_sha256=study.digest(study.RESULT),
        execution_amendment_sha256=study.digest(execution.AMENDMENT),
        mean_spell_f1_by_iou=iou_means,mean_per_class=class_means,organizer_reference_access=False,public_scoring=False,
        inference=False,fitting=False,promotion=False)
    study.write_once(study.ROOT/'experiments/results/gar_deployment_eval_v2_verification.json',result)


if __name__=='__main__':
    verify()
