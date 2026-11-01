"""Official replay of the 90 fixed original-provider metric cells."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import phase_average_guard_v1 as study
from experiments import metrics


def verify():
    declaration=study.check()
    result=json.loads(study.RESULT.read_text())
    classes,ious={},{}
    cells,max_error=0,0.
    def collect(dest,key,provider,match,value):
        dest.setdefault(key,{}).setdefault(provider,{}).setdefault(match,[]).append(float(value))
    for item in declaration['cells']:
        path=study.OUT/'cells'/f"{item['match']}__{item['view']}.json"
        receipt=json.loads(path.read_text())
        native_path=path.with_suffix('.npz')
        if study.common.digest(native_path)!=receipt['prediction_sha256']:raise ValueError('Native hash mismatch')
        with np.load(native_path,allow_pickle=False) as data:
            labels=data['labels'].astype(str)
            for mode in declaration['modes']:
                for member in declaration['members']:
                    prediction=data[f'{mode}__{member}'].astype(str)
                    measured=metrics.evaluate_tracking(labels,prediction,period=data['period'],
                        timestamps=data['timestamps'],fps=float(data['fps']))
                    saved=receipt['scores'][f'{mode}/{member}']
                    replay=dict(spell_f1=measured['spell_f1_mean'],macro_f1=measured['macro_f1'],
                        d_f1=measured['per_class']['D']['f1'],d_precision=measured['per_class']['D']['precision'],
                        d_recall=measured['per_class']['D']['recall'],graded_frames=measured['n_graded'])
                    for name,value in replay.items():
                        error=abs(float(value)-saved[name]);max_error=max(max_error,error)
                        if error>1e-12:raise ValueError('Official scalar replay failed')
                    view='clean' if item['view']=='clean' else 'feed_like'
                    prefix=f'{member}/{mode}/{view}'
                    for label,row in measured['per_class'].items():
                        for name,value in row.items():
                            if isinstance(value,(int,float,np.number)):
                                collect(classes,f'{prefix}/{label}/{name}',item['provider'],item['match'],value)
                    for iou,row in measured['spell_f1'].items():
                        for name,value in row.items():
                            if isinstance(value,(int,float,np.number)):
                                collect(ious,f'{prefix}/{iou}/{name}',item['provider'],item['match'],value)
                    cells+=1
        print('VERIFIED',item['match'],item['view'],flush=True)
    def reduce(rows):
        result={}
        for key,providers in rows.items():
            means={p:float(np.mean([np.mean(v) for v in matches.values()])) for p,matches in providers.items()}
            result[key]=dict(equal_provider_mean=float(np.mean(list(means.values()))),providers=means)
        return result
    payload=dict(study='phase_average_guard_v1',score_cells_verified=cells,max_scalar_error=max_error,
        report_sha256=study.common.digest(study.RESULT),mean_per_class=reduce(classes),
        mean_spell_by_iou=reduce(ious),organizer_reference_access=False,public_scoring=False,
        inference=False,fitting=False,promotion=False)
    study.common.write_once(study.ROOT/'experiments/results/phase_average_guard_v1_verification.json',payload)


if __name__=='__main__':
    verify()
