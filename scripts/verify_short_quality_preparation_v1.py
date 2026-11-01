"""Check new synthetic views without fitting or calculating any scores."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from experiments import short_quality_fresh_v1 as study
p=study.p


def main():
    study.check();receipt=p.read(study.OUT/'preparation.json')
    originals={(c['provider'],c['match']):c for c in p.shared._load_cells(p.read(p.DECLARATION)) if c['member']=='ensemble' and c['view']=='clean'}
    for entry in receipt['files']:
        file=ROOT/entry['path'];assert p.sha(file)==entry['sha256'];data=joblib.load(file)
        key=(p.shared._provider_for_match(entry['match']),entry['match'].removeprefix('skillcorner_had_'))
        before=originals[key];inputs=data['inputs']
        np.testing.assert_array_equal(data['labels'],before['labels'])
        for field in ('period','timestamps','fps'):np.testing.assert_array_equal(inputs[field],before[field])
        np.testing.assert_array_equal(p.supervised.model_periods(inputs),before['model_period'])
        inference=p.old.inference_record(data['record'])
        assert set(inference)=={'x','names','period','indices','native_n','fps'}
        assert np.asarray(inference['x']).shape[1]==204
    result=dict(passed=True,new_views=len(receipt['files']),native_grading_and_timebreaks_exact=True,
        inference_record_has_no_targets=True,scored_cells=0,preparation_sha256=p.sha(study.OUT/'preparation.json'),
        verifier_sha256=p.sha(Path(__file__)),organizer_reference_access=False)
    p.write(p.RESULTS/f'{study.STUDY}_preparation_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
