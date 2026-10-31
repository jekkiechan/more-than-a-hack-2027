import hashlib,json
from pathlib import Path
import numpy as np
from experiments import current_baseline as baseline, supervised
from experiments import promote_soccernet_baseline as promotion
from experiments.eval_set.protocol import sha256

manifest_path=promotion.STAGED
manifest=json.loads(manifest_path.read_text())
inputs=baseline.load_inputs(promotion.INPUTS)
models=baseline.load_models(manifest_path=manifest_path)
probability=baseline.predict_probabilities(inputs,models,manifest)
with np.load(promotion.OUT/'competition_probability.npz',allow_pickle=False) as saved:
    assert np.array_equal(probability,saved['probability']), 'Packaged probability drift'
    prediction=supervised.decode(probability,supervised.model_periods(inputs),float(inputs['fps']),baseline.lt.COST)
    prediction[np.asarray(inputs['period'])<=0]='X'
    assert np.array_equal(prediction,saved['prediction']), 'Packaged decoded prediction drift'
assert sha256(promotion.CSV)==promotion.EXPECTED_CSV, 'Actual CLI CSV bytes differ'
result=dict(package_version=promotion.PACKAGE,baseline_manifest_sha256=sha256(manifest_path),
    input_sha256=sha256(promotion.INPUTS),models_sha256=manifest['models_sha256'],
    probability_exact=True,decoded_prediction_exact=True,csv_exact=True,
    candidate_csv_sha256=sha256(promotion.CSV),rows=len(prediction),
    probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
    verifier_source_sha256=sha256(Path(__file__)),organizer_reference_accessed=False,
    organizer_scored=False,model_training_performed=False)
promotion.preserve(promotion.DIRECTORY/'probability_replay.json',promotion.encode(result))
print('PASS',result['rows'],'exact probabilities/predictions/CSV; no reference access')
