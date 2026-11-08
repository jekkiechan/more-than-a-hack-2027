"""The default CLI must preserve the frontier adapter's already-decoded D mask."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from experiments import current_baseline as cli
from experiments import spell_frontier_baseline as adapter

class FrontierCliDispatch(unittest.TestCase):
    def test_cli_uses_labels_without_a_second_joint_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest_path=root/'manifest.json';output=root/'candidate.csv'
            manifest={'model':'spell_frontier_fixture','models_sha256':{},
                      'architecture':{'kind':adapter.KIND}}
            manifest_path.write_text(json.dumps(manifest))
            inputs={'ball':np.zeros((4,2)),'period':np.asarray([0,1,1,1])}
            models={'fixture':object()};labels=np.asarray(['H','H','D','A'])
            with mock.patch('sys.argv',['predict','predict','--manifest',str(manifest_path),'--output',str(output)]), \
                 mock.patch.object(cli,'load_inputs',return_value=inputs), \
                 mock.patch.object(cli,'load_models',return_value=models), \
                 mock.patch.object(cli,'sha256',return_value='fixture-hash'), \
                 mock.patch.object(adapter,'predict_labels',return_value=labels) as predict, \
                 mock.patch.object(cli,'predict_probabilities',side_effect=AssertionError('Do not decode fixed D again')), \
                 mock.patch.object(cli.challenge,'write_submission') as write:
                cli.main()
            predict.assert_called_once_with(inputs,models,manifest)
            np.testing.assert_array_equal(write.call_args.args[0],['X','H','D','A'])
            sidecar=json.loads(output.with_suffix('.json').read_text())
            self.assertFalse(sidecar['target_labels_read'])
            self.assertFalse(sidecar['organizer_scored'])

if __name__=='__main__':unittest.main()
