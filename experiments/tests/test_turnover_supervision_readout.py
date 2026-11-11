import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import turnover_supervision_readout as readout


class TurnoverSupervisionReadoutTests(unittest.TestCase):
    def test_d_and_x_masks_are_exact(self):
        baseline = np.asarray(list('HHDDXXAA'))
        candidate = np.asarray(list('AADDXXHH'))
        readout._check_dx(candidate, baseline, 'test')

    def test_d_mask_change_fails(self):
        baseline = np.asarray(list('HHDDXXAA'))
        candidate = np.asarray(list('HHDXXXAA'))
        with self.assertRaisesRegex(ValueError, 'D mask changed'):
            readout._check_dx(candidate, baseline, 'test')

    def test_x_mask_change_fails(self):
        baseline = np.asarray(list('HHDDXXAA'))
        candidate = np.asarray(list('HHDDXHAA'))
        with self.assertRaisesRegex(ValueError, 'X mask changed'):
            readout._check_dx(candidate, baseline, 'test')

    def test_metric_tolerance(self):
        readout._close(0.5, 0.5 + 1e-11, 'within tolerance')
        with self.assertRaises(ValueError):
            readout._close(0.5, 0.5001, 'outside')

    def test_training_cohort_declaration_is_loaded_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.json'
            path.write_text(json.dumps({'study_id': readout.run.STUDY,
                                        'cohorts': {'leave_out_metrica': ['m1']}}))
            with patch.object(readout.train, 'TRAIN_DECL', path):
                declaration = readout._load_training_declaration()
        self.assertEqual(declaration['cohorts']['leave_out_metrica'], ['m1'])


if __name__ == '__main__':
    unittest.main()
