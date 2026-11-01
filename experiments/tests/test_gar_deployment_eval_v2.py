"""Match weighting and label-free probability contracts for deployment evaluation."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments import gar_deployment_eval_v2 as study
from experiments import gar_deployment_eval_v2_execution as execution


class DeploymentEvaluationTests(unittest.TestCase):
    def test_package_normalization_order_is_preserved(self):
        seeds = np.repeat(np.array([[[90., 9., 1.]], [[.01, .98, .01]]]), 100, axis=1)
        period = np.ones(100, dtype=int)
        original = execution.decode(seeds, 'original_full', period, 25.)
        calibrated = execution.decode(seeds, 'calibrated', period, 25.)
        self.assertTrue(np.all(original == 'H'))
        self.assertTrue(np.all(calibrated == 'A'))

    def test_replica_count_does_not_change_game_weight(self):
        rows = []
        for game, score in [('a', .2), ('b', .8)]:
            scores = {v: {arm: {mode: {'spell_f1': score, 'macro_f1': .7, 'd_f1': .6}
                        for mode in study.MODES} for arm in study.MANIFESTS} for v in study.VIEWS}
            rows.append({'game': game, 'scores': scores})
        out = study.aggregate(rows)
        self.assertAlmostEqual(out['v2']['original_full']['phase_avg']['mean']['spell_f1'], .5)
        self.assertEqual(set(out['v2']['original_full']['phase_avg']['per_game']), {'a', 'b'})

    def test_two_pooled_members_are_averaged_before_decoding(self):
        models = {'dense_2026': 0, 'teacher_2026': None, 'dense_2027': 1, 'teacher_2027': None}
        arrays = [np.array([[.2, .3, .5]]), np.array([[.6, .1, .3]])]
        with patch.object(study, 'checkpoint_package', return_value=({}, models)), \
             patch.object(study.old, 'inference_record', side_effect=lambda r: r), \
             patch.object(study.sn, 'predict', side_effect=lambda r, m, t: arrays[m]), \
             patch.object(study.old, 'upsample', side_effect=lambda p, r, i: p):
            result = study.native_probability({'names': ['x'], 'x': np.zeros((1, 1))}, {}, 'original_full')
        np.testing.assert_allclose(result, [[.4, .2, .4]])

    def test_bad_probabilities_fail_closed(self):
        models = {'dense_2026': 0, 'teacher_2026': None, 'dense_2027': 1, 'teacher_2027': None}
        with patch.object(study, 'checkpoint_package', return_value=({}, models)), \
             patch.object(study.old, 'inference_record', side_effect=lambda r: r), \
             patch.object(study.sn, 'predict', return_value=np.array([[np.nan, 0., 1.]])), \
             patch.object(study.old, 'upsample', side_effect=lambda p, r, i: p):
            with self.assertRaisesRegex(ValueError, 'Invalid probability'):
                study.native_probability({'names': ['x']}, {}, 'original_full')


if __name__ == '__main__':
    unittest.main()
