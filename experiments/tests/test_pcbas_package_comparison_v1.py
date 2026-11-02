import json
import unittest
from unittest import mock

import numpy as np

from experiments import pcbas_package_comparison_v1 as study


class PcbasPackageComparisonTest(unittest.TestCase):
    def test_reserve_split_is_deterministic_disjoint_and_label_free(self):
        prior = json.loads(study.PRIOR_DECLARATION.read_text())
        reserve = [r['game_id'] for r in prior['cohort']['reserve']]
        compare, sealed = study.split_reserve(reserve)
        self.assertEqual((len(compare), len(sealed)), (18, 18))
        self.assertFalse(set(compare) & set(sealed))
        self.assertEqual(set(compare) | set(sealed), set(reserve))
        self.assertEqual(study.split_reserve(list(reversed(reserve))), (compare, sealed))
        scored = {r['game_id'] for r in prior['cohort']['selected']} | study.KNOWN_VAL_GAMES
        self.assertFalse(set(compare) & scored)

    def test_split_rejects_wrong_reserve(self):
        with self.assertRaises(ValueError):
            study.split_reserve(['game_1'] * 36)
        with self.assertRaises(ValueError):
            study.split_reserve([f'game_{i}' for i in range(35)] + ['game_18'])

    def test_decode_arm_pooling_order(self):
        rng = np.random.default_rng(0)
        phases = rng.dirichlet(np.ones(3), size=(2, 5, 7))
        seen = {}

        def fake_decode(native, periods, fps, config):
            seen['native'] = native
            return np.array(['H'] * native.shape[1])
        inputs = dict(period=np.ones(7, int), timestamps=np.arange(7) / 25., fps=np.array(25.))
        with mock.patch('experiments.feed_spell_push.decode', fake_decode):
            study.decode_arm('original_full_phase', phases, inputs)
            np.testing.assert_allclose(seen['native'], phases.mean(axis=(0, 1))[None])
            study.decode_arm('calibrated_phase', phases, inputs)
            np.testing.assert_allclose(seen['native'], phases.mean(axis=1))
            study.decode_arm('calibrated_phase0', phases, inputs)
            np.testing.assert_allclose(seen['native'], phases[:, 0])

    def test_paired_bootstrap_sign(self):
        out = study.paired_bootstrap([0.5, 0.6, 0.7], [0.6, 0.7, 0.8])
        self.assertAlmostEqual(out['mean_difference'], 0.1)
        self.assertEqual(out['games_b_better'], 3)


if __name__ == '__main__':
    unittest.main()
