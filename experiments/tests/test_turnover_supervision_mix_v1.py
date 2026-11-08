import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import turnover_supervision_mix_v1 as mix


class TurnoverMixTests(unittest.TestCase):
    def test_blend_uses_only_declared_fractions(self):
        fresh = np.array([[.7, .2, .1]], dtype=np.float64)
        hard = np.array([[.1, .3, .6]], dtype=np.float64)
        np.testing.assert_allclose(mix.blend_ce(fresh, hard, .25), .75*fresh+.25*hard)
        with self.assertRaises(ValueError):
            mix.blend_ce(fresh, hard, .4)

    def test_probability_replay_rejects_tampering(self):
        expected = np.array([[.6, .4]])
        mix._assert_probability(expected, expected.copy(), 'exact')
        with self.assertRaisesRegex(ValueError, 'probability replay mismatch'):
            mix._assert_probability(np.array([[.7, .3]]), expected, 'tampered')

    def test_phase_pooling_preserves_seed_then_phase_order(self):
        archive = {}
        for seed, base in ((2026, 1.0), (2027, 11.0)):
            archive[f'prob__ce__fresh_control__seed{seed}__phase0'] = np.array([[base, base+1, base+2]])
            archive[f'prob__ce__fresh_control__seed{seed}__phase1'] = np.array([[base+4, base+5, base+6]])
        one_seed = mix._rebuild_ce(archive, 'fresh_control', 2, seed=2027)
        pooled = mix._rebuild_ce(archive, 'fresh_control', 2)
        np.testing.assert_array_equal(one_seed, np.array([[13., 14., 15.]]))
        np.testing.assert_array_equal(pooled, np.array([[8., 9., 10.]]))

    def test_freeze_refuses_partial_cell_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'out'; (out/'cells').mkdir(parents=True)
            (out/'cells'/'partial.npz').write_bytes(b'partial')
            paths = {name: root/name for name in ('decl.json','endpoint.json','result.json','readout.json')}
            with patch.object(mix, 'OUT', out), patch.object(mix, 'DECLARATION', paths['decl.json']), \
                 patch.object(mix, 'ENDPOINT', paths['endpoint.json']), patch.object(mix, 'RESULT', paths['result.json']), \
                 patch.object(mix, 'READOUT', paths['readout.json']), patch.object(mix, '_inputs', return_value={}):
                with self.assertRaisesRegex(FileExistsError, 'prediction file exists'):
                    mix.freeze()

    def test_conditional_mix_preserves_unit_mass(self):
        ce = np.array([[.6, .3, .1]])
        calibrated = np.array([[.2, .7, .1]])
        predecessor = np.array([[.4, .5, .1]])
        result = mix._mix(ce, calibrated, predecessor)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        np.testing.assert_allclose(result, .5*np.array([[2/3,1/3]])+
            .375*np.array([[2/9,7/9]])+.125*np.array([[4/9,5/9]]))


if __name__ == '__main__':
    unittest.main()
