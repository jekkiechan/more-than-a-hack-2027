import unittest
from unittest import mock
import tempfile
import json
from pathlib import Path

import numpy as np

from experiments import decoder_refinement as study, decoder_search as core
from experiments import decoder_refinement_public as public


class RefinementContracts(unittest.TestCase):
    def test_all_arms_preserve_breaks_swap_and_unavailable_classes(self):
        rng = np.random.default_rng(561)
        p = rng.random((71, 3))
        p /= p.sum(axis=1, keepdims=True)
        original = p.copy()
        period = np.r_[np.zeros(31), np.ones(40)]
        self.assertEqual(len(study.configurations()), 118)
        for config in study.configurations():
            guess = study.decode(p, period, 10, config)
            swapped = study.decode(p[:, [1, 0, 2]], period, 10, config)
            expected = np.where(guess == "H", "A", np.where(guess == "A", "H", "D"))
            np.testing.assert_array_equal(swapped, expected, err_msg=str(config))
            separate = np.r_[study.decode(p[:31], period[:31], 10, config),
                             study.decode(p[31:], period[31:], 10, config)]
            np.testing.assert_array_equal(guess, separate)
            for missing in range(3):
                no_class = p.copy()
                no_class[:, missing] = 0
                self.assertNotIn(core.supervised.LABELS[missing], study.decode(no_class, period, 10, config))
        np.testing.assert_array_equal(p, original)

    def test_previous_winner_exact(self):
        p = np.random.default_rng(6).random((100, 3))
        period = np.zeros(100)
        config = next(c for c in study.configurations() if c["family"] == "matrix" and c["ha"] == 3 and c["dead"] == 12)
        np.testing.assert_array_equal(study.decode(p, period, 25, config),
            core.decode(p, period, 25, dict(family="matrix", ha=3, dead=12)))

    def test_non_viterbi_arms_do_not_call_viterbi(self):
        p=np.random.default_rng(8).random((50,3))
        with mock.patch.object(core,"decode",side_effect=AssertionError("Viterbi called")):
            for c in study.configurations():
                if c["family"] in ("anchor_segments","total_variation"):
                    self.assertEqual(len(study.decode(p,np.zeros(50),25,c)),50)

    def test_public_gate_stays_closed_after_reference_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            reference=root/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
            reference.parent.mkdir(parents=True)
            reference.write_text('invalid reference')
            seal=root/'seal.json';seal.write_text(json.dumps({'expected_reference_sha256':'0'*64}))
            started=root/'started.json'
            with mock.patch.multiple(public,ROOT=root,SEAL=seal,STARTED=started,RESULT=root/'result.json'), mock.patch.object(public,'validate_seal'):
                with self.assertRaisesRegex(ValueError,'Reference SHA mismatch'):
                    public.score()
                self.assertTrue(started.exists())
                with self.assertRaisesRegex(ValueError,'permanently closed'):
                    public.score()


if __name__ == "__main__":
    unittest.main()
