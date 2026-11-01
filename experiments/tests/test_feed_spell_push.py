import unittest
from unittest import mock

import numpy as np

from experiments import feed_spell_push as study, decoder_refinement as prior


class FeedSpellContracts(unittest.TestCase):
    def test_all_arms_breaks_swap_missing_classes_immutability(self):
        x = np.random.default_rng(180).random((2, 73, 3))
        x /= x.sum(axis=2, keepdims=True)
        original = x.copy()
        period = np.r_[np.zeros(32), np.ones(41)]
        self.assertEqual(len(study.configurations()), 194)
        for config in study.configurations():
            y = study.decode(x, period, 10, config)
            swapped = study.decode(x[:, :, [1, 0, 2]], period, 10, config)
            np.testing.assert_array_equal(swapped,
                np.where(y == "H", "A", np.where(y == "A", "H", "D")), err_msg=str(config))
            np.testing.assert_array_equal(y, np.r_[
                study.decode(x[:, :32], period[:32], 10, config),
                study.decode(x[:, 32:], period[32:], 10, config)])
            for absent in range(3):
                masked = x.copy(); masked[:, :, absent] = 0
                self.assertNotIn("HAD"[absent], study.decode(masked, period, 10, config))
        np.testing.assert_array_equal(x, original)

    def test_reference_replay_exact(self):
        x = np.random.default_rng(17).random((2, 71, 3))
        x /= x.sum(axis=2, keepdims=True)
        period = np.zeros(71)
        for dead in (6, 12):
            np.testing.assert_array_equal(study.decode(x, period, 25,
                dict(family="reference", dead=dead)), prior.decode(x.mean(axis=0), period, 25,
                dict(family="dead_temperature", temperature=1.5, ha=3, dead=dead)))

    def test_identical_members_and_member_order(self):
        x = np.random.default_rng(18).random((2, 55, 3))
        x /= x.sum(axis=2, keepdims=True)
        period = np.zeros(55)
        for c in study.configurations():
            np.testing.assert_array_equal(study.decode(x, period, 25, c),
                                          study.decode(x[::-1], period, 25, c))
            one = study.decode(x[:1], period, 25, c)
            np.testing.assert_array_equal(one,
                study.decode(np.repeat(x[:1], 2, axis=0), period, 25, c))

    def test_anchors_never_use_viterbi(self):
        x = np.random.default_rng(19).random((2, 55, 3))
        with mock.patch.object(study.core, "decode", side_effect=AssertionError("Viterbi")):
            for c in study.configurations():
                if c["family"] == "anchors":
                    study.decode(x, np.zeros(55), 25, c)

    def test_rejects_invalid_inputs(self):
        c = study.configurations()[0]
        for x in (np.ones((2, 0, 3)), np.ones((3, 5, 3)), np.full((2, 5, 3), np.nan),
                  -np.ones((2, 5, 3)), np.zeros((2, 5, 3))):
            with self.assertRaises(ValueError):
                study.decode(x, np.zeros(x.shape[1]), 25, c)
