import itertools
import unittest

import numpy as np

from evaluation import possession_metrics as pm
from experiments import spell_mbr
from experiments import supervised


class SpellMBRTests(unittest.TestCase):
    def test_ffbs_empirical_path_mass_matches_exact_short_posterior(self):
        probabilities = np.array([
            [.62, .28, .10], [.18, .64, .18], [.34, .22, .44],
        ])
        penalty = .8
        states = np.asarray(list(itertools.product(range(3), repeat=3)))
        logp = np.log(probabilities)
        log_weights = np.array([
            sum(logp[t, state[t]] for t in range(3))
            - penalty * sum(state[t] != state[t-1] for t in range(1, 3))
            for state in states
        ])
        exact = np.exp(log_weights - log_weights.max())
        exact /= exact.sum()
        draws = spell_mbr._sample_block(
            logp, penalty, 50_000, np.random.default_rng(20261101))
        sampled_states = np.ravel_multi_index(draws.T, (3, 3, 3))
        empirical = np.bincount(sampled_states, minlength=len(states)) / len(draws)
        self.assertLess(float(np.abs(empirical - exact).sum()), .025)

    def test_linear_overlap_and_f1_match_official_including_equal_iou_ties(self):
        rng = np.random.default_rng(824)
        runs = [(0, 17), (20, 38)]
        for _ in range(400):
            truth = rng.integers(0, 4, 38, dtype=np.int8)
            pred = rng.integers(0, 4, 38, dtype=np.int8)
            gt_spells = pm.possession_spells(truth, runs)
            pred_spells = pm.possession_spells(pred, runs)
            got = spell_mbr._spell_f1_from_spells(gt_spells, pred_spells)
            expected = pm.spell_scores(truth, pred, runs)["mean_f1"]
            self.assertEqual(got, expected)

        # All four overlaps have IoU .5, so greedy tie order is observable.
        gt = np.array([0, 0, 2, 0, 0, 2, 0, 0], dtype=np.int8)
        pred = np.array([0, 2, 0, 0, 2, 0, 0, 0], dtype=np.int8)
        run = [(0, len(gt))]
        self.assertEqual(
            spell_mbr._spell_f1_from_spells(pm.possession_spells(gt, run),
                                             pm.possession_spells(pred, run)),
            pm.spell_scores(gt, pred, run)["mean_f1"])

    def test_valid_mask_and_period_breaks_are_independent(self):
        probabilities = np.array([
            [.99, .005, .005], [.99, .005, .005],
            [.005, .99, .005], [.005, .99, .005],
            [.005, .005, .99], [.005, .005, .99],
        ])
        period = np.array([4, 4, 4, 4, 9, 9])
        valid = np.array([True, True, False, True, True, True])
        output = spell_mbr.decode(probabilities, period, 25., valid_mask=valid)
        self.assertEqual(output[2], "X")
        self.assertEqual(output[0], output[1])
        self.assertEqual(output[3], "A")
        np.testing.assert_array_equal(output[4:], np.array(["D", "D"]))
        reference = supervised.decode(probabilities, period, 25., 3.)
        self.assertEqual(reference[0], reference[1])
        baseline = spell_mbr.decode(probabilities, period, 25., candidate_samples=0,
                                    valid_mask=valid)
        reference[~valid] = "X"
        np.testing.assert_array_equal(baseline, reference)

    def test_unavailable_classes_are_never_invented(self):
        for active in ([0], [1], [2], [0, 2], [1, 2]):
            probabilities = np.zeros((24, 3), dtype=np.float32)
            probabilities[:, active] = 1 / len(active)
            output = spell_mbr.decode(probabilities, np.zeros(24), 25.)
            self.assertTrue(set(output) <= {spell_mbr.OUTPUT_LABELS[k] for k in active})

    def test_team_swap_equivariance_for_asymmetric_evidence(self):
        probabilities = np.array([
            [.25, .65, .10], [.20, .70, .10], [.55, .35, .10],
            [.62, .28, .10], [.18, .72, .10], [.20, .68, .12],
        ])
        period = np.zeros(len(probabilities), dtype=int)
        left = spell_mbr.decode(probabilities, period, 25.)
        right = spell_mbr.decode(probabilities[:, [1, 0, 2]], period, 25.)
        expected = np.asarray([{"H": "A", "A": "H", "D": "D", "X": "X"}[x]
                               for x in left])
        np.testing.assert_array_equal(right, expected)

    def test_synthetic_posterior_mbr_can_beat_viterbi_expected_spell_f1(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(0, 1.1, (7, 3))
        logits[:, 2] += rng.normal(.2, .3)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        cost = .5
        paths = np.asarray(list(itertools.product(range(3), repeat=7)), dtype=np.int8)
        logp = np.log(probabilities)
        logw = logp[np.arange(7)[None, :], paths].sum(axis=1)
        logw -= cost * (paths[:, 1:] != paths[:, :-1]).sum(axis=1)
        weights = np.exp(logw - logw.max())
        posterior = weights / weights.sum()
        baseline = supervised.decode(probabilities, np.zeros(7), 10., cost)
        mbr = spell_mbr.decode(probabilities, np.zeros(7), 10., switch_cost=cost)
        label_to_code = {label: index for index, label in enumerate(spell_mbr.OUTPUT_LABELS)}
        baseline_codes = np.asarray([label_to_code[x] for x in baseline])
        mbr_codes = np.asarray([label_to_code[x] for x in mbr])
        runs = [(0, 7)]
        expected = np.asarray([
            pm.spell_scores(path, baseline_codes, runs)["mean_f1"] for path in paths
        ])
        expected_mbr = np.asarray([
            pm.spell_scores(path, mbr_codes, runs)["mean_f1"] for path in paths
        ])
        self.assertGreater(float(posterior @ expected_mbr), float(posterior @ expected))

    def test_exact_symmetric_team_evidence_uses_deterministic_h_tie(self):
        probabilities = np.tile([.45, .45, .10], (8, 1))
        labels = spell_mbr.decode(probabilities, np.zeros(8), 25.)
        np.testing.assert_array_equal(labels, spell_mbr.decode(probabilities, np.zeros(8), 25.))
        baseline = spell_mbr.decode(probabilities, np.zeros(8), 25., candidate_samples=0)
        self.assertTrue(np.all(baseline == "H"))


if __name__ == "__main__":
    unittest.main()
