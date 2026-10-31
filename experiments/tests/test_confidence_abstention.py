"""Abstention contracts: official scoring, input boundaries and team symmetry."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evaluation import possession_metrics as pm
from experiments import confidence_abstention as ca, supervised
from experiments.metrics import evaluate_tracking


class ConfidenceAbstentionTests(unittest.TestCase):
    def test_suppresses_false_split_and_hold_matches_spell(self):
        labels = np.full(100, 'H')
        decoded = labels.copy()
        decoded[50] = 'A'
        p = np.tile([.9, .05, .05], (100, 1))
        p[50] = [.45, .5, .05]
        period = np.ones(100)
        x = ca.gate(p, decoded, period, .6, 'abstain')
        hold = ca.gate(p, decoded, period, .6, 'hold')
        baseline = evaluate_tracking(labels, decoded)
        xs = evaluate_tracking(labels, x)
        hs = evaluate_tracking(labels, hold)
        self.assertGreater(xs['spell_f1_mean'], baseline['spell_f1_mean'])
        self.assertEqual(xs['spell_f1_mean'], hs['spell_f1_mean'])
        self.assertLess(xs['macro_f1'], hs['macro_f1'])
        self.assertEqual(x[50], 'X')

    def test_real_turnover_can_be_delayed(self):
        decoded = np.repeat(['H', 'A'], 50)
        p = np.tile([.9, .05, .05], (100, 1))
        p[50:70] = [.45, .5, .05]
        p[70:] = [.05, .9, .05]
        x = ca.gate(p, decoded, np.ones(100), .6, 'abstain')
        result = evaluate_tracking(decoded, x)
        self.assertLess(result['spell_f1_mean'], 1)

    def test_no_cross_half_or_time_gap_carry(self):
        inputs = dict(period=np.array([1, 1, 1, 1, 2, 2]),
                      timestamps=np.array([0, .04, 5., 5.04, 0, .04]), fps=25.)
        periods = supervised.model_periods(inputs)
        decoded = np.array(list('HHAAAA'))
        p = np.array([[.9,.05,.05], [.4,.3,.3], [.3,.4,.3], [.05,.9,.05], [.3,.4,.3], [.05,.05,.9]])
        decoded[-1] = 'D'
        out = ca.gate(p, decoded, periods, .6, 'hold')
        self.assertEqual(''.join(out), 'HHAADD')

    def test_all_uncertain_block_and_threshold_equality(self):
        p = np.tile([.5,.3,.2], (3, 1))
        decoded = np.full(3, 'H')
        periods = np.zeros(3)
        np.testing.assert_array_equal(ca.gate(p, decoded, periods, .5, 'abstain'), decoded)
        np.testing.assert_array_equal(ca.gate(p, decoded, periods, .6, 'abstain'), np.full(3, 'X'))
        np.testing.assert_array_equal(ca.gate(p, decoded, periods, .6, 'hold'), decoded)

    def test_team_swap_symmetry_and_no_mutation(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(3), 100)
        decoded = supervised.LABELS[p.argmax(axis=1)]
        original = p.copy(), decoded.copy()
        periods = np.repeat([0, 1], 50)
        swap = lambda labels: np.where(labels == 'H', 'A', np.where(labels == 'A', 'H', labels))
        for mode in ('hold', 'abstain'):
            out = ca.gate(p, decoded, periods, .6, mode)
            rev = ca.gate(p[:, [1,0,2]], swap(decoded), periods, .6, mode)
            np.testing.assert_array_equal(rev, swap(out))
        np.testing.assert_array_equal(p, original[0])
        np.testing.assert_array_equal(decoded, original[1])

    def test_reference_censoring_does_not_enter_gate(self):
        p = np.array([[.9,.05,.05], [.4,.35,.25], [.4,.35,.25], [.05,.9,.05]])
        decoded = np.array(list('HHHA'))
        periods = np.zeros(4)
        x = ca.gate(p, decoded, periods, .6, 'abstain')
        hold = ca.gate(p, decoded, periods, .6, 'hold')
        truth = pm.label_codes(list('HXAA'))
        # Official filling resets after a native reference-X gap. Model-side
        # holding must not use that hidden boundary to obtain artificial parity.
        filled = pm.fill_abstentions(pm.label_codes(x), pm.valid_runs(truth))
        self.assertEqual(filled[2], pm.A)
        self.assertEqual(hold[2], 'H')

    def test_invalid_probabilities_rejected(self):
        for p in (np.array([[np.nan,.3,.7]]), np.array([[.7,.3,.7]])):
            with self.assertRaises(ValueError):
                ca.gate(p, np.array(['H']), np.zeros(1), .6, 'abstain')

    def test_parallel_cache_matches_serial_decode_without_target_inputs(self):
        from scripts.prepare_abstention_cache import prepare_cell
        inputs = dict(period=np.ones(100), timestamps=np.arange(100)/25, fps=np.array(25.))
        labels = np.full(100, 'D')
        probability = np.tile([.8,.1,.1], (100, 1))
        def infer(tracking, models):
            self.assertNotIn('labels', tracking)
            return probability
        with tempfile.TemporaryDirectory() as tmp, patch.object(ca, 'OUT', Path(tmp)), \
                patch.object(ca.cb, 'load_models', return_value={}), \
                patch.object(ca.ce, 'load_model', return_value={}), \
                patch.object(ca.multisource_eval, 'view', return_value=(inputs, labels)), \
                patch.object(ca.cb, 'predict_probabilities', side_effect=infer):
            name = prepare_cell(('synthetic', Path('example.npz'), 'clean', {}))
            with np.load(Path(tmp)/name, allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache['baseline'], ca.supervised.decode(
                    probability, ca.supervised.model_periods(inputs), 25., ca.lt.COST))
                np.testing.assert_array_equal(cache['labels'], labels)
                np.testing.assert_array_equal(cache['probability'], probability)
            self.assertFalse(list(Path(tmp).glob('*.pending.npz')))


if __name__ == '__main__':
    unittest.main()
