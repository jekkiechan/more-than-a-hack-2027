import unittest

import numpy as np

from experiments.turnover_supervision_audit import inventory


class TurnoverInventoryTests(unittest.TestCase):
    def test_context_requires_stable_same_team_on_each_side(self):
        y = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50, dtype=np.int8)
        record = {'full_y': y, 'full_period': np.zeros(len(y), dtype=np.int32), 'fps': 25.0}
        result = inventory([record], kind='original')
        self.assertEqual(result['adjacent_ha_turnovers'], 3)
        self.assertEqual(result['turnovers_with_stable_same_team_context_both_sides']['1'], 3)
        self.assertEqual(result['turnovers_with_stable_same_team_context_both_sides']['2'], 3)

    def test_dead_unknown_and_period_breaks_stop_context(self):
        y = np.array([0] * 30 + [1] * 30 + [2] + [1] * 40 + [-1] + [0] * 30, dtype=np.int8)
        period = np.array([0] * 61 + [1] * (len(y)-61), dtype=np.int32)
        result = inventory([{'full_y': y, 'full_period': period, 'fps': 25.0}], kind='gar')
        self.assertEqual(result['adjacent_ha_turnovers'], 1)
        self.assertEqual(result['turnovers_with_stable_same_team_context_both_sides']['1'], 1)

    def test_unknown_boundary_censors_short_span(self):
        y = np.array([-1] + [0] * 20 + [1] * 20 + [2], dtype=np.int8)
        result = inventory([{'full_y': y, 'full_period': np.zeros(len(y)), 'fps': 25.0}], kind='gar')
        self.assertEqual(result['complete_constant_ha_spans_le_3s'], 1)


if __name__ == '__main__':
    unittest.main()
