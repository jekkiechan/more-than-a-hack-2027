from __future__ import annotations

import unittest

import numpy as np

from experiments import v2_calibration_ha_boundary as study


class HABoundaryTests(unittest.TestCase):
    def test_plan_grid_has_twenty_arms_and_only_hacost_varies(self):
        self.assertEqual(len(study.ARMS), 20)
        self.assertEqual(len({arm["id"] for arm in study.ARMS}), 20)
        self.assertEqual({arm["recipe"] for arm in study.ARMS}, set(study.RECIPES))
        self.assertEqual({arm["config"]["ha"] for arm in study.ARMS}, set(study.HA_COSTS))
        fixed = dict(study.BASE_CONFIG, ha=3)
        for arm in study.ARMS:
            config = dict(arm["config"], ha=3)
            self.assertEqual(config, fixed)

    def test_phase_ensemble_balances_seed_members_within_each_phase(self):
        members = [(0, 2026, np.array([[.8, .1, .1]])),
                   (0, 2027, np.array([[.6, .3, .1]])),
                   (1, 2026, np.array([[.2, .7, .1]])),
                   (1, 2027, np.array([[.4, .5, .1]]))]
        pooled = study._phase_ensemble(members, 2)
        single = study._phase_ensemble(members, 2, seed=2026)
        np.testing.assert_allclose(pooled[0], [.5, .4, .1])
        np.testing.assert_allclose(single[0], [.5, .4, .1])


if __name__ == "__main__":
    unittest.main()
