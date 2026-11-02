import unittest

import numpy as np

from experiments import gar_noball_bridge_v1 as bridge


def _record(n, runs):
    return dict(native_n=n, touch_run_columns=['start', 'end', 'team', 'end_code', 'controlled', 'censored'],
                touch_runs=np.asarray(runs, dtype=np.int64).reshape(-1, 6))


class GarNoBallBridgeTest(unittest.TestCase):
    def test_remove_ball_only_touches_ball(self):
        inputs = dict(ball=np.ones((4, 2)), ball_observed=np.ones(4, bool), players=np.zeros((4, 3, 2)),
                      player_observed=np.ones((4, 3), bool), timestamps=np.arange(4.))
        out = bridge.remove_ball(inputs)
        self.assertTrue(np.isnan(out['ball']).all())
        self.assertFalse(out['ball_observed'].any())
        self.assertIs(out['players'], inputs['players'])
        self.assertTrue(np.isfinite(inputs['ball']).all())  # input not mutated

    def test_anchor_classes_offsets_and_exclusions(self):
        records = [_record(10, [[0, 3, 0, 6, 1, 0], [4, 5, 1, 1, 1, 0], [6, 7, 1, 2, 0, 0]]),
                   _record(10, [[0, 2, 1, 5, 1, 0], [3, 4, 0, 4, 1, 1], [5, 6, 2, 1, 1, 0]])]
        anchors = bridge.anchors_from_records(records)
        self.assertEqual([(a['index'], a['team'], a['cls']) for a in anchors],
                         [(3, 0, 'drive'), (5, 1, 'pass'), (12, 1, 'cross')])

    def test_anchor_metrics_count_dead_as_wrong(self):
        anchors = [dict(index=0, team=0, cls='drive'), dict(index=1, team=1, cls='drive'),
                   dict(index=2, team=0, cls='shot')]
        prediction = np.array(['H', 'D', 'A'])
        m = bridge.anchor_metrics(anchors, prediction, np.array([14, 7, 3]))
        self.assertEqual(m['classes']['drive'], dict(n=2, correct=1, predicted_d=1))
        self.assertEqual(m['classes']['shot'], dict(n=1, correct=0, predicted_d=0))
        self.assertEqual(m['visibility']['0-5']['n'], 1)
        self.assertEqual(m['visibility']['12+']['correct'], 1)

    def test_live_metrics(self):
        m = bridge.live_metrics(np.array(['H', 'A', 'D', 'X']), np.array(['H', 'D', 'D', 'H']))
        self.assertEqual((m['live_frames'], m['live_team_correct'], m['live_predicted_d'],
                          m['dead_frames'], m['dead_predicted_d']), (2, 1, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
