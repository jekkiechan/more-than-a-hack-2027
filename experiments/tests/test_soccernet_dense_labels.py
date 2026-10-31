import unittest

import numpy as np

from experiments import soccernet_dense_labels as dl

FPS = 30.0


def stream(n, runs=(), outs=()):
    """runs: (start, end, home_flag, end_pet); outs: frame indices."""
    events = [('', -1, '')] * n
    flags = [None] * n
    for k, (s, e, flag, pet) in enumerate(runs):
        for i in range(s, e + 1):
            events[i] = ('OTB', 100 + k, pet if i == e else '')
            flags[i] = flag
    for o in outs:
        events[o] = ('OUT', 900 + o, '')
    return events, flags


def labels(n, runs=(), outs=(), mode='hold'):
    events, flags = stream(n, runs, outs)
    return dl.dense_labels(events, flags, FPS, mode)


class DenseLabelTests(unittest.TestCase):
    def test_same_team_propagates_through_gap(self):
        y, prov, _, _ = labels(40, [(5, 8, '1', 'PA'), (20, 25, '1', 'PA')])
        self.assertTrue((y[:5] == -1).all())             # before first touch
        self.assertTrue((y[5:26] == 0).all())            # H throughout
        self.assertEqual(prov[6], dl.P_TOUCH)
        self.assertEqual(prov[12], dl.P_PROPAGATED)
        self.assertTrue((y[26:] == -1).all())            # tail without event

    def test_controlled_opponent_switches_at_run_start(self):
        y, _, _, _ = labels(40, [(2, 4, '1', 'PA'), (20, 24, '0', 'PA'), (30, 32, '0', '')])
        self.assertTrue((y[2:20] == 0).all())
        self.assertTrue((y[20:33] == 1).all())

    def test_uncontrolled_challenge_does_not_switch(self):
        runs = [(2, 4, '1', 'PA'), (10, 12, '0', 'CH'), (20, 22, '1', 'PA')]
        y, prov, _, _ = labels(30, runs)
        self.assertTrue((y[2:23] == 0).all())
        self.assertEqual(prov[11], dl.P_HELD)
        literal, _, _, _ = labels(30, runs, mode='literal')
        self.assertTrue((literal[10:13] == 1).all())

    def test_single_frame_deflection_is_uncontrolled(self):
        y, _, _, _ = labels(40, [(2, 4, '1', 'PA'), (10, 10, '0', 'CL'),
                                 (30, 33, '0', 'PA')])
        self.assertTrue((y[2:30] == 0).all())             # stays H until the later controlled touch
        self.assertTrue((y[30:34] == 1).all())

    def test_uncontrolled_then_controlled_opponent_switches_late(self):
        y, _, _, _ = labels(60, [(2, 4, '1', 'PA'), (10, 12, '0', 'RE'),
                                 (30, 36, '0', 'PA')])
        self.assertTrue((y[2:30] == 0).all())
        self.assertTrue((y[30:37] == 1).all())

    def test_out_gives_dead_until_next_touch(self):
        y, prov, _, _ = labels(60, [(2, 4, '0', 'PA'), (40, 45, '1', 'PA')], outs=[15])
        self.assertTrue((y[2:15] == 1).all())
        self.assertTrue((y[15:40] == 2).all())
        self.assertEqual(prov[20], dl.P_DEAD)
        self.assertTrue((y[40:46] == 0).all())            # restart owner = next toucher

    def test_dead_runs_to_record_end(self):
        y, _, _, _ = labels(30, [(2, 4, '0', 'PA')], outs=[10])
        self.assertTrue((y[10:] == 2).all())

    def test_out_beyond_grace_leaves_gap_unknown(self):
        n = 400
        y, _, _, _ = labels(n, [(2, 4, '0', 'PA')], outs=[300])
        self.assertTrue((y[5:300] == -1).all())
        self.assertTrue((y[300:] == 2).all())

    def test_unknown_team_run_breaks_propagation(self):
        y, _, _, _ = labels(60, [(2, 4, '0', 'PA'), (10, 12, None, 'PA'),
                                 (30, 33, '0', 'PA')])
        self.assertTrue((y[2:10] == 1).all())
        self.assertTrue((y[10:30] == -1).all())           # unknown across unknown-team touch
        self.assertTrue((y[30:34] == 1).all())

    def test_first_uncontrolled_run_has_no_owner(self):
        y, _, _, _ = labels(30, [(2, 4, '1', 'CH'), (10, 12, '0', 'PA')])
        self.assertTrue((y[:10] == -1).all())
        self.assertTrue((y[10:13] == 1).all())

    def test_finalize_record_builds_consistent_dense_fields(self):
        n = 60
        events, flags = stream(n, [(2, 8, '1', 'PA'), (30, 35, '0', 'PA')], outs=[50])
        indices = np.arange(0, n, 6)
        anchor_full = np.full(n, -1, np.int8)
        anchor_full[40] = 1                               # in unknown gap? no: 36..49 is A-propagated
        record = {'_ev': events, '_home_flag': flags, 'indices': indices,
                  'y': np.full(len(indices), -1, np.int8), 'full_y': anchor_full,
                  'full_period': np.full(n, 1, np.int32),
                  'period': np.full(len(indices), 3, np.int32)}
        out = dl.finalize_record(record, FPS)
        full_y, y = out['full_y'], out['y']
        self.assertTrue(np.array_equal(y, full_y[indices]))
        self.assertTrue(np.array_equal(out['period'], out['full_period'][indices]))
        self.assertEqual(out['game_period'], 1)
        self.assertEqual(int(full_y[40]), 1)
        self.assertEqual(int(full_y[55]), 2)
        self.assertEqual(int(out['anchor_full_y'][40]), 1)
        # Existing objective helpers accept the record without raising.
        try:
            from experiments import objective_losses as ol
        except Exception:                                 # torch-free environments
            return
        ol.spell_duration_factors(out['full_y'], out['full_period'], out['indices'],
                                  out['y'], 5.0)
        ol.transition_targets(out['y'], out['period'], core_size=len(out['y']),
                              full_y=out['full_y'], full_period=out['full_period'],
                              indices=out['indices'])

    def test_all_unknown_record_is_dropped(self):
        events, flags = stream(20)
        record = {'_ev': events, '_home_flag': flags, 'indices': np.arange(0, 20, 6),
                  'y': np.full(4, -1, np.int8), 'full_y': np.full(20, -1, np.int8),
                  'full_period': np.ones(20, np.int32), 'period': np.zeros(4, np.int32)}
        self.assertIsNone(dl.finalize_record(record, FPS))


if __name__ == '__main__':
    unittest.main()
