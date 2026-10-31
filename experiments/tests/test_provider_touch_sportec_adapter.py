from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.provider_touch_calibration_v2 import (
    load_sportec_operations, predict_sportec, sportec_candidate_grid,
)


class SportecTouchAdapterTests(unittest.TestCase):
    def test_explicit_nonchanging_duel_hold_and_restart_dead_interval(self):
        kickoff = datetime(2023, 1, 1, tzinfo=timezone.utc)
        stamp = lambda seconds: (kickoff + timedelta(seconds=seconds)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = root / "match.xml"
            info.write_text(
                '<Match><General HomeTeamId="home" GuestTeamId="away" '
                f'KickoffTime="{stamp(0)}"/></Match>')
            events = root / "events.xml"
            events.write_text(
                '<Events>'
                f'<Event EventTime="{stamp(.08)}"><Play Team="home"/></Event>'
                f'<Event EventTime="{stamp(.40)}"><TacklingGame PossessionChange="false" '
                'WinnerTeam="away" WinnerRole="withoutBallControl" LoserTeam="home" '
                'LoserRole="withBallControl"/></Event>'
                f'<Event EventTime="{stamp(.80)}"><ThrowIn DecisionTimestamp="{stamp(.50)}" '
                'Team="away"><Play Team="away"/></ThrowIn></Event>'
                f'<Event EventTime="{stamp(1.20)}"><FinalWhistle/></Event>'
                '</Events>')
            timestamps = np.arange(0, 2, .04)
            ops, audit = load_sportec_operations(events, info, timestamps)
            shifted_ops, _ = load_sportec_operations(events, info, timestamps, stamp(.04))
            period = np.ones(len(timestamps), dtype=np.int8)
            hold0 = next(c for c in sportec_candidate_grid()
                         if c["nonchanging_duel"] == "hold" and c["out_tail_grace_seconds"] == 0)
            hold3 = dict(hold0, out_tail_grace_seconds=3.0)
            literal = dict(hold3, nonchanging_duel="literal")
            pred0 = predict_sportec(ops, period, len(period), hold0)
            pred3 = predict_sportec(ops, period, len(period), hold3)
            pred_literal = predict_sportec(ops, period, len(period), literal)

            self.assertEqual(audit["ops_challenge"], 1)
            self.assertEqual(audit["ops_dead_start"], 2)
            self.assertEqual(ops[0]["frame"], 2)
            self.assertEqual(shifted_ops[0]["frame"], 1)
            self.assertEqual(pred3[2], "H")
            self.assertEqual(pred3[10], "H")  # CH with no possession change holds owner
            self.assertEqual(pred_literal[10], "A")  # literal labels only the challenger touch
            self.assertEqual(pred0[10], "X")  # zero grace leaves old gap unlabelled
            self.assertTrue(np.all(pred3[13:20] == "D"))
            self.assertEqual(pred3[20], "A")
            self.assertTrue(np.all(pred3[26:30] == "A"))
            self.assertTrue(np.all(pred3[30:] == "D"))  # final whistle starts dead time


if __name__ == "__main__":
    unittest.main()
