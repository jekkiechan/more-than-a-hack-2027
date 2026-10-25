import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.metrica_open import labels_from_events
from experiments.skillcorner_had import stoppages


def event(index, team, kind, start, end, subtypes=(), period=1):
    return dict(index=index, team=team, type=kind, subtypes=set(subtypes),
                period=period, start=start, end=end)


class MetricaEventLabelTests(unittest.TestCase):
    def test_loose_ball_stays_with_last_team_and_ball_out_is_dead(self):
        events = [
            event(0, "H", "SET PIECE", 10, 10, ("KICK OFF",)),
            event(1, "H", "PASS", 10, 20),
            event(2, "H", "BALL LOST", 30, 35),
            event(3, "A", "CHALLENGE", 36, 36),
            event(4, "A", "RECOVERY", 40, 40),
            event(5, "A", "BALL OUT", 40, 50),
            event(6, "H", "SET PIECE", 60, 60, ("THROW IN",)),
            event(7, "H", "PASS", 60, 70),
        ]
        frames = np.arange(0, 80)
        labels, stats = labels_from_events(events, frames, np.ones(80, dtype=int))
        self.assertTrue((labels[:10] == "X").all())
        self.assertTrue((labels[10:40] == "H").all())      # BALL LOST/CHALLENGE do not flip
        self.assertTrue((labels[40:50] == "A").all())
        self.assertTrue((labels[50:60] == "D").all())      # BALL OUT end -> restart
        self.assertTrue((labels[60:71] == "H").all())
        self.assertTrue((labels[71:] == "X").all())        # after the last event
        self.assertEqual(stats["stoppages"], 1)

    def test_foul_whistle_and_untriggered_set_piece(self):
        events = [
            event(0, "A", "PASS", 0, 10),
            event(1, "H", "FAULT RECEIVED", 15, 15),
            event(2, "H", "SET PIECE", 30, 30, ("FREE KICK",)),
            event(3, "H", "PASS", 30, 40),
            event(4, "A", "SET PIECE", 60, 60, ("FREE KICK",)),
            event(5, "A", "PASS", 60, 70),
        ]
        labels, stats = labels_from_events(events, np.arange(71), np.ones(71, dtype=int))
        self.assertTrue((labels[15:30] == "D").all())
        self.assertTrue((labels[30:40] == "H").all())
        self.assertTrue((labels[41:60] == "D").all())      # no trigger: dead from previous end
        self.assertEqual(stats["untriggered_set_pieces"], 1)


class SkillCornerStoppageTests(unittest.TestCase):
    def test_interruption_bounds_with_margins(self):
        rows = [
            dict(event_type="player_possession", frame_start=0, frame_end=40, period=1,
                 game_interruption_after="throw_in_for"),
            dict(event_type="passing_option", frame_start=45, frame_end=50, period=1,
                 game_interruption_after=""),
            dict(event_type="player_possession", frame_start=200, frame_end=220, period=1,
                 game_interruption_after=""),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            spans, tags = stoppages(path, fps=10, margin_s=1.0)
        self.assertEqual(spans, [(41, 51, 190, 200)])
        self.assertEqual(tags, ["throw_in_for"])


if __name__ == "__main__":
    unittest.main()
