import unittest

import numpy as np

from experiments import lgbm_retention_topology as topology


class TopologyRepairTests(unittest.TestCase):
    def repair(self, baseline, refined, mode="both", period=None):
        baseline = np.asarray(list(baseline))
        refined = np.asarray(list(refined))
        if period is None:
            period = np.zeros(len(baseline), dtype=int)
        return topology.topology_repair(baseline, refined, period, mode)

    def test_start_end_and_both_modes_move_only_existing_run_edges(self):
        baseline = "HHDDAAHH"
        refined = "HDDDAAHH"
        start, ds = self.repair(baseline, refined, "start")
        end, de = self.repair(baseline, "HHDDDAHH", "end")
        both, db = self.repair(baseline, "HDDDDAHH", "both")
        self.assertEqual("".join(start), "HDDDAAHH")
        self.assertEqual("".join(end), "HHDDDAHH")
        self.assertEqual("".join(both), "HDDDDAHH")
        self.assertEqual(ds["d_runs_after"], ds["d_runs_before"])
        self.assertEqual(de["d_runs_after"], de["d_runs_before"])
        self.assertEqual(db["d_runs_after"], db["d_runs_before"])

    def test_ambiguous_split_or_merge_overlap_is_left_unchanged(self):
        baseline = "HHDDDDHH"
        refined = "HHDDHDHH"  # Two refined D runs overlap the same baseline run.
        result, diagnostics = self.repair(baseline, refined)
        np.testing.assert_array_equal(result, list(baseline))
        self.assertEqual(diagnostics["changed_frames"], 0)
        self.assertTrue(any(item["reason"] == "ambiguous_refined_overlap"
                            for item in diagnostics["rejected_runs"]))

    def test_unmatched_baseline_run_is_preserved_while_local_match_can_move(self):
        baseline = "HHDDAAHHDDAA"
        refined = "HDDDAAHHAAAA"  # First run moves; second has no refined match.
        result, diagnostics = self.repair(baseline, refined)
        self.assertEqual("".join(result), "HDDDAAHHDDAA")
        self.assertEqual(len(diagnostics["accepted_runs"]), 1)
        self.assertEqual(diagnostics["d_runs_before"], diagnostics["d_runs_after"])

    def test_HA_sequence_change_falls_back_for_whole_block(self):
        baseline = "HHDDHHAA"
        refined = "DDDDHHAA"  # Swallows the only H spell before the D run.
        result, diagnostics = self.repair(baseline, refined)
        np.testing.assert_array_equal(result, list(baseline))
        self.assertEqual(diagnostics["fallback_blocks"], [0])
        self.assertEqual(diagnostics["changed_frames"], 0)

    def test_absent_live_anchor_rejects_only_that_endpoint(self):
        baseline = "DDDHHAA"
        refined = "HDDHHAA"  # Start contraction has no preceding H/A anchor.
        result, diagnostics = self.repair(baseline, refined, mode="start")
        np.testing.assert_array_equal(result, list(baseline))
        self.assertTrue(diagnostics["proposed_runs"])
        self.assertIn("start_no_preceding_HA_anchor",
                      diagnostics["proposed_runs"][0]["rejected_endpoints"])

    def test_period_blocks_and_X_are_hard_fallback_barriers(self):
        baseline = list("HHDDAAXHHDDAA")
        refined = list("H" + "D" * 4 + "AXHHDDAA")
        period = np.array([0] * 6 + [1] * 7)
        result, diagnostics = topology.topology_repair(baseline, refined, period)
        # First block is independently repairable; X-containing block is frozen.
        self.assertEqual("".join(result[:6]), "HDDDDA")
        self.assertEqual(result[6], "X")
        np.testing.assert_array_equal(result[7:], baseline[7:])
        self.assertIn(1, diagnostics["fallback_blocks"])

    def test_team_swap_equivariance_and_identity(self):
        baseline = np.asarray(list("HHDDAAHH"))
        refined = np.asarray(list("HDDDDAHH"))
        swapped_base = np.where(baseline == "H", "A", np.where(baseline == "A", "H", baseline))
        swapped_refined = np.where(refined == "H", "A", np.where(refined == "A", "H", refined))
        result, _ = topology.topology_repair(baseline, refined, np.zeros(len(baseline)), "both")
        swapped_result, _ = topology.topology_repair(swapped_base, swapped_refined,
                                                     np.zeros(len(baseline)), "both")
        expected = np.where(result == "H", "A", np.where(result == "A", "H", result))
        np.testing.assert_array_equal(swapped_result, expected)
        identity, diag = topology.topology_repair(baseline, baseline, np.zeros(len(baseline)))
        np.testing.assert_array_equal(identity, baseline)
        self.assertEqual(diag["changed_frames"], 0)


if __name__ == "__main__":
    unittest.main()
