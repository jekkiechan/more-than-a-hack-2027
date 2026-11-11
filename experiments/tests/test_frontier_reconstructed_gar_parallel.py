import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import frontier_reconstructed_gar_parallel as execution


class ParallelGarContracts(unittest.TestCase):
    def test_fixed_four_way_partition_is_complete_disjoint_and_stable(self):
        games = [f"game-{i}" for i in range(10)]
        groups = execution.partition_games(games)
        self.assertEqual(groups, execution.partition_games(games))
        self.assertEqual([len(group) for group in groups], [3, 3, 2, 2])
        self.assertEqual(len({game for group in groups for game in group}), 10)
        self.assertEqual(set(game for group in groups for game in group), set(games))

    def test_duplicate_or_wrong_sized_test_inventory_fails_closed(self):
        games = [f"game-{i}" for i in range(10)]
        games[-1] = games[0]
        with self.assertRaisesRegex(ValueError, "ten unique"):
            execution.partition_games(games)
        with self.assertRaisesRegex(ValueError, "ten unique"):
            execution.partition_games(["only-one"])

    def test_exposure_audit_uses_full_test_inventory_before_subset_scoring(self):
        full_test = [f"game-{i}" for i in range(10)]
        partition_zero = execution.partition_games(full_test)[0]
        off_partition_exposed = next(game for game in full_test if game not in partition_zero)
        models = {"ce_2026": {"training": {"soccernet_games": ["soccernet_" + off_partition_exposed]}}}
        self.assertNotIn(off_partition_exposed, partition_zero)
        with self.assertRaisesRegex(ValueError, "TEST exposure"):
            execution.audit_model_exposure(models, full_test)

    def test_game_lock_is_exclusive_and_released_after_worker_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(execution, "OUT", Path(tmp)):
                with execution._game_locks(["g1"]):
                    with self.assertRaises(FileExistsError):
                        with execution._game_locks(["g1"]):
                            pass
                self.assertFalse((Path(tmp) / "execution_locks/g1.lock").exists())


if __name__ == "__main__":
    unittest.main()
