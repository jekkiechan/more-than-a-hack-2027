from __future__ import annotations

import unittest
from unittest import mock

from experiments import v2_weight_edit_development_execution as execution


class ExecutionDispatchTests(unittest.TestCase):
    def test_no_worker_actions_do_not_receive_worker_argument(self):
        actions = {
            "freeze": "freeze",
            "control-preflight": "control_preflight",
            "freeze-winner": "freeze_winner",
        }
        with mock.patch.object(execution, "verify_binding"):
            for action, function in actions.items():
                with self.subTest(action=action), mock.patch.object(
                    execution.development, function, return_value={"ok": action}
                ) as called:
                    self.assertEqual(execution.dispatch(action), {"ok": action})
                    called.assert_called_once_with()

    def test_scoring_actions_forward_explicit_worker_count(self):
        actions = {
            "control": "control_replay",
            "score-primary": "run_primary",
            "score-v1": "score_v1",
        }
        with mock.patch.object(execution, "verify_binding"):
            for action, function in actions.items():
                with self.subTest(action=action), mock.patch.object(
                    execution.development, function, return_value={"ok": action}
                ) as called:
                    self.assertEqual(execution.dispatch(action, workers=4), {"ok": action})
                    called.assert_called_once_with(workers=4)

    def test_worker_count_is_bounded_before_scoring(self):
        with mock.patch.object(execution, "verify_binding"), mock.patch.object(
            execution.development, "run_primary"
        ) as called:
            with self.assertRaises(ValueError):
                execution.dispatch("score-primary", workers=0)
            called.assert_not_called()

    def test_source_and_declaration_binding_are_checked_before_dispatch(self):
        with mock.patch.object(execution.development, "sha", side_effect=["bad", "bad"]), \
             mock.patch.object(execution.development, "run_primary") as called:
            with self.assertRaises(RuntimeError):
                execution.dispatch("score-primary")
            called.assert_not_called()


if __name__ == "__main__":
    unittest.main()
