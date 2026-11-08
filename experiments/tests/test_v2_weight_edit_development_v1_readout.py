from __future__ import annotations

import unittest
from unittest import mock

from experiments import v2_weight_edit_development as development
from experiments import v2_weight_edit_development_v1_readout as readout


class V1ReadoutExecutionTests(unittest.TestCase):
    def test_filters_only_archived_control_from_editable_arm_call(self):
        original = mock.Mock(return_value={"ok": True})
        readout._editable_v1_cell_call(
            original, "provider", "match", "feed_like_v1", 1,
            ["update_bias_m025", "blend_unedited_ha15", "deployed_original_full"],
            prepared=(1, 2, 3, 4, 5))
        original.assert_called_once_with(
            "provider", "match", "feed_like_v1", 1,
            ["update_bias_m025", "blend_unedited_ha15"], prepared=(1, 2, 3, 4, 5))

    def test_deduplicates_winner_when_it_is_the_unedited_control(self):
        original = mock.Mock(return_value={"ok": True})
        readout._editable_v1_cell_call(
            original, "provider", "match", "feed_like_v1", 1,
            ["blend_unedited_ha15", "blend_unedited_ha15", "deployed_original_full"])
        original.assert_called_once_with("provider", "match", "feed_like_v1", 1,
                                         ["blend_unedited_ha15"])

    def test_v1_action_uses_bound_frozen_protocol_with_scoped_adapter(self):
        with mock.patch.object(readout.execution, "verify_binding"), \
             mock.patch.object(development, "score_v1", return_value={"ok": True}) as scored:
            self.assertEqual(readout.run_v1(workers=4), {"ok": True})
            scored.assert_called_once_with(workers=4)

    def test_missing_editable_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            readout._editable_v1_cell_call(
                mock.Mock(), "provider", "match", "feed_like_v1", 1,
                ["deployed_original_full"])


if __name__ == "__main__":
    unittest.main()
