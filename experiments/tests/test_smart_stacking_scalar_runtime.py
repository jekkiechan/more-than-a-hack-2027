from unittest import TestCase
from unittest.mock import patch

import numpy as np
from experiments import smart_stacking_scalar_runtime_v1 as runtime


class ScalarAdapterTests(TestCase):
    def test_only_scalar_metadata_type_changes(self):
        value = np.array([2., 3.])
        original = dict(fps=np.asarray(25.), probabilities=value, labels=np.array(["X", "X"]))
        with patch.object(runtime, "ORIGINAL_LEVEL1", return_value=[original]):
            cell = runtime.level1("p",2026)[0]
        self.assertIs(type(cell["fps"]),float)
        self.assertEqual(cell["fps"],25.)
        self.assertIs(cell["probabilities"],value)

    def test_original_scoring_lock_cannot_be_replaced(self):
        with patch.object(runtime.study,"read",return_value={"original":"sealed"}), patch.object(runtime,"ORIGINAL_WRITE") as write:
            runtime.preserving_write(runtime.study.OUT/"score_lock.json",{"original":"sealed"})
            write.assert_not_called()
            with self.assertRaisesRegex(ValueError,"original scoring lock"):
                runtime.preserving_write(runtime.study.OUT/"score_lock.json",{"replacement":True})
