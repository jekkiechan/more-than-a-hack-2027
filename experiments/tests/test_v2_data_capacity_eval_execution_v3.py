import json
import unittest
from unittest.mock import patch

import numpy as np

from experiments import v2_data_capacity_eval_execution_v3 as execution


class JsonSafeReplayTests(unittest.TestCase):
    def test_score_row_drops_predictions_but_retains_private_probabilities(self):
        probabilities = {"cand": np.asarray([[.7, .2, .1]])}
        predictions = {"cand": np.asarray(["H"])}
        with patch.object(execution, "_ORIGINAL_SCORE_CELL", return_value={
                "metrics": {"cand": {"spell_f1": .5}},
                "predictions": predictions, "probabilities": probabilities}):
            row = execution._strip_predictions()
        self.assertNotIn("predictions", row)
        self.assertIs(row["probabilities"], probabilities)
        json_row = {k: v for k, v in row.items() if k != "probabilities"}
        json.dumps(json_row, allow_nan=False)
        np.testing.assert_array_equal(row["probabilities"]["cand"], probabilities["cand"])


if __name__ == "__main__":
    unittest.main()
