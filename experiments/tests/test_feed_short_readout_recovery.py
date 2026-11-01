import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.feed_short_calibration_readout_v2 import write_complete

class ReadoutSerialization(unittest.TestCase):
    def test_numpy_scalar_types_serialize_and_existing_result_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json'
            write_complete(path,{'guards':{'a':np.bool_(False)},'score':np.float64(.5274),'count':np.int64(435)})
            saved=path.read_bytes();self.assertEqual(json.loads(saved),{'guards':{'a':False},'score':.5274,'count':435})
            with self.assertRaises(FileExistsError):write_complete(path,{'guards':False})
            self.assertEqual(path.read_bytes(),saved)

    def test_invalid_values_fail_before_creating_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json'
            with self.assertRaises(ValueError):write_complete(path,{'score':np.float64('nan')})
            self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
