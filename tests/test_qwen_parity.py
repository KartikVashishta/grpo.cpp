import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/countdown"))
from parity import compare


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
class ParityTest(unittest.TestCase):
    def test_both_pools_are_unscaled_and_zero_reference_is_checked(self):
        import numpy as np
        specs = [{"name": pool, "pool": pool, "offset": 0, "count": 3} for pool in ("half", "float")]
        counts = {"half": 3, "float": 3}
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp)/"native", Path(tmp)/"reference"
            left.mkdir()
            right.mkdir()
            for pool in counts:
                np.array([0.25, -0.5, 0.], dtype=np.float32).tofile(right/(pool+".bin"))
                np.array([64., -128., 0.], dtype=np.float16 if pool == "half" else np.float32).tofile(left/(pool+".bin"))
            result = compare(left, right, specs, counts, left_scale=256)
            self.assertEqual(result["relative_l2"], 0.)
            self.assertTrue(all(t["passed"] for t in result["tensors"]))
            np.zeros(3, dtype=np.float32).tofile(right/"float.bin")
            result = compare(left, right, specs, counts, left_scale=256)
            self.assertFalse(result["tensors"][1]["passed"])
            np.array([np.nan, 0., 0.], dtype=np.float32).tofile(left/"float.bin")
            self.assertFalse(compare(left, right, specs, counts)["finite"])


if __name__ == "__main__":
    unittest.main()
