import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checkpoint_io_test", ROOT / "experiments/checkpoint_io.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CheckpointIOTests(unittest.TestCase):
    def test_legacy_load_does_not_forward_weights_only_to_pickle(self):
        def load(path, map_location=None, **pickle_load_args):
            self.assertEqual(path, "checkpoint.pth")
            self.assertEqual(map_location, "cuda:0")
            self.assertEqual(pickle_load_args, {})
            return {"version": 2}
        self.assertEqual(module.load_training_checkpoint(SimpleNamespace(load=load), "checkpoint.pth", "cuda:0"), {"version": 2})

    def test_modern_load_retains_full_rng_state(self):
        def load(path, map_location=None, *, weights_only=True, **kwargs):
            self.assertIs(weights_only, False)
            return {"numpy_rng": "restored"}
        self.assertEqual(module.load_training_checkpoint(SimpleNamespace(load=load), "checkpoint.pth"), {"numpy_rng": "restored"})

    def test_corrupt_file_error_propagates_without_retry(self):
        calls = []
        def load(path, map_location=None, **kwargs):
            calls.append(path)
            raise TypeError("corrupt checkpoint")
        with self.assertRaisesRegex(TypeError, "corrupt checkpoint"):
            module.load_training_checkpoint(SimpleNamespace(load=load), "checkpoint.pth")
        self.assertEqual(len(calls), 1)

if __name__ == "__main__":
    unittest.main()
