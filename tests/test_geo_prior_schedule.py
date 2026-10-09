import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = ROOT / "experiments" / "schedules.py"
_SPEC = importlib.util.spec_from_file_location("geo_schedules", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

linear_warmup_decay = _MOD.linear_warmup_decay
resolve_geo_decay_until = _MOD.resolve_geo_decay_until


def scheduled_weight(iteration, peak, final, decay_until, warmup_until=4000, decay_start=10000):
    return linear_warmup_decay(
        iteration,
        0,
        warmup_until,
        decay_start,
        decay_until,
        0.0,
        peak,
        final,
    )


class GeoPriorScheduleTests(unittest.TestCase):
    def test_shared_decay_reaches_min_at_20k(self):
        self.assertAlmostEqual(scheduled_weight(20000, 0.04, 0.01, 20000), 0.01)
        self.assertAlmostEqual(scheduled_weight(20000, 0.02, 0.0, 20000), 0.0)

    def test_extended_shared_decay_still_above_min_at_lpips_start(self):
        w_20k = scheduled_weight(20000, 0.04, 0.01, 24000)
        w_21k = scheduled_weight(21000, 0.04, 0.01, 24000)
        w_24k = scheduled_weight(24000, 0.04, 0.01, 24000)
        self.assertGreater(w_20k, 0.01)
        self.assertGreater(w_21k, 0.01)
        self.assertAlmostEqual(w_24k, 0.01)

    def test_depth_only_tail_keeps_normal_on_20k_schedule(self):
        self.assertEqual(resolve_geo_decay_until(27000, 27000), 27000)
        self.assertEqual(resolve_geo_decay_until(27000, 20000), 20000)
        depth_w = scheduled_weight(21000, 0.04, 0.01, 27000)
        normal_w = scheduled_weight(21000, 0.02, 0.0, 20000)
        self.assertGreater(depth_w, 0.01)
        self.assertAlmostEqual(normal_w, 0.0)

    def test_zero_override_inherits_shared_decay(self):
        self.assertEqual(resolve_geo_decay_until(24000, 0), 24000)
        self.assertEqual(resolve_geo_decay_until(24000, None), 24000)


if __name__ == "__main__":
    unittest.main()
