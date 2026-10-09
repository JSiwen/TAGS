import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE_PATH = ROOT / "scripts" / "summarize_vr_room_e4b.py"
_SPEC = importlib.util.spec_from_file_location("summarize_vr_room_e4b", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

depth_delta = _MOD.depth_delta
evaluate_final = _MOD.evaluate_final
evaluate_multiseed = _MOD.evaluate_multiseed
evaluate_prescreen = _MOD.evaluate_prescreen
per_view_improve_counts = _MOD.per_view_improve_counts


class Args:
    prescreen_lpips_vs_rgb = 0.12
    prescreen_psnr_drop = 0.08
    prescreen_ssim_drop = 0.001
    prescreen_ssim_improved_views = 18
    gate_lpips_vs_rgb = 0.12
    gate_lpips_vs_freeze_abs_slack = 0.002
    gate_psnr_drop = 0.05
    gate_psnr_drop_worst = 0.08
    gate_ssim_drop = 0.0002
    gate_ssim_drop_worst = 0.001
    gate_depth_log_l1_improve = 0.03
    max_points = 2_000_000
    min_fps_speedup = 1.5


def freeze_case(seed, extra=None):
    case = {
        "seed": seed,
        "psnr": 27.473,
        "ssim": 0.87780,
        "lpips": 0.19137,
        "num_points": 1_908_000,
        "fps": 165.0,
        "depth_prior": {
            "pearson": {"mean": 0.958},
            "spearman": {"mean": 0.953},
            "log_l1": {"mean": 0.0557},
        },
    }
    if extra:
        case.update(extra)
    return case


def rgb_case(seed):
    return {
        "seed": seed,
        "psnr": 27.428,
        "ssim": 0.87857,
        "lpips": 0.22296,
        "num_points": 1_916_000,
        "fps": 168.9,
    }


class SummarizeE4bGateTests(unittest.TestCase):
    def test_prescreen_passes_overlap_candidate(self):
        freeze = freeze_case(0)
        rgb = rgb_case(0)
        case = {
            "psnr": 27.48,
            "ssim": 0.87810,
            "lpips": 0.1910,
            "num_points": 1_900_000,
            "depth_prior": {
                "pearson": {"mean": 0.959},
                "spearman": {"mean": 0.954},
                "log_l1": {"mean": 0.0535},
            },
        }
        bundle = {
            "vs_freeze": _MOD.e4sum.compare_to_baseline(case, freeze),
            "vs_rgb_ssim": _MOD.e4sum.compare_to_baseline(case, rgb),
            "depth_vs_freeze": depth_delta(case["depth_prior"], freeze["depth_prior"]),
            "per_view_vs_rgb_ssim": {"ssim_improved": 52, "num_views": 42},
        }
        result = evaluate_prescreen(bundle, Args())
        self.assertTrue(result["passed"])

    def test_final_fails_when_depth_does_not_improve(self):
        freeze = freeze_case(0)
        rgb = rgb_case(0)
        case = {
            "psnr": 27.48,
            "ssim": 0.87810,
            "lpips": 0.1910,
            "num_points": 1_900_000,
            "fps": 164.0,
            "depth_prior": {
                "pearson": {"mean": 0.957},
                "spearman": {"mean": 0.952},
                "log_l1": {"mean": 0.0556},
            },
        }
        bundle = {
            "vs_freeze": _MOD.e4sum.compare_to_baseline(case, freeze),
            "vs_rgb_ssim": _MOD.e4sum.compare_to_baseline(case, rgb),
            "depth_vs_freeze": depth_delta(case["depth_prior"], freeze["depth_prior"]),
        }
        result = evaluate_final(case, bundle, Args(), unlimited_fps=80.0)
        self.assertFalse(result["quality_pass"])
        self.assertIn("depth_log_l1_improve<3pct", result["fail_reasons"])

    def test_multiseed_gate_passes_three_seeds(self):
        freeze_by_seed = {seed: freeze_case(seed) for seed in (0, 1, 2)}
        rgb_by_seed = {seed: rgb_case(seed) for seed in (0, 1, 2)}
        profile_cases = [
            freeze_case(
                0,
                {
                    "psnr": 27.49,
                    "ssim": 0.87805,
                    "lpips": 0.1908,
                    "num_points": 1_900_000,
                    "fps": 164.0,
                    "depth_prior": {
                        "pearson": {"mean": 0.960},
                        "spearman": {"mean": 0.955},
                        "log_l1": {"mean": 0.0535},
                    },
                },
            ),
            freeze_case(
                1,
                {
                    "psnr": 27.50,
                    "ssim": 0.87810,
                    "lpips": 0.1905,
                    "num_points": 1_905_000,
                    "fps": 164.5,
                    "depth_prior": {
                        "pearson": {"mean": 0.959},
                        "spearman": {"mean": 0.954},
                        "log_l1": {"mean": 0.0538},
                    },
                },
            ),
            freeze_case(
                2,
                {
                    "psnr": 27.47,
                    "ssim": 0.87795,
                    "lpips": 0.1910,
                    "num_points": 1_902_000,
                    "fps": 163.8,
                    "depth_prior": {
                        "pearson": {"mean": 0.959},
                        "spearman": {"mean": 0.954},
                        "log_l1": {"mean": 0.0536},
                    },
                },
            ),
        ]
        result = evaluate_multiseed(profile_cases, freeze_by_seed, rgb_by_seed, Args(), unlimited_fps=80.0)
        self.assertTrue(result["quality_passed"])
        self.assertTrue(result["fps_passed"])
        self.assertTrue(result["passed"])

    def test_per_view_counts_match_direction(self):
        case = {
            "views": [
                {"image_name": "a", "lpips": 0.18, "psnr": 28.0, "ssim": 0.88},
                {"image_name": "b", "lpips": 0.22, "psnr": 26.0, "ssim": 0.86},
            ]
        }
        baseline = {
            "views": [
                {"image_name": "a", "lpips": 0.20, "psnr": 27.5, "ssim": 0.87},
                {"image_name": "b", "lpips": 0.21, "psnr": 26.5, "ssim": 0.87},
            ]
        }
        counts = per_view_improve_counts(case, baseline)
        self.assertEqual(counts["lpips_improved"], 1)
        self.assertEqual(counts["psnr_improved"], 1)
        self.assertEqual(counts["ssim_improved"], 1)


if __name__ == "__main__":
    unittest.main()
