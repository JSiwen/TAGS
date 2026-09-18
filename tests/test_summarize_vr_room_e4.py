import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE_PATH = ROOT / "scripts" / "summarize_vr_room_e4.py"
_SPEC = importlib.util.spec_from_file_location("summarize_vr_room_e4", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

compare_to_baseline = _MOD.compare_to_baseline
evaluate_final = _MOD.evaluate_final
evaluate_multiseed = _MOD.evaluate_multiseed
evaluate_prescreen = _MOD.evaluate_prescreen


class Args:
    prescreen_lpips_improve = 0.03
    prescreen_psnr_drop = 0.25
    prescreen_ssim_drop = 0.003
    gate_lpips_improve = 0.05
    gate_psnr_drop = 0.2
    gate_psnr_drop_worst = 0.25
    gate_ssim_drop = 0.002
    gate_ssim_drop_worst = 0.003
    max_points = 2_000_000
    min_fps_speedup = 1.5


class SummarizeE4GateTests(unittest.TestCase):
    def test_prescreen_passes_good_candidate(self):
        baseline = {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222}
        case = {"psnr": 27.45, "ssim": 0.8785, "lpips": 0.210}
        delta = compare_to_baseline(case, baseline)
        result = evaluate_prescreen(delta, Args())
        self.assertTrue(result["passed"])

    def test_multiseed_gate_passes_three_seeds(self):
        baseline_by_seed = {
            0: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
            1: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
            2: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
        }
        profile_cases = [
            {"seed": 0, "psnr": 27.45, "ssim": 0.8788, "lpips": 0.208, "num_points": 1_900_000},
            {"seed": 1, "psnr": 27.44, "ssim": 0.8787, "lpips": 0.207, "num_points": 1_910_000},
            {"seed": 2, "psnr": 27.43, "ssim": 0.8786, "lpips": 0.209, "num_points": 1_905_000},
        ]
        result = evaluate_multiseed(profile_cases, baseline_by_seed, Args())
        self.assertTrue(result["quality_passed"])
        self.assertFalse(result["fps_passed"])
        self.assertFalse(result["passed"])

    def test_multiseed_gate_fails_when_one_seed_regresses_lpips(self):
        baseline_by_seed = {
            0: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
            1: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
            2: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
        }
        profile_cases = [
            {"seed": 0, "psnr": 27.45, "ssim": 0.8788, "lpips": 0.208, "num_points": 1_900_000},
            {"seed": 1, "psnr": 27.44, "ssim": 0.8787, "lpips": 0.223, "num_points": 1_910_000},
            {"seed": 2, "psnr": 27.43, "ssim": 0.8786, "lpips": 0.209, "num_points": 1_905_000},
        ]
        result = evaluate_multiseed(profile_cases, baseline_by_seed, Args())
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["all_seeds_lpips_not_worse"])

    def test_evaluate_final_uses_unlimited_fps_argument(self):
        baseline = {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222, "fps": 160.0}
        case = {
            "psnr": 27.45,
            "ssim": 0.8788,
            "lpips": 0.208,
            "num_points": 1_900_000,
            "fps": 120.0,
        }
        delta = compare_to_baseline(case, baseline)
        result = evaluate_final(case, baseline, delta, {}, Args(), unlimited_fps=80.0)
        self.assertTrue(result["quality_pass"])
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(result["fps_speedup_vs_unlimited"], 1.5)

    def test_multiseed_quality_and_fps_gates_are_separate(self):
        baseline_by_seed = {
            0: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
            1: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
            2: {"psnr": 27.5, "ssim": 0.879, "lpips": 0.222},
        }
        profile_cases = [
            {"seed": 0, "psnr": 27.45, "ssim": 0.8788, "lpips": 0.208, "num_points": 1_900_000, "fps": 120.0},
            {"seed": 1, "psnr": 27.44, "ssim": 0.8787, "lpips": 0.207, "num_points": 1_910_000, "fps": 120.0},
            {"seed": 2, "psnr": 27.43, "ssim": 0.8786, "lpips": 0.209, "num_points": 1_905_000, "fps": 120.0},
        ]
        result = evaluate_multiseed(profile_cases, baseline_by_seed, Args(), unlimited_fps=80.0)
        self.assertTrue(result["quality_passed"])
        self.assertTrue(result["fps_passed"])
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
