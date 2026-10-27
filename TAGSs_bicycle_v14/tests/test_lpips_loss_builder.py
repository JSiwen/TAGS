import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE_PATH = ROOT / "experiments" / "loss_builder.py"
_SPEC = importlib.util.spec_from_file_location("loss_builder", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

LossBuilder = _MOD.LossBuilder
_normalize_weights = _MOD._normalize_weights
_parse_float_list = _MOD._parse_float_list


class DummyGeoPrior:
    def __call__(self, *args, **kwargs):
        import torch

        return torch.tensor(0.0), {}


def make_builder(
    *,
    schedule="step",
    final=0.03,
    start=21000,
    multiscale=False,
    lambda_dssim=0.2,
):
    opt = SimpleNamespace(iterations=30000, lambda_dssim=lambda_dssim)
    cfg = SimpleNamespace(
        detail_loss_start_iter=0,
        detail_loss_weight_init=0.0,
        detail_loss_weight_final=0.0,
        detail_loss_max_weight=0.05,
        detail_loss_edge_quantile=0.75,
        lpips_loss_start_iter=start,
        lpips_loss_weight_init=0.0,
        lpips_loss_weight_final=final,
        lpips_loss_max_weight=max(final, 0.05),
        lpips_loss_interval=1,
        lpips_loss_schedule=schedule,
        enable_multiscale_lpips=multiscale,
        lpips_multiscale_scales="1.0,0.5,0.25",
        lpips_multiscale_weights="0.5,0.3,0.2",
        psnr_loss_start_iter=0,
        psnr_loss_weight_init=0.0,
        psnr_loss_weight_final=0.0,
        psnr_loss_max_weight=0.05,
        color_align_start_iter=0,
        color_align_weight_init=0.0,
        color_align_weight_final=0.0,
        color_align_max_weight=0.03,
        color_align_std_weight=0.5,
        color_align_hard_view_boost=1.5,
        color_align_low_grad_quantile=0.60,
        color_align_min_mask_fraction=0.20,
        color_align_require_fullres=True,
        hard_view_boost_max=3.0,
        enable_geo_prior=False,
    )
    return LossBuilder(opt, cfg, DummyGeoPrior())


class LPIPSLossBuilderTests(unittest.TestCase):
    def test_step_schedule_boundaries(self):
        builder = make_builder(schedule="step", final=0.03, start=21000)
        self.assertEqual(builder._lpips_weight(20999), 0.0)
        self.assertAlmostEqual(builder._lpips_weight(21000), 0.03)
        self.assertAlmostEqual(builder._lpips_weight(30000), 0.03)

    def test_linear_schedule_ramps(self):
        builder = make_builder(schedule="linear", final=0.03, start=21000)
        self.assertEqual(builder._lpips_weight(20999), 0.0)
        self.assertAlmostEqual(builder._lpips_weight(25500), 0.015)
        self.assertAlmostEqual(builder._lpips_weight(30000), 0.03)

    def test_multiscale_weights_normalize(self):
        weights = _normalize_weights([0.5, 0.3, 0.2])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0], 0.5)

    def test_parse_float_list(self):
        self.assertEqual(_parse_float_list("1.0,0.5,0.25", ()), [1.0, 0.5, 0.25])

    def test_multiscale_builder_parses_scales(self):
        builder = make_builder(multiscale=True)
        self.assertEqual(builder._lpips_scales, [1.0, 0.5, 0.25])
        self.assertAlmostEqual(sum(builder._lpips_scale_weights), 1.0)

    def test_lpips_criterion_is_cached(self):
        import torch
        from lpipsPyTorch import get_lpips_criterion

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for LPIPS cache test")
        first = get_lpips_criterion("vgg", "0.1", "cuda")
        second = get_lpips_criterion("vgg", "0.1", "cuda")
        self.assertIs(first, second)
        for param in first.parameters():
            self.assertFalse(param.requires_grad)

    def test_lpips_forward_has_finite_grad(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for LPIPS gradient test")
        builder = make_builder(schedule="step", final=0.01, start=0, multiscale=False)
        pred = torch.rand(3, 64, 64, device="cuda", requires_grad=True)
        target = torch.rand(3, 64, 64, device="cuda")
        loss, _ = builder._compute_lpips_loss(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())


if __name__ == "__main__":
    unittest.main()
