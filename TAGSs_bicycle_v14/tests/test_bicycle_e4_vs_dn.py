import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


export_mod = _load("export_bicycle_mono_depth", "scripts/export_bicycle_mono_depth.py")
summary_mod = _load("summarize_bicycle_e4_vs_dn", "scripts/summarize_bicycle_e4_vs_dn.py")


class DepthExportTests(unittest.TestCase):
    def test_uint8_inverse_depth_matches_make_depth_scale(self):
        raw = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        normalized = export_mod.normalize_inverse_depth(raw)
        np.testing.assert_allclose(normalized, raw.astype(np.float32) / 255.0)

    def test_scale_offset_then_metric(self):
        inverse = np.array([[0.5, 0.0], [0.25, 1.0]], dtype=np.float32)
        aligned = export_mod.apply_scale_offset(inverse, scale=2.0, offset=-0.5)
        metric = export_mod.inverse_to_metric(aligned)
        self.assertAlmostEqual(float(metric[0, 0]), 1.0 / 0.5)
        self.assertEqual(float(metric[0, 1]), 0.0)
        self.assertAlmostEqual(float(metric[1, 1]), 1.0 / 1.5)

    def test_holdout_matches_sorted_llffhold(self):
        names = [f"_DSC{index:04d}.JPG" for index in range(16)]
        holdout = export_mod.holdout_names(names, 8)
        self.assertEqual(holdout, ["_DSC0000.JPG", "_DSC0008.JPG"])

    def test_tags_resolution_1_keeps_images_4(self):
        self.assertEqual(export_mod.tags_train_resolution(1237, 822, 1), (1237, 822))

    def test_tags_resolution_caps_width_at_800(self):
        self.assertEqual(export_mod.tags_train_resolution(1237, 822, 800), (800, 531))

    def test_rewrite_transforms_scales_intrinsics_once(self):
        payload = {
            "w": 4946,
            "h": 3286,
            "fl_x": 4649.5,
            "fl_y": 4627.3,
            "cx": 2473.0,
            "cy": 1643.0,
            "frames": [{"file_path": "images/_DSC8679.JPG", "transform_matrix": [[1, 0, 0, 0]]}],
        }
        rewritten, changed = export_mod.rewrite_transforms(payload, "images_4", (1237, 822))
        self.assertTrue(changed)
        self.assertEqual(rewritten["w"], 1237)
        self.assertEqual(rewritten["h"], 822)
        self.assertAlmostEqual(rewritten["fl_x"], 4649.5 * 1237 / 4946)
        self.assertEqual(rewritten["frames"][0]["file_path"], "images_4/_DSC8679.JPG")
        self.assertEqual(rewritten["frames"][0]["w"], 1237)
        again, changed_again = export_mod.rewrite_transforms(rewritten, "images_4", (1237, 822))
        self.assertFalse(changed_again)
        self.assertEqual(again["fl_x"], rewritten["fl_x"])


class SummaryTests(unittest.TestCase):
    def test_delta_and_ply_count_and_stage(self):
        tags = {"PSNR": 25.0, "SSIM": 0.80, "LPIPS": 0.20, "num_points": 1_900_000}
        dn = {"PSNR": 24.0, "SSIM": 0.75, "LPIPS": 0.18, "num_points": 1_000_000}
        delta = summary_mod.compare_metrics(tags, dn)
        self.assertAlmostEqual(delta["psnr"], 1.0)
        self.assertAlmostEqual(delta["ssim"], 0.05)
        self.assertAlmostEqual(delta["lpips"], 0.02)
        self.assertAlmostEqual(delta["num_points_ratio"], 1.9)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply = root / "point_cloud.ply"
            ply.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nend_header\n0 0 0\n")
            self.assertEqual(summary_mod.count_ply_vertices(ply), 3)

            src = root / "eval"
            (src / "pred" / "rgb").mkdir(parents=True)
            (src / "gt" / "rgb").mkdir(parents=True)
            (src / "pred" / "rgb" / "_DSC8679.png").write_bytes(b"pred")
            (src / "gt" / "rgb" / "_DSC8679.png").write_bytes(b"gt")
            copied = summary_mod.stage_dn_renders(src, root / "test" / "dn-splatter")
            self.assertEqual(copied, 1)
            self.assertEqual((root / "test" / "dn-splatter" / "renders" / "_DSC8679.png").read_bytes(), b"pred")

            tags_dir = root / "tags"
            dn_dir = root / "dn"
            tags_dir.mkdir()
            dn_dir.mkdir()
            (tags_dir / "per_view.json").write_text(
                json.dumps({"ours_30000": {"PSNR": {"_DSC8679.JPG": 25.0}, "SSIM": {"_DSC8679.JPG": 0.8}, "LPIPS": {"_DSC8679.JPG": 0.2}}}),
                encoding="utf-8",
            )
            (dn_dir / "per_view.json").write_text(
                json.dumps({"dn-splatter": {"PSNR": {"_DSC8679.png": 24.0}, "SSIM": {"_DSC8679.png": 0.7}, "LPIPS": {"_DSC8679.png": 0.3}}}),
                encoding="utf-8",
            )
            wins = summary_mod.per_view_wins(
                summary_mod.read_per_view(tags_dir, "ours_30000"),
                summary_mod.read_per_view(dn_dir, "dn-splatter"),
            )
            self.assertEqual(wins["num_views"], 1)
            self.assertEqual(wins["psnr_tags_better"], 1)
            self.assertEqual(wins["lpips_tags_better"], 1)


if __name__ == "__main__":
    unittest.main()
