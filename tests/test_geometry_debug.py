"""Deterministic geometric invariants and failure-path regression tests."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import contextlib
import io
import json
import tempfile
import numpy as np

spec = importlib.util.spec_from_file_location("geometry", str(Path(__file__).resolve().parents[1] / "scripts/geometry_debug.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[10., 0, 3.], [0, 12., 2.], [0, 0, 1.]])
        self.W = np.eye(4)
        self.W[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        self.W[:3, 3] = [.3, -.2, 1.]
        self.depth = np.full((5, 7), 2.)
        self.mask = np.ones((5, 7), dtype=bool)
        self.b = dict(depth=self.depth, mask=self.mask, K=self.K, w2c=self.W)
        self.cfg = dict(stride=1, relative_depth_tolerance=.02, radius_thresholds=[.01, .05])

    def test_roundtrip_nontrivial_pose(self):
        xyz, uv = g.backproject(self.depth, self.K, self.W, self.mask)
        actual_uv, z = g.project(xyz, self.K, self.W)
        np.testing.assert_allclose(actual_uv, uv, atol=1e-12)
        np.testing.assert_allclose(z, 2.)

    def test_plane_camera_translation(self):
        target = dict(self.b, w2c=self.W.copy())
        target["w2c"][0, 3] += .2
        result = g.consistency(self.b, target, self.cfg)
        self.assertGreater(result["scored"], 0)
        self.assertAlmostEqual(result["consistent_fraction"], 1.)

    def test_occlusion_and_front_disagreement(self):
        behind = g.consistency(self.b, dict(self.b, depth=self.depth * .5), self.cfg)
        self.assertEqual(behind["occluded"], 35)
        self.assertIsNone(behind["visible_absrel"])
        front = g.consistency(self.b, dict(self.b, depth=self.depth * 2), self.cfg)
        self.assertEqual(front["scored"], 35)
        self.assertEqual(front["consistent_fraction"], 0.)
        self.assertAlmostEqual(front["visible_absrel"], .5)

    def test_invalid_depth_and_mask(self):
        depth = self.depth.copy()
        depth[0, :3] = [np.nan, np.inf, 0]
        mask = self.mask.copy()
        mask[1, 0] = False
        xyz, _ = g.backproject(depth, self.K, self.W, mask)
        self.assertEqual(len(xyz), 31)
        result = g.consistency(self.b, dict(self.b, mask=np.zeros_like(mask)), self.cfg)
        self.assertEqual(result["target_valid"], 0)
        self.assertIsNone(result["consistent_fraction"])

    def test_sparse_perfect_and_scaled(self):
        xyz, _ = g.backproject(self.depth, self.K, self.W, self.mask)
        points = dict(enumerate(xyz))
        view = dict(observed_ids=list(points))
        result = g.sparse_metrics(self.b, view, points, 1., self.cfg)
        self.assertLess(result["depth_absrel"], 1e-12)
        self.assertAlmostEqual(result["sparse_to_surface_coverage"]["0.01"], 1.)
        result = g.sparse_metrics(dict(self.b, depth=self.depth * 2), view, points, 1., self.cfg)
        self.assertAlmostEqual(result["depth_absrel"], 1.)

    def test_empty_metrics_are_null(self):
        summary = g.summarize(dict(views=[], directed_pairs=[], surface_points=0))
        self.assertIsNone(summary["consistency_fraction_sample_weighted"])
        self.assertIsNone(summary["sfm_depth_absrel_observation_weighted"])


class RecordTests(unittest.TestCase):
    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return g.main(args)

    def assert_record(self, folder, status):
        record = json.loads((folder / "record.json").read_text())
        self.assertEqual(record["status"], status)
        self.assertIn("finished_utc", record)
        self.assertTrue((folder / "console.log").is_file())
        self.assertTrue((folder / "config_snapshot.json").is_file())
        self.assertTrue((folder / "summary.json").is_file())
        inventory = json.loads((folder / "artifacts.json").read_text())
        for item in inventory["files"]:
            self.assertEqual(g.sha256(folder / item["path"]), item["sha256"])
        return record

    def test_repeated_preflight_unique_records(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cfg = base / "config.json"
            cfg.write_text('{}')
            args = ["--config", str(cfg), "--output", str(base / "runs"), "--stage", "preflight"]
            with mock.patch.object(g, "execute", return_value=dict(counts={}, selected_views=[])):
                self.assertEqual(self.invoke(args), 0)
                first = next((base / "runs").iterdir())
                before = (first / "record.json").read_bytes()
                self.assertEqual(self.invoke(args), 0)
            folders = list((base / "runs").iterdir())
            self.assertEqual(len(folders), 2)
            self.assertEqual(before, (first / "record.json").read_bytes())
            for folder in folders:
                self.assert_record(folder, "ok")

    def test_invalid_json_and_non_object_recorded(self):
        for content in ['{bad json', '[]']:
            with tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                cfg = base / "bad.json"
                cfg.write_text(content)
                self.assertEqual(self.invoke(["--config", str(cfg), "--output", str(base / "runs")]), 1)
                folder = next((base / "runs").iterdir())
                record = self.assert_record(folder, "failed")
                self.assertIn("traceback", record)
                self.assertFalse((folder / "metrics.json").exists())

    def test_missing_config_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.assertEqual(self.invoke(["--config", str(base / "missing.json"), "--output", str(base / "runs")]), 1)
            self.assert_record(next((base / "runs").iterdir()), "failed")

    def test_missing_checkpoint_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template = Path(__file__).resolve().parents[1] / "configs/bicycle_v16_25views.json"
            self.assertEqual(self.invoke(["--repo", str(base), "--config", str(template), "--output", str(base / "runs"), "--stage", "preflight"]), 1)
            record = self.assert_record(next((base / "runs").iterdir()), "failed")
            self.assertEqual(record["error_type"], "FileNotFoundError")

    def test_bad_cli_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with mock.patch.object(Path, "cwd", return_value=base):
                self.assertEqual(self.invoke(["--unknown-option"]), 1)
            folder = next((base / "geometry_records").iterdir())
            record = self.assert_record(folder, "failed")
            self.assertEqual(record["stage"], "cli_error")

    def test_interrupt_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cfg = base / "config.json"
            cfg.write_text('{}')
            with mock.patch.object(g, "execute", side_effect=KeyboardInterrupt):
                self.assertEqual(self.invoke(["--config", str(cfg), "--output", str(base / "runs")]), 130)
            self.assert_record(next((base / "runs").iterdir()), "interrupted")

    def test_success_writes_metrics_csv_report(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cfg = base / "config.json"
            cfg.write_text('{}')
            def success(args, repo, config, out):
                result = dict(views=[], directed_pairs=[], surface_points=0, caveat="synthetic fixture", depth_semantics="Z")
                return dict(metrics=g.save_measurements(out, result))
            with mock.patch.object(g, "execute", side_effect=success):
                self.assertEqual(self.invoke(["--config", str(cfg), "--output", str(base / "runs")]), 0)
            folder = next((base / "runs").iterdir())
            self.assert_record(folder, "ok")
            for name in ['metrics.json', 'per_view.csv', 'per_pair.csv', 'REPORT.md']:
                self.assertTrue((folder / name).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
