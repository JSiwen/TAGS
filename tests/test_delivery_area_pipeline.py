"""CPU checks for data encoding and scene conversion, not a GPU training test."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pipeline', str(REPO / 'scripts/delivery_area/pipeline.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class DeliveryAreaTests(unittest.TestCase):
    def test_uint16_and_affine_calibration(self):
        raw = np.arange(100, dtype=np.float32).reshape(10, 10)
        encoded = p.encode_depth(raw)
        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual((encoded.min(), encoded.max()), (0, 65535))
        decoded = encoded.ravel().astype(np.float64) / 65535
        fit = p.fit_scale(decoded, decoded * 2.7 + .2)
        self.assertAlmostEqual(fit['scale'], 2.7)
        self.assertAlmostEqual(fit['offset'], .2)
        with self.assertRaises(AssertionError):
            p.encode_depth(np.ones((2, 2)))

    def test_scene_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = p.RAW, p.SCENE, p.CODE
            try:
                p.RAW, p.SCENE, p.CODE = Path(tmp) / 'raw', Path(tmp) / 'scene', REPO
                src = p.RAW / 'images/dslr_images_undistorted'
                src.mkdir(parents=True)
                calibration = p.RAW / 'dslr_calibration_undistorted'
                calibration.mkdir()
                c = p.colmap()
                cams = {1: c.Camera(1, 'PINHOLE', 41, 29, np.array([30., 31., 19., 13.]))}
                images = {}
                for i in range(44):
                    name = '%04d.JPG' % i
                    Image.new('RGB', (41, 29), (i, 0, 0)).save(str(src / name))
                    images[i+1] = c.Image(i+1, np.array([1., 0, 0, 0]), np.zeros(3), 1,
                                         'dslr_images_undistorted/' + name, np.array([[10., 12.]]), np.array([-1]))
                c.write_model(cams, images, {}, str(calibration), '.txt')
                p.prepare()
                newcams, newimages, _ = p.model()
                np.testing.assert_array_equal(newcams[1].params, cams[1].params)
                np.testing.assert_array_equal(newimages[1].xys, images[1].xys)
                with Image.open(str(p.SCENE / 'images_4/0000.JPG')) as img:
                    self.assertEqual(img.size, (10, 7))
                saved = p.load(p.SCENE / 'preparation.json')
                self.assertEqual((len(saved['train']), len(saved['test'])), (38, 6))
                p.prepare()  # Verified idempotent reuse.
                (p.SCENE / 'images_4/0000.JPG').write_bytes(b'changed')
                with self.assertRaises(RuntimeError):
                    p.prepare()
            finally:
                p.RAW, p.SCENE, p.CODE = previous

    def test_scan_parser_rejects_partial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scan.log'
            path.write_text('F1-scores: 0.01 0.02 0.03 0.04 0.05 0.06\n')
            self.assertAlmostEqual(p.scan_f1(path)[1], .02)
            path.write_text('F1-scores: 0.01 0.02\n')
            with self.assertRaises(AssertionError):
                p.scan_f1(path)

    def test_summary_pairs_matching_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = p.OUT, p.gate
            try:
                p.OUT, p.gate = Path(tmp), lambda: None
                for seed in range(3):
                    for group in p.GROUPS:
                        name = '%s_seed%d' % (group, seed)
                        value = .1 + .01*seed
                        if group == 'tags_depth_dsine_w1':
                            value += .01*(seed+1)
                        m = p.OUT / 'garden' / name
                        p.save(m / 'experiment_manifest.json', dict(status='trained', wall_seconds=60))
                        p.save(m / 'followup_eval_30000/complete.json',
                               dict(views=6, points=100, rgb=dict(PSNR=20, SSIM=.8, LPIPS=.2)))
                        scan = p.OUT / 'geometry' / name
                        p.save(scan / 'complete.json', dict(files={}))
                        (scan / 'scan.log').write_text('F1-scores: 0.01 %s 0.3 0.4 0.5 0.6\n' % value)
                with contextlib.redirect_stdout(io.StringIO()):
                    p.summarize()
                summary = p.load(p.OUT / 'summary.json')
                comparison = summary['paired']['tags_depth_dsine_w1_minus_tags_depth_only']['F1_2cm_percent']
                np.testing.assert_allclose(comparison['seed_deltas'], [1, 2, 3])
                self.assertAlmostEqual(comparison['mean'], 2)
                self.assertAlmostEqual(comparison['sample_sd'], 1)
            finally:
                p.OUT, p.gate = previous


if __name__ == '__main__':
    unittest.main()
