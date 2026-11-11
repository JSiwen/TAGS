import unittest

import numpy as np

from utils.depth_encoding import decode_inverse_depth
from utils.make_depth_scale import sample_inverse_depth


class DepthInputDecodingTests(unittest.TestCase):
    def test_auto_decodes_uint8_as_monocular_inverse_depth(self):
        raw = np.array([[0, 50, 51, 255]], dtype=np.uint8)
        actual = decode_inverse_depth(raw, encoding="auto")
        expected = raw.astype(np.float32) / 255.0
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.all(np.diff(actual[0]) >= 0))

    def test_legacy_threshold_reverses_uint8_inverse_depth_order(self):
        raw = np.array([[50, 51, 255]], dtype=np.uint8)
        actual = decode_inverse_depth(raw, encoding="legacy_threshold")
        np.testing.assert_allclose(
            actual, [[20.0, 1000.0 / 51.0, 1000.0 / 255.0]], rtol=1e-6
        )
        self.assertTrue(np.all(np.diff(actual[0]) < 0))

    def test_uint16_sensor_depth_is_decoded_as_inverse_meters(self):
        raw = np.array([[1000, 2000]], dtype=np.uint16)
        actual = decode_inverse_depth(raw, encoding="sensor_mm")
        np.testing.assert_allclose(actual, [[1.0, 0.5]])

    def test_depth_scale_sampling_returns_all_samples(self):
        depth = np.array([[0, 1], [2, 3]], dtype=np.float32)
        xy = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
        actual = sample_inverse_depth(depth, xy)
        np.testing.assert_allclose(actual, [1.5, 3.0])


if __name__ == "__main__":
    unittest.main()
