"""Decode supported depth image encodings into inverse-depth maps."""

import numpy as np


DEPTH_INPUT_ENCODINGS = (
    "auto",
    "monocular_inverse",
    "sensor_mm",
    "legacy_threshold",
)


def decode_inverse_depth(raw, encoding="auto", is_nerf_synthetic=False):
    """Return a float32 inverse-depth map without applying COLMAP calibration.

    ``auto`` treats uint8 maps as normalized monocular inverse depth (raw/255).
    For backwards compatibility it keeps the historical uint16 heuristic:
    values above 50 are millimeters, otherwise normalized by 65535. Use an
    explicit encoding when the uint16 source is known.
    """
    array = np.asarray(raw)
    if array.ndim == 3:
        if array.shape[2] < 1:
            raise ValueError("Depth image has no channels")
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("Depth image must be HxW or HxWxC; got {}".format(array.shape))

    mode = str(encoding or "auto").lower()
    if mode not in DEPTH_INPUT_ENCODINGS:
        raise ValueError(
            "Unsupported depth_input_encoding={!r}; choose one of {}".format(
                mode, ", ".join(DEPTH_INPUT_ENCODINGS)
            )
        )

    values = array.astype(np.float32, copy=False)
    if is_nerf_synthetic:
        return values / 512.0

    if mode == "legacy_threshold":
        if float(np.nanmax(values)) > 50.0:
            depth_m = values * 1e-3
            invdepth = np.zeros_like(values, dtype=np.float32)
            valid = np.isfinite(depth_m) & (depth_m > 1e-6)
            invdepth[valid] = 1.0 / depth_m[valid]
            return invdepth
        return values / float(2 ** 16)

    if mode == "sensor_mm":
        depth_m = values * 1e-3
        invdepth = np.zeros_like(values, dtype=np.float32)
        valid = np.isfinite(depth_m) & (depth_m > 1e-6)
        invdepth[valid] = 1.0 / depth_m[valid]
        return invdepth

    if mode == "monocular_inverse":
        if np.issubdtype(array.dtype, np.integer):
            return values / float(np.iinfo(array.dtype).max)
        if not np.isfinite(values).all():
            raise ValueError("Monocular inverse-depth image contains non-finite values")
        return values

    # Auto mode uses dtype to distinguish the known Bicycle uint8 prior from
    # uint16 sensor depth. It never decides that uint8 values are millimeters.
    if array.dtype == np.uint8:
        return values / 255.0
    if array.dtype == np.uint16:
        if float(np.nanmax(values)) > 50.0:
            depth_m = values * 1e-3
            invdepth = np.zeros_like(values, dtype=np.float32)
            valid = np.isfinite(depth_m) & (depth_m > 1e-6)
            invdepth[valid] = 1.0 / depth_m[valid]
            return invdepth
        return values / float(2 ** 16)
    if np.issubdtype(array.dtype, np.integer):
        return values / float(np.iinfo(array.dtype).max)
    raise ValueError(
        "Auto depth decoding does not recognize dtype {}; use an explicit "
        "depth_input_encoding".format(array.dtype)
    )
