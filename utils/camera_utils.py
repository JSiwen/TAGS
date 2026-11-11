#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal
from utils.depth_encoding import decode_inverse_depth
from PIL import Image
import cv2
import errno
import io
import time

WARNED = False
DEPTH_DECODING_LOGGED = False


def _load_pil_image_eager(path, max_attempts=8):
    """
    Read the file into memory and decode fully. Avoids lazy JPEG loads that fail later in resize()
    with transient EIO on network filesystems (e.g. JuiceFS). Retries only on errno.EIO.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            with open(path, "rb") as f:
                data = f.read()
            image = Image.open(io.BytesIO(data))
            image.load()
            return image
        except OSError as e:
            last_err = e
            if e.errno != errno.EIO:
                raise
            time.sleep(min(2.0, 0.15 * (2 ** attempt)))
    raise OSError(
        errno.EIO,
        f"{max_attempts}× EIO reading image {path}; copy the scene to local disk or retry when storage is stable."
    ) from last_err


def _cv2_imread_retry(path, flags, max_attempts=8):
    last_err = None
    for attempt in range(max_attempts):
        try:
            with open(path, "rb") as f:
                buf = f.read()
            im = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), flags)
            if im is None:
                raise IOError(f"cv2.imdecode returned None for {path}")
            return im
        except OSError as e:
            last_err = e
            if e.errno != errno.EIO:
                raise
            time.sleep(min(2.0, 0.15 * (2 ** attempt)))
    raise OSError(
        errno.EIO,
        f"{max_attempts}× EIO reading depth {path}; copy the scene to local disk or retry when storage is stable.",
    ) from last_err


def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = _load_pil_image_eager(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            raw = _cv2_imread_retry(cam_info.depth_path, cv2.IMREAD_UNCHANGED)
            encoding = getattr(args, "depth_input_encoding", "auto")
            invdepthmap = decode_inverse_depth(
                raw, encoding=encoding, is_nerf_synthetic=is_nerf_synthetic
            )
            global DEPTH_DECODING_LOGGED
            if not DEPTH_DECODING_LOGGED:
                print(
                    "[DepthInput] encoding={} dtype={} raw_range=[{}, {}] "
                    "inverse_range=[{:.6g}, {:.6g}]".format(
                        encoding,
                        raw.dtype,
                        float(np.nanmin(raw)),
                        float(np.nanmax(raw)),
                        float(np.nanmin(invdepthmap)),
                        float(np.nanmax(invdepthmap)),
                    )
                )
                DEPTH_DECODING_LOGGED = True

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    normal_map = None
    normal_mask = None
    if getattr(cam_info, "normal_path", ""):
        normal_np = np.load(cam_info.normal_path, allow_pickle=False)
        if normal_np.dtype != np.float32:
            raise TypeError(
                f"Normal map must be float32: {cam_info.normal_path} has {normal_np.dtype}"
            )
        if normal_np.shape != (orig_h, orig_w, 3):
            raise ValueError(
                f"Normal map size mismatch for {cam_info.image_name}: "
                f"got {normal_np.shape}, expected {(orig_h, orig_w, 3)}"
            )
        finite = np.isfinite(normal_np).all(axis=2)
        lengths = np.linalg.norm(np.nan_to_num(normal_np, nan=0.0, posinf=0.0, neginf=0.0), axis=2)
        valid = finite & (lengths > 1e-6)
        normal_np = np.nan_to_num(normal_np, nan=0.0, posinf=0.0, neginf=0.0)
        normal_np = normal_np / np.maximum(lengths[..., None], 1e-6)
        normal_t = torch.from_numpy(normal_np).permute(2, 0, 1).unsqueeze(0).float()
        mask_t = torch.from_numpy(valid.astype(np.float32))[None, None]
        target_size = (resolution[1], resolution[0])
        if normal_t.shape[-2:] != target_size:
            normal_t = F.interpolate(
                normal_t, size=target_size, mode="bilinear", align_corners=False
            )
            mask_t = F.interpolate(mask_t, size=target_size, mode="nearest")
        normal_t = F.normalize(normal_t, dim=1, eps=1e-6)
        normal_map = normal_t[0]
        normal_mask = mask_t[0]

    intrinsics = None
    if getattr(cam_info, "intrinsics", None) is not None:
        intrinsics = np.asarray(cam_info.intrinsics, dtype=np.float32).copy()
        intrinsics[0, :] *= float(resolution[0]) / float(cam_info.width)
        intrinsics[1, :] *= float(resolution[1]) / float(cam_info.height)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  normal_map=normal_map, normal_mask=normal_mask, intrinsics=intrinsics)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
