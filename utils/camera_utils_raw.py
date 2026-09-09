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
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import errno
import io
import time

WARNED = False


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
            if is_nerf_synthetic:
                invdepthmap = _cv2_imread_retry(cam_info.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 512
            else:
                invdepthmap = _cv2_imread_retry(cam_info.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / float(2**16)

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

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

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