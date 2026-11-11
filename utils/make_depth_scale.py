import numpy as np
import argparse
import cv2
import json
import os

try:
    from .read_write_model import *
except ImportError:
    from read_write_model import *

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    d = invmonodepthmap.astype(np.float32)
    # Depth Anything grayscale PNG is typically uint8; MiDaS-style exports often uint16.
    if np.issubdtype(invmonodepthmap.dtype, np.integer):
        if invmonodepthmap.dtype == np.uint8:
            invmonodepthmap = d / 255.0
        else:
            invmonodepthmap = d / (2**16)
    else:
        mx = float(d.max()) or 1.0
        invmonodepthmap = d / mx if mx > 1.0 else d
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = sample_inverse_depth(invmonodepthmap, maps)
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        if not np.isfinite(s_mono) or s_mono <= 1e-8:
            scale = 0.0
            offset = 0.0
        else:
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}


def sample_inverse_depth(depth_map, image_xy):
    """Sample a 2-D depth map at COLMAP xy coordinates, returning a 1-D array."""
    points = np.asarray(image_xy, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return np.empty((0,), dtype=np.float32)
    sampled = cv2.remap(
        np.asarray(depth_map, dtype=np.float32),
        points[:, 0].reshape(-1, 1),
        points[:, 1].reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument('--model_type', default="bin")
    parser.add_argument('--output', default="", help="Optional output JSON path; default is sparse/0/depth_params.json")
    args = parser.parse_args()


    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    depth_param_list = [
        get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args)
        for key in images_metas
    ]

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    output_path = args.output or f"{args.base_dir}/sparse/0/depth_params.json"
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(depth_params, f, indent=2)

    print("Wrote {} calibration entries to {}".format(len(depth_params), output_path))
