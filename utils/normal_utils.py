import torch
import torch.nn.functional as F


def depth_to_normals(inv_depth, eps=1e-6, normalize_gradients=True):
    """
    Approximate image-space normals from inverse depth using finite differences.
    inv_depth: [1, H, W] or [H, W]
    returns: [3, H, W] in camera-like image space.
    """
    if inv_depth.dim() == 3:
        d = inv_depth[0]
    else:
        d = inv_depth

    dzdx = torch.zeros_like(d)
    dzdy = torch.zeros_like(d)
    dzdx[:, 1:-1] = 0.5 * (d[:, 2:] - d[:, :-2])
    dzdy[1:-1, :] = 0.5 * (d[2:, :] - d[:-2, :])
    if d.shape[1] > 2:
        dzdx[:, 0] = dzdx[:, 1]
        dzdx[:, -1] = dzdx[:, -2]
    if d.shape[0] > 2:
        dzdy[0, :] = dzdy[1, :]
        dzdy[-1, :] = dzdy[-2, :]

    if normalize_gradients:
        valid = torch.isfinite(d) & (d > eps)
        if valid.any():
            scale = d[valid].median().abs().clamp_min(eps)
            dzdx = dzdx / scale
            dzdy = dzdy / scale

    nx = -dzdx
    ny = -dzdy
    nz = torch.ones_like(d)
    n = torch.stack([nx, ny, nz], dim=0)
    n = n / (torch.norm(n, dim=0, keepdim=True) + eps)
    return n


def camera_z_to_normals(camera_z, intrinsics, valid_mask=None, eps=1e-6):
    """Compute differentiable camera-frame normals from camera-space Z depth.

    Args:
        camera_z: [1,H,W] or [H,W] positive camera-space Z depth.
        intrinsics: [3,3] pixel-space K for this rendered resolution.
        valid_mask: optional [1,H,W] mask, e.g. alpha/depth validity.

    Returns:
        normals: [3,H,W], oriented by dP/du x dP/dv for image coordinates
                 (u right, v down, +Z forward).
        mask: [1,H,W] valid central-difference pixels.
    """
    z = camera_z[0] if camera_z.ndim == 3 else camera_z
    if z.ndim != 2:
        raise ValueError(f"camera_z must be [H,W] or [1,H,W], got {tuple(camera_z.shape)}")
    K = torch.as_tensor(intrinsics, dtype=z.dtype, device=z.device)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be [3,3], got {tuple(K.shape)}")

    h, w = z.shape
    if h < 3 or w < 3:
        raise ValueError(f"camera_z is too small for central differences: {(h, w)}")

    v, u = torch.meshgrid(
        torch.arange(h, dtype=z.dtype, device=z.device),
        torch.arange(w, dtype=z.dtype, device=z.device),
        indexing="ij",
    )
    fx = K[0, 0].clamp_min(eps)
    fy = K[1, 1].clamp_min(eps)
    x = (u - K[0, 2]) / fx
    y = (v - K[1, 2]) / fy
    points = torch.stack((x * z, y * z, z), dim=0)

    d_p_du = 0.5 * (points[:, :, 2:] - points[:, :, :-2])
    d_p_du = F.pad(d_p_du, (1, 1, 0, 0), mode="replicate")
    d_p_dv = 0.5 * (points[:, 2:, :] - points[:, :-2, :])
    d_p_dv = F.pad(d_p_dv, (0, 0, 1, 1), mode="replicate")

    normals = torch.cross(d_p_du, d_p_dv, dim=0)
    lengths = torch.linalg.vector_norm(normals, dim=0, keepdim=True)
    normals = normals / lengths.clamp_min(eps)

    mask = torch.isfinite(z) & (z > eps) & torch.isfinite(lengths[0]) & (lengths[0] > eps)
    if valid_mask is not None:
        vm = valid_mask[0] if valid_mask.ndim == 3 else valid_mask
        mask = mask & (vm > 0.5)
    # A central difference depends on neighboring 3D points; require all 3x3
    # samples to be valid so silhouettes/background do not create false edges.
    neighborhood = F.avg_pool2d(
        mask.to(z.dtype)[None, None], kernel_size=3, stride=1, padding=1,
        count_include_pad=False,
    )[0, 0]
    mask = mask & (neighborhood > 0.999)
    return normals, mask[None].to(z.dtype)
