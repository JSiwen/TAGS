import torch


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
