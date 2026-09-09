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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def _empty_sh_dc_like(means3D: torch.Tensor) -> torch.Tensor:
    """Sentinel empty tensor for fused (P, M, 3) SH layout (no Python torch.cat)."""
    return torch.empty(0, device=means3D.device, dtype=torch.float32)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    sh_dc=None,
):
    if sh_dc is None:
        sh_dc = _empty_sh_dc_like(means3D)
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        sh_dc,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        sh_dc,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        ctx.sh_split = bool(sh_dc is not None and sh_dc.numel() > 0)

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            sh_dc,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            sh_dc,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            sh_dc,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_depth,
            sh,
            sh_dc,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = (
            _C.rasterize_gaussians_backward(*args)
        )

        if ctx.sh_split:
            grad_sh_param = grad_sh[:, 1:, :]
            grad_sh_dc = grad_sh[:, :1, :]
        else:
            grad_sh_param = grad_sh
            grad_sh_dc = None

        return (
            grad_means3D,
            grad_means2D,
            grad_sh_param,
            grad_sh_dc,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        shs_dc=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if shs is None:
            shs = torch.Tensor([])
        if shs_dc is None:
            shs_dc = _empty_sh_dc_like(means3D)
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            sh_dc=shs_dc,
        )
