import torch


class GeometryController:
    def __init__(self, cfg):
        self.cfg = cfg

    def _gradient_signal(self, gaussians):
        if getattr(gaussians, "xyz_gradient_accum", None) is None or getattr(gaussians, "denom", None) is None:
            return None
        grad = gaussians.xyz_gradient_accum.squeeze() / gaussians.denom.squeeze().clamp_min(1.0)
        return grad.detach()

    def apply(self, iteration, gaussians):
        if not getattr(self.cfg, "enable_geometry_controller", False):
            return {}
        if iteration < int(getattr(self.cfg, "geometry_start_iter", 0)):
            return {}
        interval = max(1, int(getattr(self.cfg, "geometry_interval", 250)))
        if iteration % interval != 0:
            return {}

        stats = {}
        with torch.no_grad():
            scales = gaussians.get_scaling.max(dim=1).values
            opacity = gaussians.get_opacity.squeeze()
            if scales.numel() < 8:
                return {}

            q = float(getattr(self.cfg, "geometry_scale_cap_quantile", 0.98))
            q = min(1.0, max(0.5, q))
            scale_cap = torch.quantile(scales, q).item()
            scale_cap = min(scale_cap, float(getattr(self.cfg, "geometry_scale_cap_max", 0.12)))
            scale_cap = max(scale_cap, 1e-4)

            grad = self._gradient_signal(gaussians)
            protected = torch.zeros_like(scales, dtype=torch.bool)
            if grad is not None and grad.numel() == scales.numel():
                gq = float(getattr(self.cfg, "geometry_protect_grad_quantile", 0.95))
                gq = min(1.0, max(0.5, gq))
                gthr = torch.quantile(grad, gq)
                protected = grad >= gthr

            oversized = scales > scale_cap
            min_opacity = float(getattr(self.cfg, "geometry_min_opacity", 0.008))
            prune_scale_factor = float(getattr(self.cfg, "geometry_prune_scale_factor", 0.85))
            prune_mask = oversized & (opacity < min_opacity) & (scales > (scale_cap * prune_scale_factor)) & (~protected)

            if prune_mask.any():
                before = int(scales.numel())
                gaussians.prune_points(prune_mask)
                after = int(gaussians.get_xyz.shape[0])
                stats["geometry/pruned_points"] = float(before - after)
                scales = gaussians.get_scaling.max(dim=1).values
                opacity = gaussians.get_opacity.squeeze()

            stats["geometry/scale_cap"] = float(scale_cap)
            stats["geometry/scale_mean"] = float(scales.mean().item())
            stats["geometry/scale_p95"] = float(torch.quantile(scales, 0.95).item())
            stats["geometry/opacity_mean"] = float(opacity.mean().item())
            if grad is not None and grad.numel() > 0:
                stats["geometry/grad_mean"] = float(grad.mean().item())
        return stats
