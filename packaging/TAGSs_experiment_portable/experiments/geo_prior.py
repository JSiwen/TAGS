import torch
import torch.nn.functional as F
from .schedules import linear_warmup_decay
from utils.normal_utils import depth_to_normals


class GeometricPriorLoss:
    def __init__(self, cfg):
        self.cfg = cfg

    def _scheduled_weight(self, iteration, init_value, peak_value, final_value):
        start_iter = max(0, int(getattr(self.cfg, "geo_prior_start_iter", 0)))
        if iteration < start_iter:
            return 0.0
        shifted_iter = iteration - start_iter
        warmup_until = max(0, int(getattr(self.cfg, "geo_prior_warmup_until", 0)) - start_iter)
        decay_start = max(warmup_until, int(getattr(self.cfg, "geo_prior_decay_start", 0)) - start_iter)
        decay_until = max(decay_start, int(getattr(self.cfg, "geo_prior_decay_until", 0)) - start_iter)
        return linear_warmup_decay(
            shifted_iter,
            0,
            warmup_until,
            decay_start,
            decay_until,
            init_value,
            peak_value,
            final_value,
        )

    def _fullres_relax_factor(self, iteration):
        decay_until = max(1, int(getattr(self.cfg, "geo_prior_decay_until", 24000)))
        decay_start = max(0, int(getattr(self.cfg, "geo_prior_decay_start", 12000)))
        if iteration <= decay_start:
            return 1.0
        progress = min(1.0, max(0.0, float(iteration - decay_start) / float(decay_until - decay_start + 1)))
        # Preserve early geometry shaping while reducing over-smoothing in late refinement.
        return 1.0 - 0.35 * progress

    def _depth_loss(self, render_pkg, viewpoint_cam, iteration):
        if not hasattr(viewpoint_cam, "depth_reliable") or not viewpoint_cam.depth_reliable:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0
        if getattr(viewpoint_cam, "invdepthmap", None) is None:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0

        weight = self._scheduled_weight(
            iteration,
            self.cfg.depth_prior_weight_init,
            self.cfg.depth_prior_weight_final,
            self.cfg.depth_prior_weight_min,
        )
        if weight <= 0:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0

        pred = render_pkg["depth"]
        gt = viewpoint_cam.invdepthmap.cuda()
        base_mask = viewpoint_cam.depth_mask.cuda() if getattr(viewpoint_cam, "depth_mask", None) is not None else torch.ones_like(gt)
        mask, mask_fraction = self._reliability_mask(pred, gt, viewpoint_cam.original_image.cuda(), base_mask)
        if mask.sum() < 2:
            return torch.tensor(0.0, device="cuda"), weight, 0.0

        loss_type = self.cfg.depth_prior_loss_type.lower()
        if loss_type == "edge_aware_log_l1":
            base_loss = self._edge_aware_log_l1(pred, gt, viewpoint_cam.original_image.cuda(), mask)
        elif loss_type == "log_l1":
            base_loss = self._masked_mean(torch.log1p(torch.abs(pred - gt)), mask)
        elif loss_type == "l1":
            base_loss = self._masked_mean(torch.abs(pred - gt), mask)
        else:
            # Charbonnier-like robust objective is the conservative default.
            diff = pred - gt
            base_loss = self._masked_mean(torch.sqrt(diff * diff + 1e-6), mask)

        if self.cfg.depth_prior_pearson_weight > 0:
            base_loss = base_loss + self.cfg.depth_prior_pearson_weight * self._pearson_depth_loss(pred, gt, mask)
        if self.cfg.depth_prior_local_pearson_weight > 0:
            base_loss = base_loss + self.cfg.depth_prior_local_pearson_weight * self._local_pearson_depth_loss(pred, gt, mask)

        texture_relax = self._low_texture_relax_factor(viewpoint_cam.original_image.cuda())
        weight = weight * self._fullres_relax_factor(iteration) * texture_relax
        loss = base_loss * weight
        return loss, weight, mask_fraction

    def _normal_loss(self, render_pkg, viewpoint_cam, iteration):
        if not hasattr(viewpoint_cam, "depth_reliable") or not viewpoint_cam.depth_reliable:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0
        if getattr(viewpoint_cam, "invdepthmap", None) is None:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0

        weight = self._scheduled_weight(
            iteration,
            self.cfg.normal_prior_weight_init,
            self.cfg.normal_prior_weight_final,
            self.cfg.normal_prior_weight_min,
        )
        if weight <= 0:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0

        gt_depth = viewpoint_cam.invdepthmap.cuda()
        base_mask = viewpoint_cam.depth_mask.cuda() if getattr(viewpoint_cam, "depth_mask", None) is not None else torch.ones_like(gt_depth)
        normal_mask, normal_mask_fraction = self._reliability_mask(
            render_pkg["depth"],
            gt_depth,
            viewpoint_cam.original_image.cuda(),
            base_mask,
        )
        if normal_mask.sum() < 2:
            return torch.tensor(0.0, device="cuda"), 0.0, 0.0

        pred_n = depth_to_normals(render_pkg["depth"])
        gt_n = depth_to_normals(gt_depth)
        if self.cfg.normal_prior_loss_type.lower() == "l1":
            base_loss = self._masked_mean(torch.abs(pred_n - gt_n).mean(dim=0, keepdim=True), normal_mask)
        else:
            cos_sim = torch.sum(pred_n * gt_n, dim=0).clamp(-1.0, 1.0)
            base_loss = self._masked_mean((1.0 - cos_sim).unsqueeze(0), normal_mask)

        if self.cfg.normal_prior_tv_weight > 0:
            tv_mask = self._smooth_region_mask(gt_depth, viewpoint_cam.original_image.cuda(), base_mask)
            tv_decay = max(0.2, self._fullres_relax_factor(iteration))
            base_loss = base_loss + (self.cfg.normal_prior_tv_weight * tv_decay) * self._tv_loss(pred_n, tv_mask)

        texture_relax = self._low_texture_relax_factor(viewpoint_cam.original_image.cuda())
        weight = weight * self._fullres_relax_factor(iteration) * texture_relax
        loss = base_loss * weight
        return loss, weight, normal_mask_fraction

    def _low_texture_relax_factor(self, rgb):
        strength = max(0.0, float(getattr(self.cfg, "geo_prior_low_texture_relax_strength", 0.0)))
        if strength <= 0:
            return 1.0
        gray = rgb.mean(dim=0)
        grad = self._gradient_magnitude(gray)
        flat = grad.reshape(-1)
        if flat.numel() < 8:
            return 1.0
        p50 = torch.quantile(flat, 0.50)
        p90 = torch.quantile(flat, 0.90)
        texture_ratio = float((p50 / (p90 + 1e-6)).clamp(0.0, 1.0).item())
        min_scale = min(1.0, max(0.1, float(getattr(self.cfg, "geo_prior_low_texture_relax_min", 0.75))))
        relax = 1.0 - strength * (1.0 - texture_ratio)
        return max(min_scale, min(1.0, relax))

    def _edge_aware_log_l1(self, pred, gt, rgb, mask):
        diff = torch.log1p(torch.abs(pred - gt))
        rgb = rgb.unsqueeze(0)
        diff = diff.unsqueeze(0)
        mask = mask.unsqueeze(0)

        grad_img_x = torch.mean(torch.abs(rgb[..., :, 1:] - rgb[..., :, :-1]), dim=1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(rgb[..., 1:, :] - rgb[..., :-1, :]), dim=1, keepdim=True)
        loss_x = torch.exp(-grad_img_x) * diff[..., :, 1:]
        loss_y = torch.exp(-grad_img_y) * diff[..., 1:, :]
        return self._masked_mean(loss_x, mask[..., :, 1:]) + self._masked_mean(loss_y, mask[..., 1:, :])

    def _masked_mean(self, values, mask):
        mask = mask.to(dtype=values.dtype, device=values.device)
        while mask.dim() < values.dim():
            mask = mask.unsqueeze(0)
        mask = torch.broadcast_to(mask, values.shape)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def _normal_reliability_mask(self, depth, rgb, base_mask):
        return self._reliability_mask(depth, depth, rgb, base_mask)[0]

    def _reliability_mask(self, pred_depth, gt_depth, rgb, base_mask):
        valid = self._valid_depth_mask(gt_depth, base_mask)
        if pred_depth is not None:
            pred = pred_depth[0] if pred_depth.dim() == 3 else pred_depth
            valid = valid & torch.isfinite(pred)
        if not valid.any():
            empty = torch.zeros_like(base_mask[:1] if base_mask.dim() == 3 else base_mask.unsqueeze(0))
            return empty.to(dtype=gt_depth.dtype, device=gt_depth.device), 0.0

        depth_edge = self._gradient_magnitude(gt_depth)
        rgb_edge = self._gradient_magnitude(rgb.mean(dim=0))
        local_std = self._local_std(gt_depth)

        depth_thresh = self._masked_quantile(depth_edge, valid, getattr(self.cfg, "depth_prior_edge_quantile", 0.92))
        rgb_thresh = self._masked_quantile(rgb_edge, valid, getattr(self.cfg, "depth_prior_rgb_edge_quantile", 0.95))
        var_thresh = self._masked_quantile(local_std, valid, getattr(self.cfg, "depth_prior_var_quantile", 0.92))

        reliable = valid & (depth_edge <= depth_thresh) & (rgb_edge <= rgb_thresh) & (local_std <= var_thresh)
        score = self._normalize_map(depth_edge) + self._normalize_map(rgb_edge) + self._normalize_map(local_std)
        reliable = self._ensure_min_fraction(reliable, valid, score)
        mask = reliable.unsqueeze(0).to(dtype=gt_depth.dtype, device=gt_depth.device)
        return mask, float(reliable.float().mean().item())

    def _valid_depth_mask(self, depth, base_mask):
        d = depth[0] if depth.dim() == 3 else depth
        valid_depth = base_mask[0] > 0 if base_mask.dim() == 3 else base_mask > 0
        return valid_depth & torch.isfinite(d) & (d > 0)

    def _smooth_region_mask(self, depth, rgb, base_mask):
        valid = self._valid_depth_mask(depth, base_mask)
        if not valid.any():
            return torch.zeros_like(base_mask)
        depth_edge = self._gradient_magnitude(depth)
        rgb_edge = self._gradient_magnitude(rgb.mean(dim=0))
        depth_thresh = self._masked_quantile(depth_edge, valid, getattr(self.cfg, "normal_tv_depth_edge_quantile", 0.70))
        rgb_thresh = self._masked_quantile(rgb_edge, valid, getattr(self.cfg, "normal_tv_rgb_edge_quantile", 0.70))
        smooth = valid & (depth_edge <= depth_thresh) & (rgb_edge <= rgb_thresh)
        return smooth.unsqueeze(0).to(dtype=depth.dtype, device=depth.device)

    def _gradient_magnitude(self, tensor):
        t = tensor[0] if tensor.dim() == 3 else tensor
        dx = torch.zeros_like(t)
        dy = torch.zeros_like(t)
        dx[:, 1:-1] = 0.5 * torch.abs(t[:, 2:] - t[:, :-2])
        dy[1:-1, :] = 0.5 * torch.abs(t[2:, :] - t[:-2, :])
        if t.shape[1] > 2:
            dx[:, 0] = dx[:, 1]
            dx[:, -1] = dx[:, -2]
        if t.shape[0] > 2:
            dy[0, :] = dy[1, :]
            dy[-1, :] = dy[-2, :]
        return torch.sqrt(dx * dx + dy * dy)

    def _local_std(self, tensor, kernel_size=5):
        t = tensor[0] if tensor.dim() == 3 else tensor
        x = t.unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        mean_sq = F.avg_pool2d(x * x, kernel_size, stride=1, padding=padding)
        return torch.sqrt((mean_sq - mean * mean).clamp_min(0.0)).squeeze(0).squeeze(0)

    def _masked_quantile(self, values, mask, quantile):
        selected = values[mask]
        if selected.numel() == 0:
            return values.new_tensor(0.0)
        return torch.quantile(selected.float(), max(0.0, min(1.0, float(quantile)))).to(values.dtype)

    def _normalize_map(self, values):
        v_min = values.min()
        v_max = values.max()
        return (values - v_min) / (v_max - v_min).clamp_min(1e-8)

    def _ensure_min_fraction(self, reliable, valid, score):
        min_fraction = max(0.0, min(1.0, float(getattr(self.cfg, "depth_prior_min_mask_fraction", 0.20))))
        valid_count = int(valid.sum().item())
        min_count = int(min_fraction * valid.numel())
        if valid_count == 0 or int(reliable.sum().item()) >= min_count:
            return reliable
        keep_count = min(valid_count, max(1, min_count))
        threshold = torch.topk(score[valid], k=keep_count, largest=False).values.max()
        return valid & (score <= threshold)

    def _pearson_depth_loss(self, pred, gt, mask):
        valid = mask > 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=pred.device)
        pred_valid = pred[valid]
        gt_valid = gt[valid]
        pred_norm = (pred_valid - pred_valid.mean()) / (pred_valid.std() + 1e-6)
        gt_norm = (gt_valid - gt_valid.mean()) / (gt_valid.std() + 1e-6)
        return 1.0 - (pred_norm * gt_norm).mean()

    def _local_pearson_depth_loss(self, pred, gt, mask, patch_size=128, samples=4):
        _, height, width = pred.shape
        if height < patch_size or width < patch_size:
            return self._pearson_depth_loss(pred, gt, mask)

        losses = []
        for _ in range(samples):
            top = torch.randint(0, height - patch_size + 1, (1,), device=pred.device).item()
            left = torch.randint(0, width - patch_size + 1, (1,), device=pred.device).item()
            patch = (slice(None), slice(top, top + patch_size), slice(left, left + patch_size))
            losses.append(self._pearson_depth_loss(pred[patch], gt[patch], mask[patch]))
        return torch.stack(losses).mean()

    def _tv_loss(self, tensor, mask=None):
        tensor = tensor.unsqueeze(0)
        diff_h = torch.abs(tensor[..., 1:, :] - tensor[..., :-1, :]).mean(dim=1, keepdim=True)
        diff_w = torch.abs(tensor[..., :, 1:] - tensor[..., :, :-1]).mean(dim=1, keepdim=True)
        if mask is None:
            return diff_h.mean() + diff_w.mean()
        m = mask.unsqueeze(0).to(dtype=tensor.dtype, device=tensor.device)
        mask_h = m[..., 1:, :] * m[..., :-1, :]
        mask_w = m[..., :, 1:] * m[..., :, :-1]
        return self._masked_mean(diff_h, mask_h) + self._masked_mean(diff_w, mask_w)

    def __call__(self, render_pkg, viewpoint_cam, iteration):
        total = torch.tensor(0.0, device="cuda")
        stats = {}

        if self.cfg.geo_prior_depth:
            depth_loss, depth_w, depth_mask_fraction = self._depth_loss(render_pkg, viewpoint_cam, iteration)
            total = total + depth_loss
            stats["geo/depth_weight"] = depth_w
            stats["geo/depth_loss"] = float(depth_loss.item())
            stats["geo/depth_mask_fraction"] = depth_mask_fraction

        if self.cfg.geo_prior_normal:
            normal_loss, normal_w, normal_mask_fraction = self._normal_loss(render_pkg, viewpoint_cam, iteration)
            total = total + normal_loss
            stats["geo/normal_weight"] = normal_w
            stats["geo/normal_loss"] = float(normal_loss.item())
            stats["geo/normal_mask_fraction"] = normal_mask_fraction

        return total, stats
