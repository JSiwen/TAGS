import torch

from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


class LossBuilder:
    def __init__(self, opt, cfg, geo_prior, sparse_controller=None):
        self.opt = opt
        self.cfg = cfg
        self.geo_prior = geo_prior
        self.sparse_controller = sparse_controller

    def _detail_weight(self, iteration):
        start_iter = max(0, int(getattr(self.cfg, "detail_loss_start_iter", 0)))
        init_w = float(getattr(self.cfg, "detail_loss_weight_init", 0.0))
        final_w = float(getattr(self.cfg, "detail_loss_weight_final", 0.0))
        max_w = max(0.0, float(getattr(self.cfg, "detail_loss_max_weight", 0.05)))
        init_w = max(0.0, min(init_w, max_w))
        final_w = max(0.0, min(final_w, max_w))
        if iteration <= start_iter:
            return init_w
        span = max(1, int(self.opt.iterations) - start_iter)
        progress = min(1.0, max(0.0, float(iteration - start_iter) / float(span)))
        return init_w + (final_w - init_w) * progress

    def _detail_mask(self, image):
        gray = image.mean(dim=0, keepdim=True)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[..., :, 1:-1] = 0.5 * (gray[..., :, 2:] - gray[..., :, :-2])
        gy[..., 1:-1, :] = 0.5 * (gray[..., 2:, :] - gray[..., :-2, :])
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        quantile = max(0.0, min(1.0, float(getattr(self.cfg, "detail_loss_edge_quantile", 0.75))))
        threshold = torch.quantile(grad_mag.reshape(-1), quantile)
        mask = (grad_mag >= threshold).to(dtype=image.dtype, device=image.device)
        return mask, grad_mag

    def _detail_loss(self, pred, target):
        mask, _ = self._detail_mask(target)
        pred_gx = torch.zeros_like(pred)
        pred_gy = torch.zeros_like(pred)
        gt_gx = torch.zeros_like(target)
        gt_gy = torch.zeros_like(target)
        pred_gx[..., :, 1:-1] = 0.5 * (pred[..., :, 2:] - pred[..., :, :-2])
        pred_gy[..., 1:-1, :] = 0.5 * (pred[..., 2:, :] - pred[..., :-2, :])
        gt_gx[..., :, 1:-1] = 0.5 * (target[..., :, 2:] - target[..., :, :-2])
        gt_gy[..., 1:-1, :] = 0.5 * (target[..., 2:, :] - target[..., :-2, :])
        grad_diff = torch.abs(pred_gx - gt_gx) + torch.abs(pred_gy - gt_gy)
        weighted = grad_diff * mask
        denom = (mask.sum() * pred.shape[0]).clamp_min(1.0)
        return weighted.sum() / denom, float(mask.mean().item())

    def _lpips_weight(self, iteration):
        start_iter = max(0, int(getattr(self.cfg, "lpips_loss_start_iter", 0)))
        init_w = float(getattr(self.cfg, "lpips_loss_weight_init", 0.0))
        final_w = float(getattr(self.cfg, "lpips_loss_weight_final", 0.0))
        max_w = max(0.0, float(getattr(self.cfg, "lpips_loss_max_weight", 0.02)))
        init_w = max(0.0, min(init_w, max_w))
        final_w = max(0.0, min(final_w, max_w))
        if iteration <= start_iter:
            return init_w
        span = max(1, int(self.opt.iterations) - start_iter)
        progress = min(1.0, max(0.0, float(iteration - start_iter) / float(span)))
        return init_w + (final_w - init_w) * progress

    def _psnr_weight(self, iteration):
        start_iter = max(0, int(getattr(self.cfg, "psnr_loss_start_iter", 0)))
        init_w = float(getattr(self.cfg, "psnr_loss_weight_init", 0.0))
        final_w = float(getattr(self.cfg, "psnr_loss_weight_final", 0.0))
        max_w = max(0.0, float(getattr(self.cfg, "psnr_loss_max_weight", 0.05)))
        init_w = max(0.0, min(init_w, max_w))
        final_w = max(0.0, min(final_w, max_w))
        if iteration <= start_iter:
            return init_w
        span = max(1, int(self.opt.iterations) - start_iter)
        progress = min(1.0, max(0.0, float(iteration - start_iter) / float(span)))
        return init_w + (final_w - init_w) * progress

    def _psnr_loss(self, pred, target):
        mse = torch.mean((pred - target) * (pred - target)).clamp_min(1e-8)
        psnr_value = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
        # A positive, monotonic proxy for maximizing PSNR with usable scale.
        loss = torch.log1p(100.0 * mse)
        return loss, psnr_value

    def _color_align_weight(self, iteration):
        start_iter = max(0, int(getattr(self.cfg, "color_align_start_iter", 0)))
        init_w = float(getattr(self.cfg, "color_align_weight_init", 0.0))
        final_w = float(getattr(self.cfg, "color_align_weight_final", 0.0))
        max_w = max(0.0, float(getattr(self.cfg, "color_align_max_weight", 0.03)))
        init_w = max(0.0, min(init_w, max_w))
        final_w = max(0.0, min(final_w, max_w))
        if iteration <= start_iter:
            return init_w
        span = max(1, int(self.opt.iterations) - start_iter)
        progress = min(1.0, max(0.0, float(iteration - start_iter) / float(span)))
        return init_w + (final_w - init_w) * progress

    def _color_align_loss(self, pred, target):
        gray = target.mean(dim=0, keepdim=True)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[..., :, 1:-1] = 0.5 * torch.abs(gray[..., :, 2:] - gray[..., :, :-2])
        gy[..., 1:-1, :] = 0.5 * torch.abs(gray[..., 2:, :] - gray[..., :-2, :])
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        q = min(1.0, max(0.0, float(getattr(self.cfg, "color_align_low_grad_quantile", 0.60))))
        threshold = torch.quantile(grad_mag.reshape(-1), q)
        low_grad_mask = grad_mag <= threshold
        min_fraction = min(1.0, max(0.0, float(getattr(self.cfg, "color_align_min_mask_fraction", 0.20))))
        if float(low_grad_mask.float().mean().item()) < min_fraction:
            # Fallback to global stats if low-gradient region is too small.
            low_grad_mask = torch.ones_like(low_grad_mask, dtype=torch.bool)

        ch_mean_losses = []
        ch_std_losses = []
        ch_mask = low_grad_mask[0]
        for c in range(pred.shape[0]):
            pred_vals = pred[c][ch_mask]
            gt_vals = target[c][ch_mask]
            pred_mean = pred_vals.mean()
            gt_mean = gt_vals.mean()
            pred_std = pred_vals.std(unbiased=False)
            gt_std = gt_vals.std(unbiased=False)
            ch_mean_losses.append(torch.abs(pred_mean - gt_mean))
            ch_std_losses.append(torch.abs(pred_std - gt_std))
        mean_loss = torch.stack(ch_mean_losses).mean()
        std_loss = torch.stack(ch_std_losses).mean()
        std_w = max(0.0, float(getattr(self.cfg, "color_align_std_weight", 0.5)))
        return mean_loss + std_w * std_loss

    @staticmethod
    def _late_state(train_state):
        if not train_state:
            return {}
        return train_state.get("late_controller", {}) or {}

    def __call__(self, render_pkg, viewpoint_cam, iteration, gaussians=None, train_state=None):
        image = render_pkg["render"]
        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        photo_loss = (1.0 - self.opt.lambda_dssim) * ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
        psnr_loss, psnr_value = self._psnr_loss(image, gt_image)
        stats = {
            "train/l1": float(ll1.item()),
            "train/ssim": float(ssim_value.item()) if hasattr(ssim_value, "item") else float(ssim_value),
            "train/psnr": float(psnr_value.item()),
            # Cheap proxy used by late-stage controller for per-view guard signals.
            "train/lpips_proxy": float((0.7 * ll1 + 0.3 * (1.0 - ssim_value)).item()),
        }
        late_state = self._late_state(train_state)

        view_loss_weight = 1.0
        if train_state is not None:
            view_loss_weight = float(train_state.get("view_loss_weight", 1.0))
        view_loss_weight = max(1.0, view_loss_weight)

        psnr_w = self._psnr_weight(iteration)
        if psnr_w > 0:
            psnr_w *= float(late_state.get("psnr_multiplier", 1.0))
            photo_loss = photo_loss + psnr_w * psnr_loss
            stats["train/psnr_loss"] = float(psnr_loss.item())
            stats["train/psnr_weight"] = float(psnr_w * view_loss_weight)

        total = view_loss_weight * photo_loss
        stats["train/photo_loss"] = float(photo_loss.item())
        stats["train/photo_weight"] = float(view_loss_weight)

        detail_w = self._detail_weight(iteration)
        if detail_w > 0:
            detail_loss, detail_mask_fraction = self._detail_loss(image, gt_image)
            effective_detail_w = detail_w * view_loss_weight
            total = total + effective_detail_w * detail_loss
            stats["train/detail_loss"] = float(detail_loss.item())
            stats["train/detail_weight"] = float(effective_detail_w)
            stats["train/detail_mask_fraction"] = detail_mask_fraction

        lpips_w = self._lpips_weight(iteration)
        lpips_interval = max(1, int(getattr(self.cfg, "lpips_loss_interval", 4)))
        if lpips_w > 0 and (iteration % lpips_interval == 0):
            lpips_loss = lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type="vgg")
            effective_lpips_w = (
                lpips_w
                * float(late_state.get("lpips_guard_multiplier", 1.0))
                * view_loss_weight
            )
            total = total + effective_lpips_w * lpips_loss
            stats["train/lpips_loss"] = float(lpips_loss.item())
            stats["train/lpips_weight"] = float(effective_lpips_w)

        color_align_w = self._color_align_weight(iteration)
        if color_align_w > 0:
            if bool(getattr(self.cfg, "color_align_require_fullres", True)):
                if train_state is not None and not bool(train_state.get("fullres_ready", False)):
                    color_align_w = 0.0
            if color_align_w <= 0:
                stats["train/color_align_weight"] = 0.0
            else:
                if train_state is not None:
                    sample_w = float(train_state.get("view_sampling_weight", 1.0))
                else:
                    sample_w = 1.0
                hard_boost = max(1.0, float(getattr(self.cfg, "color_align_hard_view_boost", 1.5)))
                max_boost = max(1.0, float(getattr(self.cfg, "hard_view_boost_max", 3.0)))
                if max_boost > 1.0:
                    normalized = min(1.0, max(0.0, (sample_w - 1.0) / (max_boost - 1.0)))
                else:
                    normalized = 0.0
                effective_w = color_align_w * (1.0 + (hard_boost - 1.0) * normalized)
                effective_w *= float(late_state.get("color_align_multiplier", 1.0))
                color_align_loss = self._color_align_loss(image, gt_image)
                total = total + effective_w * color_align_loss
                stats["train/color_align_loss"] = float(color_align_loss.item())
                stats["train/color_align_weight"] = float(effective_w)

        if self.cfg.enable_geo_prior:
            geo_loss, geo_stats = self.geo_prior(render_pkg, viewpoint_cam, iteration)
            total = total + geo_loss
            stats.update(geo_stats)

        if self.sparse_controller is not None and gaussians is not None:
            sparse_loss, sparse_stats = self.sparse_controller.penalty(iteration, gaussians, train_state)
            total = total + sparse_loss
            stats.update(sparse_stats)

        stats["train/total_loss"] = float(total.item())
        return total, ll1, stats
