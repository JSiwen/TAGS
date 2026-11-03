import torch
import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import get_lpips_criterion

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


def _parse_float_list(spec, default):
    text = str(spec or "").strip()
    if not text:
        return list(default)
    values = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values if values else list(default)


def _normalize_weights(weights):
    total = sum(weights)
    if total <= 0:
        raise ValueError("LPIPS multiscale weights must sum to a positive value.")
    return [weight / total for weight in weights]


class LossBuilder:
    def __init__(self, opt, cfg, geo_prior, sparse_controller=None):
        self.opt = opt
        self.cfg = cfg
        self.geo_prior = geo_prior
        self.sparse_controller = sparse_controller
        self._lpips_criterion = None
        self._lpips_scales = None
        self._lpips_scale_weights = None
        if bool(getattr(self.cfg, "enable_multiscale_lpips", False)):
            self._lpips_scales = _parse_float_list(
                getattr(self.cfg, "lpips_multiscale_scales", "1.0,0.5,0.25"),
                (1.0, 0.5, 0.25),
            )
            raw_weights = _parse_float_list(
                getattr(self.cfg, "lpips_multiscale_weights", "0.5,0.3,0.2"),
                (0.5, 0.3, 0.2),
            )
            if len(raw_weights) != len(self._lpips_scales):
                raise ValueError(
                    "lpips_multiscale_weights must match lpips_multiscale_scales length."
                )
            self._lpips_scale_weights = _normalize_weights(raw_weights)

    def _ensure_lpips_criterion(self, device):
        if self._lpips_criterion is None:
            self._lpips_criterion = get_lpips_criterion("vgg", "0.1", device)
        return self._lpips_criterion

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
        if iteration < start_iter:
            return init_w
        schedule = str(getattr(self.cfg, "lpips_loss_schedule", "linear")).strip().lower()
        if schedule == "step":
            return final_w
        if schedule == "staged":
            # 分阶段调度: 形如 "iter:weight,iter:weight,..."
            # 例: "0:0.004,8000:0.001,25000:0.0" (TAGS §4.3.2 staged 调度)
            spec = str(getattr(self.cfg, "lpips_loss_staged_iters", "")).strip()
            if not spec:
                return final_w
            checkpoints = []
            for token in spec.replace(";", ",").split(","):
                token = token.strip()
                if not token or ":" not in token:
                    continue
                parts = token.split(":", 1)
                try:
                    checkpoints.append((int(parts[0].strip()), float(parts[1].strip())))
                except ValueError:
                    continue
            checkpoints.sort(key=lambda x: x[0])
            if not checkpoints:
                return final_w
            # 找到 iter 之前最近的 checkpoint
            current = checkpoints[0][1]
            for iter_thresh, weight in checkpoints:
                if iteration >= iter_thresh:
                    current = weight
                else:
                    break
            return current
        span = max(1, int(self.opt.iterations) - start_iter)
        progress = min(1.0, max(0.0, float(iteration - start_iter) / float(span)))
        return init_w + (final_w - init_w) * progress

    def _resize_for_scale(self, image, scale):
        if scale >= 1.0 - 1e-6:
            return image
        height = max(1, int(round(image.shape[-2] * scale)))
        width = max(1, int(round(image.shape[-1] * scale)))
        return F.interpolate(
            image.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

    def _compute_lpips_loss(self, image, gt_image):
        criterion = self._ensure_lpips_criterion(image.device)
        if not self._lpips_scales:
            loss = criterion(image.unsqueeze(0), gt_image.unsqueeze(0))
            return loss, {}

        per_scale = {}
        total = torch.tensor(0.0, device=image.device, dtype=image.dtype)
        for scale, weight in zip(self._lpips_scales, self._lpips_scale_weights):
            pred = self._resize_for_scale(image, scale)
            target = self._resize_for_scale(gt_image, scale)
            scale_loss = criterion(pred.unsqueeze(0), target.unsqueeze(0))
            total = total + float(weight) * scale_loss
            per_scale[f"train/lpips_scale_{scale:g}"] = float(scale_loss.item())
        return total, per_scale

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
        lambda_dssim = float(getattr(self.opt, "lambda_dssim", 0.0))
        if lambda_dssim > 0.0:
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            photo_loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)
            ssim_scalar = float(ssim_value.item()) if hasattr(ssim_value, "item") else float(ssim_value)
            lpips_proxy = float((0.7 * ll1 + 0.3 * (1.0 - ssim_value)).item())
        else:
            photo_loss = ll1
            ssim_scalar = float("nan")
            lpips_proxy = float(ll1.item())

        psnr_loss, psnr_value = self._psnr_loss(image, gt_image)
        stats = {
            "train/l1": float(ll1.item()),
            "train/ssim": ssim_scalar,
            "train/psnr": float(psnr_value.item()),
            "train/lpips_proxy": lpips_proxy,
        }
        late_state = self._late_state(train_state)

        view_loss_weight = 1.0
        if train_state is not None:
            view_loss_weight = float(train_state.get("view_loss_weight", 1.0))
        view_loss_weight = max(1.0, view_loss_weight)

        psnr_w = self._psnr_weight(iteration)
        stats["train/psnr_weight_original"] = psnr_w
        stats["train/psnr_multiplier"] = float(late_state.get("psnr_multiplier", 1.0))
        stats["train/psnr_loss_raw"] = float(psnr_loss.item())
        stats["train/psnr_loss_weighted"] = float(psnr_loss.item()) * psnr_w * stats["train/psnr_multiplier"] * view_loss_weight
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
        stats["train/lpips_weight_original"] = lpips_w
        stats["train/lpips_multiplier"] = float(late_state.get("lpips_loss_multiplier", 1.0))
        lpips_interval = max(1, int(getattr(self.cfg, "lpips_loss_interval", 4)))
        if lpips_w > 0 and (iteration % lpips_interval == 0):
            lpips_loss, lpips_scale_stats = self._compute_lpips_loss(image, gt_image)
            effective_lpips_w = (
                lpips_w
                * float(late_state.get("lpips_guard_multiplier", 1.0))
                * stats["train/lpips_multiplier"]
                * view_loss_weight
            )
            total = total + effective_lpips_w * lpips_loss
            stats["train/lpips_loss"] = float(lpips_loss.item())
            stats["train/lpips_weight"] = float(effective_lpips_w)
            stats["train/lpips_loss_weighted"] = float((effective_lpips_w * lpips_loss).item())
            stats.update(lpips_scale_stats)

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
