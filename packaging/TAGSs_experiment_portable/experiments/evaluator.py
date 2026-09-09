import torch

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from utils.loss_utils import ssim
from .region_eval import RegionEvaluator


class EvaluationReporter:
    def __init__(self, scene, render_func, render_args, tb_writer=None, train_test_exp=False, cfg=None):
        self.scene = scene
        self.render_func = render_func
        self.render_args = render_args
        self.tb_writer = tb_writer
        self.train_test_exp = train_test_exp
        self.cfg = cfg
        self.region_evaluator = RegionEvaluator(cfg) if cfg and getattr(cfg, "enable_region_eval", False) else None

    def log_train_scalars(self, iteration, ll1, loss, elapsed_ms, stats):
        if not self.tb_writer:
            return
        self.tb_writer.add_scalar("train/l1", ll1.item(), iteration)
        self.tb_writer.add_scalar("train/total_loss", loss.item(), iteration)
        self.tb_writer.add_scalar("train/iter_time", elapsed_ms, iteration)
        for key, value in stats.items():
            self.tb_writer.add_scalar(key, value, iteration)

    def maybe_evaluate(self, iteration, testing_iterations):
        if iteration not in testing_iterations:
            return {}

        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": self.scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        eval_metrics = {}
        for config in validation_configs:
            metrics = self._evaluate_config(iteration, testing_iterations, config)
            if metrics:
                eval_metrics[config["name"]] = metrics

        if self.tb_writer:
            self.tb_writer.add_histogram("scene/opacity_histogram", self.scene.gaussians.get_opacity, iteration)
            self.tb_writer.add_scalar("scene/total_points", self.scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        return eval_metrics

    def _evaluate_config(self, iteration, testing_iterations, config):
        cameras = config["cameras"]
        if not cameras:
            return {}

        l1_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        region_acc = {}
        for idx, viewpoint in enumerate(cameras):
            image = torch.clamp(self.render_func(viewpoint, self.scene.gaussians, *self.render_args)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if self.train_test_exp:
                image = image[..., image.shape[-1] // 2 :]
                gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
            if self.tb_writer and idx < 5:
                self.tb_writer.add_images(
                    f"{config['name']}_view_{viewpoint.image_name}/render",
                    image[None],
                    global_step=iteration,
                )
                if iteration == testing_iterations[0]:
                    self.tb_writer.add_images(
                        f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                        gt_image[None],
                        global_step=iteration,
                    )
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
            lpips_test += lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type="vgg").double()
            if self.region_evaluator is not None:
                per_region = self.region_evaluator.evaluate(image, gt_image)
                for key, value in per_region.items():
                    region_acc[key] = region_acc.get(key, 0.0) + float(value)

        psnr_test /= len(cameras)
        l1_test /= len(cameras)
        ssim_test /= len(cameras)
        lpips_test /= len(cameras)
        print(
            f"\n[ITER {iteration}] Evaluating {config['name']}: "
            f"L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}"
        )
        if self.tb_writer:
            self.tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
            self.tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)
            self.tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - ssim", ssim_test, iteration)
            self.tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - lpips", lpips_test, iteration)
            if self.region_evaluator is not None:
                for key, value in region_acc.items():
                    self.tb_writer.add_scalar(f"{config['name']}/region - {key}", value / len(cameras), iteration)
        metrics = {
            "l1": float(l1_test.item()),
            "psnr": float(psnr_test.item()),
            "ssim": float(ssim_test.item()),
            "lpips": float(lpips_test.item()),
            "num_cameras": int(len(cameras)),
        }
        if self.region_evaluator is not None:
            metrics["region"] = {key: float(value / len(cameras)) for key, value in region_acc.items()}
        return metrics
