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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        # COLMAP train/test split; used when eval=True. k>0: every k-th image (sorted by filename) is test (LLFF-style).
        # 0: load test names from sparse/0/test.txt if present; if missing, falls back to the same split as k=8.
        # test.txt lines must match COLMAP image names exactly (including extension).
        self.llffhold = 8
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 18_000
        self.densify_grad_threshold = 0.00016
        # Upper bound on newly added Gaussians per densification step.
        # 0 keeps original behavior (no cap).
        self.densify_max_new_points_per_iter = 0
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

class ExperimentParams(ParamGroup):
    def __init__(self, parser):
        # Global toggles
        self.enable_geo_prior = False
        self.enable_resolution_adaptive = False
        self.enable_sparse_optimization = False

        # Geometric prior settings
        self.geo_prior_depth = False
        self.geo_prior_normal = False
        self.depth_prior_loss_type = "edge_aware_log_l1"
        self.normal_prior_loss_type = "cosine"
        self.depth_prior_pearson_weight = 0.0
        self.depth_prior_local_pearson_weight = 0.05
        self.normal_prior_tv_weight = 0.003
        self.depth_prior_edge_quantile = 0.92
        self.depth_prior_rgb_edge_quantile = 0.95
        self.depth_prior_var_quantile = 0.92
        self.depth_prior_min_mask_fraction = 0.20
        self.normal_tv_depth_edge_quantile = 0.70
        self.normal_tv_rgb_edge_quantile = 0.70
        self.depth_prior_weight_init = 0.0
        self.depth_prior_weight_final = 0.05
        self.depth_prior_weight_min = 0.01
        self.normal_prior_weight_init = 0.0
        self.normal_prior_weight_final = 0.02
        self.normal_prior_weight_min = 0.0
        self.geo_prior_start_iter = 0
        self.geo_prior_warmup_until = 5000
        self.geo_prior_decay_start = 12000
        self.geo_prior_decay_until = 24000

        # Detail-preserving image loss settings
        self.detail_loss_weight_init = 0.0
        self.detail_loss_weight_final = 0.0
        self.detail_loss_start_iter = 4000
        self.detail_loss_edge_quantile = 0.75
        self.detail_loss_max_weight = 0.05
        self.edge_guided_densify_factor = 0.90

        # Late-stage LPIPS perceptual loss. Disabled by default after v7 showed
        # weak/strong LPIPS variants were slower and did not improve PSNR balance.
        self.lpips_loss_weight_init = 0.0
        self.lpips_loss_weight_final = 0.0
        self.lpips_loss_start_iter = 16000
        self.lpips_loss_max_weight = 0.02
        self.lpips_loss_interval = 4

        # Differentiable PSNR supervision. Minimizes a scaled log-MSE proxy,
        # which preserves PSNR ordering while keeping gradients well-scaled.
        self.psnr_loss_weight_init = 0.0
        self.psnr_loss_weight_final = 0.0
        self.psnr_loss_start_iter = 0
        self.psnr_loss_max_weight = 0.05

        # Hard-view mining: reweight camera sampling by recent per-view difficulty.
        self.enable_hard_view_mining = False
        self.hard_view_start_iter = 4000
        self.hard_view_ema_momentum = 0.90
        self.hard_view_sample_alpha = 1.5
        self.hard_view_boost_max = 3.0
        self.hard_view_ssim_weight = 0.5
        self.hard_view_ramp_iters = 6000
        self.hard_view_uniform_mix = 0.5
        self.hard_view_require_fullres = True
        self.hard_view_psnr_weight = 0.10
        # Apply mild hard-view loss reweighting without changing camera sampling.
        self.enable_hard_view_loss_reweight = False
        self.hard_view_loss_reweight_start_iter = 18000
        self.hard_view_loss_reweight_max = 1.12

        # Color/exposure alignment: stabilize per-view color statistics to reduce PSNR drop cases.
        self.color_align_weight_init = 0.0
        self.color_align_weight_final = 0.0
        self.color_align_start_iter = 8000
        self.color_align_max_weight = 0.03
        self.color_align_std_weight = 0.5
        self.color_align_hard_view_boost = 1.5
        self.color_align_low_grad_quantile = 0.60
        self.color_align_min_mask_fraction = 0.20
        self.color_align_require_fullres = True

        # Optional geo-prior relaxation on low-texture views to avoid over-constraint.
        self.geo_prior_low_texture_relax_strength = 0.0
        self.geo_prior_low_texture_relax_min = 0.75

        # Resolution schedule: comma separated "iter:scale" pairs.
        # Example: "1200:4,4000:2,30000:1"
        self.resolution_schedule = "1200:4,4000:2,30000:1"

        # Grouped LR schedule for geometry/color branches
        self.enable_group_lr_schedule = False
        self.lr_stage1_end = 6000
        self.lr_stage2_end = 18000
        self.lr_geo_mult_stage1 = 1.10
        self.lr_geo_mult_stage2 = 1.00
        self.lr_geo_mult_stage3 = 0.75
        self.lr_color_mult_stage1 = 0.75
        self.lr_color_mult_stage2 = 1.05
        self.lr_color_mult_stage3 = 0.80

        # Geometry controller: clamp/prune oversized noisy Gaussians and log stats
        self.enable_geometry_controller = False
        self.geometry_start_iter = 9000
        self.geometry_interval = 250
        self.geometry_min_opacity = 0.008
        self.geometry_scale_cap_quantile = 0.98
        self.geometry_scale_cap_max = 0.12
        self.geometry_prune_scale_factor = 0.85
        self.geometry_protect_grad_quantile = 0.95

        # Region-wise evaluation (full / subject / background)
        self.enable_region_eval = False
        self.region_eval_boxes_json = ""

        # Sparse optimization settings
        self.sparse_mode = "none"  # one of: none, dash, spa, dash_spa
        self.sparse_budget_upperbound = 0  # <=0 disables budget pruning
        self.spa_interval = 1000
        self.spa_start_iter = 13000
        self.spa_end_iter = 25000
        self.spa_prune_ratio = 0.05
        self.prune_ratio2 = 0.04
        self.rho_lr = 0.0005
        self.spa_metric = "hybrid"  # one of: opacity, hybrid
        self.spa_require_full_resolution = True
        self.spa_after_fullres_delay = 3000
        self.spa_max_prune_points = 30000
        self.spa_protect_top_fraction = 0.05
        super().__init__(parser, "Experiment Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
