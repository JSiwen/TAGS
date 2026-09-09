from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    enable_geo_prior: bool
    enable_resolution_adaptive: bool
    enable_sparse_optimization: bool
    geo_prior_depth: bool
    geo_prior_normal: bool
    depth_prior_loss_type: str
    normal_prior_loss_type: str
    depth_prior_pearson_weight: float
    depth_prior_local_pearson_weight: float
    normal_prior_tv_weight: float
    depth_prior_edge_quantile: float
    depth_prior_rgb_edge_quantile: float
    depth_prior_var_quantile: float
    depth_prior_min_mask_fraction: float
    normal_tv_depth_edge_quantile: float
    normal_tv_rgb_edge_quantile: float
    depth_prior_weight_init: float
    depth_prior_weight_final: float
    normal_prior_weight_init: float
    normal_prior_weight_final: float
    depth_prior_weight_min: float
    normal_prior_weight_min: float
    geo_prior_start_iter: int
    geo_prior_warmup_until: int
    geo_prior_decay_start: int
    geo_prior_decay_until: int
    detail_loss_weight_init: float
    detail_loss_weight_final: float
    detail_loss_start_iter: int
    detail_loss_edge_quantile: float
    detail_loss_max_weight: float
    edge_guided_densify_factor: float
    lpips_loss_weight_init: float
    lpips_loss_weight_final: float
    lpips_loss_start_iter: int
    lpips_loss_max_weight: float
    lpips_loss_interval: int
    psnr_loss_weight_init: float
    psnr_loss_weight_final: float
    psnr_loss_start_iter: int
    psnr_loss_max_weight: float
    enable_hard_view_mining: bool
    hard_view_start_iter: int
    hard_view_ema_momentum: float
    hard_view_sample_alpha: float
    hard_view_boost_max: float
    hard_view_ssim_weight: float
    hard_view_ramp_iters: int
    hard_view_uniform_mix: float
    hard_view_require_fullres: bool
    enable_hard_view_loss_reweight: bool
    hard_view_loss_reweight_start_iter: int
    hard_view_loss_reweight_max: float
    color_align_weight_init: float
    color_align_weight_final: float
    color_align_start_iter: int
    color_align_max_weight: float
    color_align_std_weight: float
    color_align_hard_view_boost: float
    color_align_low_grad_quantile: float
    color_align_min_mask_fraction: float
    color_align_require_fullres: bool
    enable_balanced_late_controller: bool
    late_controller_structure_end_iter: int
    late_controller_handoff_end_iter: int
    late_controller_refine_end_iter: int
    late_controller_lpips_guard_start_iter: int
    late_controller_lpips_guard_max_boost: float
    late_controller_lpips_guard_view_boost: float
    late_controller_lpips_guard_chronic_boost: float
    late_controller_lpips_guard_margin: float
    late_controller_lpips_guard_ema_momentum: float
    late_controller_lpips_proxy_interval: int
    late_controller_color_stage_boost: float
    late_controller_color_polish_scale: float
    late_controller_chronic_views: str
    geo_prior_low_texture_relax_strength: float
    geo_prior_low_texture_relax_min: float
    resolution_schedule: str
    enable_group_lr_schedule: bool
    lr_stage1_end: int
    lr_stage2_end: int
    lr_geo_mult_stage1: float
    lr_geo_mult_stage2: float
    lr_geo_mult_stage3: float
    lr_color_mult_stage1: float
    lr_color_mult_stage2: float
    lr_color_mult_stage3: float
    enable_geometry_controller: bool
    geometry_start_iter: int
    geometry_interval: int
    geometry_min_opacity: float
    geometry_scale_cap_quantile: float
    geometry_scale_cap_max: float
    geometry_prune_scale_factor: float
    geometry_protect_grad_quantile: float
    enable_region_eval: bool
    region_eval_boxes_json: str
    enable_live_loss_logging: bool
    live_loss_log_interval: int
    live_stage_iterations: str
    sparse_mode: str
    sparse_budget_upperbound: int
    spa_interval: int
    spa_start_iter: int
    spa_end_iter: int
    spa_prune_ratio: float
    prune_ratio2: float
    rho_lr: float
    spa_metric: str
    spa_require_full_resolution: bool
    spa_after_fullres_delay: int
    spa_max_prune_points: int
    spa_protect_top_fraction: float
    enable_resolution_aware_schedule: bool
    enable_adaptive_stage_controller: bool
    ras_ref_pixels: int
    ras_scale_min: float
    ras_scale_max: float
    ras_densify_base: int
    ras_densify_min: int
    ras_densify_max: int
    ras_spa_delay: int
    adaptive_densify_extend_step: int
    adaptive_densify_max_iter: int
    adaptive_point_budget: int

    @staticmethod
    def from_args(args):
        return ExperimentConfig(
            enable_geo_prior=args.enable_geo_prior,
            enable_resolution_adaptive=args.enable_resolution_adaptive,
            enable_sparse_optimization=args.enable_sparse_optimization,
            geo_prior_depth=args.geo_prior_depth,
            geo_prior_normal=args.geo_prior_normal,
            depth_prior_loss_type=args.depth_prior_loss_type,
            normal_prior_loss_type=args.normal_prior_loss_type,
            depth_prior_pearson_weight=args.depth_prior_pearson_weight,
            depth_prior_local_pearson_weight=args.depth_prior_local_pearson_weight,
            normal_prior_tv_weight=args.normal_prior_tv_weight,
            depth_prior_edge_quantile=args.depth_prior_edge_quantile,
            depth_prior_rgb_edge_quantile=args.depth_prior_rgb_edge_quantile,
            depth_prior_var_quantile=args.depth_prior_var_quantile,
            depth_prior_min_mask_fraction=args.depth_prior_min_mask_fraction,
            normal_tv_depth_edge_quantile=args.normal_tv_depth_edge_quantile,
            normal_tv_rgb_edge_quantile=args.normal_tv_rgb_edge_quantile,
            depth_prior_weight_init=args.depth_prior_weight_init,
            depth_prior_weight_final=args.depth_prior_weight_final,
            normal_prior_weight_init=args.normal_prior_weight_init,
            normal_prior_weight_final=args.normal_prior_weight_final,
            depth_prior_weight_min=args.depth_prior_weight_min,
            normal_prior_weight_min=args.normal_prior_weight_min,
            geo_prior_start_iter=args.geo_prior_start_iter,
            geo_prior_warmup_until=args.geo_prior_warmup_until,
            geo_prior_decay_start=args.geo_prior_decay_start,
            geo_prior_decay_until=args.geo_prior_decay_until,
            detail_loss_weight_init=args.detail_loss_weight_init,
            detail_loss_weight_final=args.detail_loss_weight_final,
            detail_loss_start_iter=args.detail_loss_start_iter,
            detail_loss_edge_quantile=args.detail_loss_edge_quantile,
            detail_loss_max_weight=args.detail_loss_max_weight,
            edge_guided_densify_factor=args.edge_guided_densify_factor,
            lpips_loss_weight_init=args.lpips_loss_weight_init,
            lpips_loss_weight_final=args.lpips_loss_weight_final,
            lpips_loss_start_iter=args.lpips_loss_start_iter,
            lpips_loss_max_weight=args.lpips_loss_max_weight,
            lpips_loss_interval=args.lpips_loss_interval,
            psnr_loss_weight_init=args.psnr_loss_weight_init,
            psnr_loss_weight_final=args.psnr_loss_weight_final,
            psnr_loss_start_iter=args.psnr_loss_start_iter,
            psnr_loss_max_weight=args.psnr_loss_max_weight,
            enable_hard_view_mining=args.enable_hard_view_mining,
            hard_view_start_iter=args.hard_view_start_iter,
            hard_view_ema_momentum=args.hard_view_ema_momentum,
            hard_view_sample_alpha=args.hard_view_sample_alpha,
            hard_view_boost_max=args.hard_view_boost_max,
            hard_view_ssim_weight=args.hard_view_ssim_weight,
            hard_view_ramp_iters=args.hard_view_ramp_iters,
            hard_view_uniform_mix=args.hard_view_uniform_mix,
            hard_view_require_fullres=args.hard_view_require_fullres,
            enable_hard_view_loss_reweight=args.enable_hard_view_loss_reweight,
            hard_view_loss_reweight_start_iter=args.hard_view_loss_reweight_start_iter,
            hard_view_loss_reweight_max=args.hard_view_loss_reweight_max,
            color_align_weight_init=args.color_align_weight_init,
            color_align_weight_final=args.color_align_weight_final,
            color_align_start_iter=args.color_align_start_iter,
            color_align_max_weight=args.color_align_max_weight,
            color_align_std_weight=args.color_align_std_weight,
            color_align_hard_view_boost=args.color_align_hard_view_boost,
            color_align_low_grad_quantile=args.color_align_low_grad_quantile,
            color_align_min_mask_fraction=args.color_align_min_mask_fraction,
            color_align_require_fullres=args.color_align_require_fullres,
            enable_balanced_late_controller=args.enable_balanced_late_controller,
            late_controller_structure_end_iter=args.late_controller_structure_end_iter,
            late_controller_handoff_end_iter=args.late_controller_handoff_end_iter,
            late_controller_refine_end_iter=args.late_controller_refine_end_iter,
            late_controller_lpips_guard_start_iter=args.late_controller_lpips_guard_start_iter,
            late_controller_lpips_guard_max_boost=args.late_controller_lpips_guard_max_boost,
            late_controller_lpips_guard_view_boost=args.late_controller_lpips_guard_view_boost,
            late_controller_lpips_guard_chronic_boost=args.late_controller_lpips_guard_chronic_boost,
            late_controller_lpips_guard_margin=args.late_controller_lpips_guard_margin,
            late_controller_lpips_guard_ema_momentum=args.late_controller_lpips_guard_ema_momentum,
            late_controller_lpips_proxy_interval=args.late_controller_lpips_proxy_interval,
            late_controller_color_stage_boost=args.late_controller_color_stage_boost,
            late_controller_color_polish_scale=args.late_controller_color_polish_scale,
            late_controller_chronic_views=args.late_controller_chronic_views,
            geo_prior_low_texture_relax_strength=args.geo_prior_low_texture_relax_strength,
            geo_prior_low_texture_relax_min=args.geo_prior_low_texture_relax_min,
            resolution_schedule=args.resolution_schedule,
            enable_group_lr_schedule=args.enable_group_lr_schedule,
            lr_stage1_end=args.lr_stage1_end,
            lr_stage2_end=args.lr_stage2_end,
            lr_geo_mult_stage1=args.lr_geo_mult_stage1,
            lr_geo_mult_stage2=args.lr_geo_mult_stage2,
            lr_geo_mult_stage3=args.lr_geo_mult_stage3,
            lr_color_mult_stage1=args.lr_color_mult_stage1,
            lr_color_mult_stage2=args.lr_color_mult_stage2,
            lr_color_mult_stage3=args.lr_color_mult_stage3,
            enable_geometry_controller=args.enable_geometry_controller,
            geometry_start_iter=args.geometry_start_iter,
            geometry_interval=args.geometry_interval,
            geometry_min_opacity=args.geometry_min_opacity,
            geometry_scale_cap_quantile=args.geometry_scale_cap_quantile,
            geometry_scale_cap_max=args.geometry_scale_cap_max,
            geometry_prune_scale_factor=args.geometry_prune_scale_factor,
            geometry_protect_grad_quantile=args.geometry_protect_grad_quantile,
            enable_region_eval=args.enable_region_eval,
            region_eval_boxes_json=args.region_eval_boxes_json,
            enable_live_loss_logging=args.enable_live_loss_logging,
            live_loss_log_interval=args.live_loss_log_interval,
            live_stage_iterations=args.live_stage_iterations,
            sparse_mode=args.sparse_mode,
            sparse_budget_upperbound=args.sparse_budget_upperbound,
            spa_interval=args.spa_interval,
            spa_start_iter=args.spa_start_iter,
            spa_end_iter=args.spa_end_iter,
            spa_prune_ratio=args.spa_prune_ratio,
            prune_ratio2=args.prune_ratio2,
            rho_lr=args.rho_lr,
            spa_metric=args.spa_metric,
            spa_require_full_resolution=args.spa_require_full_resolution,
            spa_after_fullres_delay=args.spa_after_fullres_delay,
            spa_max_prune_points=args.spa_max_prune_points,
            spa_protect_top_fraction=args.spa_protect_top_fraction,
            enable_resolution_aware_schedule=args.enable_resolution_aware_schedule,
            enable_adaptive_stage_controller=args.enable_adaptive_stage_controller,
            ras_ref_pixels=args.ras_ref_pixels,
            ras_scale_min=args.ras_scale_min,
            ras_scale_max=args.ras_scale_max,
            ras_densify_base=args.ras_densify_base,
            ras_densify_min=args.ras_densify_min,
            ras_densify_max=args.ras_densify_max,
            ras_spa_delay=args.ras_spa_delay,
            adaptive_densify_extend_step=args.adaptive_densify_extend_step,
            adaptive_densify_max_iter=args.adaptive_densify_max_iter,
            adaptive_point_budget=args.adaptive_point_budget,
        )
