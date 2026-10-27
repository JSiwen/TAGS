# Bicycle re-run: official 3DGS baseline + E4 winner + v14 winner, all at 1237x822.
#
# Protocol (identical for all three runs):
#   scene      : Mip-NeRF 360 bicycle
#   images     : images_4 (1237x822 native), --resolution 1
#   split      : --eval, llffhold 8 -> 25 test views / 169 train views
#   iterations : 30000, seed 0
#   metrics    : metrics.py, VGG LPIPS (byte-identical lpipsPyTorch in both repos)
#
# Case A: official graphdeco-inria/gaussian-splatting, stock defaults.
#         (baseline "normal values": densify_until 15000, grad 0.0002,
#          opacity prune 0.005, screen-size 20, no max_gaussians, no density cap)
# Case B: TAGS E4 winner (lpips_ms_w005) via train_optimized.py, recipe unchanged.
# Case C: TAGS v14 v8 monitor_28500_select via train_optimized.py, recipe unchanged
#         except images_4/-r 1 to align the pixel budget.
#
# Usage (repo root):
#   .\scripts\run_bicycle_rerun_1237.ps1
#
# Run a single case:
#   $env:RERUN_CASE = "official"   # A
#   $env:RERUN_CASE = "e4"         # B
#   $env:RERUN_CASE = "v14"        # C
#   .\scripts\run_bicycle_rerun_1237.ps1
#
# Case A (official graphdeco-inria 3DGS) is OPT-IN. It is skipped unless
# RERUN_OFFICIAL=1 is set, because the baseline numbers we compare against
# already exist (see scripts/summarize_bicycle_3way.py) and the official run
# costs ~70-75 min of GPU time.
#
# Skip training (render + metrics only):
#   $env:RERUN_SKIP_TRAIN = "1"; .\scripts\run_bicycle_rerun_1237.ps1

$ErrorActionPreference = "Continue"

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$PY = if ($env:RERUN_PYTHON) {
    $env:RERUN_PYTHON
} else {
    "C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe"
}
$OFFICIAL_ROOT = Join-Path $ROOT "third_party\GaussianSplatting"

$SOURCE = if ($env:RERUN_SOURCE) { $env:RERUN_SOURCE } else { Join-Path $ROOT "data\bicycle" }
$IMAGES = if ($env:RERUN_IMAGES) { $env:RERUN_IMAGES } else { "images_4" }
$RESOLUTION = if ($env:RERUN_RESOLUTION) { $env:RERUN_RESOLUTION } else { "1" }
$ITERATIONS = if ($env:RERUN_ITERATIONS) { [int]$env:RERUN_ITERATIONS } else { 30000 }
$SEED = if ($env:RERUN_SEED) { [int]$env:RERUN_SEED } else { 0 }
$SKIP_TRAIN = if ($env:RERUN_SKIP_TRAIN) { $env:RERUN_SKIP_TRAIN -eq "1" } else { $false }
$ONLY = if ($env:RERUN_CASE) { $env:RERUN_CASE.ToLower() } else { "" }
# Case A (official graphdeco-inria 3DGS) is opt-in: set RERUN_OFFICIAL=1 to include it.
$RUN_OFFICIAL = if ($env:RERUN_OFFICIAL) { $env:RERUN_OFFICIAL -eq "1" } else { $false }

# Shared env for every child process.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:TAGGS_CPU_KNN_INIT = "1"
$env:PYTHONUNBUFFERED = "1"
$env:TAGGS_DENSIFY_EMPTY_CACHE = "1"

# Official repo needs its own dir on sys.path; TAGS runs need the repo root.
$env:PYTHONPATH = "$OFFICIAL_ROOT;$ROOT"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "=============================================================="
    Write-Host "  $msg"
    Write-Host "=============================================================="
}

function Get-LatestCheckpoint {
    param([string]$ModelDir)
    if (-not (Test-Path $ModelDir)) { return $null }
    $best = $null
    $bestIter = -1
    Get-ChildItem -Path $ModelDir -Filter "chkpnt*.pth" -File -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.BaseName -match '^chkpnt(\d+)$') {
            $it = [int]$Matches[1]
            if ($it -gt $bestIter) { $bestIter = $it; $best = $_.FullName }
        }
    }
    if ($null -eq $best) { return $null }
    return @{ Path = $best; Iteration = $bestIter }
}

function Invoke-Logged {
    param(
        [string]$WorkingDir,
        [string]$LogPath,
        [string]$Label,
        [string[]]$CmdArgs
    )
    Write-Host "[$Label] cwd=$WorkingDir"
    Write-Host "[$Label] log=$LogPath"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        Push-Location $WorkingDir
        & $PY @CmdArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0) {
        throw "[$Label] failed with exit code $code. Tail of ${LogPath}:"
        Get-Content $LogPath -Tail 40
    }
}

# ---------------------------------------------------------------------------
# Case A: official graphdeco-inria baseline
# ---------------------------------------------------------------------------
function Invoke-OfficialBaseline {
    $modelDir = Join-Path $ROOT "output\bicycle_official_3dgs_1237"
    $logPath = Join-Path $ROOT "output\bicycle_official_3dgs_1237.log"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    $finalPly = Join-Path $modelDir "point_cloud\iteration_$ITERATIONS\point_cloud.ply"
    if ($SKIP_TRAIN -or (Test-Path $finalPly)) {
        Write-Host "[official] final ply present, skipping training."
    } else {
        $resume = Get-LatestCheckpoint $modelDir
        $trainArgs = @(
            "train.py",
            "-s", $SOURCE,
            "-m", $modelDir,
            "-r", $RESOLUTION,
            "--images", $IMAGES,
            "--eval",
            "--data_device", "cpu",
            "--iterations", "$ITERATIONS",
            "--test_iterations", "7000", "$ITERATIONS",
            "--save_iterations", "7000", "$ITERATIONS",
            "--checkpoint_iterations", "3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "$ITERATIONS",
            "--disable_viewer"
        )
        if ($null -ne $resume) {
            $trainArgs += @("--start_checkpoint", $resume.Path)
            Write-Host "[official] resume from $($resume.Path)"
        }
        Invoke-Logged -WorkingDir $OFFICIAL_ROOT -LogPath $logPath -Label "official" -CmdArgs $trainArgs
    }

    Write-Host "[official] rendering test split"
    Invoke-Logged -WorkingDir $OFFICIAL_ROOT -LogPath $logPath -Label "official-render" -CmdArgs @(
        "render.py", "-m", $modelDir, "--iteration", "$ITERATIONS", "--skip_train", "--quiet"
    )
    Write-Host "[official] computing metrics"
    Invoke-Logged -WorkingDir $OFFICIAL_ROOT -LogPath $logPath -Label "official-metrics" -CmdArgs @(
        "metrics.py", "-m", $modelDir
    )
}

# ---------------------------------------------------------------------------
# Case B: TAGS E4 winner (lpips_ms_w005)
# ---------------------------------------------------------------------------
function Get-E4Args {
    $modelDir = $script:modelDir
    $evalIters = @("7000", "10000", "15100", "18100", "19500", "21100", "24000", "$ITERATIONS")
    $ckptIters = @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "$ITERATIONS")
    return @(
        "train_optimized.py",
        "-s", $SOURCE,
        "-m", $modelDir,
        "--images", $IMAGES,
        "--depths", (Join-Path $SOURCE "depth_any_out"),
        "--resolution", $RESOLUTION,
        "--data_device", "cpu",
        "--eval",
        "--llffhold", "8",
        "--seed", "$SEED",
        "--iterations", "$ITERATIONS",
        "--position_lr_init", "0.00016",
        "--position_lr_final", "0.0000016",
        "--position_lr_delay_mult", "0.01",
        "--position_lr_max_steps", "$ITERATIONS",
        "--feature_lr", "0.0025",
        "--opacity_lr", "0.025",
        "--scaling_lr", "0.005",
        "--rotation_lr", "0.001",
        "--exposure_lr_init", "0.0",
        "--exposure_lr_final", "0.0",
        "--percent_dense", "0.01",
        "--lambda_dssim", "0.2",
        "--densification_interval", "100",
        "--opacity_reset_interval", "3000",
        "--densify_from_iter", "500",
        "--densify_until_iter", "18000",
        "--densify_until_fraction", "0.50",
        "--densify_grad_threshold", "0.00016",
        "--max_gaussians", "2500000",
        "--prune_opacity_threshold", "0.02",
        "--prune_min_visibility", "0",
        "--prune_max_screen_size", "-1",
        "--prune_grad_threshold", "0.0",
        "--prune_grad_min_denom", "1.0",
        "--prune_only_from_fraction", "0.50",
        "--prune_until_fraction", "1.0",
        "--enable_geo_prior",
        "--geo_prior_depth",
        "--geo_prior_normal",
        "--depth_prior_loss_type", "edge_aware_log_l1",
        "--normal_prior_loss_type", "cosine",
        "--depth_prior_pearson_weight", "0.0",
        "--depth_prior_local_pearson_weight", "0.0",
        "--normal_prior_tv_weight", "0.0015",
        "--depth_prior_weight_init", "0.0",
        "--depth_prior_weight_final", "0.04",
        "--depth_prior_weight_min", "0.01",
        "--geo_prior_start_iter", "0",
        "--geo_prior_warmup_until", "4000",
        "--geo_prior_decay_start", "10000",
        "--geo_prior_decay_until", "20000",
        "--lpips_loss_weight_init", "0.0",
        "--lpips_loss_weight_final", "0.05",
        "--lpips_loss_start_iter", "21000",
        "--lpips_loss_max_weight", "0.05",
        "--lpips_loss_interval", "1",
        "--lpips_loss_schedule", "step",
        "--enable_multiscale_lpips",
        "--lpips_multiscale_scales", "1.0,0.5,0.25",
        "--lpips_multiscale_weights", "0.5,0.3,0.2",
        "--psnr_loss_weight_init", "0.0",
        "--psnr_loss_weight_final", "0.0",
        "--detail_loss_weight_init", "0.0",
        "--detail_loss_weight_final", "0.0",
        "--color_align_weight_init", "0.0",
        "--color_align_weight_final", "0.0",
        "--enable_live_loss_logging",
        "--live_loss_log_interval", "100",
        "--live_stage_iterations", ($evalIters -join ","),
        "--test_iterations"
    ) + $evalIters + @(
        "--save_iterations", "$ITERATIONS",
        "--checkpoint_iterations"
    ) + $ckptIters + @(
        "--disable_viewer"
    )
}

function Invoke-E4 {
    $script:modelDir = Join-Path $ROOT "output\bicycle_e4_lpips_ms_w005_rerun_1237"
    $modelDir = $script:modelDir
    $logPath = Join-Path $ROOT "output\bicycle_e4_lpips_ms_w005_rerun_1237.log"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    $finalPly = Join-Path $modelDir "point_cloud\iteration_$ITERATIONS\point_cloud.ply"
    if ($SKIP_TRAIN -or (Test-Path $finalPly)) {
        Write-Host "[e4] final ply present, skipping training."
    } else {
        $args = Get-E4Args
        $resume = Get-LatestCheckpoint $modelDir
        if ($null -ne $resume) {
            $args += @("--start_checkpoint", $resume.Path)
            Write-Host "[e4] resume from $($resume.Path)"
        }
        Invoke-Logged -WorkingDir $ROOT -LogPath $logPath -Label "e4" -CmdArgs $args
    }

    Write-Host "[e4] rendering test split"
    Invoke-Logged -WorkingDir $ROOT -LogPath $logPath -Label "e4-render" -CmdArgs @(
        "render.py", "-m", $modelDir, "--iteration", "$ITERATIONS", "--skip_train", "--quiet"
    )
    Write-Host "[e4] computing metrics"
    Invoke-Logged -WorkingDir $ROOT -LogPath $logPath -Label "e4-metrics" -CmdArgs @(
        "metrics.py", "-m", $modelDir, "--split", "test"
    )
}

# ---------------------------------------------------------------------------
# Case C: TAGS v14 v8 monitor_28500_select
# ---------------------------------------------------------------------------
function Get-V14Args {
    $modelDir = $script:modelDir
    $stageIters = @("7000", "18000", "24000", "25500", "28500", "$ITERATIONS")
    $ckptIters = @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "28500", "$ITERATIONS")
    return @(
        "train_optimized.py",
        "-s", $SOURCE,
        "-m", $modelDir,
        "--images", $IMAGES,
        "--depths", (Join-Path $SOURCE "depth_any_out"),
        "--resolution", $RESOLUTION,
        "--data_device", "cpu",
        "--eval",
        "--llffhold", "8",
        "--seed", "$SEED",
        "--iterations", "$ITERATIONS",
        "--position_lr_init", "0.00016",
        "--position_lr_final", "0.0000016",
        "--position_lr_delay_mult", "0.01",
        "--position_lr_max_steps", "$ITERATIONS",
        "--feature_lr", "0.0025",
        "--opacity_lr", "0.025",
        "--scaling_lr", "0.005",
        "--rotation_lr", "0.001",
        "--exposure_lr_init", "0.01",
        "--exposure_lr_final", "0.001",
        "--exposure_lr_delay_steps", "0",
        "--exposure_lr_delay_mult", "0.0",
        "--percent_dense", "0.01",
        "--lambda_dssim", "0.2",
        "--densification_interval", "100",
        "--opacity_reset_interval", "3000",
        "--densify_from_iter", "500",
        "--densify_until_iter", "18000",
        "--densify_grad_threshold", "0.00016",
        "--depth_l1_weight_init", "1.0",
        "--depth_l1_weight_final", "0.01",
        "--enable_geo_prior",
        "--geo_prior_depth",
        "--geo_prior_normal",
        "--depth_prior_loss_type", "edge_aware_log_l1",
        "--normal_prior_loss_type", "cosine",
        "--depth_prior_pearson_weight", "0.0",
        "--depth_prior_local_pearson_weight", "0.05",
        "--normal_prior_tv_weight", "0.003",
        "--depth_prior_edge_quantile", "0.92",
        "--depth_prior_rgb_edge_quantile", "0.95",
        "--depth_prior_var_quantile", "0.92",
        "--depth_prior_min_mask_fraction", "0.2",
        "--normal_tv_depth_edge_quantile", "0.7",
        "--normal_tv_rgb_edge_quantile", "0.7",
        "--depth_prior_weight_init", "0.0",
        "--depth_prior_weight_final", "0.05",
        "--depth_prior_weight_min", "0.01",
        "--normal_prior_weight_init", "0.0",
        "--normal_prior_weight_final", "0.02",
        "--normal_prior_weight_min", "0.0",
        "--geo_prior_low_texture_relax_strength", "0.0",
        "--geo_prior_low_texture_relax_min", "0.90",
        "--geo_prior_start_iter", "0",
        "--geo_prior_warmup_until", "5000",
        "--geo_prior_decay_start", "12000",
        "--geo_prior_decay_until", "25000",
        "--detail_loss_weight_init", "0.0",
        "--detail_loss_weight_final", "0.0",
        "--detail_loss_start_iter", "4000",
        "--detail_loss_edge_quantile", "0.75",
        "--detail_loss_max_weight", "0.05",
        "--edge_guided_densify_factor", "0.90",
        "--lpips_loss_weight_init", "0.0",
        "--lpips_loss_weight_final", "0.010",
        "--lpips_loss_start_iter", "16000",
        "--lpips_loss_interval", "4",
        "--lpips_loss_max_weight", "0.02",
        "--color_align_weight_init", "0.0",
        "--color_align_weight_final", "0.0015",
        "--color_align_start_iter", "26000",
        "--color_align_max_weight", "0.004",
        "--color_align_std_weight", "0.25",
        "--color_align_hard_view_boost", "1.0",
        "--color_align_low_grad_quantile", "0.50",
        "--color_align_min_mask_fraction", "0.20",
        "--color_align_require_fullres",
        "--enable_resolution_adaptive",
        "--resolution_schedule", "1200:4,4000:2,30000:1",
        "--enable_group_lr_schedule",
        "--lr_stage1_end", "6000",
        "--lr_stage2_end", "18000",
        "--lr_geo_mult_stage1", "1.10",
        "--lr_geo_mult_stage2", "1.00",
        "--lr_geo_mult_stage3", "0.75",
        "--lr_color_mult_stage1", "0.75",
        "--lr_color_mult_stage2", "1.05",
        "--lr_color_mult_stage3", "0.80",
        "--enable_geometry_controller",
        "--geometry_start_iter", "9000",
        "--geometry_interval", "250",
        "--geometry_min_opacity", "0.008",
        "--geometry_scale_cap_quantile", "0.98",
        "--geometry_scale_cap_max", "0.12",
        "--geometry_prune_scale_factor", "0.85",
        "--geometry_protect_grad_quantile", "0.95",
        "--enable_sparse_optimization",
        "--sparse_mode", "spa",
        "--spa_interval", "1000",
        "--spa_start_iter", "13000",
        "--spa_end_iter", "25000",
        "--spa_prune_ratio", "0.05",
        "--prune_ratio2", "0.04",
        "--rho_lr", "0.0005",
        "--spa_metric", "hybrid",
        "--spa_require_full_resolution",
        "--spa_after_fullres_delay", "3000",
        "--spa_max_prune_points", "30000",
        "--spa_protect_top_fraction", "0.05",
        "--enable_live_loss_logging",
        "--live_loss_log_interval", "100",
        "--live_stage_iterations", ($stageIters -join ","),
        "--test_iterations"
    ) + $stageIters + @(
        "--save_iterations"
    ) + $stageIters + @(
        "--checkpoint_iterations"
    ) + $ckptIters + @(
        "--disable_viewer"
    )
}

function Invoke-V14 {
    $script:modelDir = Join-Path $ROOT "output\bicycle_v14_v8_monitor_28500_select_1237"
    $modelDir = $script:modelDir
    $logPath = Join-Path $ROOT "output\bicycle_v14_v8_monitor_28500_select_1237.log"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    $finalPly = Join-Path $modelDir "point_cloud\iteration_$ITERATIONS\point_cloud.ply"
    if ($SKIP_TRAIN -or (Test-Path $finalPly)) {
        Write-Host "[v14] final ply present, skipping training."
    } else {
        $args = Get-V14Args
        $resume = Get-LatestCheckpoint $modelDir
        if ($null -ne $resume) {
            $args += @("--start_checkpoint", $resume.Path)
            Write-Host "[v14] resume from $($resume.Path)"
        }
        Invoke-Logged -WorkingDir $ROOT -LogPath $logPath -Label "v14" -CmdArgs $args
    }

    $stageIters = @("7000", "18000", "24000", "25500", "28500", "$ITERATIONS")
    foreach ($it in $stageIters) {
        $ply = Join-Path $modelDir "point_cloud\iteration_$it\point_cloud.ply"
        if (-not (Test-Path $ply)) {
            Write-Host "[v14] skip render $it (missing ply)"
            continue
        }
        Invoke-Logged -WorkingDir $ROOT -LogPath $logPath -Label "v14-render-$it" -CmdArgs @(
            "render.py", "-m", $modelDir, "--iteration", "$it", "--skip_train", "--quiet"
        )
    }
    Write-Host "[v14] computing metrics"
    Invoke-Logged -WorkingDir $ROOT -LogPath $logPath -Label "v14-metrics" -CmdArgs @(
        "metrics.py", "-m", $modelDir, "--split", "test"
    )
}

# ---------------------------------------------------------------------------
Write-Step "bicycle re-run | official baseline + E4 + v14 | $IMAGES @ r$RESOLUTION | $ITERATIONS iters | seed $SEED"
Write-Host "python : $PY"
Write-Host "source : $SOURCE"
Write-Host "only   : $(if ($ONLY) { $ONLY } else { '<all>' })"

if (-not (Test-Path $PY)) { throw "Python not found: $PY" }
if (-not (Test-Path (Join-Path $SOURCE "sparse\0\cameras.bin"))) {
    throw "COLMAP sparse/0 not found under $SOURCE"
}
if (-not (Test-Path (Join-Path $SOURCE $IMAGES))) {
    throw "Image folder not found: $(Join-Path $SOURCE $IMAGES)"
}

$start = Get-Date
if ($RUN_OFFICIAL -and ($ONLY -eq "" -or $ONLY -eq "official")) {
    Write-Step "Case A: official graphdeco-inria 3DGS (stock defaults)"
    Invoke-OfficialBaseline
} elseif ($ONLY -eq "official") {
    Write-Step "Case A skipped: official 3DGS is opt-in. Set RERUN_OFFICIAL=1 to run it."
}
if ($ONLY -eq "" -or $ONLY -eq "e4") {
    Write-Step "Case B: TAGS E4 winner (lpips_ms_w005)"
    Invoke-E4
}
if ($ONLY -eq "" -or $ONLY -eq "v14") {
    Write-Step "Case C: TAGS v14 v8 monitor_28500_select"
    Invoke-V14
}

Write-Step "Done in $([math]::Round(((Get-Date) - $start).TotalMinutes, 1)) minutes"
Write-Host "Results:"
Write-Host "  A official : output\bicycle_official_3dgs_1237\results.json  (only if RERUN_OFFICIAL=1)"
Write-Host "  B E4       : output\bicycle_e4_lpips_ms_w005_rerun_1237\results.json"
Write-Host "  C v14      : output\bicycle_v14_v8_monitor_28500_select_1237\results.json"
