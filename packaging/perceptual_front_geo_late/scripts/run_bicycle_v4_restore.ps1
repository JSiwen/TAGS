# Train TAGSs on Mip-NeRF 360 bicycle with a perceptual-front / geo-late config.
#
# Prerequisites:
#   1. COLMAP data at data/bicycle/  (images/ + sparse/0/)
#   2. Monocular depths at data/bicycle/depth_any_out/
#   3. Depth scale file at data/bicycle/sparse/0/depth_params.json
#
# Usage (PowerShell, from repo root):
#   .\scripts\run_bicycle_v4_restore.ps1
#
# Override paths (use TAGGS_MODEL to avoid clashing with HuggingFace $env:MODEL):
#   $env:SOURCE = "D:\TAGSs\TAGSs_core\data\bicycle"
#   $env:DEPTHS = "D:\TAGSs\TAGSs_core\data\bicycle\depth_any_out"
#   $env:TAGGS_MODEL = "D:\TAGSs\TAGSs_core\output\bicycle_integrated_perceptual_front_geo_late"
#   .\scripts\run_bicycle_v4_restore.ps1
#
# Skip render/metrics after training:
#   $env:RUN_RENDER_METRICS = "0"
#   .\scripts\run_bicycle_v4_restore.ps1

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

function Get-OutputModelPath {
    if ($env:TAGGS_MODEL) { return $env:TAGGS_MODEL }
    if ($env:MODEL -and ($env:MODEL -match '^[A-Za-z]:[\\/]' -or $env:MODEL -match '^\\\\')) {
        return $env:MODEL
    }
    if ($env:MODEL) {
        Write-Warning "Ignoring `$env:MODEL='$($env:MODEL)' (not a filesystem path). Set `$env:TAGGS_MODEL for the training output directory."
    }
    return Join-Path $ROOT "output\bicycle_integrated_perceptual_front_geo_late"
}

function Invoke-TrainPython {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CommandArgs
    )
    $condaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "gaussian_splatting" }
    if ($env:CONDA_DEFAULT_ENV -eq $condaEnv) {
        & python @CommandArgs
        return
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        & conda run -n $condaEnv --no-capture-output python @CommandArgs
        return
    }
    & python @CommandArgs
}

$SOURCE = if ($env:SOURCE) { $env:SOURCE } else { Join-Path $ROOT "data\bicycle" }
$DEPTHS = if ($env:DEPTHS) { $env:DEPTHS } else { Join-Path $ROOT "data\bicycle\depth_any_out" }
$MODEL = Get-OutputModelPath
$ITERATIONS = if ($env:ITERATIONS) { $env:ITERATIONS } else { "30000" }
$RUN_RENDER_METRICS = if ($env:RUN_RENDER_METRICS) { $env:RUN_RENDER_METRICS } else { "1" }

if (-not $env:TAGGS_CPU_KNN_INIT) { $env:TAGGS_CPU_KNN_INIT = "1" }
if (-not $env:KMP_DUPLICATE_LIB_OK) { $env:KMP_DUPLICATE_LIB_OK = "TRUE" }
# Speed-first defaults for this heavy full-res stage.
# You can override before launch, e.g.:
#   $env:TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="1"   # disable this speedup
if (-not $env:TAGGS_DENSIFY_INTERVAL_MULT_FULLRES) { $env:TAGGS_DENSIFY_INTERVAL_MULT_FULLRES = "2" }
if (-not $env:TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD) { $env:TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD = "50000" }
# Avoid frequent empty_cache synchronization during densify unless explicitly needed for OOM mitigation.
if (-not $env:TAGGS_DENSIFY_EMPTY_CACHE) { $env:TAGGS_DENSIFY_EMPTY_CACHE = "0" }

function Ensure-PillowImport {
    python -c "from PIL import Image" 2>$null
    if ($LASTEXITCODE -eq 0) { return }
    $bin = Join-Path $env:CONDA_PREFIX "Library\bin"
    $deflate = Join-Path $bin "deflate.dll"
    $libdeflate = Join-Path $bin "libdeflate.dll"
    if ((Test-Path $deflate) -and -not (Test-Path $libdeflate)) {
        Write-Warning "Applying Windows Pillow fix: copy deflate.dll -> libdeflate.dll"
        Copy-Item $deflate $libdeflate
    }
    python -c "from PIL import Image" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pillow import failed. Run: .\scripts\fix_gaussian_splatting_pillow.ps1"
    }
}

if ($env:CONDA_PREFIX) {
    Ensure-PillowImport
}

$TrainPythonLabel = if ($env:CONDA_DEFAULT_ENV -eq $(if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "gaussian_splatting" })) {
    "python"
} elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    "conda run -n gaussian_splatting python"
} else {
    "python"
}

Write-Host "=== bicycle perceptual-front geo-late training ==="
Write-Host "SOURCE:     $SOURCE"
Write-Host "DEPTHS:     $DEPTHS"
Write-Host "MODEL:      $MODEL"
Write-Host "ITERATIONS: $ITERATIONS"
Write-Host "ROOT:       $ROOT"
Write-Host "PYTHON:     $TrainPythonLabel"
Write-Host ""

Invoke-TrainPython -CommandArgs @("-c", "import torch") 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyTorch not found. Run: conda activate gaussian_splatting"
}

if (-not (Test-Path $SOURCE)) {
    Write-Error "SOURCE directory not found: $SOURCE"
}

Invoke-TrainPython -CommandArgs @(
  "train_optimized.py",
  "-s", $SOURCE,
  "-m", $MODEL,
  "--images", "images",
  "--depths", $DEPTHS,
  "--resolution", "-1",
  "--data_device", "cpu",
  "--eval",
  "--llffhold", "8",
  "--iterations", $ITERATIONS,
  "--position_lr_init", "0.00016",
  "--position_lr_final", "0.0000016",
  "--position_lr_delay_mult", "0.01",
  "--position_lr_max_steps", $ITERATIONS,
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
  "--normal_prior_tv_weight", "0.0",
  "--depth_prior_edge_quantile", "0.92",
  "--depth_prior_rgb_edge_quantile", "0.95",
  "--depth_prior_var_quantile", "0.92",
  "--depth_prior_min_mask_fraction", "0.2",
  "--normal_tv_depth_edge_quantile", "0.7",
  "--normal_tv_rgb_edge_quantile", "0.7",
  "--depth_prior_weight_init", "0.0",
  "--depth_prior_weight_final", "0.04",
  "--depth_prior_weight_min", "0.01",
  "--normal_prior_weight_init", "0.0",
  "--normal_prior_weight_final", "0.012",
  "--normal_prior_weight_min", "0.0",
  "--geo_prior_low_texture_relax_strength", "0.0",
  "--geo_prior_low_texture_relax_min", "0.90",
  "--geo_prior_start_iter", "26000",
  "--geo_prior_warmup_until", "28500",
  "--geo_prior_decay_start", "30000",
  "--geo_prior_decay_until", "30000",
  # Keep geometry late; use perceptual/PSNR image supervision in the early/mid stages.
  "--detail_loss_weight_init", "0.0",
  "--detail_loss_weight_final", "0.0",
  "--detail_loss_start_iter", "4000",
  "--detail_loss_edge_quantile", "0.75",
  "--detail_loss_max_weight", "0.05",
  "--lpips_loss_weight_init", "0.004",
  "--lpips_loss_weight_final", "0.0",
  "--lpips_loss_start_iter", "1",
  "--lpips_loss_max_weight", "0.008",
  "--lpips_loss_interval", "8",
  "--psnr_loss_weight_init", "0.020",
  "--psnr_loss_weight_final", "0.010",
  "--psnr_loss_start_iter", "1",
  "--psnr_loss_max_weight", "0.030",
  # Use loss reweighting only; keep camera sampling uniform to avoid unstable hard-view feedback.
  "--enable_hard_view_loss_reweight",
  "--hard_view_loss_reweight_start_iter", "8000",
  "--hard_view_loss_reweight_max", "1.20",
  "--hard_view_psnr_weight", "0.15",
  "--color_align_weight_init", "0.0",
  # Late color align had tiny aggregate gains with per-view trade-offs; disable by default for speed.
  "--color_align_weight_final", "0.0",
  "--color_align_start_iter", "24000",
  "--color_align_max_weight", "0.0",
  "--color_align_std_weight", "0.0",
  "--color_align_hard_view_boost", "1.0",
  "--color_align_low_grad_quantile", "0.60",
  "--color_align_min_mask_fraction", "0.20",
  "--color_align_require_fullres",
  "--enable_resolution_adaptive",
  "--resolution_schedule", "1200:4,4000:2,30000:1",
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
  "--test_iterations", "30000",
  "--save_iterations", "30000",
  "--disable_viewer"
)

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RUN_RENDER_METRICS -eq "1") {
  Write-Host ""
  Write-Host "=== render & metrics ==="
  Invoke-TrainPython -CommandArgs @("render.py", "-m", $MODEL, "--skip_train", "--quiet")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Invoke-TrainPython -CommandArgs @("metrics.py", "-m", $MODEL, "--split", "test")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "Done. Model saved to: $MODEL"
