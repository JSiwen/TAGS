# Frozen E4 winner (lpips_ms_w005) on Mip-NeRF 360 bicycle.
#
# Full 30k, seed 0:
#   .\scripts\run_bicycle_e4_winner.ps1
#
# Smoke (1k iters, LPIPS starts at 700):
#   $env:E4_SMOKE = "1"
#   .\scripts\run_bicycle_e4_winner.ps1
#
# Output: output/bicycle_e4_lpips_ms_w005/

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume = if ($env:E4_AUTO_RESUME) { $env:E4_AUTO_RESUME -ne "0" } else { $true }
$python = if ($env:E4_PYTHON) { $env:E4_PYTHON } else { "python" }
$smoke = if ($env:E4_SMOKE) { $env:E4_SMOKE -ne "0" } else { $false }
$iterations = if ($smoke) { 1000 } else { 30000 }
$seedList = if ($env:E4_SEEDS) {
    $env:E4_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    @(0)
}
$source = if ($env:BICYCLE_SOURCE) { $env:BICYCLE_SOURCE } else { Join-Path $ROOT "data\bicycle" }
$images = if ($env:BICYCLE_IMAGES) { $env:BICYCLE_IMAGES } else { "images_4" }
$depths = if ($env:BICYCLE_DEPTHS) { $env:BICYCLE_DEPTHS } else { Join-Path $source "depth_any_out" }
# 1 = images_4 native (1237x822). Matches DN --downscale-factor 1.
# 800 would scale the long edge down to about 800x531.
$resolution = if ($env:BICYCLE_RESOLUTION) { $env:BICYCLE_RESOLUTION } else { "1" }

if (-not (Test-Path (Join-Path $source "sparse\0\cameras.bin"))) {
    throw "COLMAP not found under $source\sparse\0"
}
if (-not (Test-Path (Join-Path $source $images))) {
    throw "Image folder not found: $(Join-Path $source $images)"
}
if (-not (Test-Path $depths)) {
    throw "Depth folder not found: $depths"
}
if (-not (Test-Path (Join-Path $source "sparse\0\depth_params.json"))) {
    throw "depth_params.json not found under $source\sparse\0"
}

$checkpointIterations = if ($smoke) { @("1000") } else {
    @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "30000")
}
$evalIterations = if ($smoke) { @("1000") } else {
    @("7000", "10000", "15100", "18100", "19500", "21100", "24000", "30000")
}
$liveStageIterations = ($evalIterations -join ",")
$lpipsStartIter = if ($smoke) { 700 } else { 21000 }

function Get-LatestCheckpoint {
    param([string]$ModelDir)
    if (-not (Test-Path $ModelDir)) { return $null }
    $bestPath = $null
    $bestIter = -1
    Get-ChildItem -Path $ModelDir -Filter "chkpnt*.pth" -File -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.BaseName -match '^chkpnt(\d+)$') {
            $iter = [int]$Matches[1]
            if ($iter -gt $bestIter) {
                $bestIter = $iter
                $bestPath = $_.FullName
            }
        }
    }
    if ($null -eq $bestPath) { return $null }
    return @{ Path = $bestPath; Iteration = $bestIter }
}

Write-Host "[bicycle E4] smoke=$smoke iterations=$iterations seeds=$($seedList -join ',') images=$images resolution=$resolution"

foreach ($seed in $seedList) {
    $suffix = if ($seed -eq 0) { "" } else { "_seed$seed" }
    $caseName = "bicycle_e4_lpips_ms_w005$suffix"
    $modelDir = Join-Path $ROOT "output\$caseName"
    $finalPly = Join-Path $modelDir "point_cloud\iteration_$iterations\point_cloud.ply"

    if ($skipCompleted -and (Test-Path $finalPly)) {
        Write-Host "[bicycle E4] Skip completed $caseName"
        continue
    }

    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    $resumeCkpt = $null
    if ($autoResume) {
        $resumeCkpt = Get-LatestCheckpoint -ModelDir $modelDir
        if ($null -ne $resumeCkpt -and [int]$resumeCkpt.Iteration -ge $iterations -and (Test-Path $finalPly)) {
            Write-Host "[bicycle E4] Skip completed $caseName (chkpnt$($resumeCkpt.Iteration) + final ply)"
            continue
        }
    }

    $trainArgs = @(
        "train_optimized.py",
        "-s", $source,
        "-m", $modelDir,
        "--images", $images,
        "--depths", $depths,
        "--resolution", $resolution,
        "--data_device", "cpu",
        "--eval",
        "--llffhold", "8",
        "--seed", "$seed",
        "--iterations", "$iterations",
        "--position_lr_init", "0.00016",
        "--position_lr_final", "0.0000016",
        "--position_lr_delay_mult", "0.01",
        "--position_lr_max_steps", "$iterations",
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
        "--lpips_loss_start_iter", "$lpipsStartIter",
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
        "--live_stage_iterations", $liveStageIterations,
        "--test_iterations"
    ) + $evalIterations + @(
        "--save_iterations", "$iterations",
        "--checkpoint_iterations"
    ) + $checkpointIterations + @(
        "--disable_viewer"
    )

    if ($null -ne $resumeCkpt) {
        $trainArgs += @("--start_checkpoint", "$($resumeCkpt.Path)")
        Write-Host "[bicycle E4] Resume $caseName from chkpnt$($resumeCkpt.Iteration)"
    } else {
        Write-Host "[bicycle E4] Starting $caseName from scratch"
    }

    $meta = @{
        profile = "lpips_ms_w005"
        scene = "bicycle"
        seed = $seed
        images = $images
        resolution = $resolution
        llffhold = 8
        smoke = $smoke
        iterations = $iterations
        lpips_start_iter = $lpipsStartIter
        model_dir = $modelDir
        note = "Frozen E4 winner. No RAS, SPA, or late_stabilize."
    }
    ($meta | ConvertTo-Json -Depth 4) | Set-Content -Path (Join-Path $modelDir "bicycle_e4_profile.json") -Encoding UTF8

    $env:TAGGS_CPU_KNN_INIT = "1"
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    $logPath = Join-Path $modelDir "bicycle_e4_train_console.log"
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($null -ne $resumeCkpt -and (Test-Path $logPath)) {
            "`n==== RESUME $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') from chkpnt$($resumeCkpt.Iteration) ====`n" |
                Add-Content -Path $logPath -Encoding UTF8
            & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath -Append
        } else {
            & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath
        }
        $trainExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    if ($trainExitCode -ne 0) {
        throw "bicycle E4 $caseName failed with exit code $trainExitCode"
    }
}

Write-Host "[bicycle E4] Done."
