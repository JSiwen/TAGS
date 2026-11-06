# =============================================================================
# §4.4 几何先验介入时机与约束强度 (6 独立组 × 4 场景 × 5 seed = 120 runs)
# =============================================================================
# 对应论文 §4.4.2 + §4.4.3
#
# §4.4.T0 = §4.4.S0 (No-Geo, 不重复跑)
# §4.4.T3 = §4.4.S2 (Base-Geo α=1.0, 26000-30000, 不重复跑)
# 实际独立: T0, T1, T2, T3, S1, S3 = 6 组
#
# 用法:
#   .\scripts\run_exp4_4_geometry.ps1
#   $env:E4_SCENES = "bicycle,garden,bonsai,room"
#   $env:E4_SEEDS = "0,1,2,3,4"
#   .\scripts\run_exp4_4_geometry.ps1
#
#   # 只跑 T1 + T3 (快速验证)
#   $env:E4_GROUPS = "T1,T3"
#   .\scripts\run_exp4_4_geometry.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume   = if ($env:E4_AUTO_RESUME)   { $env:E4_AUTO_RESUME   -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# §4.4 6 个独立组
$ALL_GROUPS = @(
    # T0/S0: No-Geo 基线
    @{ id="T0_NoGeo";     geo="False";       geo_start="0";     warmup_until="0";     decay_start="0";     decay_until="0";     alpha_init="0.0"; alpha_final="0.0" },
    # T1: Early 1000-5000
    @{ id="T1_Early";     geo="True";        geo_start="1000";  warmup_until="5000";  decay_start="5000";  decay_until="5000";  alpha_init="1.0"; alpha_final="1.0" },
    # T2: Mid 13000-17000 (与 SPA 重叠期)
    @{ id="T2_Mid";       geo="True";        geo_start="13000"; warmup_until="17000"; decay_start="17000"; decay_until="17000"; alpha_init="1.0"; alpha_final="1.0" },
    # T3/S2: Late 26000-30000 (Base-Geo α=1.0)
    @{ id="T3_S2_LateBase"; geo="True";      geo_start="26000"; warmup_until="30000"; decay_start="26000"; decay_until="30000"; alpha_init="1.0"; alpha_final="1.0" },
    # S1: Weak-Geo α=0.5, 26000-30000
    @{ id="S1_WeakGeo";   geo="True";        geo_start="26000"; warmup_until="30000"; decay_start="26000"; decay_until="30000"; alpha_init="0.5"; alpha_final="0.5" },
    # S3: Strong-Geo α=1.5, 26000-30000
    @{ id="S3_StrongGeo"; geo="True";        geo_start="26000"; warmup_until="30000"; decay_start="26000"; decay_until="30000"; alpha_init="1.5"; alpha_final="1.5" },
)

$selectedGroups = if ($env:E4_GROUPS) {
    $env:E4_GROUPS.Split(",") | ForEach-Object { $_.Trim() }
} else {
    $ALL_GROUPS | ForEach-Object { $_.id }
}

# 4 场景全跑
$ALL_SCENES = @("bicycle", "garden", "bonsai", "room")
$selectedScenes = if ($env:E4_SCENES) {
    $env:E4_SCENES.Split(",") | ForEach-Object { $_.Trim() }
} else {
    $ALL_SCENES
}

$seedList = if ($env:E4_SEEDS) {
    $env:E4_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    @(0)
}

# §4.2 选出的 prune_ratio2 (默认 0.04)
$DEFAULT_PRUNE_RATIO = if ($env:E4_PRUNE_RATIO) { $env:E4_PRUNE_RATIO } else { "0.04" }
# §4.3 选出的 LPIPS 调度 (默认 staged)
$DEFAULT_LPIPS_SCHEDULE = if ($env:E4_LPIPS_SCHEDULE) { $env:E4_LPIPS_SCHEDULE } else { "staged" }

# v3: 主实验统一采用 1/2 分辨率
$DEFAULT_IMAGES_DIR = if ($env:E4_IMAGES_DIR) { $env:E4_IMAGES_DIR } else { "images_2" }

# ---- 辅助函数 ----
function Get-LatestCheckpoint {
    param([string]$ModelDir)
    if (-not (Test-Path $ModelDir)) { return $null }
    $bestPath = $null; $bestIter = -1
    Get-ChildItem -Path $ModelDir -Filter "chkpnt*.pth" -File -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.BaseName -match '^chkpnt(\d+)$') {
            $iter = [int]$Matches[1]
            if ($iter -gt $bestIter) { $bestIter = $iter; $bestPath = $_.FullName }
        }
    }
    if ($null -eq $bestPath) { return $null }
    return @{ Path = $bestPath; Iteration = $bestIter; SizeMB = [math]::Round((Get-Item $bestPath).Length / 1MB, 1) }
}

# §4.4 固定超参 (复用 §4.2/§4.3 选出的值)
$FIXED_ARGS = @(
    "--data_device", "cpu",
    "--eval",
    "--llffhold", "8",
    "--images", $DEFAULT_IMAGES_DIR,
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
    # §4.3 选出的 LPIPS (staged 或 step)
    "--enable_multiscale_lpips",
    "--lpips_loss_weight_init", "0.004",
    "--lpips_loss_weight_final", "0.004",
    "--lpips_loss_start_iter", "0",
    "--lpips_loss_schedule", $DEFAULT_LPIPS_SCHEDULE,
    "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0",
    # §4.2 选出的 SPA
    "--enable_sparse_optimization",
    "--sparse_mode", "spa",
    "--prune_ratio2", $DEFAULT_PRUNE_RATIO,
    "--spa_start_iter", "13000",
    "--spa_end_iter", "25000",
    "--spa_interval", "1000",
    "--rho_lr", "0.0005",
    "--spa_metric", "hybrid",
    "--spa_require_full_resolution", "True",
    "--spa_after_fullres_delay", "3000",
    "--spa_max_prune_points", "30000",
    "--spa_protect_top_fraction", "0.05",
    # 学习率
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
    "--enable_adaptive_stage_controller", "False",
)

$checkpointIterations = if ($smoke) { @("1000") } else {
    @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "30000")
}

# ---- 主循环 ----
$totalRuns = $selectedScenes.Count * $selectedGroups.Count * $seedList.Count
$runIdx = 0

foreach ($scene in $selectedScenes) {
    $source = Join-Path $ROOT "data\$scene"
    $depths = Join-Path $source "depth_any_out"

    if (-not (Test-Path (Join-Path $source "sparse\0\cameras.bin"))) {
        Write-Warning "[exp4_4] COLMAP not found. Skipping scene=$scene"
        continue
    }
    if (-not (Test-Path (Join-Path $source $DEFAULT_IMAGES_DIR))) {
        Write-Warning "[exp4_4] $DEFAULT_IMAGES_DIR not found. Skipping scene=$scene"
        continue
    }
    if (-not (Test-Path $depths)) {
        Write-Warning "[exp4_4] depths not found. Skipping scene=$scene"
        continue
    }

    foreach ($groupId in $selectedGroups) {
        $groupDef = $ALL_GROUPS | Where-Object { $_.id -eq $groupId } | Select-Object -First 1
        if ($null -eq $groupDef) {
            Write-Warning "[exp4_4] Unknown group: $groupId. Skip."
            continue
        }

        $groupDir = Join-Path $ROOT "output\exp4_$scene\4_4_$($groupDef.id)"

        foreach ($seed in $seedList) {
            $runIdx++
            $trainDir = Join-Path $groupDir "train\seed$seed"
            $finalPly = Join-Path $trainDir "point_cloud\iteration_$iterations\point_cloud.ply"

            if ($skipCompleted -and (Test-Path $finalPly)) {
                Write-Host "[exp4_4] [$runIdx/$totalRuns] Skip $scene/$($groupDef.id)/seed$seed (done)"
                continue
            }

            $resumeCkpt = $null
            if ($autoResume) {
                $resumeCkpt = Get-LatestCheckpoint -ModelDir $trainDir
                if ($null -ne $resumeCkpt -and (Test-Path $finalPly)) {
                    Write-Host "[exp4_4] [$runIdx/$totalRuns] Skip $scene/$($groupDef.id)/seed$seed (done+chkpnt)"
                    continue
                }
            }

            New-Item -ItemType Directory -Force -Path $trainDir | Out-Null

            # ---- 构建 train args ----
            $trainArgs = @(
                $python, "train_optimized.py",
                "-s", $source,
                "-m", $trainDir,
                "--images", $DEFAULT_IMAGES_DIR,
                "--depths", $depths,
                "--resolution", "1",
                "--seed", "$seed",
                "--iterations", "$iterations",
            ) + $FIXED_ARGS

            # ---- §4.4 核心: Geo 开关 + 几何参数 ----
            $trainArgs += @(
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
                "--depth_prior_min_mask_fraction", "0.20",
                "--normal_tv_depth_edge_quantile", "0.70",
                "--normal_tv_rgb_edge_quantile", "0.70",
                "--geo_prior_start_iter", $groupDef.geo_start,
                "--geo_prior_warmup_until", $groupDef.warmup_until,
                "--geo_prior_decay_start", $groupDef.decay_start,
                "--geo_prior_decay_until", $groupDef.decay_until,
                "--depth_prior_weight_init", $groupDef.alpha_init,
                "--depth_prior_weight_final", $groupDef.alpha_final,
                "--depth_prior_weight_min", "0.01",
                "--normal_prior_weight_init", $groupDef.alpha_init,
                "--normal_prior_weight_final", $groupDef.alpha_final,
                "--normal_prior_weight_min", "0.0",
                "--geo_prior_fullres_relax_max", "0.35",
            )

            if ($groupDef.geo -eq "False") {
                $trainArgs += @("--enable_geo_prior", "False")
            } else {
                $trainArgs += "--enable_geo_prior"
            }

            if ($null -ne $resumeCkpt) {
                $trainArgs += @("--start_checkpoint", $resumeCkpt.Path)
                Write-Host "[exp4_4] [$runIdx/$totalRuns] Resume $scene/$($groupDef.id)/seed$seed from chkpnt$($resumeCkpt.Iteration)"
            } else {
                Write-Host "[exp4_4] [$runIdx/$totalRuns] Start $scene/$($groupDef.id)/seed$seed from scratch"
            }

            $trainArgs += @("--checkpoint_iterations") + $checkpointIterations

            $logPath = Join-Path $trainDir "exp4_4_console.log"
            $env:TAGGS_CPU_KNN_INIT = "1"
            $env:KMP_DUPLICATE_LIB_OK = "TRUE"
            $env:PYTHONIOENCODING = "utf-8"
            $env:PYTHONUTF8 = "1"

            $savedEA = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                if ($null -ne $resumeCkpt -and (Test-Path $logPath)) {
                    "`n==== RESUME $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') from chkpnt$($resumeCkpt.Iteration) ====`n" |
                        Add-Content -Path $logPath -Encoding UTF8
                    & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath -Append
                } else {
                    & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath
                }
            } finally {
                $ErrorActionPreference = $savedEA
            }

            if ($LASTEXITCODE -ne 0) {
                Write-Warning "[exp4_4] Train failed: $scene/$($groupDef.id)/seed$seed (exit=$LASTEXITCODE)"
                continue
            }

            $renderCmd = @($python, "render.py", "-s", $source, "-m", $trainDir,
                           "--iteration", "$iterations", "--skip_train")
            Write-Host "[exp4_4] Render $scene/$($groupDef.id)/seed$seed"
            & $renderCmd 2>&1 | Out-Null

            $metricsCmd = @($python, "metrics.py", "-m", $trainDir)
            Write-Host "[exp4_4] Metrics $scene/$($groupDef.id)/seed$seed"
            & $metricsCmd 2>&1 | Out-Null

            Write-Host "[exp4_4] [$runIdx/$totalRuns] Done $scene/$($groupDef.id)/seed$seed"
        }
    }
}

Write-Host "[exp4_4] All $totalRuns runs complete."
