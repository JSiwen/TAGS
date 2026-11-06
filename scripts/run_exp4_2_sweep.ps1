# =============================================================================
# §4.2 高斯稀疏化强度选择 (5 组 × 3 场景 × 5 seed = 75 runs)
# =============================================================================
# 对应论文 §4.2.3, 实验组 A/B/C/D/E
#
# 用法:
#   # 跑 5 组 × 5 seed (默认 bicycle)
#   .\scripts\run_exp4_2_sweep.ps1
#
#   # 只跑 1 组
#   $env:E4_GROUPS = "0.04"
#   $env:E4_SEEDS = "0,1,2,3,4"
#   .\scripts\run_exp4_2_sweep.ps1
#
#   # 跑 garden 场景
#   $env:E4_SCENE = "garden"
#   .\scripts\run_exp4_2_sweep.ps1
#
#   # Smoke test (1k iter)
#   $env:E4_SMOKE = "1"
#   .\scripts\run_exp4_2_sweep.ps1
#
# 输出: output/exp4_{scene}/4_2_{group_id}/train/seed{N}/

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume   = if ($env:E4_AUTO_RESUME)   { $env:E4_AUTO_RESUME   -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# §4.2 5 组 prune_ratio2 值 (group ID)
# 0 = Baseline (SPA off)
# 0.02/0.04/0.06/0.08 = SPA sparse strengths
$ALL_GROUPS = @(
    @{ id="0.00_A_Baseline";   prune_ratio2="0.00"; spa_mode="none"    },
    @{ id="0.02_B_Weak";      prune_ratio2="0.02"; spa_mode="spa"     },
    @{ id="0.04_C_Reference"; prune_ratio2="0.04"; spa_mode="spa"     },
    @{ id="0.06_D_MedStr";    prune_ratio2="0.06"; spa_mode="spa"     },
    @{ id="0.08_E_Strong";    prune_ratio2="0.08"; spa_mode="spa"     },
)

$selectedGroups = if ($env:E4_GROUPS) {
    $env:E4_GROUPS.Split(",") | ForEach-Object { $_.Trim() }
} else {
    $ALL_GROUPS | ForEach-Object { $_.id }
}

# 3 场景: bicycle (主) + garden + bonsai
$ALL_SCENES = @("bicycle", "garden", "bonsai")
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

# v3: 主实验统一采用 1/2 分辨率 (--images images_2)
# 若 1/2 超出显存/时间预算, 改用 1/4 (--images images_4)
# 标准分辨率验证另见 run_exp4_std_resolution.ps1
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

# §4.2 固定超参 (不随 prune_ratio2 变化)
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
    # §4.2: LPIPS 关, Geo 关
    "--lpips_loss_weight_init", "0.0",
    "--enable_geo_prior", "False",
    # 学习率 (与 E4 winner 一致)
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
    # SPA 固定时段 (13000-25000)
    "--spa_start_iter", "13000",
    "--spa_end_iter", "25000",
    "--spa_interval", "1000",
    "--rho_lr", "0.0005",
    "--spa_metric", "hybrid",
    "--spa_require_full_resolution", "True",
    "--spa_after_fullres_delay", "3000",
    "--spa_max_prune_points", "30000",
    "--spa_protect_top_fraction", "0.05",
    # 关闭 adaptive stage controller (保证确定性)
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

    # 场景路径检查
    if (-not (Test-Path (Join-Path $source "sparse\0\cameras.bin"))) {
        Write-Warning "[exp4_2] COLMAP not found: $source\sparse\0. Skipping scene=$scene"
        continue
    }
    if (-not (Test-Path (Join-Path $source $DEFAULT_IMAGES_DIR))) {
        Write-Warning "[exp4_2] $DEFAULT_IMAGES_DIR not found. Skipping scene=$scene"
        continue
    }
    if (-not (Test-Path $depths)) {
        Write-Warning "[exp4_2] depths not found: $depths. Skipping scene=$scene"
        continue
    }

    foreach ($groupId in $selectedGroups) {
        $groupDef = $ALL_GROUPS | Where-Object { $_.id -eq $groupId } | Select-Object -First 1
        if ($null -eq $groupDef) {
            Write-Warning "[exp4_2] Unknown group: $groupId. Skip."
            continue
        }

        $groupDir = Join-Path $ROOT "output\exp4_$scene\4_2_$($groupDef.id)"

        foreach ($seed in $seedList) {
            $runIdx++
            $trainDir = Join-Path $groupDir "train\seed$seed"
            $finalPly = Join-Path $trainDir "point_cloud\iteration_$iterations\point_cloud.ply"

            # --- 跳过已完成 ---
            if ($skipCompleted -and (Test-Path $finalPly)) {
                Write-Host "[exp4_2] [$runIdx/$totalRuns] Skip $scene/$($groupDef.id)/seed$seed (done)"
                continue
            }

            # --- 找续跑 chkpnt ---
            $resumeCkpt = $null
            if ($autoResume) {
                $resumeCkpt = Get-LatestCheckpoint -ModelDir $trainDir
                if ($null -ne $resumeCkpt -and (Test-Path $finalPly)) {
                    Write-Host "[exp4_2] [$runIdx/$totalRuns] Skip $scene/$($groupDef.id)/seed$seed (done+chkpnt)"
                    continue
                }
            }

            New-Item -ItemType Directory -Force -Path $trainDir | Out-Null

            # --- 构建 train args ---
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

            # §4.2 核心参数: SPA on/off + prune_ratio2
            if ($groupDef.spa_mode -eq "spa") {
                $trainArgs += @("--enable_sparse_optimization", "--sparse_mode", "spa",
                                "--prune_ratio2", $groupDef.prune_ratio2)
            }
            # SPA off 时 sparse_mode 为 none (默认), 不需要显式设

            if ($null -ne $resumeCkpt) {
                $trainArgs += @("--start_checkpoint", $resumeCkpt.Path)
                Write-Host "[exp4_2] [$runIdx/$totalRuns] Resume $scene/$($groupDef.id)/seed$seed from chkpnt$($resumeCkpt.Iteration) ($($resumeCkpt.SizeMB) MB)"
            } else {
                Write-Host "[exp4_2] [$runIdx/$totalRuns] Start $scene/$($groupDef.id)/seed$seed from scratch"
            }

            # --- checkpoint 计划 ---
            $trainArgs += @("--checkpoint_iterations") + $checkpointIterations

            # --- log ---
            $logPath = Join-Path $trainDir "exp4_2_console.log"
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

            # --- 渲染 + 评分 (skip 模式下跳过) ---
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "[exp4_2] Train failed for $scene/$($groupDef.id)/seed$seed (exit=$LASTEXITCODE)"
                continue
            }

            $renderCmd = @(
                $python, "render.py",
                "-s", $source,
                "-m", $trainDir,
                "--iteration", "$iterations",
                "--skip_train",
            )
            Write-Host "[exp4_2] Render $scene/$($groupDef.id)/seed$seed"
            & $renderCmd 2>&1 | Out-Null

            $metricsCmd = @($python, "metrics.py", "-m", $trainDir)
            Write-Host "[exp4_2] Metrics $scene/$($groupDef.id)/seed$seed"
            & $metricsCmd 2>&1 | Out-Null

            Write-Host "[exp4_2] [$runIdx/$totalRuns] Done $scene/$($groupDef.id)/seed$seed"
        }
    }
}

Write-Host "[exp4_2] All $totalRuns runs complete."
