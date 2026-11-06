# =============================================================================
# §4.5 多模块协同消融 (7 组 × 4 场景 × 5 seed = 140 runs)
# =============================================================================
# 对应论文 §4.5.2 表 4-X
# 2^k 析因设计: LPIPS × SPA × Geo × Resolution
#
# 用法:
#   .\scripts\run_exp4_5_ablation.ps1
#   $env:E4_SCENES = "bicycle,garden,bonsai,room"
#   $env:E4_SEEDS = "0,1,2,3,4"
#   .\scripts\run_exp4_5_ablation.ps1
#
#   # 只跑 1 组
#   $env:E4_GROUPS = "Full"
#   .\scripts\run_exp4_5_ablation.ps1
#
#   # Smoke
#   $env:E4_SMOKE = "1"
#   .\scripts\run_exp4_5_ablation.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume   = if ($env:E4_AUTO_RESUME)   { $env:E4_AUTO_RESUME   -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# §4.5 7 个实验组
$ALL_GROUPS = @(
    # A: Baseline (3DGS) - all off
    @{
        id="A_Baseline3DGS"; lpips=$false; spa=$false; geo=$false; resolution=$false;
        desc="原始 3DGS 基线, 无任何增强"
    },
    # B: w/o Perceptual - SPA on, Geo on, LPIPS off, Resolution on
    @{
        id="B_woPerceptual"; lpips=$false; spa=$true; geo=$true; resolution=$true;
        desc="验证感知损失贡献"
    },
    # C: w/o SPA - SPA off, Geo on, LPIPS on, Resolution on
    @{
        id="C_woSPA"; lpips=$true; spa=$false; geo=$true; resolution=$true;
        desc="验证稀疏化贡献"
    },
    # D: w/o Geo - SPA on, Geo off, LPIPS on, Resolution on
    @{
        id="D_woGeo"; lpips=$true; spa=$true; geo=$false; resolution=$true;
        desc="验证几何先验贡献"
    },
    # E: w/o Resolution - SPA on, Geo on, LPIPS on, Resolution off
    @{
        id="E_woResolution"; lpips=$true; spa=$true; geo=$true; resolution=$false;
        desc="验证多分辨率调度贡献"
    },
    # F: Full TAGS - all on
    @{
        id="F_FullTAGS"; lpips=$true; spa=$true; geo=$true; resolution=$true;
        desc="完整方法"
    },
    # G: Negative Control - SPA 反向 (25000-30000, 即几何窗口之后)
    @{
        id="G_NegCtrl"; lpips=$true; spa=$true; geo=$true; resolution=$true; spa_reverse=$true;
        desc="SPA 反向, 验证消融方向正确性"
    },
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

# §4.2 选出的 prune_ratio2
$DEFAULT_PRUNE_RATIO = if ($env:E4_PRUNE_RATIO) { $env:E4_PRUNE_RATIO } else { "0.04" }

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

# §4.5 基础超参 (复用 §4.2/§4.3 选出的配置)
function Get-BaseArgs {
    param([bool]$lpips, [bool]$spa, [bool]$geo, [bool]$resolution, [bool]$spa_reverse)
    $args = @(
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

    # --- LPIPS ---
    if ($lpips) {
        $args += @(
            "--enable_multiscale_lpips",
            "--lpips_loss_weight_init", "0.004",
            "--lpips_loss_weight_final", "0.004",
            "--lpips_loss_start_iter", "0",
            "--lpips_loss_schedule", "staged",
            "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0"
        )
    } else {
        $args += @("--lpips_loss_weight_init", "0.0")
    }

    # --- SPA ---
    if ($spa) {
        $args += @(
            "--enable_sparse_optimization",
            "--sparse_mode", "spa",
            "--prune_ratio2", $DEFAULT_PRUNE_RATIO,
            "--spa_interval", "1000",
            "--rho_lr", "0.0005",
            "--spa_metric", "hybrid",
            "--spa_require_full_resolution", "True",
            "--spa_after_fullres_delay", "3000",
            "--spa_max_prune_points", "30000",
            "--spa_protect_top_fraction", "0.05",
        )
        if ($spa_reverse) {
            # Negative Control: SPA 反向到 25000-30000
            $args += @("--spa_start_iter", "25000", "--spa_end_iter", "30000")
        } else {
            $args += @("--spa_start_iter", "13000", "--spa_end_iter", "25000")
        }
    } else {
        # SPA 关
        $args += @("--enable_sparse_optimization", "False", "--sparse_mode", "none")
    }

    # --- Geo ---
    if ($geo) {
        $args += @(
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
            "--depth_prior_min_mask_fraction", "0.20",
            "--normal_tv_depth_edge_quantile", "0.70",
            "--normal_tv_rgb_edge_quantile", "0.70",
            "--geo_prior_start_iter", "26000",
            "--geo_prior_warmup_until", "30000",
            "--geo_prior_decay_start", "26000",
            "--geo_prior_decay_until", "30000",
            "--depth_prior_weight_init", "0.04",
            "--depth_prior_weight_final", "0.04",
            "--depth_prior_weight_min", "0.01",
            "--normal_prior_weight_init", "0.02",
            "--normal_prior_weight_final", "0.02",
            "--normal_prior_weight_min", "0.0",
            "--geo_prior_fullres_relax_max", "0.35",
        )
    } else {
        $args += @("--enable_geo_prior", "False")
    }

    # --- Resolution ---
    if ($resolution) {
        $args += @(
            "--enable_resolution_aware_schedule",
            "--resolution_schedule", "1200:4,4000:2,$iterations`:1"
        )
    } else {
        $args += @("--enable_resolution_aware_schedule", "False")
    }

    return $args
}

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
        Write-Warning "[exp4_5] COLMAP not found. Skipping scene=$scene"
        continue
    }
    if (-not (Test-Path (Join-Path $source $DEFAULT_IMAGES_DIR))) {
        Write-Warning "[exp4_5] $DEFAULT_IMAGES_DIR not found. Skipping scene=$scene"
        continue
    }
    if (-not (Test-Path $depths)) {
        Write-Warning "[exp4_5] depths not found. Skipping scene=$scene"
        continue
    }

    foreach ($groupId in $selectedGroups) {
        $groupDef = $ALL_GROUPS | Where-Object { $_.id -eq $groupId } | Select-Object -First 1
        if ($null -eq $groupDef) {
            Write-Warning "[exp4_5] Unknown group: $groupId. Skip."
            continue
        }

        $groupDir = Join-Path $ROOT "output\exp4_$scene\4_5_$($groupDef.id)"

        foreach ($seed in $seedList) {
            $runIdx++
            $trainDir = Join-Path $groupDir "train\seed$seed"
            $finalPly = Join-Path $trainDir "point_cloud\iteration_$iterations\point_cloud.ply"

            if ($skipCompleted -and (Test-Path $finalPly)) {
                Write-Host "[exp4_5] [$runIdx/$totalRuns] Skip $scene/$($groupDef.id)/seed$seed (done)"
                continue
            }

            $resumeCkpt = $null
            if ($autoResume) {
                $resumeCkpt = Get-LatestCheckpoint -ModelDir $trainDir
                if ($null -ne $resumeCkpt -and (Test-Path $finalPly)) {
                    Write-Host "[exp4_5] [$runIdx/$totalRuns] Skip $scene/$($groupDef.id)/seed$seed (done+chkpnt)"
                    continue
                }
            }

            New-Item -ItemType Directory -Force -Path $trainDir | Out-Null

            # 获取该组的 base args
            $groupArgs = Get-BaseArgs `
                -lpips $groupDef.lpips `
                -spa $groupDef.spa `
                -geo $groupDef.geo `
                -resolution $groupDef.resolution `
                -spa_reverse ($groupDef.ContainsKey("spa_reverse") -and $groupDef.spa_reverse)

            $trainArgs = @(
                $python, "train_optimized.py",
                "-s", $source,
                "-m", $trainDir,
                "--images", $DEFAULT_IMAGES_DIR,
                "--depths", $depths,
                "--resolution", "1",
                "--seed", "$seed",
                "--iterations", "$iterations",
            ) + $groupArgs

            if ($null -ne $resumeCkpt) {
                $trainArgs += @("--start_checkpoint", $resumeCkpt.Path)
                Write-Host "[exp4_5] [$runIdx/$totalRuns] Resume $scene/$($groupDef.id)/seed$seed from chkpnt$($resumeCkpt.Iteration)"
            } else {
                Write-Host "[exp4_5] [$runIdx/$totalRuns] Start $scene/$($groupDef.id)/seed$seed from scratch"
            }

            $trainArgs += @("--checkpoint_iterations") + $checkpointIterations

            $logPath = Join-Path $trainDir "exp4_5_console.log"
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
                Write-Warning "[exp4_5] Train failed: $scene/$($groupDef.id)/seed$seed (exit=$LASTEXITCODE)"
                continue
            }

            $renderCmd = @($python, "render.py", "-s", $source, "-m", $trainDir,
                           "--iteration", "$iterations", "--skip_train")
            Write-Host "[exp4_5] Render $scene/$($groupDef.id)/seed$seed"
            & $renderCmd 2>&1 | Out-Null

            $metricsCmd = @($python, "metrics.py", "-m", $trainDir)
            Write-Host "[exp4_5] Metrics $scene/$($groupDef.id)/seed$seed"
            & $metricsCmd 2>&1 | Out-Null

            Write-Host "[exp4_5] [$runIdx/$totalRuns] Done $scene/$($groupDef.id)/seed$seed ($($groupDef.desc))"
        }
    }
}

Write-Host "[exp4_5] All $totalRuns runs complete."
