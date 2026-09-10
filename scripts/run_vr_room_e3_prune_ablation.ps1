# E3: vr_room pruning-strategy ablation.
#
# Phase A — single-seed screening (default seed=0):
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# Phase B — top-2 configs x 3 seeds:
#   $env:E3_PHASE = "multiseed"
#   $env:E3_TOP_CONFIGS = "opacity_0.01,densify_0.50"
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# Selected profiles only:
#   $env:E3_PROFILES = "opacity_0.005,opacity_0.01,densify_0.50"
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# After training, FPS + gate summary:
#   python scripts\summarize_vr_room_e3.py --measure-fps 1
#
# Resume / checkpoints:
#   - If point_cloud/iteration_30000 exists, the case is skipped (unless E3_SKIP_COMPLETED=0).
#   - Otherwise the newest chkpnt*.pth under the case dir is passed as --start_checkpoint.
#   - Default checkpoint cadence is every 3k iters (override with E3_CHECKPOINT_ITERS).
#   - Disable auto-resume: $env:E3_AUTO_RESUME = "0"

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$phase = if ($env:E3_PHASE) { $env:E3_PHASE.Trim().ToLower() } else { "screen" }
$skipCompleted = if ($env:E3_SKIP_COMPLETED) { $env:E3_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume = if ($env:E3_AUTO_RESUME) { $env:E3_AUTO_RESUME -ne "0" } else { $true }
$python = if ($env:E3_PYTHON) { $env:E3_PYTHON } else { "python" }
$seedList = if ($env:E3_SEEDS) {
    $env:E3_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    if ($phase -eq "multiseed") { @(0, 1, 2) } else { @(0) }
}
$checkpointIterations = if ($env:E3_CHECKPOINT_ITERS) {
    $env:E3_CHECKPOINT_ITERS.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
    # Dense cadence for safer resume (every 3k through 30k).
    @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "30000")
}

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

function Get-BaseProfile {
    return @{
        Opacity = "0.005"
        MinVisibility = "0"
        MaxScreenSize = "-1"
        GradThreshold = "0.0"
        GradMinDenom = "1.0"
        DensifyUntilFraction = "0.6"
        PruneOnlyFromFraction = "0.0"
        PruneUntilFraction = "1.0"
        MaxGaussians = "0"
    }
}

function Get-AllProfiles {
    $profiles = [ordered]@{}

    foreach ($opacity in @("0.005", "0.01", "0.02")) {
        $p = Get-BaseProfile
        $p.Opacity = $opacity
        $profiles["opacity_$opacity"] = $p
    }

    foreach ($vis in @("1", "3", "5")) {
        $p = Get-BaseProfile
        $p.MinVisibility = $vis
        $profiles["minvis_$vis"] = $p
    }

    foreach ($screen in @("0", "10", "20", "40")) {
        $p = Get-BaseProfile
        $p.MaxScreenSize = $screen
        $profiles["screen_$screen"] = $p
    }

    foreach ($grad in @("1e-7", "5e-7", "1e-6")) {
        $p = Get-BaseProfile
        $p.GradThreshold = $grad
        $profiles["grad_$grad"] = $p
    }

    foreach ($frac in @("0.50", "0.65", "0.80", "0.90")) {
        $p = Get-BaseProfile
        $p.DensifyUntilFraction = $frac
        $p.PruneOnlyFromFraction = $frac
        $p.PruneUntilFraction = "1.0"
        $profiles["densify_$frac"] = $p
    }

    # Promising combo from E2: early densify stop + mild opacity raise, optional 2M safety cap.
    $comboA = Get-BaseProfile
    $comboA.Opacity = "0.01"
    $comboA.DensifyUntilFraction = "0.50"
    $comboA.PruneOnlyFromFraction = "0.50"
    $comboA.PruneUntilFraction = "1.0"
    $profiles["combo_opacity0.01_densify0.50"] = $comboA

    $comboB = Get-BaseProfile
    $comboB.Opacity = "0.01"
    $comboB.DensifyUntilFraction = "0.65"
    $comboB.PruneOnlyFromFraction = "0.65"
    $comboB.PruneUntilFraction = "1.0"
    $comboB.MaxGaussians = "2000000"
    $profiles["combo_opacity0.01_densify0.65_cap2m"] = $comboB

    return $profiles
}

function Resolve-SelectedProfiles {
    param([System.Collections.Specialized.OrderedDictionary]$All)

    if ($phase -eq "multiseed") {
        if (-not $env:E3_TOP_CONFIGS) {
            throw "E3_PHASE=multiseed requires E3_TOP_CONFIGS=cfg1,cfg2"
        }
        $names = $env:E3_TOP_CONFIGS.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    } elseif ($env:E3_PROFILES) {
        $names = $env:E3_PROFILES.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    } else {
        $names = @($All.Keys)
    }

    $selected = [ordered]@{}
    foreach ($name in $names) {
        if (-not $All.Contains($name)) {
            throw "Unknown E3 profile '$name'. Known: $($All.Keys -join ', ')"
        }
        $selected[$name] = $All[$name]
    }
    return $selected
}

$allProfiles = Get-AllProfiles
$profiles = Resolve-SelectedProfiles -All $allProfiles

Write-Host "[E3] phase=$phase profiles=$($profiles.Keys -join ', ') seeds=$($seedList -join ',')"

foreach ($name in $profiles.Keys) {
    $profile = $profiles[$name]
    foreach ($seed in $seedList) {
        $caseName = if ($phase -eq "multiseed") { "vr_room_e3_${name}_seed$seed" } else { "vr_room_e3_${name}" }
        if ($phase -ne "multiseed" -and $seed -ne 0) {
            $caseName = "vr_room_e3_${name}_seed$seed"
        }
        $modelDir = Join-Path $ROOT "output\$caseName"
        $finalPly = Join-Path $modelDir "point_cloud\iteration_30000\point_cloud.ply"
        if ($skipCompleted -and (Test-Path $finalPly)) {
            Write-Host "[E3] Skip completed $caseName"
            continue
        }

        New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
        $resumeCkpt = $null
        if ($autoResume) {
            $resumeCkpt = Get-LatestCheckpoint -ModelDir $modelDir
            if ($null -ne $resumeCkpt -and [int]$resumeCkpt.Iteration -ge 30000 -and (Test-Path $finalPly)) {
                Write-Host "[E3] Skip completed $caseName (chkpnt$($resumeCkpt.Iteration) + final ply)"
                continue
            }
            if ($null -ne $resumeCkpt -and [int]$resumeCkpt.Iteration -ge 30000 -and -not (Test-Path $finalPly)) {
                Write-Host "[E3] Found chkpnt$($resumeCkpt.Iteration) but missing final ply; will resume and re-save."
            }
        }

        $trainArgs = @(
            "train_optimized.py",
            "-s", (Join-Path $ROOT "data\vr_room"),
            "-m", $modelDir,
            "--images", "images",
            "--depths", (Join-Path $ROOT "data\vr_room\depth"),
            "--resolution", "1",
            "--data_device", "cpu",
            "--eval",
            "--llffhold", "0",
            "--seed", "$seed",
            "--iterations", "30000",
            "--position_lr_init", "0.00016",
            "--position_lr_final", "0.0000016",
            "--position_lr_delay_mult", "0.01",
            "--position_lr_max_steps", "30000",
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
            "--densify_until_fraction", "$($profile.DensifyUntilFraction)",
            "--densify_grad_threshold", "0.00016",
            "--max_gaussians", "$($profile.MaxGaussians)",
            "--prune_opacity_threshold", "$($profile.Opacity)",
            "--prune_min_visibility", "$($profile.MinVisibility)",
            "--prune_max_screen_size", "$($profile.MaxScreenSize)",
            "--prune_grad_threshold", "$($profile.GradThreshold)",
            "--prune_grad_min_denom", "$($profile.GradMinDenom)",
            "--prune_only_from_fraction", "$($profile.PruneOnlyFromFraction)",
            "--prune_until_fraction", "$($profile.PruneUntilFraction)",
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
            "--live_stage_iterations", "7000,10000,15000,18000,24000,30000",
            "--test_iterations", "7000", "15000", "24000", "30000",
            "--save_iterations", "30000",
            "--checkpoint_iterations"
        ) + $checkpointIterations + @(
            "--disable_viewer"
        )

        if ($null -ne $resumeCkpt) {
            $trainArgs += @("--start_checkpoint", "$($resumeCkpt.Path)")
            Write-Host "[E3] Resume $caseName from chkpnt$($resumeCkpt.Iteration) ($($resumeCkpt.Path))"
        } else {
            Write-Host "[E3] Starting $caseName from scratch"
        }

        $env:TAGGS_CPU_KNN_INIT = "1"
        $env:KMP_DUPLICATE_LIB_OK = "TRUE"
        $logPath = Join-Path $modelDir "e3_train_console.log"
        $meta = @{
            profile = $name
            seed = $seed
            settings = $profile
            model_dir = $modelDir
            auto_resume = $autoResume
            resume_checkpoint = if ($null -ne $resumeCkpt) { $resumeCkpt.Path } else { $null }
            resume_iteration = if ($null -ne $resumeCkpt) { $resumeCkpt.Iteration } else { $null }
            checkpoint_iterations = $checkpointIterations
        }
        ($meta | ConvertTo-Json -Depth 5) | Set-Content -Path (Join-Path $modelDir "e3_profile.json") -Encoding UTF8

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
            throw "E3 $caseName failed with exit code $trainExitCode"
        }
    }
}

Write-Host "[E3] Requested runs complete. Next: python scripts\summarize_vr_room_e3.py --measure-fps 1"
