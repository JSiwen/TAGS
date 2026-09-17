# E4: vr_room perceptual-loss ablation on the fixed E3 2.5M-cap schedule.
#
# Phase A — seed=0 screening (8 variants):
#   .\scripts\run_vr_room_e4_perceptual_ablation.ps1
#
# Smoke test (1k iters, LPIPS sanity):
#   $env:E4_SMOKE = "1"
#   $env:E4_PROFILES = "lpips_w0.01"
#   .\scripts\run_vr_room_e4_perceptual_ablation.ps1
#
# Phase B — multiseed finalists:
#   $env:E4_PHASE = "multiseed"
#   $env:E4_TOP_CONFIGS = "rgb_l1,rgb_ssim,lpips_w0.03,lpips_ms_w0.03"
#   $env:E4_SEEDS = "0,1,2"
#   .\scripts\run_vr_room_e4_perceptual_ablation.ps1
#
# After training:
#   python scripts\summarize_vr_room_e4.py --measure-fps 1
#
# Outputs: output/vr_room_e4_<profile>[_seedN]/

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$phase = if ($env:E4_PHASE) { $env:E4_PHASE.Trim().ToLower() } else { "screen" }
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume = if ($env:E4_AUTO_RESUME) { $env:E4_AUTO_RESUME -ne "0" } else { $true }
$python = if ($env:E4_PYTHON) { $env:E4_PYTHON } else { "python" }
$smoke = if ($env:E4_SMOKE) { $env:E4_SMOKE -ne "0" } else { $false }
$iterations = if ($smoke) { 1000 } else { 30000 }
$seedList = if ($env:E4_SEEDS) {
    $env:E4_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    if ($phase -eq "multiseed") { @(0, 1, 2) } else { @(0) }
}
$checkpointIterations = if ($env:E4_CHECKPOINT_ITERS) {
    $env:E4_CHECKPOINT_ITERS.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} elseif ($smoke) {
    @("1000")
} else {
    @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "30000")
}
$evalIterations = if ($smoke) {
    @("1000")
} else {
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

function Get-StructureProfile {
    return @{
        Opacity = "0.02"
        MinVisibility = "0"
        MaxScreenSize = "-1"
        GradThreshold = "0.0"
        GradMinDenom = "1.0"
        DensifyUntilFraction = "0.50"
        PruneOnlyFromFraction = "0.50"
        PruneUntilFraction = "1.0"
        MaxGaussians = "2500000"
    }
}

function Get-PerceptualProfiles {
    $profiles = [ordered]@{}
    $base = @{
        LambdaDssim = "0.2"
        LpipsFinal = "0.0"
        LpipsMax = "0.0"
        Multiscale = $false
    }

    $rgbL1 = $base.Clone()
    $rgbL1.LambdaDssim = "0.0"
    $profiles["rgb_l1"] = $rgbL1

    $rgbSsim = $base.Clone()
    $profiles["rgb_ssim"] = $rgbSsim

    foreach ($weight in @("0.01", "0.03", "0.05")) {
        $name = "lpips_w$($weight.Replace('.', ''))"
        $p = $base.Clone()
        $p.LpipsFinal = $weight
        $p.LpipsMax = $weight
        $profiles[$name] = $p
    }

    foreach ($weight in @("0.01", "0.03", "0.05")) {
        $name = "lpips_ms_w$($weight.Replace('.', ''))"
        $p = $base.Clone()
        $p.LpipsFinal = $weight
        $p.LpipsMax = $weight
        $p.Multiscale = $true
        $profiles[$name] = $p
    }

    return $profiles
}

function Copy-Profile {
    param($Profile)
    $copy = @{}
    foreach ($key in $Profile.Keys) {
        $copy[$key] = $Profile[$key]
    }
    return $copy
}

function Get-ProfileSettingValue {
    param($Settings, [string]$Key)
    if ($null -eq $Settings) { return $null }
    if ($Settings -is [hashtable] -or $Settings -is [System.Collections.IDictionary]) {
        if ($Settings.Contains($Key)) { return [string]$Settings[$Key] }
        return $null
    }
    $prop = $Settings.PSObject.Properties[$Key]
    if ($null -eq $prop) { return $null }
    return [string]$prop.Value
}

function Assert-ResumeProfileMatches {
    param(
        [string]$ModelDir,
        [string]$Name,
        [int]$Seed,
        $Structure,
        $Perceptual
    )
    $metaPath = Join-Path $ModelDir "e4_profile.json"
    if (-not (Test-Path $metaPath)) { return }
    $existing = Get-Content -Path $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($existing.profile -and [string]$existing.profile -ne $Name) {
        throw "E4 resume mismatch in $ModelDir : stored profile '$($existing.profile)' != '$Name'"
    }
    if ($null -ne $existing.seed -and [int]$existing.seed -ne $Seed) {
        throw "E4 resume mismatch in $ModelDir : stored seed $($existing.seed) != $Seed"
    }
    foreach ($key in @(
        "Opacity", "MinVisibility", "MaxScreenSize", "GradThreshold", "GradMinDenom",
        "DensifyUntilFraction", "PruneOnlyFromFraction", "PruneUntilFraction", "MaxGaussians"
    )) {
        $expected = [string]$Structure[$key]
        $actual = Get-ProfileSettingValue -Settings $existing.structure -Key $key
        if ($null -ne $actual -and $actual -ne $expected) {
            throw "E4 resume mismatch in $ModelDir : structure.$key '$actual' != '$expected'"
        }
    }
    foreach ($key in @("LambdaDssim", "LpipsFinal", "LpipsMax", "Multiscale")) {
        $expected = [string]$Perceptual[$key]
        $actual = Get-ProfileSettingValue -Settings $existing.perceptual -Key $key
        if ($null -ne $actual -and $actual -ne $expected) {
            throw "E4 resume mismatch in $ModelDir : perceptual.$key '$actual' != '$expected'"
        }
    }
}

function Resolve-SelectedProfiles {
    param([System.Collections.Specialized.OrderedDictionary]$All)

    if ($phase -eq "multiseed") {
        if (-not $env:E4_TOP_CONFIGS) {
            throw "E4_PHASE=multiseed requires E4_TOP_CONFIGS=cfg1,cfg2,..."
        }
        $names = $env:E4_TOP_CONFIGS.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    } elseif ($env:E4_PROFILES) {
        $names = $env:E4_PROFILES.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    } else {
        $names = @($All.Keys)
    }

    $selected = [ordered]@{}
    foreach ($name in $names) {
        if (-not $All.Contains($name)) {
            throw "Unknown E4 profile '$name'. Known: $($All.Keys -join ', ')"
        }
        $selected[$name] = $All[$name]
    }
    return $selected
}

$structure = Get-StructureProfile
$allProfiles = Get-PerceptualProfiles
$profiles = Resolve-SelectedProfiles -All $allProfiles

Write-Host "[E4] phase=$phase smoke=$smoke iterations=$iterations profiles=$($profiles.Keys -join ', ') seeds=$($seedList -join ',')"

foreach ($name in $profiles.Keys) {
    $perceptual = $profiles[$name]
    foreach ($seed in $seedList) {
        $suffix = if ($seed -eq 0) { "" } else { "_seed$seed" }
        $caseName = "vr_room_e4_${name}${suffix}"
        $modelDir = Join-Path $ROOT "output\$caseName"
        $finalPly = Join-Path $modelDir "point_cloud\iteration_$iterations\point_cloud.ply"

        if ($skipCompleted -and (Test-Path $finalPly)) {
            Write-Host "[E4] Skip completed $caseName"
            continue
        }

        New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
        $resumeCkpt = $null
        if ($autoResume) {
            $resumeCkpt = Get-LatestCheckpoint -ModelDir $modelDir
            if ($null -ne $resumeCkpt -and [int]$resumeCkpt.Iteration -ge $iterations -and (Test-Path $finalPly)) {
                Write-Host "[E4] Skip completed $caseName (chkpnt$($resumeCkpt.Iteration) + final ply)"
                continue
            }
        }
        Assert-ResumeProfileMatches -ModelDir $modelDir -Name $name -Seed $seed -Structure $structure -Perceptual $perceptual

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
            "--lambda_dssim", "$($perceptual.LambdaDssim)",
            "--densification_interval", "100",
            "--opacity_reset_interval", "3000",
            "--densify_from_iter", "500",
            "--densify_until_iter", "18000",
            "--densify_until_fraction", "$($structure.DensifyUntilFraction)",
            "--densify_grad_threshold", "0.00016",
            "--max_gaussians", "$($structure.MaxGaussians)",
            "--prune_opacity_threshold", "$($structure.Opacity)",
            "--prune_min_visibility", "$($structure.MinVisibility)",
            "--prune_max_screen_size", "$($structure.MaxScreenSize)",
            "--prune_grad_threshold", "$($structure.GradThreshold)",
            "--prune_grad_min_denom", "$($structure.GradMinDenom)",
            "--prune_only_from_fraction", "$($structure.PruneOnlyFromFraction)",
            "--prune_until_fraction", "$($structure.PruneUntilFraction)",
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
            "--lpips_loss_weight_final", "$($perceptual.LpipsFinal)",
            "--lpips_loss_start_iter", "$lpipsStartIter",
            "--lpips_loss_max_weight", "$($perceptual.LpipsMax)",
            "--lpips_loss_interval", "1",
            "--lpips_loss_schedule", "step",
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

        if ([bool]$perceptual.Multiscale) {
            $trainArgs += @(
                "--enable_multiscale_lpips",
                "--lpips_multiscale_scales", "1.0,0.5,0.25",
                "--lpips_multiscale_weights", "0.5,0.3,0.2"
            )
        }

        if ($null -ne $resumeCkpt) {
            $trainArgs += @("--start_checkpoint", "$($resumeCkpt.Path)")
            Write-Host "[E4] Resume $caseName from chkpnt$($resumeCkpt.Iteration)"
        } else {
            Write-Host "[E4] Starting $caseName from scratch"
        }

        $env:TAGGS_CPU_KNN_INIT = "1"
        $env:KMP_DUPLICATE_LIB_OK = "TRUE"
        $logPath = Join-Path $modelDir "e4_train_console.log"
        $meta = @{
            profile = $name
            seed = $seed
            structure = $structure
            perceptual = @{
                LambdaDssim = $perceptual.LambdaDssim
                LpipsFinal = $perceptual.LpipsFinal
                LpipsMax = $perceptual.LpipsMax
                Multiscale = [bool]$perceptual.Multiscale
                LpipsStartIter = $lpipsStartIter
                LpipsSchedule = "step"
            }
            model_dir = $modelDir
            auto_resume = $autoResume
            resume_checkpoint = if ($null -ne $resumeCkpt) { $resumeCkpt.Path } else { $null }
            resume_iteration = if ($null -ne $resumeCkpt) { $resumeCkpt.Iteration } else { $null }
            checkpoint_iterations = $checkpointIterations
            eval_iterations = $evalIterations
            smoke = $smoke
            iterations = $iterations
        }
        ($meta | ConvertTo-Json -Depth 6) | Set-Content -Path (Join-Path $modelDir "e4_profile.json") -Encoding UTF8

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
            throw "E4 $caseName failed with exit code $trainExitCode"
        }
    }
}

Write-Host "[E4] Done."
