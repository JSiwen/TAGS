# E4b: vr_room geo-prior overlap with the frozen E4 winner recipe (lpips_ms_w005).
#
# Default Wave 1 — seed=0 for decay24k / decay27k:
#   .\scripts\run_vr_room_e4b_geo_overlap.ps1
#
# Smoke test:
#   $env:E4B_SMOKE = "1"
#   $env:E4B_PROFILES = "decay24k"
#   .\scripts\run_vr_room_e4b_geo_overlap.ps1
#
# Wave 2 — three-seed finalist:
#   $env:E4B_PHASE = "multiseed"
#   $env:E4B_TOP_CONFIGS = "decay24k"
#   $env:E4B_SEEDS = "0,1,2"
#   .\scripts\run_vr_room_e4b_geo_overlap.ps1
#
# Conditional depth-only tail (only if B/C over-smooth):
#   $env:E4B_PROFILES = "depth_tail27k"
#   .\scripts\run_vr_room_e4b_geo_overlap.ps1
#
# Outputs: output/vr_room_e4b_<profile>[_seedN]/
# Frozen E4 winner is reused as decay20k unless E4B_RETRAIN_FREEZE=1.

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$phase = if ($env:E4B_PHASE) { $env:E4B_PHASE.Trim().ToLower() } else { "screen" }
$skipCompleted = if ($env:E4B_SKIP_COMPLETED) { $env:E4B_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume = if ($env:E4B_AUTO_RESUME) { $env:E4B_AUTO_RESUME -ne "0" } else { $true }
$python = if ($env:E4B_PYTHON) { $env:E4B_PYTHON } elseif ($env:E4_PYTHON) { $env:E4_PYTHON } else { "python" }
$smoke = if ($env:E4B_SMOKE) { $env:E4B_SMOKE -ne "0" } else { $false }
$retrainFreeze = if ($env:E4B_RETRAIN_FREEZE) { $env:E4B_RETRAIN_FREEZE -ne "0" } else { $false }
$iterations = if ($smoke) { 1000 } else { 30000 }
$seedList = if ($env:E4B_SEEDS) {
    $env:E4B_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    if ($phase -eq "multiseed") { @(0, 1, 2) } else { @(0) }
}
$checkpointIterations = if ($env:E4B_CHECKPOINT_ITERS) {
    $env:E4B_CHECKPOINT_ITERS.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} elseif ($smoke) {
    @("1000")
} else {
    @("3000", "6000", "9000", "12000", "15000", "18000", "21000", "24000", "27000", "30000")
}
$evalIterations = if ($smoke) {
    @("1000")
} else {
    @("7000", "10000", "15100", "18100", "19500", "21100", "24100", "27100", "30000")
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

function Get-GeoProfiles {
    $profiles = [ordered]@{}
    $base = @{
        GeoDecayUntil = 20000
        DepthDecayUntil = 0
        NormalDecayUntil = 0
        FullresRelaxMax = 0.35
        FreezeBaseline = $false
    }

    $decay20k = $base.Clone()
    $decay20k.FreezeBaseline = $true
    $profiles["decay20k"] = $decay20k

    $decay24k = $base.Clone()
    $decay24k.GeoDecayUntil = 24000
    $profiles["decay24k"] = $decay24k

    $decay27k = $base.Clone()
    $decay27k.GeoDecayUntil = 27000
    $profiles["decay27k"] = $decay27k

    $depthTail = $base.Clone()
    $depthTail.GeoDecayUntil = 27000
    $depthTail.DepthDecayUntil = 27000
    $depthTail.NormalDecayUntil = 20000
    $profiles["depth_tail27k"] = $depthTail

    $depth27Normal24 = $base.Clone()
    $depth27Normal24.GeoDecayUntil = 27000
    $depth27Normal24.NormalDecayUntil = 24000
    $profiles["depth27k_normal24k"] = $depth27Normal24

    $depth24Normal27 = $base.Clone()
    $depth24Normal27.GeoDecayUntil = 24000
    $depth24Normal27.NormalDecayUntil = 27000
    $profiles["depth24k_normal27k"] = $depth24Normal27

    $depth255Normal24 = $base.Clone()
    $depth255Normal24.GeoDecayUntil = 25500
    $depth255Normal24.NormalDecayUntil = 24000
    $profiles["depth25500_normal24k"] = $depth255Normal24

    $depth27Normal255 = $base.Clone()
    $depth27Normal255.GeoDecayUntil = 27000
    $depth27Normal255.NormalDecayUntil = 25500
    $profiles["depth27k_normal25500"] = $depth27Normal255

    $decay25500 = $base.Clone()
    $decay25500.GeoDecayUntil = 25500
    $profiles["decay25500"] = $decay25500

    $depth27Normal24Relax50 = $depth27Normal24.Clone()
    $depth27Normal24Relax50.FullresRelaxMax = 0.50
    $profiles["depth27k_normal24k_relax50"] = $depth27Normal24Relax50

    return $profiles
}

function Get-E4FreezeCaseName {
    param([int]$Seed)
    if ($Seed -eq 0) { return "vr_room_e4_lpips_ms_w005" }
    return "vr_room_e4_lpips_ms_w005_seed$Seed"
}

function Test-E4FreezeComplete {
    param([int]$Seed, [int]$Iters)
    $caseName = Get-E4FreezeCaseName -Seed $Seed
    $ply = Join-Path $ROOT "output\$caseName\point_cloud\iteration_$Iters\point_cloud.ply"
    return Test-Path $ply
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
        $Geo
    )
    $metaPath = Join-Path $ModelDir "e4b_profile.json"
    if (-not (Test-Path $metaPath)) { return }
    $existing = Get-Content -Path $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($existing.profile -and [string]$existing.profile -ne $Name) {
        throw "E4b resume mismatch in $ModelDir : stored profile '$($existing.profile)' != '$Name'"
    }
    if ($null -ne $existing.seed -and [int]$existing.seed -ne $Seed) {
        throw "E4b resume mismatch in $ModelDir : stored seed $($existing.seed) != $Seed"
    }
    foreach ($key in @("GeoDecayUntil", "DepthDecayUntil", "NormalDecayUntil", "FullresRelaxMax")) {
        $expected = [string]$Geo[$key]
        $actual = Get-ProfileSettingValue -Settings $existing.geo -Key $key
        if ($null -ne $actual -and $actual -ne $expected) {
            throw "E4b resume mismatch in $ModelDir : geo.$key '$actual' != '$expected'"
        }
    }
}

function Resolve-SelectedProfiles {
    param([System.Collections.Specialized.OrderedDictionary]$All)

    if ($phase -eq "multiseed") {
        if (-not $env:E4B_TOP_CONFIGS) {
            throw "E4B_PHASE=multiseed requires E4B_TOP_CONFIGS=cfg1,cfg2,..."
        }
        $names = $env:E4B_TOP_CONFIGS.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    } elseif ($env:E4B_PROFILES) {
        $names = $env:E4B_PROFILES.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    } else {
        $names = @("decay24k", "decay27k")
    }

    $selected = [ordered]@{}
    foreach ($name in $names) {
        if (-not $All.Contains($name)) {
            throw "Unknown E4b profile '$name'. Known: $($All.Keys -join ', ')"
        }
        $selected[$name] = $All[$name]
    }
    return $selected
}

$allProfiles = Get-GeoProfiles
$profiles = Resolve-SelectedProfiles -All $allProfiles

Write-Host "[E4b] phase=$phase smoke=$smoke iterations=$iterations profiles=$($profiles.Keys -join ', ') seeds=$($seedList -join ',')"

foreach ($name in $profiles.Keys) {
    $geo = $profiles[$name]
    foreach ($seed in $seedList) {
        $suffix = if ($seed -eq 0) { "" } else { "_seed$seed" }
        $caseName = "vr_room_e4b_${name}${suffix}"
        $modelDir = Join-Path $ROOT "output\$caseName"
        $finalPly = Join-Path $modelDir "point_cloud\iteration_$iterations\point_cloud.ply"

        if (-not $retrainFreeze -and [bool]$geo.FreezeBaseline -and -not $smoke) {
            if (Test-E4FreezeComplete -Seed $seed -Iters $iterations) {
                Write-Host "[E4b] Reuse E4 freeze $(Get-E4FreezeCaseName -Seed $seed) as $caseName"
                continue
            }
            Write-Warning "[E4b] E4 freeze missing for seed $seed; training $caseName from the frozen recipe"
        }

        if ($skipCompleted -and (Test-Path $finalPly)) {
            Write-Host "[E4b] Skip completed $caseName"
            continue
        }

        New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
        $resumeCkpt = $null
        if ($autoResume) {
            $resumeCkpt = Get-LatestCheckpoint -ModelDir $modelDir
            if ($null -ne $resumeCkpt -and [int]$resumeCkpt.Iteration -ge $iterations -and (Test-Path $finalPly)) {
                Write-Host "[E4b] Skip completed $caseName (chkpnt$($resumeCkpt.Iteration) + final ply)"
                continue
            }
        }
        Assert-ResumeProfileMatches -ModelDir $modelDir -Name $name -Seed $seed -Geo $geo

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
            "--geo_prior_decay_until", "$($geo.GeoDecayUntil)",
            "--geo_prior_depth_decay_until", "$($geo.DepthDecayUntil)",
            "--geo_prior_normal_decay_until", "$($geo.NormalDecayUntil)",
            "--geo_prior_fullres_relax_max", "$($geo.FullresRelaxMax)",
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
            Write-Host "[E4b] Resume $caseName from chkpnt$($resumeCkpt.Iteration)"
        } else {
            Write-Host "[E4b] Starting $caseName from scratch (geo decay until $($geo.GeoDecayUntil))"
        }

        $env:TAGGS_CPU_KNN_INIT = "1"
        $env:KMP_DUPLICATE_LIB_OK = "TRUE"
        $logPath = Join-Path $modelDir "e4b_train_console.log"
        $meta = @{
            profile = $name
            seed = $seed
            frozen_perceptual = "lpips_ms_w005"
            geo = @{
                GeoDecayUntil = $geo.GeoDecayUntil
                DepthDecayUntil = $geo.DepthDecayUntil
                NormalDecayUntil = $geo.NormalDecayUntil
                FullresRelaxMax = $geo.FullresRelaxMax
                FreezeBaseline = [bool]$geo.FreezeBaseline
            }
            perceptual = @{
                LambdaDssim = "0.2"
                LpipsFinal = "0.05"
                LpipsMax = "0.05"
                Multiscale = $true
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
        ($meta | ConvertTo-Json -Depth 6) | Set-Content -Path (Join-Path $modelDir "e4b_profile.json") -Encoding UTF8

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
            throw "E4b $caseName failed with exit code $trainExitCode"
        }
    }
}

Write-Host "[E4b] Done."
