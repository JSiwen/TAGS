# E3 v2: vr_room pruning-strategy ablation.
#
# Fixes vs v1: prune-only no longer applies oversized screen/world prune;
# eval/snapshots run after densify/prune; test evals include 10k/18k/19.5k.
# All screen runs use a 2M safety cap (E2 budget) so 3080 training stays feasible.
#
# Phase A — single-seed screening (default seed=0), all factors:
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# Selected profiles:
#   $env:E3_PROFILES = "opacity_0.005,opacity_0.01,densify_0.50,densify_0.80"
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# Phase B — top-2 configs x 3 seeds:
#   $env:E3_PHASE = "multiseed"
#   $env:E3_TOP_CONFIGS = "densify_0.80,opacity_0.01"
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# Wave 4 orthogonal screen (seed=0; keeps v2 names, new profiles only):
#   $env:E3_PROFILES = "combo_opacity0.02_densify0.65,densify0.50_pruneonly0.80,densify0.60_pruneonly0.60,combo_opacity0.02_densify0.50_pruneonly0.80,combo_opacity0.02_densify0.60_pruneonly0.60"
#   .\scripts\run_vr_room_e3_prune_ablation.ps1
#
# Cap isolation later:
#   $env:E3_RUN_TAG = "v3cap"
#   $env:E3_PROFILES = "combo_opacity0.02_densify0.50,e2_style_cap2m"
#   $env:E3_MAX_GAUSSIANS = "2500000"   # or "0" to force uncap
#
# After training:
#   python scripts\summarize_vr_room_e3.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
#
# Outputs: output/vr_room_e3_v2_<profile>/  (override with E3_RUN_TAG)
# Resume skips completed 30k plys unless E3_SKIP_COMPLETED=0.
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
$maxGaussiansOverride = 0
$capOverrideSet = $false
if ($null -ne $env:E3_MAX_GAUSSIANS -and $env:E3_MAX_GAUSSIANS.Trim() -ne "") {
    $capOverrideSet = $true
    $maxGaussiansOverride = [int]$env:E3_MAX_GAUSSIANS.Trim()
    if ($maxGaussiansOverride -lt 0) {
        throw "E3_MAX_GAUSSIANS must be >= 0, got $maxGaussiansOverride"
    }
}
$runTag = if ($env:E3_RUN_TAG) { $env:E3_RUN_TAG.Trim() } else { "v2" }
$evalIterations = @("7000", "10000", "15100", "18100", "19500", "21100", "24000", "30000")
$liveStageIterations = ($evalIterations -join ",")

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
        # E2-like densify horizon (18k), then last 20% prune-only.
        DensifyUntilFraction = "0.6"
        PruneOnlyFromFraction = "0.80"
        PruneUntilFraction = "1.0"
        MaxGaussians = "2000000"
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

    foreach ($frac in @("0.50", "0.65", "0.80")) {
        $p = Get-BaseProfile
        $p.DensifyUntilFraction = $frac
        $p.PruneOnlyFromFraction = $frac
        $p.PruneUntilFraction = "1.0"
        $profiles["densify_$frac"] = $p
    }

    # Last 10% prune-only, densify follows the E2 60% horizon.
    $late10 = Get-BaseProfile
    $late10.PruneOnlyFromFraction = "0.90"
    $profiles["lateprune_0.90"] = $late10

    $comboA = Get-BaseProfile
    $comboA.Opacity = "0.01"
    $comboA.DensifyUntilFraction = "0.50"
    $comboA.PruneOnlyFromFraction = "0.50"
    $profiles["combo_opacity0.01_densify0.50"] = $comboA

    $comboB = Get-BaseProfile
    $comboB.Opacity = "0.01"
    $comboB.DensifyUntilFraction = "0.80"
    $comboB.PruneOnlyFromFraction = "0.80"
    $profiles["combo_opacity0.01_densify0.80"] = $comboB

    $comboC = Get-BaseProfile
    $comboC.Opacity = "0.02"
    $comboC.DensifyUntilFraction = "0.50"
    $comboC.PruneOnlyFromFraction = "0.50"
    $profiles["combo_opacity0.02_densify0.50"] = $comboC

    $comboD = Get-BaseProfile
    $comboD.Opacity = "0.02"
    $comboD.DensifyUntilFraction = "0.65"
    $comboD.PruneOnlyFromFraction = "0.65"
    $profiles["combo_opacity0.02_densify0.65"] = $comboD

    # Orthogonal schedule corners. densify_* above still couples both knobs.
    $orthA = Get-BaseProfile
    $orthA.DensifyUntilFraction = "0.50"
    $orthA.PruneOnlyFromFraction = "0.80"
    $profiles["densify0.50_pruneonly0.80"] = $orthA

    $orthB = Get-BaseProfile
    $orthB.DensifyUntilFraction = "0.60"
    $orthB.PruneOnlyFromFraction = "0.60"
    $profiles["densify0.60_pruneonly0.60"] = $orthB

    $orthC = Get-BaseProfile
    $orthC.Opacity = "0.02"
    $orthC.DensifyUntilFraction = "0.50"
    $orthC.PruneOnlyFromFraction = "0.80"
    $profiles["combo_opacity0.02_densify0.50_pruneonly0.80"] = $orthC

    $orthD = Get-BaseProfile
    $orthD.Opacity = "0.02"
    $orthD.DensifyUntilFraction = "0.60"
    $orthD.PruneOnlyFromFraction = "0.60"
    $profiles["combo_opacity0.02_densify0.60_pruneonly0.60"] = $orthD

    $e2style = Get-BaseProfile
    $e2style.DensifyUntilFraction = "0.6"
    $e2style.PruneOnlyFromFraction = "0.6"
    $e2style.PruneUntilFraction = "0.6"
    $profiles["e2_style_cap2m"] = $e2style

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

function Get-CapSuffix([int]$Budget) {
    if ($Budget -le 0) { return "" }
    if ($Budget % 1000000 -eq 0) { return "cap$($Budget / 1000000)m" }
    $millions = ($Budget / 1000000.0).ToString(
        "0.##", [System.Globalization.CultureInfo]::InvariantCulture
    )
    return "cap${millions}m"
}

function Apply-MaxGaussiansCap {
    param(
        [System.Collections.Specialized.OrderedDictionary]$Profiles,
        [int]$Budget,
        [bool]$ForceOverride = $false
    )
    if (-not $ForceOverride) { return $Profiles }
    $capped = [ordered]@{}
    foreach ($name in $Profiles.Keys) {
        $profile = Copy-Profile $Profiles[$name]
        $baseName = $name -replace '_cap[\d.]+m$', '' -replace '_uncap$', ''
        if ($Budget -le 0) {
            $profile.MaxGaussians = "0"
            $capped["${baseName}_uncap"] = $profile
            continue
        }
        $existing = 0
        if ($profile.MaxGaussians) { $existing = [int]$profile.MaxGaussians }
        if ($existing -eq $Budget) {
            $capped[$name] = $profile
            continue
        }
        $profile.MaxGaussians = "$Budget"
        $suffix = Get-CapSuffix $Budget
        $capped["${baseName}_$suffix"] = $profile
    }
    return $capped
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
        $Profile
    )
    $metaPath = Join-Path $ModelDir "e3_profile.json"
    if (-not (Test-Path $metaPath)) { return }
    $existing = Get-Content -Path $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($existing.profile -and [string]$existing.profile -ne $Name) {
        throw "E3 resume mismatch in $ModelDir : stored profile '$($existing.profile)' != '$Name'"
    }
    if ($null -ne $existing.seed -and [int]$existing.seed -ne $Seed) {
        throw "E3 resume mismatch in $ModelDir : stored seed $($existing.seed) != $Seed"
    }
    $keys = @(
        "Opacity", "MinVisibility", "MaxScreenSize", "GradThreshold", "GradMinDenom",
        "DensifyUntilFraction", "PruneOnlyFromFraction", "PruneUntilFraction", "MaxGaussians"
    )
    foreach ($key in $keys) {
        $expected = [string]$Profile[$key]
        $actual = Get-ProfileSettingValue -Settings $existing.settings -Key $key
        if ($null -ne $actual -and $actual -ne $expected) {
            throw "E3 resume mismatch in $ModelDir : settings.$key '$actual' != '$expected'"
        }
    }
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
$profiles = Apply-MaxGaussiansCap -Profiles $profiles -Budget $maxGaussiansOverride -ForceOverride $capOverrideSet

Write-Host "[E3] phase=$phase tag=$runTag profiles=$($profiles.Keys -join ', ') seeds=$($seedList -join ',')"
if ($capOverrideSet) {
    Write-Host "[E3] Gaussian cap override E3_MAX_GAUSSIANS=$maxGaussiansOverride"
}

foreach ($name in $profiles.Keys) {
    $profile = $profiles[$name]
    foreach ($seed in $seedList) {
        $caseName = if ($phase -eq "multiseed") {
            "vr_room_e3_${runTag}_${name}_seed$seed"
        } else {
            "vr_room_e3_${runTag}_${name}"
        }
        if ($phase -ne "multiseed" -and $seed -ne 0) {
            $caseName = "vr_room_e3_${runTag}_${name}_seed$seed"
        }
        $modelDir = Join-Path $ROOT "output\$caseName"
        $finalPly = Join-Path $modelDir "point_cloud\iteration_30000\point_cloud.ply"
        if (Test-Path $modelDir) {
            Assert-ResumeProfileMatches -ModelDir $modelDir -Name $name -Seed $seed -Profile $profile
        }
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
            "--live_stage_iterations", $liveStageIterations,
            "--test_iterations"
        ) + $evalIterations + @(
            "--save_iterations", "30000",
            "--checkpoint_iterations"
        ) + $checkpointIterations + @(
            "--disable_viewer"
        )

        $capNote = if ([int]$profile.MaxGaussians -gt 0) { " max_gaussians=$($profile.MaxGaussians)" } else { "" }
        if ($null -ne $resumeCkpt) {
            $trainArgs += @("--start_checkpoint", "$($resumeCkpt.Path)")
            Write-Host "[E3] Resume $caseName from chkpnt$($resumeCkpt.Iteration) ($($resumeCkpt.Path))$capNote"
        } else {
            Write-Host "[E3] Starting $caseName from scratch$capNote"
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
            e2_max_gaussians_override = $maxGaussiansOverride
            cap_override_set = $capOverrideSet
            run_tag = $runTag
            eval_iterations = $evalIterations
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

Write-Host "[E3] Requested runs complete. Next: python scripts\summarize_vr_room_e3.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30"
