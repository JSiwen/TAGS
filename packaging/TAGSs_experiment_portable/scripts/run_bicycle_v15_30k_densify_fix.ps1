# v15 30k densify-fix experiments (RTX 3080, 12k-protocol)
#
# Implements the post-30k failure plan:
#   E0  diagnostic baseline (densify until 18k, dense 12k-20k checkpoints)
#   E1  early stop densify at 12k                         [DEFAULT]
#   E1b align densify until 7.2k (12k promotion ratio)   [recommended next]
#   E2  E1 + tighter point-growth gate
#   E3  E1 + stronger SPA prune
#   E4  E1 + lighter late LPIPS
#
# Protocol locked to 12k winner settings:
#   images=images_4, resolution=800, late_stabilize, iterations=30000
#
# Usage (repo root):
#   .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# With env file:
#   . .\scripts\configs\bicycle_v15_30k_densify_fix.env.ps1
#   .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# Run a specific experiment:
#   $env:EXPERIMENT = "E1";  .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#   $env:EXPERIMENT = "E1b"; .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# Run diagnostic + main fix sequentially:
#   $env:EXPERIMENT = "all"; .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# Force retrain:
#   $env:SKIP_COMPLETED = "0"; .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# OOM fallback:
#   $env:RESOLUTION = "640"; $env:IMAGES = "images_2"
#   .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# Render/metrics/summary only:
#   $env:RUN_TRAIN = "0"; $env:RUN_RENDER_METRICS = "1"
#   .\scripts\run_bicycle_v15_30k_densify_fix.ps1

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

foreach ($name in @(
    "OUTPUT_ROOT", "ITERATIONS", "RAW_STAGE_ITERATIONS",
    "CASE_NAME", "ONLY_CASE", "CONFIG_LABEL", "STAGE_SUMMARY_JSON", "WRITE_STAGE_JSON",
    "DENSIFY_UNTIL_FULL", "DENSIFY_FROM_FULL",
    "SPA_PRUNE_RATIO", "PRUNE_RATIO2",
    "TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD", "TAGGS_DENSIFY_INTERVAL_MULT_FULLRES",
    "LR_GEO_MULT_STAGE3", "LR_COLOR_MULT_STAGE3",
    "LPIPS_START_FULL", "LPIPS_WEIGHT_FINAL", "LPIPS_MAX_WEIGHT", "LPIPS_INTERVAL",
    "COLOR_ALIGN_START_FULL", "COLOR_ALIGN_WEIGHT_FINAL", "COLOR_ALIGN_MAX_WEIGHT",
    "PSNR_START_FULL", "PSNR_WEIGHT_FINAL", "PSNR_MAX_WEIGHT"
)) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$EXPERIMENT = if ($env:EXPERIMENT) { $env:EXPERIMENT.Trim().ToUpper() } else { "E1" }
$VARIANT = if ($env:VARIANT) { $env:VARIANT.Trim() } else { "late_stabilize" }
$ITERATIONS = if ($env:ITERATIONS) { $env:ITERATIONS } else { "30000" }
$RESOLUTION = if ($env:RESOLUTION) { $env:RESOLUTION } else { "800" }
$IMAGES = if ($env:IMAGES) { $env:IMAGES } else { "images_4" }
$RUN_TRAIN = if ($env:RUN_TRAIN) { $env:RUN_TRAIN } else { "1" }
$RUN_RENDER_METRICS = if ($env:RUN_RENDER_METRICS) { $env:RUN_RENDER_METRICS } else { "1" }
$SKIP_COMPLETED = if ($env:SKIP_COMPLETED) { $env:SKIP_COMPLETED } else { "1" }

$RAW_STAGE_ITERATIONS = if ($env:RAW_STAGE_ITERATIONS) {
    $env:RAW_STAGE_ITERATIONS
} else {
    "7000,12000,14000,15000,16000,17000,18000,20000,24000,27000,28500,30000"
}

$PSNR_DROP_THRESHOLD = if ($env:PSNR_DROP_THRESHOLD) { $env:PSNR_DROP_THRESHOLD } else { "0.05" }
$PERCEPTUAL_PSNR_BUDGET = if ($env:PERCEPTUAL_PSNR_BUDGET) { $env:PERCEPTUAL_PSNR_BUDGET } else { "0.10" }
$CHRONIC_TRIPLE_LIMIT = if ($env:CHRONIC_TRIPLE_LIMIT) { $env:CHRONIC_TRIPLE_LIMIT } else { "1" }

function Get-EnvOrDefault {
    param([string]$Name, [string]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    return $Default
}

function Get-ExperimentProfile {
    param([string]$Id)
    switch ($Id) {
        "E0" {
            return @{
                Id = "E0"
                Label = "diagnostic baseline (densify until 18k)"
                OutputRoot = "output\v15_30k_fix_e0_diag"
                CaseName = "bicycle_v15_3080_e0_diag_baseline"
                DensifyUntilFull = "18000"
                SpaPruneRatio = "0.05"
                PruneRatio2 = "0.04"
                DensifyPointThreshold = "50000"
                DensifyIntervalMult = "2"
                LateStabilizeLpipsStart = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_START_FULL" "24000")
                LateStabilizeLpipsWeight = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_WEIGHT_FINAL" "0.006")
            }
        }
        "E1" {
            return @{
                Id = "E1"
                Label = "early stop densify at 12k"
                OutputRoot = "output\v15_30k_fix_e1_densify12k"
                CaseName = "bicycle_v15_3080_e1_densify12k"
                DensifyUntilFull = "12000"
                SpaPruneRatio = "0.05"
                PruneRatio2 = "0.04"
                DensifyPointThreshold = "50000"
                DensifyIntervalMult = "2"
                LateStabilizeLpipsStart = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_START_FULL" "24000")
                LateStabilizeLpipsWeight = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_WEIGHT_FINAL" "0.006")
                StageIterations = $null
            }
        }
        "E1B" {
            return @{
                Id = "E1b"
                Label = "align densify until 7.2k (12k promotion ratio 18000*12/30)"
                OutputRoot = "output\v15_30k_fix_e1b_densify7k"
                CaseName = "bicycle_v15_3080_e1b_densify7k"
                DensifyUntilFull = "7200"
                SpaPruneRatio = "0.05"
                PruneRatio2 = "0.04"
                DensifyPointThreshold = "50000"
                DensifyIntervalMult = "2"
                LateStabilizeLpipsStart = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_START_FULL" "24000")
                LateStabilizeLpipsWeight = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_WEIGHT_FINAL" "0.006")
                StageIterations = "7000,7200,8000,10000,12000,14000,15000,16000,17000,18000,20000,24000,27000,28500,30000"
            }
        }
        "E2" {
            return @{
                Id = "E2"
                Label = "E1 + tighter point-growth gate"
                OutputRoot = "output\v15_30k_fix_e2_pointgate"
                CaseName = "bicycle_v15_3080_e2_pointgate"
                DensifyUntilFull = "12000"
                SpaPruneRatio = "0.05"
                PruneRatio2 = "0.04"
                DensifyPointThreshold = "800000"
                DensifyIntervalMult = "3"
                LateStabilizeLpipsStart = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_START_FULL" "24000")
                LateStabilizeLpipsWeight = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_WEIGHT_FINAL" "0.006")
            }
        }
        "E3" {
            return @{
                Id = "E3"
                Label = "E1 + stronger SPA prune"
                OutputRoot = "output\v15_30k_fix_e3_strongprune"
                CaseName = "bicycle_v15_3080_e3_strongprune"
                DensifyUntilFull = "12000"
                SpaPruneRatio = "0.08"
                PruneRatio2 = "0.06"
                DensifyPointThreshold = "50000"
                DensifyIntervalMult = "2"
                LateStabilizeLpipsStart = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_START_FULL" "24000")
                LateStabilizeLpipsWeight = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_WEIGHT_FINAL" "0.006")
            }
        }
        "E4" {
            return @{
                Id = "E4"
                Label = "E1 + lighter late LPIPS"
                OutputRoot = "output\v15_30k_fix_e4_lightlpips"
                CaseName = "bicycle_v15_3080_e4_lightlpips"
                DensifyUntilFull = "12000"
                SpaPruneRatio = "0.05"
                PruneRatio2 = "0.04"
                DensifyPointThreshold = "50000"
                DensifyIntervalMult = "2"
                LateStabilizeLpipsStart = "26000"
                LateStabilizeLpipsWeight = "0.004"
            }
        }
        default {
            throw "Unknown EXPERIMENT='$Id'. Use E0|E1|E1b|E2|E3|E4|all"
        }
    }
}

function Apply-LateStabilizeOverrides {
    param(
        [string]$LpipsStart,
        [string]$LpipsWeight
    )
    return @{
        LR_GEO_MULT_STAGE3 = (Get-EnvOrDefault "LATE_STABILIZE_LR_GEO_MULT_STAGE3" "0.40")
        LR_COLOR_MULT_STAGE3 = (Get-EnvOrDefault "LATE_STABILIZE_LR_COLOR_MULT_STAGE3" "0.55")
        LPIPS_START_FULL = $LpipsStart
        LPIPS_WEIGHT_FINAL = $LpipsWeight
        LPIPS_MAX_WEIGHT = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_MAX_WEIGHT" "0.010")
        LPIPS_INTERVAL = (Get-EnvOrDefault "LATE_STABILIZE_LPIPS_INTERVAL" "10")
        COLOR_ALIGN_START_FULL = (Get-EnvOrDefault "LATE_STABILIZE_COLOR_ALIGN_START_FULL" "25000")
        COLOR_ALIGN_WEIGHT_FINAL = (Get-EnvOrDefault "LATE_STABILIZE_COLOR_ALIGN_WEIGHT_FINAL" "0.0010")
        COLOR_ALIGN_MAX_WEIGHT = (Get-EnvOrDefault "LATE_STABILIZE_COLOR_ALIGN_MAX_WEIGHT" "0.0020")
        PSNR_START_FULL = (Get-EnvOrDefault "LATE_STABILIZE_PSNR_START_FULL" "22000")
        PSNR_WEIGHT_FINAL = (Get-EnvOrDefault "LATE_STABILIZE_PSNR_WEIGHT_FINAL" "0.008")
        PSNR_MAX_WEIGHT = (Get-EnvOrDefault "LATE_STABILIZE_PSNR_MAX_WEIGHT" "0.012")
    }
}

function Invoke-Experiment {
    param([hashtable]$Profile)

    $outputRoot = Join-Path $ROOT $Profile.OutputRoot
    $caseName = $Profile.CaseName
    $modelDir = Join-Path $outputRoot $caseName
    $stageSummaryJson = Join-Path $outputRoot "${caseName}_stage_selection_summary.json"
    $stageIters = if ($Profile.StageIterations) { $Profile.StageIterations } else { $RAW_STAGE_ITERATIONS }
    $configLabel = "exp=$($Profile.Id), $($Profile.Label), images=$IMAGES, resolution=$RESOLUTION, iterations=$ITERATIONS, densify_until_full=$($Profile.DensifyUntilFull)"
    $overrides = Apply-LateStabilizeOverrides $Profile.LateStabilizeLpipsStart $Profile.LateStabilizeLpipsWeight

    Write-Host ""
    Write-Host "=== v15 30k densify-fix | $($Profile.Id) | $caseName ==="
    Write-Host "LABEL:       $($Profile.Label)"
    Write-Host "OUTPUT_ROOT: $outputRoot"
    Write-Host "MODEL_DIR:   $modelDir"
    Write-Host "CONFIG:      $configLabel"
    Write-Host "STAGES:      $stageIters"
    Write-Host "DENSIFY:     until_full=$($Profile.DensifyUntilFull)"
    Write-Host "SPA:         prune=$($Profile.SpaPruneRatio)/$($Profile.PruneRatio2)"
    Write-Host "POINT GATE:  threshold=$($Profile.DensifyPointThreshold), interval_mult=$($Profile.DensifyIntervalMult)"
    Write-Host "LPIPS:       start=$($Profile.LateStabilizeLpipsStart), weight=$($Profile.LateStabilizeLpipsWeight)"
    Write-Host ""

    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

    $env:OUTPUT_ROOT = $outputRoot
    $env:ITERATIONS = "$ITERATIONS"
    $env:RESOLUTION = "$RESOLUTION"
    $env:IMAGES = "$IMAGES"
    $env:RAW_STAGE_ITERATIONS = $stageIters
    $env:CASE_NAME = $caseName
    $env:CONFIG_LABEL = $configLabel
    $env:RUN_TRAIN = $RUN_TRAIN
    $env:RUN_RENDER_METRICS = $RUN_RENDER_METRICS
    $env:SKIP_COMPLETED = $SKIP_COMPLETED
    $env:WRITE_STAGE_JSON = "1"
    $env:STAGE_SUMMARY_JSON = $stageSummaryJson
    $env:PSNR_DROP_THRESHOLD = $PSNR_DROP_THRESHOLD
    $env:PERCEPTUAL_PSNR_BUDGET = $PERCEPTUAL_PSNR_BUDGET
    $env:CHRONIC_TRIPLE_LIMIT = $CHRONIC_TRIPLE_LIMIT

    $env:DENSIFY_UNTIL_FULL = $Profile.DensifyUntilFull
    $env:SPA_PRUNE_RATIO = $Profile.SpaPruneRatio
    $env:PRUNE_RATIO2 = $Profile.PrunRatio2
    $env:TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD = $Profile.DensifyPointThreshold
    $env:TAGGS_DENSIFY_INTERVAL_MULT_FULLRES = $Profile.DensifyIntervalMult
    if (-not $env:TAGGS_DENSIFY_MIN_FREE_MB) { $env:TAGGS_DENSIFY_MIN_FREE_MB = "1500" }
    if (-not $env:TAGGS_DENSIFY_EMPTY_CACHE) { $env:TAGGS_DENSIFY_EMPTY_CACHE = "1" }

    foreach ($entry in $overrides.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, "Process")
    }

    try {
        & "$PSScriptRoot\run_bicycle_v15_fast_test_3080.ps1"
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $($Profile.Id) failed (exit $LASTEXITCODE)"
        }
    } finally {
        foreach ($entry in $overrides.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $null, "Process")
        }
        foreach ($key in @(
            "CASE_NAME", "CONFIG_LABEL", "STAGE_SUMMARY_JSON", "WRITE_STAGE_JSON",
            "DENSIFY_UNTIL_FULL", "SPA_PRUNE_RATIO", "PRUNE_RATIO2",
            "TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD", "TAGGS_DENSIFY_INTERVAL_MULT_FULLRES"
        )) {
            [Environment]::SetEnvironmentVariable($key, $null, "Process")
        }
    }

    return $stageSummaryJson
}

function Write-ExperimentGateHints {
    param(
        [string]$StageSummaryMd,
        [string]$ExperimentId = ""
    )

    Write-Host ""
    Write-Host "=== Gate hints (manual check) ==="
    Write-Host "  18k: PSNR >= 23.9 dB, vs 7k drop <= 0.3 dB, regressed views < 20/25"
    Write-Host "  24k: PSNR >= 24.3 dB (recover to early-stage level)"
    if ($ExperimentId -eq "E1b") {
        Write-Host "  30k: PSNR >= 26.0 dB stretch (vs 12k 26.3053), LPIPS <= 0.175"
        Write-Host "  7.2k: check no large dip at densify cutoff (compare E1 12k dip -2.06 dB)"
    } else {
        Write-Host "  30k: PSNR >= 25.8 dB, LPIPS <= 0.19 (stretch vs 12k 26.3053 / 0.1747)"
    }
    if (Test-Path $StageSummaryMd) {
        Write-Host "  Review: $StageSummaryMd"
    }
}

Write-Host "=== bicycle v15 30k densify-fix (RTX 3080) ==="
Write-Host "EXPERIMENT:   $EXPERIMENT"
Write-Host "VARIANT:      $VARIANT (late_stabilize only in this script)"
Write-Host "PROTOCOL:     images_4 @ resolution=800"
Write-Host "ITERATIONS:   $ITERATIONS"
Write-Host "RESOLUTION:   $RESOLUTION"
Write-Host "IMAGES:       $IMAGES"
Write-Host "STAGES:       $RAW_STAGE_ITERATIONS"
Write-Host "Compare to:   output/v16_local_3080 (30k PSNR 26.91)"
Write-Host "E1 baseline:  output/v15_30k_fix_e1_densify12k (30k PSNR 25.8548)"
Write-Host "E1b current:  output/v15_30k_fix_e1b_densify7k (30k PSNR 26.69)"
Write-Host ""

if ($RESOLUTION -ne "800" -or $IMAGES -ne "images_4") {
    Write-Host "NOTE: protocol softened to images=$IMAGES resolution=$RESOLUTION"
    Write-Host ""
}

$experimentIds = switch ($EXPERIMENT) {
    "ALL" { @("E0", "E1") }
    default { @($EXPERIMENT) }
}

$stageJsons = @()
foreach ($id in $experimentIds) {
    $profile = Get-ExperimentProfile $id
    $stageJson = Invoke-Experiment $profile
    $stageJsons += $stageJson
    Write-ExperimentGateHints (Join-Path (Join-Path $ROOT $profile.OutputRoot) "${profile.CaseName}_stage_selection_summary.md") $profile.Id
}

$existing = $stageJsons | Where-Object { Test-Path $_ }
if ($existing.Count -gt 1) {
    $promotionRoot = Join-Path $ROOT "output\v15_30k_fix"
    New-Item -ItemType Directory -Force -Path $promotionRoot | Out-Null
    $promotionMd = Join-Path $promotionRoot "promotion_summary.md"
    $promotionJson = Join-Path $promotionRoot "promotion_summary.json"
    python "scripts\summarize_v15_promotion.py" `
        --stage_jsons $existing `
        --out_md $promotionMd `
        --out_json $promotionJson
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Combined promotion summary: $promotionMd"
    } else {
        Write-Warning "Combined promotion summary incomplete."
    }
}

Write-Host ""
Write-Host "Done."
Write-Host "Next:"
Write-Host "  1) Open *_stage_selection_summary.md and check 18k / 24k / 30k gates"
Write-Host "  2) Compare E1b vs E1, and optionally vs v16_local_3080"
Write-Host "  3) Check the case directory under output\v15_30k_fix_*"
