# v15 30k densify-fix experiments — optional env loader
#
# Usage (repo root):
#   . .\scripts\configs\bicycle_v15_30k_densify_fix.env.ps1
#   .\scripts\run_bicycle_v15_30k_densify_fix.ps1
#
# Select experiment (default E1):
#   $env:EXPERIMENT = "E0"   # diagnostic baseline, densify until 18k
#   $env:EXPERIMENT = "E1"   # early stop densify at 12k
#   $env:EXPERIMENT = "E1b"  # align densify until 7.2k (recommended next)
#   $env:EXPERIMENT = "E2"   # E1 + tighter point-growth gate
#   $env:EXPERIMENT = "E3"   # E1 + stronger SPA prune
#   $env:EXPERIMENT = "E4"   # E1 + lighter late LPIPS
#   $env:EXPERIMENT = "all"  # run E0 then E1 (skip if already completed)
#
# OOM fallback:
#   $env:RESOLUTION = "640"; $env:IMAGES = "images_2"

foreach ($name in @(
    "OUTPUT_ROOT", "ITERATIONS", "RESOLUTION", "IMAGES", "RAW_STAGE_ITERATIONS",
    "CASE_NAME", "ONLY_CASE", "VARIANT", "CONFIG_LABEL", "STAGE_SUMMARY_JSON",
    "WRITE_STAGE_JSON", "EXPERIMENT", "DENSIFY_UNTIL_FULL", "DENSIFY_FROM_FULL",
    "SPA_PRUNE_RATIO", "PRUNE_RATIO2",
    "TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD", "TAGGS_DENSIFY_INTERVAL_MULT_FULLRES",
    "LATE_STABILIZE_LPIPS_START_FULL", "LATE_STABILIZE_LPIPS_WEIGHT_FINAL"
)) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:EXPERIMENT = if ($env:EXPERIMENT) { $env:EXPERIMENT.Trim() } else { "E1" }
$env:VARIANT = "late_stabilize"
$env:ITERATIONS = "30000"
$env:RESOLUTION = if ($env:RESOLUTION) { $env:RESOLUTION } else { "800" }
$env:IMAGES = if ($env:IMAGES) { $env:IMAGES } else { "images_4" }
$env:RUN_TRAIN = if ($env:RUN_TRAIN) { $env:RUN_TRAIN } else { "1" }
$env:RUN_RENDER_METRICS = if ($env:RUN_RENDER_METRICS) { $env:RUN_RENDER_METRICS } else { "1" }
$env:SKIP_COMPLETED = if ($env:SKIP_COMPLETED) { $env:SKIP_COMPLETED } else { "1" }

# Dense mid-stage checkpoints to localize 7k->18k collapse.
$env:RAW_STAGE_ITERATIONS = "7000,12000,14000,15000,16000,17000,18000,20000,24000,27000,28500,30000"

$env:PSNR_DROP_THRESHOLD = "0.05"
$env:PERCEPTUAL_PSNR_BUDGET = "0.10"
$env:CHRONIC_TRIPLE_LIMIT = "1"

# late_stabilize overrides (same as 12k-protocol 30k winner recipe)
$env:LATE_STABILIZE_LR_GEO_MULT_STAGE3 = "0.40"
$env:LATE_STABILIZE_LR_COLOR_MULT_STAGE3 = "0.55"
$env:LATE_STABILIZE_LPIPS_START_FULL = "24000"
$env:LATE_STABILIZE_LPIPS_WEIGHT_FINAL = "0.006"
$env:LATE_STABILIZE_LPIPS_MAX_WEIGHT = "0.010"
$env:LATE_STABILIZE_LPIPS_INTERVAL = "10"
$env:LATE_STABILIZE_COLOR_ALIGN_START_FULL = "25000"
$env:LATE_STABILIZE_COLOR_ALIGN_WEIGHT_FINAL = "0.0010"
$env:LATE_STABILIZE_COLOR_ALIGN_MAX_WEIGHT = "0.0020"
$env:LATE_STABILIZE_PSNR_START_FULL = "22000"
$env:LATE_STABILIZE_PSNR_WEIGHT_FINAL = "0.008"
$env:LATE_STABILIZE_PSNR_MAX_WEIGHT = "0.012"

Write-Host "Loaded densify-fix env: EXPERIMENT=$($env:EXPERIMENT)"
Write-Host "  IMAGES=$($env:IMAGES) RESOLUTION=$($env:RESOLUTION) ITERATIONS=$($env:ITERATIONS)"
