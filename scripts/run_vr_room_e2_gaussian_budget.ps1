# E2: vr_room fixed-Gaussian-budget quality/scale experiment.
#
# Run the four fixed budgets from an activated gaussian_splatting environment:
#   .\scripts\run_vr_room_e2_gaussian_budget.ps1
#
# Run selected budgets:
#   $env:E2_BUDGETS = "1000000,1500000"; .\scripts\run_vr_room_e2_gaussian_budget.ps1
#
# The existing 5.27M run is reused as the unlimited control by the summarizer.
# Set E2_BUDGETS=0 explicitly only when a fresh unlimited rerun is needed.
#
# A budget of 0 means unlimited. Fixed-budget runs intentionally start from
# scratch; resuming a checkpoint above the requested budget is rejected.

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$budgets = if ($env:E2_BUDGETS) {
    $env:E2_BUDGETS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    @(1000000, 1500000, 2000000, 3000000)
}
$skipCompleted = if ($env:E2_SKIP_COMPLETED) { $env:E2_SKIP_COMPLETED -ne "0" } else { $true }
$python = if ($env:E2_PYTHON) { $env:E2_PYTHON } else { "python" }

function Get-BudgetLabel([int]$budget) {
    if ($budget -eq 0) { return "unlimited" }
    if ($budget % 1000000 -eq 0) { return "$($budget / 1000000)m" }
    $millions = ($budget / 1000000.0).ToString(
        "0.##", [System.Globalization.CultureInfo]::InvariantCulture
    )
    return "${millions}m"
}

foreach ($budget in $budgets) {
    if ($budget -lt 0) {
        throw "Gaussian budget must be >= 0, got $budget"
    }

    $label = Get-BudgetLabel $budget
    $modelDir = Join-Path $ROOT "output\vr_room_e2_budget_$label"
    $finalPly = Join-Path $modelDir "point_cloud\iteration_30000\point_cloud.ply"
    if ($skipCompleted -and (Test-Path $finalPly)) {
        Write-Host "[E2] Skip completed budget=$label at $modelDir"
        continue
    }

    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
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
        "--densify_grad_threshold", "0.00016",
        "--max_gaussians", "$budget",
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
        "--test_iterations", "7000", "10000", "15000", "18000", "24000", "30000",
        "--save_iterations", "30000",
        "--checkpoint_iterations", "15000", "24000", "30000",
        "--disable_viewer"
    )

    $env:TAGGS_CPU_KNN_INIT = "1"
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    $logPath = Join-Path $modelDir "e2_train_console.log"
    Write-Host "[E2] Starting budget=$label ($budget), output=$modelDir"
    # Windows PowerShell turns native stderr (including tqdm progress) into
    # ErrorRecord objects. Keep those non-terminating and judge by exit code.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath
        $trainExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    if ($trainExitCode -ne 0) {
        throw "E2 budget=$label failed with exit code $trainExitCode"
    }
}

Write-Host "[E2] Requested budget runs complete."
