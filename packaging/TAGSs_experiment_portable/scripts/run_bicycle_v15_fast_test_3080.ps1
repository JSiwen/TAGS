# Shared RTX 3080 bicycle trainer (env-configured worker).
# Also used by run_bicycle_v15_30k_densify_fix.ps1 — do not delete.
#
# Default standalone mode: scaled-iteration low-VRAM smoke test.
#
# Usage (PowerShell, repo root):
#   .\scripts\run_bicycle_v15_fast_test_3080.ps1
#
# Tiers:
#   FAST_TIER=6k   ~2-4h  (default, recommended)
#   FAST_TIER=8k   ~4-6h  (more stable late-stage trend)
#
# Custom:
#   $env:ITERATIONS = "6000"
#   $env:RESOLUTION = "640"
#   $env:IMAGES = "images_2"
#   .\scripts\run_bicycle_v15_fast_test_3080.ps1
#
# Skip training (render/metrics/summary only):
#   $env:RUN_TRAIN = "0"; .\scripts\run_bicycle_v15_fast_test_3080.ps1

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

function Invoke-TrainPython {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CommandArgs
    )
    if (-not $env:KMP_DUPLICATE_LIB_OK) { $env:KMP_DUPLICATE_LIB_OK = "TRUE" }
    $condaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "gaussian_splatting" }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($env:CONDA_DEFAULT_ENV -eq $condaEnv) {
            & python @CommandArgs 2>&1
            return $LASTEXITCODE
        }
        if (Get-Command conda -ErrorAction SilentlyContinue) {
            & conda run -n $condaEnv --no-capture-output python @CommandArgs 2>&1
            return $LASTEXITCODE
        }
        & python @CommandArgs 2>&1
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Test-PythonSnippet {
    param([string]$Code)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & python -c $Code 2>&1
        return @{
            ExitCode = $LASTEXITCODE
            Output = ($out | Out-String).Trim()
        }
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Assert-TrainEnvironment {
    $checks = @()

    if (-not $env:CONDA_PREFIX) {
        $checks += "Conda env not active. Run: conda activate gaussian_splatting"
    }

    $torch = Test-PythonSnippet "import torch; print(torch.__version__)"
    if ($torch.ExitCode -ne 0) {
        if ($torch.Output -match "4551|AppControl|应用程序控制|fbgemm") {
            $checks += @(
                "PyTorch blocked by Windows App Control (fbgemm.dll).",
                "  -> Run: .\scripts\test_train_environment.ps1",
                "  -> Or disable Smart App Control / use WSL2 Linux env"
            )
        } else {
            $checks += "PyTorch import failed: $($torch.Output)"
        }
    }

    $pillow = Test-PythonSnippet "from PIL import Image; print(Image.__version__)"
    if ($pillow.ExitCode -ne 0) {
        $checks += "Pillow import failed. Run: .\scripts\fix_gaussian_splatting_pillow.ps1"
    }

    if ($checks.Count -gt 0) {
        Write-Host ""
        Write-Host "Environment preflight failed:"
        foreach ($line in $checks) { Write-Host $line }
        throw "Training environment is not ready."
    }
}

function Scale-Iter {
    param([int]$FullIter)
    return [Math]::Max(1, [int][Math]::Round($FullIter * $script:ITERATIONS / $script:FULL_ITERATIONS))
}

function Get-EnvValue {
    param([string]$Name, [string]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    return $Default
}

function Get-ScaledFullIterEnv {
    param([string]$Name, [int]$DefaultFullIter)
    return Scale-Iter ([int](Get-EnvValue $Name "$DefaultFullIter"))
}

function Get-FloatEnv {
    param([string]$Name, [string]$Default)
    return Get-EnvValue $Name $Default
}

function Parse-StageIterations {
    param([string]$Spec, [bool]$ScaleFromFull)
    $items = $Spec -split '[,;\s]+' | Where-Object { $_ -and $_.Trim() }
    if (-not $items -or $items.Count -eq 0) {
        Write-Error "No iterations found in stage iteration spec: '$Spec'"
    }
    $iters = foreach ($item in $items) {
        $iter = [int]$item
        if ($ScaleFromFull) { Scale-Iter $iter } else { $iter }
    }
    return $iters | Sort-Object -Unique
}

$FULL_ITERATIONS = 30000
$FAST_TIER = if ($env:FAST_TIER) { $env:FAST_TIER.ToLower() } else { "6k" }
$ITERATIONS = if ($env:ITERATIONS) {
    [int]$env:ITERATIONS
} elseif ($FAST_TIER -eq "8k") {
    8000
} elseif ($FAST_TIER -eq "6k") {
    6000
} else {
    Write-Error "Unknown FAST_TIER='$FAST_TIER'. Use 6k or 8k, or set ITERATIONS explicitly."
}

$SOURCE = if ($env:SOURCE) { $env:SOURCE } else { Join-Path $ROOT "data\bicycle" }
$DEPTHS = if ($env:DEPTHS) { $env:DEPTHS } else { Join-Path $ROOT "data\bicycle\depth_any_out" }
$OUTPUT_ROOT = if ($env:OUTPUT_ROOT) { $env:OUTPUT_ROOT } else { Join-Path $ROOT "output\v15_fast_3080" }
$RESOLUTION = if ($env:RESOLUTION) { $env:RESOLUTION } else { "800" }
$IMAGES = if ($env:IMAGES) { $env:IMAGES } else { "images_4" }
$RUN_TRAIN = if ($env:RUN_TRAIN) { $env:RUN_TRAIN } else { "1" }
$RUN_RENDER_METRICS = if ($env:RUN_RENDER_METRICS) { $env:RUN_RENDER_METRICS } else { "1" }
$SKIP_COMPLETED = if ($env:SKIP_COMPLETED) { $env:SKIP_COMPLETED } else { "1" }
$CASE_NAME = if ($env:CASE_NAME) { $env:CASE_NAME } else { "bicycle_v15_fast_${ITERATIONS}" }

if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "0" }
if (-not $env:TAGGS_CPU_KNN_INIT) { $env:TAGGS_CPU_KNN_INIT = "1" }
if (-not $env:TAGGS_DENSIFY_MIN_FREE_MB) { $env:TAGGS_DENSIFY_MIN_FREE_MB = "1500" }
if (-not $env:TAGGS_DENSIFY_EMPTY_CACHE) { $env:TAGGS_DENSIFY_EMPTY_CACHE = "1" }
if (-not $env:TAGGS_DENSIFY_INTERVAL_MULT_FULLRES) { $env:TAGGS_DENSIFY_INTERVAL_MULT_FULLRES = "2" }
if (-not $env:TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD) { $env:TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD = "50000" }
if (-not $env:KMP_DUPLICATE_LIB_OK) { $env:KMP_DUPLICATE_LIB_OK = "TRUE" }

$densifyFrom = Get-ScaledFullIterEnv "DENSIFY_FROM_FULL" 500
$densifyUntil = Get-ScaledFullIterEnv "DENSIFY_UNTIL_FULL" 18000
$geoWarmupUntil = Get-ScaledFullIterEnv "GEO_WARMUP_UNTIL_FULL" 5000
$geoDecayStart = Get-ScaledFullIterEnv "GEO_DECAY_START_FULL" 12000
$geoDecayUntil = Get-ScaledFullIterEnv "GEO_DECAY_UNTIL_FULL" 25000
$lpipsStart = Get-ScaledFullIterEnv "LPIPS_START_FULL" 16000
$colorAlignStart = Get-ScaledFullIterEnv "COLOR_ALIGN_START_FULL" 26000
$spaStart = Get-ScaledFullIterEnv "SPA_START_FULL" 13000
$spaEnd = Get-ScaledFullIterEnv "SPA_END_FULL" 25000
$lrStage1End = Get-ScaledFullIterEnv "LR_STAGE1_END_FULL" 6000
$lrStage2End = Get-ScaledFullIterEnv "LR_STAGE2_END_FULL" 18000
$lrGeoMultStage1 = Get-FloatEnv "LR_GEO_MULT_STAGE1" "1.10"
$lrGeoMultStage2 = Get-FloatEnv "LR_GEO_MULT_STAGE2" "1.00"
$lrGeoMultStage3 = Get-FloatEnv "LR_GEO_MULT_STAGE3" "0.75"
$lrColorMultStage1 = Get-FloatEnv "LR_COLOR_MULT_STAGE1" "0.75"
$lrColorMultStage2 = Get-FloatEnv "LR_COLOR_MULT_STAGE2" "1.05"
$lrColorMultStage3 = Get-FloatEnv "LR_COLOR_MULT_STAGE3" "0.80"
$lpipsWeightFinal = Get-FloatEnv "LPIPS_WEIGHT_FINAL" "0.01"
$lpipsMaxWeight = Get-FloatEnv "LPIPS_MAX_WEIGHT" "0.02"
$lpipsInterval = Get-EnvValue "LPIPS_INTERVAL" "4"
$psnrStart = Get-ScaledFullIterEnv "PSNR_START_FULL" 0
$psnrWeightFinal = Get-FloatEnv "PSNR_WEIGHT_FINAL" "0.0"
$psnrMaxWeight = Get-FloatEnv "PSNR_MAX_WEIGHT" "0.05"
$colorAlignWeightFinal = Get-FloatEnv "COLOR_ALIGN_WEIGHT_FINAL" "0.0015"
$colorAlignMaxWeight = Get-FloatEnv "COLOR_ALIGN_MAX_WEIGHT" "0.004"
$colorAlignStdWeight = Get-FloatEnv "COLOR_ALIGN_STD_WEIGHT" "0.25"
$spaPruneRatio = Get-FloatEnv "SPA_PRUNE_RATIO" "0.05"
$pruneRatio2 = Get-FloatEnv "PRUNE_RATIO2" "0.04"
$liveLogInterval = Get-EnvValue "LIVE_LOG_INTERVAL" "1"
$schedule = "{0}:4,{1}:2,{2}:1" -f (Scale-Iter 1200), (Scale-Iter 4000), $ITERATIONS

if ($env:RAW_STAGE_ITERATIONS) {
    $stageIters = Parse-StageIterations $env:RAW_STAGE_ITERATIONS $false
} elseif ($env:STAGE_ITERATIONS) {
    $stageIters = Parse-StageIterations $env:STAGE_ITERATIONS $true
} else {
    $stageIters = @(
        (Scale-Iter 7000),
        (Scale-Iter 18000),
        (Scale-Iter 24000),
        (Scale-Iter 25500),
        (Scale-Iter 28500),
        $ITERATIONS
    ) | Sort-Object -Unique
}

$primaryCheckpoint = Scale-Iter 28500
$finalCheckpoint = $ITERATIONS
$modelDir = Join-Path $OUTPUT_ROOT $CASE_NAME
$logDir = Join-Path $ROOT "output\v15_logs_fast_3080"
$logPath = Join-Path $logDir "$CASE_NAME.log"
$doneMarker = Join-Path $modelDir ".v15_fast_3080_done"
$summaryJson = Join-Path $OUTPUT_ROOT "${CASE_NAME}_checkpoint_summary.json"
$summaryMd = Join-Path $OUTPUT_ROOT "${CASE_NAME}_checkpoint_summary.md"
$stageSummaryMd = Join-Path $OUTPUT_ROOT "${CASE_NAME}_stage_selection_summary.md"
$stageSummaryJson = if ($env:STAGE_SUMMARY_JSON) {
    $env:STAGE_SUMMARY_JSON
} else {
    Join-Path $OUTPUT_ROOT "${CASE_NAME}_stage_selection_summary.json"
}
$WRITE_STAGE_JSON = if ($env:WRITE_STAGE_JSON) { $env:WRITE_STAGE_JSON } else { "1" }
$configLabel = if ($env:CONFIG_LABEL) {
    $env:CONFIG_LABEL
} else {
    "images=$IMAGES, resolution=$RESOLUTION, iterations=$ITERATIONS, RTX 3080"
}
$psnrDropThreshold = if ($env:PSNR_DROP_THRESHOLD) { $env:PSNR_DROP_THRESHOLD } else { "0.05" }
$perceptualPsnrBudget = if ($env:PERCEPTUAL_PSNR_BUDGET) { $env:PERCEPTUAL_PSNR_BUDGET } else { "0.10" }
$chronicTripleLimit = if ($env:CHRONIC_TRIPLE_LIMIT) { $env:CHRONIC_TRIPLE_LIMIT } else { "1" }
$stageTraceSpec = ($stageIters | ForEach-Object { "$_" }) -join ","

Write-Host "=== bicycle v15 fast test (RTX 3080) ==="
Write-Host "FAST_TIER:           $FAST_TIER"
Write-Host "CASE_NAME:           $CASE_NAME"
Write-Host "SOURCE:              $SOURCE"
Write-Host "DEPTHS:              $DEPTHS"
Write-Host "OUTPUT_ROOT:         $OUTPUT_ROOT"
Write-Host "MODEL:               $modelDir"
Write-Host "ITERATIONS:          $ITERATIONS (scaled from $FULL_ITERATIONS)"
Write-Host "RESOLUTION:          $RESOLUTION"
Write-Host "IMAGES:              $IMAGES"
Write-Host "CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES"
Write-Host "28500 proxy:         ours_$primaryCheckpoint"
Write-Host "30000 proxy:         ours_$finalCheckpoint"
Write-Host "STAGE_ITERATIONS:    $($stageIters -join ', ')"
Write-Host "SCHEDULE:            $schedule"
Write-Host "DENSIFY:             $densifyFrom -> $densifyUntil"
Write-Host "SPA:                 $spaStart -> $spaEnd, prune=$spaPruneRatio/$pruneRatio2"
Write-Host "LPIPS:               start=$lpipsStart, weight=$lpipsWeightFinal, max=$lpipsMaxWeight, interval=$lpipsInterval"
Write-Host "PSNR:                start=$psnrStart, weight=$psnrWeightFinal, max=$psnrMaxWeight"
Write-Host "GROUP_LR:            s1=$lrStage1End, s2=$lrStage2End, geo=[$lrGeoMultStage1,$lrGeoMultStage2,$lrGeoMultStage3], color=[$lrColorMultStage1,$lrColorMultStage2,$lrColorMultStage3]"
Write-Host "LIVE_LOG:            interval=$liveLogInterval, stages=$stageTraceSpec"
Write-Host ""

if (-not (Test-Path $SOURCE)) {
    Write-Error "SOURCE not found: $SOURCE"
}
if (-not (Test-Path (Join-Path $SOURCE "sparse\0\cameras.bin"))) {
    Write-Error "COLMAP sparse/0 not found under $SOURCE"
}
if (-not (Test-Path $DEPTHS)) {
    Write-Error "DEPTHS not found: $DEPTHS"
}

New-Item -ItemType Directory -Force -Path $OUTPUT_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ($RUN_TRAIN -eq "1") {
    Assert-TrainEnvironment
}

function Test-CaseHasMetrics {
    param([string]$Dir)
    $results = Join-Path $Dir "results.json"
    if (-not (Test-Path $results)) { return $false }
    $content = Get-Content $results -Raw | ConvertFrom-Json
    $primaryMethod = "ours_$primaryCheckpoint"
    $finalMethod = "ours_$finalCheckpoint"
    return ($null -ne $content.$primaryMethod) -and ($null -ne $content.$finalMethod)
}

if ($SKIP_COMPLETED -eq "1" -and ((Test-Path $doneMarker) -or (Test-CaseHasMetrics $modelDir))) {
    Write-Host "Skip training (already completed): $CASE_NAME"
} elseif ($RUN_TRAIN -eq "1") {
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    $trainArgs = @(
        "train_optimized.py",
        "-s", $SOURCE,
        "-m", $modelDir,
        "--images", $IMAGES,
        "--depths", $DEPTHS,
        "--resolution", $RESOLUTION,
        "--data_device", "cpu",
        "--eval",
        "--llffhold", "8",
        "--iterations", "$ITERATIONS",
        "--densify_from_iter", "$densifyFrom",
        "--densify_until_iter", "$densifyUntil",
        "--densify_grad_threshold", "0.00016",
        "--enable_geo_prior",
        "--geo_prior_depth",
        "--geo_prior_normal",
        "--depth_prior_weight_final", "0.05",
        "--normal_prior_weight_final", "0.02",
        "--normal_prior_tv_weight", "0.003",
        "--geo_prior_start_iter", "0",
        "--geo_prior_warmup_until", "$geoWarmupUntil",
        "--geo_prior_decay_start", "$geoDecayStart",
        "--geo_prior_decay_until", "$geoDecayUntil",
        "--lpips_loss_start_iter", "$lpipsStart",
        "--lpips_loss_weight_final", "$lpipsWeightFinal",
        "--lpips_loss_max_weight", "$lpipsMaxWeight",
        "--lpips_loss_interval", "$lpipsInterval",
        "--psnr_loss_start_iter", "$psnrStart",
        "--psnr_loss_weight_final", "$psnrWeightFinal",
        "--psnr_loss_max_weight", "$psnrMaxWeight",
        "--color_align_start_iter", "$colorAlignStart",
        "--color_align_weight_final", "$colorAlignWeightFinal",
        "--color_align_max_weight", "$colorAlignMaxWeight",
        "--color_align_std_weight", "$colorAlignStdWeight",
        "--enable_resolution_adaptive",
        "--resolution_schedule", $schedule,
        "--enable_group_lr_schedule",
        "--lr_stage1_end", "$lrStage1End",
        "--lr_stage2_end", "$lrStage2End",
        "--lr_geo_mult_stage1", "$lrGeoMultStage1",
        "--lr_geo_mult_stage2", "$lrGeoMultStage2",
        "--lr_geo_mult_stage3", "$lrGeoMultStage3",
        "--lr_color_mult_stage1", "$lrColorMultStage1",
        "--lr_color_mult_stage2", "$lrColorMultStage2",
        "--lr_color_mult_stage3", "$lrColorMultStage3",
        "--enable_geometry_controller",
        "--enable_live_loss_logging",
        "--live_loss_log_interval", "$liveLogInterval",
        "--live_stage_iterations", "$stageTraceSpec",
        "--enable_sparse_optimization",
        "--sparse_mode", "spa",
        "--spa_start_iter", "$spaStart",
        "--spa_end_iter", "$spaEnd",
        "--spa_prune_ratio", "$spaPruneRatio",
        "--prune_ratio2", "$pruneRatio2",
        "--spa_metric", "hybrid",
        "--test_iterations"
    ) + ($stageIters | ForEach-Object { "$_" }) + @(
        "--save_iterations"
    ) + ($stageIters | ForEach-Object { "$_" }) + @(
        "--disable_viewer"
    )

    Write-Host "Training -> $modelDir"
    Write-Host "Log: $logPath"
    Write-Host "Watch progress in another terminal:"
    Write-Host "  .\scripts\watch_train_progress.ps1 -ModelDir `"$modelDir`" -FollowLog"
    if (Test-Path $logPath) {
        Remove-Item $logPath -Force -ErrorAction SilentlyContinue
    }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $trainExit = 0
    try {
        Invoke-TrainPython -CommandArgs $trainArgs 2>&1 | ForEach-Object {
            $line = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { "$_" }
            Write-Host $line
            Add-Content -Path $logPath -Value $line -Encoding utf8
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                if ($_.Exception.Message -match "Traceback|Error|Exception") {
                    # keep going so full traceback reaches log/console
                }
            }
        }
        $trainExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($trainExit -ne 0) {
        if (Test-Path $logPath) {
            Get-Content $logPath -Tail 30 | Write-Host
        }
        Write-Error "Training failed (exit $trainExit). See $logPath"
    }
}

if ($RUN_RENDER_METRICS -eq "1") {
    Write-Host ""
    Write-Host "=== Render stage checkpoints ==="
    foreach ($iter in $stageIters) {
        $ply = Join-Path $modelDir "point_cloud\iteration_$iter\point_cloud.ply"
        if (-not (Test-Path $ply)) {
            Write-Host "Skip render iteration $iter (missing point cloud)"
            continue
        }
        Write-Host "Render iteration $iter"
        Invoke-TrainPython -CommandArgs @(
            "render.py", "-m", $modelDir, "--iteration", "$iter", "--skip_train", "--quiet"
        )
        if ($LASTEXITCODE -ne 0) {
            Write-Error "render.py failed at iteration $iter"
        }
    }

    Write-Host ""
    Write-Host "=== Metrics ==="
    Invoke-TrainPython -CommandArgs @("metrics.py", "-m", $modelDir, "--split", "test")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "metrics.py failed for $CASE_NAME"
    }
}

Write-Host ""
Write-Host "=== Checkpoint summary ==="
Invoke-TrainPython -CommandArgs @(
    "scripts/summarize_v15_fast_checkpoint.py",
    "--model_dir", $modelDir,
    "--primary_iter", "$primaryCheckpoint",
    "--final_iter", "$finalCheckpoint",
    "--out_json", $summaryJson,
    "--out_md", $summaryMd
)
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Checkpoint summary incomplete. Check results.json after render/metrics."
} else {
    Write-Host "Summary: $summaryMd"
}

$stageArgs = @(
    "scripts/summarize_v15_stage_selection.py",
    "--model_dir", $modelDir,
    "--out_md", $stageSummaryMd,
    "--config_label", $configLabel,
    "--baseline_method", "ours_$(($stageIters | Select-Object -First 1))",
    "--final_method", "ours_$finalCheckpoint",
    "--psnr_drop_threshold", "$psnrDropThreshold",
    "--perceptual_psnr_budget", "$perceptualPsnrBudget",
    "--chronic_triple_limit", "$chronicTripleLimit"
)
if ($WRITE_STAGE_JSON -eq "1") {
    $stageArgs += @("--out_json", $stageSummaryJson)
}
Invoke-TrainPython -CommandArgs $stageArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stage selection summary incomplete. Check results.json/per_view.json after render/metrics."
} else {
    Write-Host "Stage summary: $stageSummaryMd"
    if ($WRITE_STAGE_JSON -eq "1") {
        Write-Host "Stage JSON:    $stageSummaryJson"
    }
}

New-Item -ItemType File -Force -Path $doneMarker | Out-Null

Write-Host ""
Write-Host "Done."
Write-Host "  Log:     $logPath"
Write-Host "  Model:   $modelDir"
Write-Host "  Summary: $summaryMd"
Write-Host "  Stage:   $stageSummaryMd"
Write-Host ""
Write-Host "OOM fallback:"
Write-Host '  $env:RESOLUTION="640"; $env:IMAGES="images_2"; .\scripts\run_bicycle_v15_fast_test_3080.ps1'
