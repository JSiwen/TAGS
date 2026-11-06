# =============================================================================
# v3 阶段 2: QA Baseline Check - no-pretrain 3DGS baseline 验证 PSNR 阈值 (1/2 分辨率)
# =============================================================================
# 跑 4 场景 × N seed (默认 5) × 30k iter, 验证 PSNR 达到 1/2 分辨率阈值
#
# 用法:
#   .\scripts\run_qa_baseline.ps1
#   $env:E4_SCENES = "bicycle"; $env:E4_SEEDS = "0"  ; .\scripts\run_qa_baseline.ps1
#   $env:E4_IMAGES_DIR = "images_4"  ; .\scripts\run_qa_baseline.ps1  (1/4 备选)
#   $env:E4_DRY_RUN = "1"           ; .\scripts\run_qa_baseline.ps1  (只打印命令)
#
# 输出:
#   output/exp4_stats/qa_baseline_results.csv
#   output/exp4_stats/baseline_thresholds.json
#   output/exp4_stats/qa_baseline_report.md
#   output/exp4_stats/qa_baseline_report.json

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$python      = if ($env:E4_PYTHON)      { $env:E4_PYTHON }      else { "python" }
$scenesEnv   = if ($env:E4_SCENES)      { $env:E4_SCENES }      else { "bicycle,garden,bonsai,room" }
$seedsEnv    = if ($env:E4_SEEDS)       { $env:E4_SEEDS }       else { "0,1,2,3,4" }
$imagesDir   = if ($env:E4_IMAGES_DIR)  { $env:E4_IMAGES_DIR }  else { "images_2" }
$iterations  = if ($env:E4_ITERATIONS)  { [int]$env:E4_ITERATIONS } else { 30000 }
$skipExisting = if ($env:E4_SKIP_EXISTING) { $env:E4_SKIP_EXISTING -ne "0" } else { $true }
$dryRun      = if ($env:E4_DRY_RUN)     { $env:E4_DRY_RUN -ne "0" } else { $false }
$outputDir   = if ($env:E4_OUTPUT)      { $env:E4_OUTPUT }      else { "output/exp4_stats" }

# wandb 集成 (qa_baseline 不直接上 wandb, 但每行结果可后续 sync)
$wandbEnabled = if ($env:WANDB_ENABLED) { $env:WANDB_ENABLED -ne "0" } else { $false }
$resTag = if ($imagesDir -eq "images") { "std_res" } elseif ($imagesDir -eq "images_2") { "1/2_res" } elseif ($imagesDir -eq "images_4") { "1/4_res" } else { $imagesDir }
if ($wandbEnabled) {
    if (-not $env:WANDB_PROJECT) { $env:WANDB_PROJECT = "tags-ablation" }
    if (-not $env:WANDB_RUN_GROUP) { $env:WANDB_RUN_GROUP = "exp4v3_qa_baseline" }
    if (-not $env:WANDB_TAGS) { $env:WANDB_TAGS = "v3,qa_baseline,$resTag" }
    Write-Host "  (wandb透传) project=$env:WANDB_PROJECT group=$env:WANDB_RUN_GROUP tags=$env:WANDB_TAGS"
}

# 环境变量
$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ---- 打印当前阶段信息 ----
Write-Host "============================================================"
Write-Host "[v3 Stage 2] QA Baseline Check (no-pretrain 3DGS)"
Write-Host "  Scenes    : $scenesEnv"
Write-Host "  Seeds     : $seedsEnv"
Write-Host "  images_dir: $imagesDir (v3: 1/2 分辨率)"
Write-Host "  Iterations: $iterations"
Write-Host "  Output    : $outputDir"
Write-Host "  Wandb     : $wandbEnabled"
Write-Host "============================================================"

# ---- 构造 python 命令 ----
$pyArgs = @('scripts\qa_baseline_check.py', '--scenes', $scenesEnv, '--seeds', $seedsEnv, '--images-dir', $imagesDir, '--iterations', "$iterations", '--output', $outputDir, '--python', $python)
if ($skipExisting) { $pyArgs += '--skip-existing' }

if ($dryRun) {
    Write-Host "[DRY-RUN] Command: $python $($pyArgs -join ' ')"
    exit 0
}

# ---- 跑 ----
& $python @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Warning "[v3 Stage 2] qa_baseline_check failed (exit=$LASTEXITCODE)"
    Write-Warning "  阈值未达标的场景需要回退到 Stage 1 调预处理"
    Write-Warning "  或改用 1/4 分辨率: `$env:E4_IMAGES_DIR = 'images_4'"
    exit $LASTEXITCODE
}

# ---- 读报告并展示 ----
$reportPath = Join-Path $ROOT "$outputDir\qa_baseline_report.md"
if (Test-Path $reportPath) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[v3 Stage 2] Report:"
    Get-Content $reportPath
    Write-Host "============================================================"
}

# ---- 提示下一步 ----
Write-Host ""
Write-Host "============================================================"
Write-Host "[v3 Stage 2] DONE"
Write-Host "  下一步 (Stage 4 - Group A):"
Write-Host "    .\scripts\run_exp4v3_A_method_comparison.ps1"
Write-Host "  详细说明: docs\EXP4_RUN_BOOK.md §4"
Write-Host "============================================================"
