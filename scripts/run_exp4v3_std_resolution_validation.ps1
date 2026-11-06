# =============================================================================
# v3 标准分辨率验证 (3 场景 × 完整 TAGS × 3 seed = 9 runs)
# =============================================================================
# 挑选代表性场景 (bicycle, bonsai, garden), 用完整 TAGSs 在标准分辨率下训练
# 验证 1/2 分辨率得出的结论是否能扩展到标准分辨率
#
# 输出: output/exp4v3_std_{scene}/Std_FullTAGS/train/seed{N}/
# 汇总: output/exp4_stats/exp4v3_std_resolution_validation.csv
#
# 用法:
#   .\scripts\run_exp4v3_std_resolution_validation.ps1
#   $env:E4_SCENES = "bicycle"  # 单场景
#   $env:E4_SEEDS = "0"  # 单 seed
#   .\scripts\run_exp4v3_std_resolution_validation.ps1
#
#   # Smoke
#   $env:E4_SMOKE = "1"
#   .\scripts\run_exp4v3_std_resolution_validation.ps1
#
# 注意: 标准分辨率需要 ~3-4x 显存/时间 vs 1/2 分辨率
#       建议先跑完 1/2 主实验再决定是否跑这个

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$dryRun       = if ($env:E4_DRY_RUN)        { $env:E4_DRY_RUN -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# 2-3 个代表性场景
$ALL_SCENES = @("bicycle", "bonsai", "garden")
$selectedScenes = if ($env:E4_SCENES) {
    $env:E4_SCENES.Split(",") | ForEach-Object { $_.Trim() }
} else {
    $ALL_SCENES
}

# 3 seed (减半)
$seedList = if ($env:E4_SEEDS) {
    $env:E4_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    @(0, 1, 2)
}

# 标准分辨率
$DEFAULT_IMAGES_DIR = if ($env:E4_IMAGES_DIR) { $env:E4_IMAGES_DIR } else { "images" }
if ([string]::IsNullOrEmpty($DEFAULT_IMAGES_DIR)) { $DEFAULT_IMAGES_DIR = "images" }

# 调用 Python 脚本
$pyArgs = @('scripts\run_exp4v3_std_resolution_validation.py', '--scenes', ($selectedScenes -join ','), '--seeds', ($seedList -join ','), '--iterations', "$iterations", '--images-dir', $DEFAULT_IMAGES_DIR, '--output-root', 'output')
if ($skipCompleted) { $pyArgs += '--skip-existing' }
if ($smoke) { $pyArgs += '--dry-run' }
if ($dryRun) { $pyArgs += '--dry-run' }

# ---- wandb 集成 (env 透传给子进程) ----
$wandbEnabled = if ($env:WANDB_ENABLED) { $env:WANDB_ENABLED -ne "0" } else { $false }
$resTag = if ($DEFAULT_IMAGES_DIR -eq "images") { "std_res" } elseif ($DEFAULT_IMAGES_DIR -eq "images_2") { "1/2_res" } elseif ($DEFAULT_IMAGES_DIR -eq "images_4") { "1/4_res" } else { $DEFAULT_IMAGES_DIR }
if ($wandbEnabled) {
    if (-not $env:WANDB_PROJECT) { $env:WANDB_PROJECT = "tags-ablation" }
    if (-not $env:WANDB_RUN_GROUP) { $env:WANDB_RUN_GROUP = "exp4v3_std_resolution_validation" }
    if (-not $env:WANDB_TAGS) { $env:WANDB_TAGS = "v3,$resTag,$($seedList -join '_')-seed" }
    Write-Host "[v3 std] Wandb ON  project=$env:WANDB_PROJECT group=$env:WANDB_RUN_GROUP tags=$env:WANDB_TAGS"
} else {
    Write-Host "[v3 std] Wandb OFF (set `$env:WANDB_ENABLED=1 to enable)"
}

$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& $python @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Warning "[v3 std] Python script exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if (-not $smoke -and -not $dryRun) {
    Write-Host "[v3 std] Summarizing..."
    & $python "scripts\run_exp4v3_std_resolution_validation.py" `
        --scenes "bicycle,bonsai,garden" `
        --seeds "0" `
        --summarize `
        --output-root "output"
}

Write-Host "[v3 std] All runs complete."
