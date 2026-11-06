# =============================================================================
# v3 Group B: 模块消融实验 (5 组逐项启用 × 4 场景 × 5 seed = 100 runs)
# =============================================================================
# 1/2 分辨率, 逐项启用 Geo / SPA / Staged 模块
# 输出: output/exp4v3_B_{scene}/B.X_*/train/seed{N}/
# 汇总: output/exp4_stats/exp4v3_B_module_ablation.csv
#
# 用法:
#   .\scripts\run_exp4v3_B_module_ablation.ps1
#   $env:E4_SCENES = "bicycle,garden"
#   $env:E4_GROUPS = "B4"
#   .\scripts\run_exp4v3_B_module_ablation.ps1
#
#   # Smoke
#   $env:E4_SMOKE = "1"
#   .\scripts\run_exp4v3_B_module_ablation.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$dryRun       = if ($env:E4_DRY_RUN)        { $env:E4_DRY_RUN -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# 5 组逐项启用 (短名, 与 Python argparse default="B0,B1,B2,B3,B4" 一致)
$ALL_GROUPS = @("B0", "B1", "B2", "B3", "B4")
$selectedGroups = if ($env:E4_GROUPS) {
    $env:E4_GROUPS.Split(",") | ForEach-Object { $_.Trim() }
} else {
    $ALL_GROUPS
}

# 4 场景
$ALL_SCENES = @("bicycle", "garden", "bonsai", "room")
$selectedScenes = if ($env:E4_SCENES) {
    $env:E4_SCENES.Split(",") | ForEach-Object { $_.Trim() }
} else {
    $ALL_SCENES
}

$seedList = if ($env:E4_SEEDS) {
    $env:E4_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    @(0)
}

# v3: 主实验统一采用 1/2 分辨率
$DEFAULT_IMAGES_DIR = if ($env:E4_IMAGES_DIR) { $env:E4_IMAGES_DIR } else { "images_2" }
if ([string]::IsNullOrEmpty($DEFAULT_IMAGES_DIR)) { $DEFAULT_IMAGES_DIR = "images_2" }

# 调用 Python 脚本
$pyArgs = @('scripts\run_exp4v3_B_module_ablation.py', '--scenes', ($selectedScenes -join ','), '--groups', ($selectedGroups -join ','), '--seeds', ($seedList -join ','), '--iterations', "$iterations", '--images-dir', $DEFAULT_IMAGES_DIR, '--output-root', 'output')
if ($skipCompleted) { $pyArgs += '--skip-existing' }
if ($smoke) { $pyArgs += '--dry-run' }
if ($dryRun) { $pyArgs += '--dry-run' }

# ---- wandb 集成 (env 透传给子进程) ----
$wandbEnabled = if ($env:WANDB_ENABLED) { $env:WANDB_ENABLED -ne "0" } else { $false }
$resTag = if ($DEFAULT_IMAGES_DIR -eq "images") { "std_res" } elseif ($DEFAULT_IMAGES_DIR -eq "images_2") { "1/2_res" } elseif ($DEFAULT_IMAGES_DIR -eq "images_4") { "1/4_res" } else { $DEFAULT_IMAGES_DIR }
if ($wandbEnabled) {
    if (-not $env:WANDB_PROJECT) { $env:WANDB_PROJECT = "tags-ablation" }
    if (-not $env:WANDB_RUN_GROUP) { $env:WANDB_RUN_GROUP = "exp4v3_B_module_ablation" }
    if (-not $env:WANDB_TAGS) { $env:WANDB_TAGS = "v3,$resTag,$($seedList -join '_')-seed" }
    Write-Host "[v3 Group B] Wandb ON  project=$env:WANDB_PROJECT group=$env:WANDB_RUN_GROUP tags=$env:WANDB_TAGS"
} else {
    Write-Host "[v3 Group B] Wandb OFF (set `$env:WANDB_ENABLED=1 to enable)"
}

$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& $python @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Warning "[v3 Group B] Python script exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if (-not $smoke -and -not $dryRun) {
    Write-Host "[v3 Group B] Summarizing..."
    & $python "scripts\run_exp4v3_B_module_ablation.py" `
        --scenes "bicycle,garden,bonsai,room" `
        --groups "B0,B1,B2,B3,B4" `
        --seeds "0" `
        --summarize `
        --output-root "output"
}

Write-Host "[v3 Group B] All runs complete."
