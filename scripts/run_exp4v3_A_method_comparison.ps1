# =============================================================================
# v3 Group A: 主要方法对比 (4 方法 × 4 场景 × 5 seed = 80 runs)
# =============================================================================
# 1/2 分辨率, 4 方法, 4 场景, 5 seed
# 输出: output/exp4v3_A_{scene}/A.X_{method}/train/seed{N}/
# 汇总: output/exp4_stats/exp4v3_A_method_comparison.csv
#
# 用法:
#   .\scripts\run_exp4v3_A_method_comparison.ps1
#   $env:E4_SCENES = "bicycle,garden"
#   $env:E4_GROUPS = "A4"
#   .\scripts\run_exp4v3_A_method_comparison.ps1
#
#   # Smoke
#   $env:E4_SMOKE = "1"
#   .\scripts\run_exp4v3_A_method_comparison.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume   = if ($env:E4_AUTO_RESUME)   { $env:E4_AUTO_RESUME   -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$dryRun       = if ($env:E4_DRY_RUN)        { $env:E4_DRY_RUN -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# 4 方法 (PS1 端用短名 A1/A2/A3/A4 匹配 Python GROUPS 字典 key)
$ALL_GROUPS = @("A1", "A2", "A3", "A4")
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

# 2026-10-17 修订: 统一 --images images_2, --resolution 控制物理像素
$DEFAULT_IMAGES_DIR = if ($env:E4_IMAGES_DIR) { $env:E4_IMAGES_DIR } else { "images_2" }
if ([string]::IsNullOrEmpty($DEFAULT_IMAGES_DIR)) { $DEFAULT_IMAGES_DIR = "images_2" }
$DEFAULT_RESOLUTION = if ($env:E4_RESOLUTION) { [int]$env:E4_RESOLUTION } else { 2 }
if ($DEFAULT_RESOLUTION -notin @(1,2,4,8)) { throw "E4_RESOLUTION must be 1, 2, 4 or 8 (got $DEFAULT_RESOLUTION)" }

# 调用 Python 脚本
$pyArgs = @('scripts\run_exp4v3_A_method_comparison.py', '--scenes', ($selectedScenes -join ','), '--groups', ($selectedGroups -join ','), '--seeds', ($seedList -join ','), '--iterations', "$iterations", '--images-dir', $DEFAULT_IMAGES_DIR, '--resolution', "$DEFAULT_RESOLUTION", '--output-root', 'output')
if ($skipCompleted) { $pyArgs += '--skip-existing' }
if ($smoke) { $pyArgs += "--dry-run" }  # smoke 模式只打印不实际跑
if ($dryRun) { $pyArgs += "--dry-run" }  # dry-run 也只打印不跑

# ---- wandb 集成 (env 透传给子进程) ----
$wandbEnabled = if ($env:WANDB_ENABLED) { $env:WANDB_ENABLED -ne "0" } else { $false }
# 2026-10-17 修订: tag 由 --resolution 决定 (1/2/4/8 -> 1/2/1/4/1/8/1/16 物理)
$resTag = switch ($DEFAULT_RESOLUTION) {
    1 { "1/2_res" }   # 1/2 物理像素
    2 { "1/4_res" }   # 1/4 物理像素 (主实验默认)
    4 { "1/8_res" }   # 1/8 物理像素
    8 { "1/16_res" }  # 1/16 物理像素
    default { "res_$DEFAULT_RESOLUTION" }
}
if ($wandbEnabled) {
    if (-not $env:WANDB_PROJECT) { $env:WANDB_PROJECT = "tags-ablation" }
    if (-not $env:WANDB_RUN_GROUP) { $env:WANDB_RUN_GROUP = "exp4v3_A_method_comparison" }
    if (-not $env:WANDB_TAGS) { $env:WANDB_TAGS = "v3,$resTag,$($seedList -join '_')-seed" }
    Write-Host "[v3 Group A] Wandb ON  project=$env:WANDB_PROJECT group=$env:WANDB_RUN_GROUP tags=$env:WANDB_TAGS"
} else {
    Write-Host "[v3 Group A] Wandb OFF (set `$env:WANDB_ENABLED=1 to enable)"
}

# 完成后做汇总
$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& $python @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Warning "[v3 Group A] Python script exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if (-not $smoke -and -not $dryRun) {
    # 跑汇总
    Write-Host "[v3 Group A] Summarizing..."
    & $python "scripts\run_exp4v3_A_method_comparison.py" `
        --scenes "bicycle,garden,bonsai,room" `
        --groups "A1,A2,A3,A4" `
        --seeds "0" `
        --summarize `
        --output-root "output"
}

Write-Host "[v3 Group A] All runs complete."
