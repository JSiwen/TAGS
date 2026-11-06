# =============================================================================
# v3 Group C: 分辨率敏感性实验 (3 分辨率 × 4 场景 × 3 seed = 36 runs)
# =============================================================================
# 完整 TAGSs 在 1/4, 1/2, 标准分辨率下训练评估
# 输出: output/exp4v3_C_{scene}/C.X_Res_*/train/seed{N}/
# 汇总: output/exp4_stats/exp4v3_C_resolution_sensitivity.csv
#
# 用法:
#   .\scripts\run_exp4v3_C_resolution_sensitivity.ps1
#   $env:E4_SCENES = "bicycle"
#   $env:E4_GROUPS = "C.3"  # 只跑标准分辨率
#   .\scripts\run_exp4v3_C_resolution_sensitivity.ps1
#
#   # Smoke
#   $env:E4_SMOKE = "1"
#   .\scripts\run_exp4v3_C_resolution_sensitivity.ps1
#
#   # 注意: 标准分辨率 (--images images) 需要大量显存和时间
#   建议先跑 1/2 (Group A/B 主实验), 1/4 和标准分辨率仅在时间充裕时跑

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$python       = if ($env:E4_PYTHON)       { $env:E4_PYTHON }       else { "python" }
$smoke        = if ($env:E4_SMOKE)         { $env:E4_SMOKE -ne "0" } else { $false }
$dryRun       = if ($env:E4_DRY_RUN)        { $env:E4_DRY_RUN -ne "0" } else { $false }
$iterations   = if ($smoke) { 1000 } else { 30000 }

# 3 分辨率组 (短名, 与 Python argparse default="C.1,C.2,C.3" 一致)
$ALL_GROUPS = @("C.1", "C.2", "C.3")
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

# 3 seed (v3: 减半以平衡工作量)
$seedList = if ($env:E4_SEEDS) {
    $env:E4_SEEDS.Split(",") | ForEach-Object { [int]$_.Trim() }
} else {
    @(0, 1, 2)
}

# 调用 Python 脚本
$pyArgs = @('scripts\run_exp4v3_C_resolution_sensitivity.py', '--scenes', ($selectedScenes -join ','), '--groups', ($selectedGroups -join ','), '--seeds', ($seedList -join ','), '--iterations', "$iterations", '--output-root', 'output')
if ($skipCompleted) { $pyArgs += '--skip-existing' }
if ($smoke) { $pyArgs += '--dry-run' }
if ($dryRun) { $pyArgs += '--dry-run' }

# ---- wandb 集成 (env 透传给子进程) ----
$wandbEnabled = if ($env:WANDB_ENABLED) { $env:WANDB_ENABLED -ne "0" } else { $false }
if ($wandbEnabled) {
    if (-not $env:WANDB_PROJECT) { $env:WANDB_PROJECT = "tags-ablation" }
    if (-not $env:WANDB_RUN_GROUP) { $env:WANDB_RUN_GROUP = "exp4v3_C_resolution_sensitivity" }
    if (-not $env:WANDB_TAGS) { $env:WANDB_TAGS = "v3,$($selectedGroups -join '_'),$($seedList -join '_')-seed" }
    Write-Host "[v3 Group C] Wandb ON  project=$env:WANDB_PROJECT group=$env:WANDB_RUN_GROUP tags=$env:WANDB_TAGS"
} else {
    Write-Host "[v3 Group C] Wandb OFF (set `$env:WANDB_ENABLED=1 to enable)"
}

$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& $python @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Warning "[v3 Group C] Python script exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if (-not $smoke -and -not $dryRun) {
    Write-Host "[v3 Group C] Summarizing..."
    & $python "scripts\run_exp4v3_C_resolution_sensitivity.py" `
        --scenes "bicycle,garden,bonsai,room" `
        --groups "C.1,C.2,C.3" `
        --seeds "0" `
        --summarize `
        --output-root "output"
}

Write-Host "[v3 Group C] All runs complete."
