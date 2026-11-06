# =============================================================================
# v3 阶段 1: 预处理 (Preprocess) - 对 garden/bonsai/room 跑完整预处理
# =============================================================================
# 跑:
#   - Depth Anything V2 推理 -> data/{scene}/depth_any_out/
#   - 法线估计 -> data/{scene}/normals_from_pretrain/
#   - depth_params.json 标定 -> data/{scene}/sparse/0/depth_params.json
#   - mono_depth/*_aligned.npy 导出 -> data/{scene}/mono_depth/
#   - transforms.json 改写 (含 mono_depth_path 字段)
#   - 4 场景统一自检
#
# 用法:
#   .\scripts\run_prepare_extra_scenes.ps1
#   $env:E4_STAGE = "all"  ; .\scripts\run_prepare_extra_scenes.ps1
#   $env:E4_STAGE = "depth"; $env:E4_SCENES = "garden"; .\scripts\run_prepare_extra_scenes.ps1
#   $env:E4_DRY_RUN = "1"  ; .\scripts\run_prepare_extra_scenes.ps1  (只打印命令)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ---- 参数解析 ----
$python     = if ($env:E4_PYTHON)   { $env:E4_PYTHON }   else { "python" }
$stage      = if ($env:E4_STAGE)    { $env:E4_STAGE }    else { "all" }
$hold       = if ($env:E4_HOLD)     { [int]$env:E4_HOLD } else { 8 }
$scenesEnv  = if ($env:E4_SCENES)   { $env:E4_SCENES }   else { "garden,bonsai,room" }
$dryRun     = if ($env:E4_DRY_RUN)  { $env:E4_DRY_RUN -ne "0" } else { $false }

# wandb 集成 (预处理本身不上 wandb, 但留 hook 供后续阶段使用)
$wandbEnabled = if ($env:WANDB_ENABLED) { $env:WANDB_ENABLED -ne "0" } else { $false }

# 环境变量 (与现有 PS1 脚本一致)
$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ---- 打印当前阶段信息 ----
Write-Host "============================================================"
Write-Host "[v3 Stage 1] Preprocess for $scenesEnv"
Write-Host "  Stage : $stage"
Write-Host "  Hold  : $hold (LLFF holdout)"
Write-Host "  Wandb : $wandbEnabled"
Write-Host "============================================================"

# ---- 构造 python 命令 ----
$pyArgs = @('scripts\prepare_extra_scenes.py', '--stage', $stage, '--scenes', $scenesEnv, '--hold', "$hold")

if ($dryRun) {
    Write-Host "[DRY-RUN] Command: $python $($pyArgs -join ' ')"
    exit 0
}

# ---- 跑 ----
& $python @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Warning "[v3 Stage 1] prepare_extra_scenes failed (exit=$LASTEXITCODE)"
    Write-Warning "  建议排查: 见 docs\EXP4_RUN_BOOK.md §7 失败回退"
    exit $LASTEXITCODE
}

# ---- 跑完后写 ledger (本地) ----
$ledgerPath = Join-Path $ROOT "output\exp4_stats\preprocess_ledger.jsonl"
$ledgerPathParent = Split-Path -Parent $ledgerPath
if (-not (Test-Path $ledgerPathParent)) {
    New-Item -ItemType Directory -Force -Path $ledgerPathParent | Out-Null
}

$timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
$ledgerEntry = @{
    timestamp = $timestamp
    stage     = $stage
    scenes    = $scenesEnv -split ","
    hold      = $hold
    wandb     = $wandbEnabled
} | ConvertTo-Json -Compress

Add-Content -Path $ledgerPath -Value $ledgerEntry -Encoding UTF8
Write-Host "[v3 Stage 1] Ledger appended: $ledgerPath"

# ---- 写 stage report ----
$reportPath = Join-Path $ROOT "output\exp4_stats\preprocess_stage_check.json"
$report = @{
    timestamp = $timestamp
    stage     = $stage
    scenes    = $scenesEnv -split ","
    hold      = $hold
    wandb     = $wandbEnabled
    status    = "ok"
} | ConvertTo-Json

Set-Content -Path $reportPath -Value $report -Encoding UTF8
Write-Host "[v3 Stage 1] Report written: $reportPath"

# ---- 提示下一步 ----
Write-Host ""
Write-Host "============================================================"
Write-Host "[v3 Stage 1] DONE"
Write-Host "  下一步 (Stage 2):"
Write-Host "    python scripts\qa_baseline_check.py --images-dir images_2"
Write-Host "    或: .\scripts\run_qa_baseline.ps1"
Write-Host "  详细说明: docs\EXP4_RUN_BOOK.md §2"
Write-Host "============================================================"
