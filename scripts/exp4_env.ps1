# TAGSs exp4 one-shot env config (PowerShell)
#
# Usage (dot-source in PowerShell, does NOT pollute your $PROFILE):
#   . D:\TAGSs\TAGSs_core\scripts\exp4_env.ps1
#
# Effect: current session auto-sets
#   - WANDB_ENABLED=1
#   - WANDB_PROJECT=tags-ablation
#   - WANDB_ENTITY=your_username       <-- WARNING: change here
#   - E4_SEEDS=0,1,2,3,4           (5-seed default)
#   - E4_SKIP_COMPLETED=1          (on)
#   - E4_AUTO_RESUME=1             (on)
#   - TAGSs_root alias points to repo root
#
# To make it permanent: append the body to $PROFILE

# ===== CHANGE HERE =====
$env:WANDB_ENTITY = "your_username"   # WARNING: replace with your wandb entity
# ======================

$env:WANDB_ENABLED = "1"
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_RUN_GROUP = "exp4_5_Full"
$env:E4_SEEDS = "0,1,2,3,4"
$env:E4_SKIP_COMPLETED = "1"
$env:E4_AUTO_RESUME = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Aliases
$TAGSs_root = "D:\TAGSs\TAGSs_core"
Set-Alias -Name tags_root -Value $TAGSs_root -Scope Global -Force
function tags-cd { Set-Location $TAGSs_root }
Set-Alias -Name tcd -Value tags-cd -Scope Global -Force

# Status print (English only to avoid encoding issues)
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " TAGSs exp4 env loaded" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "WANDB_ENABLED   = $env:WANDB_ENABLED"
Write-Host "WANDB_PROJECT   = $env:WANDB_PROJECT"
Write-Host "WANDB_ENTITY    = $env:WANDB_ENTITY"
Write-Host "WANDB_RUN_GROUP = $env:WANDB_RUN_GROUP"
Write-Host "E4_SEEDS        = $env:E4_SEEDS"
Write-Host "E4_SKIP_COMPLETED = $env:E4_SKIP_COMPLETED (1=skip done, 0=rerun)"
Write-Host "E4_AUTO_RESUME  = $env:E4_AUTO_RESUME (1=resume from chkpnt, 0=from scratch)"
Write-Host "Repo root = $TAGSs_root (alias: tcd / tags-cd)"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Common commands:" -ForegroundColor Yellow
Write-Host "  tcd                                       cd to repo root"
Write-Host "  .\scripts\run_bicycle_e4_winner.ps1       run bicycle 5-seed"
Write-Host "  python scripts\summarize_bicycle_e4_vs_dn.py    aggregate after 5-seed done"
Write-Host "  python scripts\build_section_4_6.py        generate section 4.6 table"
Write-Host ""
