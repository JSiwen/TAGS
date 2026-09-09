# Example: COLMAP scene under $Source, trained model under $Model.
# 1) (once) split train/test -> sparse/0/test.txt
# 2) train with --eval --llffhold 0
# 3) render test views only
# 4) metrics on test split
#
# Adjust paths and training args before running.

param(
    [string]$Source = "data",
    [string]$Model  = "data/output"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== [1/4] Split (skip if test.txt already set) ===" -ForegroundColor Cyan
python scripts/split_colmap_train_test.py -s $Source --strategy random --test_ratio 0.1 --seed 0

Write-Host "=== [2/4] Train (add your flags: train.py or train_optimized.py) ===" -ForegroundColor Cyan
Write-Host "Example: python train_optimized.py -s $Source -m $Model --eval --llffhold 0 --iterations 30000"
Write-Host "Run training manually, then press Enter to continue to render + metrics."
Read-Host

Write-Host "=== [3/4] Render test views ===" -ForegroundColor Cyan
python render.py -m $Model --skip_train --quiet

Write-Host "=== [4/4] Metrics (test) ===" -ForegroundColor Cyan
python metrics.py -m $Model --split test

Write-Host "Done. See results.json under $Model" -ForegroundColor Green
