# Post-processing for vr_room E4b geo-overlap ablation.
#
# Usage:
#   .\scripts\run_e4b_post.ps1 -Stage summarize
#   .\scripts\run_e4b_post.ps1 -Stage depth
#   .\scripts\run_e4b_post.ps1 -Stage per_view
#   .\scripts\run_e4b_post.ps1 -Stage final

param(
    [ValidateSet("summarize", "depth", "per_view", "final")]
    [string]$Stage = "final",
    [string[]]$Cases = @()
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$python = if ($env:E4B_PYTHON) { $env:E4B_PYTHON } elseif ($env:E4_PYTHON) { $env:E4_PYTHON } else { "python" }

function Get-E4bCaseDirs {
    param([string[]]$Selected)
    if ($Selected -and $Selected.Count -gt 0) {
        return $Selected | ForEach-Object { Join-Path $ROOT "output\$_" } | Where-Object { Test-Path $_ }
    }
    $dirs = @()
    $dirs += Get-ChildItem -Path (Join-Path $ROOT "output") -Directory -Filter "vr_room_e4b_*" | ForEach-Object { $_.FullName }
    foreach ($seed in 0, 1, 2) {
        $suffix = if ($seed -eq 0) { "" } else { "_seed$seed" }
        foreach ($name in @("vr_room_e4_lpips_ms_w005$suffix", "vr_room_e4_rgb_ssim$suffix")) {
            $path = Join-Path $ROOT "output\$name"
            if (Test-Path $path) { $dirs += $path }
        }
    }
    return $dirs | Select-Object -Unique
}

$caseDirs = Get-E4bCaseDirs -Selected $Cases

switch ($Stage) {
    "summarize" {
        & $python scripts\summarize_vr_room_e4b.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
    }
    "depth" {
        foreach ($dir in $caseDirs) {
            Write-Host "[E4b post] depth consistency for $dir"
            & $python scripts\evaluate_depth_per_view.py -m $dir --iteration 30000
        }
    }
    "per_view" {
        foreach ($dir in $caseDirs) {
            Write-Host "[E4b post] per-view NVS for $dir"
            & $python scripts\evaluate_nvs_per_view.py -m $dir --iteration 30000
        }
    }
    "final" {
        foreach ($dir in $caseDirs) {
            & $python scripts\evaluate_nvs_per_view.py -m $dir --iteration 30000
            & $python scripts\evaluate_depth_per_view.py -m $dir --iteration 30000
        }
        & $python scripts\summarize_vr_room_e4b.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
    }
}

Write-Host "[E4b post] Stage '$Stage' complete."
