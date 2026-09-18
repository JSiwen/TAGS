# Orchestrate E4 multiseed finals: quality gate, optional backup, FPS/diagnostics, winner freeze.
#
# Usage:
#   .\scripts\run_e4_multiseed_finals.ps1 -Stage quality
#   .\scripts\run_e4_multiseed_finals.ps1 -Stage backup
#   .\scripts\run_e4_multiseed_finals.ps1 -Stage diagnostics -Profiles lpips_w005,lpips_ms_w005
#   .\scripts\run_e4_multiseed_finals.ps1 -Stage all

param(
    [ValidateSet("quality", "backup", "diagnostics", "all")]
    [string]$Stage = "all",
    [string[]]$Profiles = @(),
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

if (-not $Python) {
    if ($env:E4_PYTHON) {
        $Python = $env:E4_PYTHON
    } else {
        $Python = "python"
    }
}

$MainProfiles = @("lpips_w005", "lpips_ms_w005")
$BaselineProfile = "rgb_ssim"
$UnlimitedCase = "output/vr_room_indoor_sensor_depth"

function Invoke-QualitySummary {
    Write-Host "[E4 finals] quality-only summary (no FPS measurement)"
    & $Python scripts\summarize_vr_room_e4.py `
        --case-glob "vr_room_e4_*" `
        --json-out "output/vr_room_e4_summary.json" `
        --csv-out "output/vr_room_e4_summary.csv"
}

function Get-QualityPassingMainProfiles {
    $summaryPath = Join-Path $ROOT "output/vr_room_e4_summary.json"
    if (-not (Test-Path $summaryPath)) {
        throw "Missing summary: $summaryPath"
    }
    $summary = Get-Content -Path $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $passed = @()
    foreach ($profile in $MainProfiles) {
        $agg = $summary.profile_aggregates.$profile
        if ($null -eq $agg) { continue }
        if ($agg.multiseed_gate.quality_passed -eq $true) {
            $passed += $profile
        }
    }
    return ,$passed
}

function Invoke-BackupWave {
    Write-Host "[E4 finals] running backup wave: rgb_ssim + lpips_ms_w003 (seed1/2 only if needed)"
    $env:E4_PHASE = "multiseed"
    $env:E4_TOP_CONFIGS = "rgb_ssim,lpips_ms_w003"
    $env:E4_SEEDS = "0,1,2"
    $env:E4_SKIP_COMPLETED = "1"
    $env:E4_AUTO_RESUME = "1"
    $env:E4_PYTHON = $Python
    .\scripts\run_vr_room_e4_perceptual_ablation.ps1
    Invoke-QualitySummary
}

function Get-CaseDirsForProfiles {
    param([string[]]$SelectedProfiles)
    $dirs = @()
    foreach ($profile in $SelectedProfiles) {
        foreach ($seed in 0, 1, 2) {
            $suffix = if ($seed -eq 0) { "" } else { "_seed$seed" }
            $name = "vr_room_e4_${profile}${suffix}"
            $path = Join-Path $ROOT "output\$name"
            if (Test-Path $path) { $dirs += $path }
        }
        if ($profile -eq $BaselineProfile) { continue }
    }
    $baselineDirs = @()
    foreach ($seed in 0, 1, 2) {
        $suffix = if ($seed -eq 0) { "" } else { "_seed$seed" }
        $name = "vr_room_e4_${BaselineProfile}${suffix}"
        $path = Join-Path $ROOT "output\$name"
        if (Test-Path $path) { $baselineDirs += $path }
    }
    return ($baselineDirs + $dirs) | Select-Object -Unique
}

function Invoke-Diagnostics {
    param([string[]]$SelectedProfiles)
    if (-not $SelectedProfiles -or $SelectedProfiles.Count -eq 0) {
        $SelectedProfiles = Get-QualityPassingMainProfiles
        if ($SelectedProfiles.Count -eq 0) {
            Write-Host "[E4 finals] no quality-passing main profiles; skip diagnostics"
            return
        }
    }
    $allProfiles = @($BaselineProfile) + $SelectedProfiles
    $caseDirs = Get-CaseDirsForProfiles -SelectedProfiles $allProfiles
    foreach ($dir in $caseDirs) {
        Write-Host "[E4 finals] per-view NVS for $dir"
        & $Python scripts\evaluate_nvs_per_view.py -m $dir --iteration 30000
        Write-Host "[E4 finals] depth consistency for $dir"
        & $Python scripts\evaluate_depth_per_view.py -m $dir --iteration 30000
    }
    $unlimitedDir = Join-Path $ROOT $UnlimitedCase
    if (Test-Path $unlimitedDir) {
        Write-Host "[E4 finals] measuring FPS for finalists + unlimited"
        & $Python scripts\summarize_vr_room_e4.py `
            --measure-fps 1 `
            --fps-warmup 10 `
            --fps-repeats 30 `
            --unlimited-case $UnlimitedCase `
            --json-out "output/vr_room_e4_summary.json" `
            --csv-out "output/vr_room_e4_summary.csv"
    } else {
        Write-Warning "Unlimited case missing at $UnlimitedCase; skipping FPS measurement"
        Invoke-QualitySummary
    }
}

function Write-WinnerFreeze {
    $summaryPath = Join-Path $ROOT "output/vr_room_e4_summary.json"
    $summary = Get-Content -Path $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $candidates = @()
    foreach ($profile in $MainProfiles) {
        $agg = $summary.profile_aggregates.$profile
        if ($null -eq $agg) { continue }
        if ($agg.multiseed_gate.passed -eq $true) {
            $candidates += [pscustomobject]@{
                profile = $profile
                lpips_improve = [double]$agg.lpips_improve_ratio_mean
                psnr_drop = [double]$agg.psnr_drop_mean
                ssim_drop = [double]$agg.ssim_drop_mean
                fps_speedup = if ($agg.multiseed_gate.fps_speedup_mean) { [double]$agg.multiseed_gate.fps_speedup_mean } else { $null }
            }
        }
    }
    $winner = $null
    $reason = "none_passed_full_gate"
    if ($candidates.Count -eq 1) {
        $winner = $candidates[0].profile
        $reason = "only_full_pass"
    } elseif ($candidates.Count -ge 2) {
        $single = $candidates | Where-Object { $_.profile -eq "lpips_w005" } | Select-Object -First 1
        $multi = $candidates | Where-Object { $_.profile -eq "lpips_ms_w005" } | Select-Object -First 1
        if ($single.lpips_improve -ge ($multi.lpips_improve + 0.01) -and $single.ssim_drop -le ($multi.ssim_drop + 0.0003)) {
            $winner = "lpips_w005"
            $reason = "best_lpips_with_acceptable_ssim"
        } else {
            $winner = "lpips_ms_w005"
            $reason = "more_stable_quality_tradeoff"
        }
    }
    $freeze = [ordered]@{
        winner_profile = $winner
        selection_reason = $reason
        full_pass_candidates = @($candidates | ForEach-Object { $_.profile })
        quality_passing_profiles = @($summary.quality_passing_profiles)
        fps_passing_profiles = @($summary.fps_passing_profiles)
        pass_criteria = $summary.pass_criteria
        generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    $outPath = Join-Path $ROOT "output/vr_room_e4_winner.json"
    ($freeze | ConvertTo-Json -Depth 8) | Set-Content -Path $outPath -Encoding UTF8
    Write-Host "[E4 finals] wrote $outPath"
    if ($winner) {
        Write-Host "[E4 finals] winner: $winner ($reason)"
    } else {
        Write-Host "[E4 finals] no winner selected yet"
    }
}

switch ($Stage) {
    "quality" {
        Invoke-QualitySummary
    }
    "backup" {
        $passed = Get-QualityPassingMainProfiles
        if ($passed.Count -gt 0) {
            Write-Host "[E4 finals] main profiles passed quality gate: $($passed -join ', '); skip backup"
        } else {
            Invoke-BackupWave
        }
    }
    "diagnostics" {
        Invoke-Diagnostics -SelectedProfiles $Profiles
        Write-WinnerFreeze
    }
    "all" {
        Invoke-QualitySummary
        $passed = Get-QualityPassingMainProfiles
        if ($passed.Count -eq 0) {
            Invoke-BackupWave
        }
        Invoke-Diagnostics -SelectedProfiles $Profiles
        Write-WinnerFreeze
    }
}

Write-Host "[E4 finals] Stage '$Stage' complete."
