# E3 Wave 4+ orchestration: summarize -> cap isolation -> multiseed -> final eval.
#
# Prerequisites: Wave 4 orthogonal profiles finished under output/vr_room_e3_v2_*.
#
# Usage (from repo root, gaussian_splatting env active):
#   .\scripts\run_e3_post_wave4.ps1 -Stage summarize
#   .\scripts\run_e3_post_wave4.ps1 -Stage cap
#   .\scripts\run_e3_post_wave4.ps1 -Stage multiseed -TopConfigs "combo_opacity0.02_densify0.50,densify0.50_pruneonly0.80"
#   .\scripts\run_e3_post_wave4.ps1 -Stage final -Models "output\vr_room_e3_v2_combo_opacity0.02_densify0.50,..."

param(
    [ValidateSet("summarize", "cap", "multiseed", "final", "all")]
    [string]$Stage = "all",
    [string]$TopConfigs = "",
    [string]$Models = "",
    [int]$FpsTrials = 5
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$python = if ($env:E3_PYTHON) { $env:E3_PYTHON } else { "python" }

function Invoke-Summarize {
    Write-Host "[post-wave4] summarize v2 + v3cap cases"
    & $python scripts\summarize_vr_room_e3.py `
        --case-glob "vr_room_e3_v2_*,vr_room_e3_v3cap_*" `
        --measure-fps 1 `
        --fps-warmup 10 `
        --fps-repeats 30
}

function Get-TopConfigsFromSummary {
    $summaryPath = Join-Path $ROOT "output\vr_room_e3_summary.json"
    if (-not (Test-Path $summaryPath)) {
        throw "Missing $summaryPath. Run -Stage summarize first."
    }
    $summary = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $top = @($summary.recommended_top2_for_multiseed)
    if ($top.Count -lt 1) {
        throw "No recommended_top2_for_multiseed in summary."
    }
    return ($top -join ",")
}

function Invoke-CapIsolation {
    Write-Host "[post-wave4] cap isolation at 2.5M (3080-safe step)"
    $env:E3_RUN_TAG = "v3cap"
    $env:E3_PROFILES = "combo_opacity0.02_densify0.50,e2_style_cap2m"
    $env:E3_MAX_GAUSSIANS = "2500000"
    Remove-Item Env:E3_PHASE -ErrorAction SilentlyContinue
    Remove-Item Env:E3_SEEDS -ErrorAction SilentlyContinue
    & "$ROOT\scripts\run_vr_room_e3_prune_ablation.ps1"
    Invoke-Summarize
}

function Invoke-Multiseed {
    param([string]$Configs)
    if (-not $Configs) {
        $Configs = Get-TopConfigsFromSummary
    }
    Write-Host "[post-wave4] multiseed configs=$Configs seeds=0,1,2"
    $env:E3_PHASE = "multiseed"
    $env:E3_TOP_CONFIGS = $Configs
    $env:E3_SEEDS = "0,1,2"
    Remove-Item Env:E3_RUN_TAG -ErrorAction SilentlyContinue
    Remove-Item Env:E3_MAX_GAUSSIANS -ErrorAction SilentlyContinue
    & "$ROOT\scripts\run_vr_room_e3_prune_ablation.ps1"
    # Also multiseed e2_style under v3cap tag for paired comparison.
    $env:E3_RUN_TAG = "v3cap"
    $env:E3_PROFILES = "e2_style_cap2m"
    $env:E3_SEEDS = "1,2"
    Remove-Item Env:E3_PHASE -ErrorAction SilentlyContinue
    & "$ROOT\scripts\run_vr_room_e3_prune_ablation.ps1"
    Invoke-Summarize
}

function Invoke-FinalEval {
    param([string]$ModelList, [int]$Trials)
    if (-not $ModelList) {
        $summaryPath = Join-Path $ROOT "output\vr_room_e3_summary.json"
        $summary = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $names = @()
        foreach ($profile in $summary.recommended_top2_for_multiseed) {
            $match = @($summary.cases | Where-Object { $_.profile -eq $profile -and $_.seed -eq 0 })
            if ($match.Count -gt 0) { $names += "output\$($match[0].case)" }
        }
        $e2 = Get-ChildItem -Path (Join-Path $ROOT "output") -Directory -Filter "vr_room_e3_v3cap_e2_style_cap2m*" |
            Select-Object -First 1
        if ($e2) { $names += $e2.FullName }
        $unlimited = Join-Path $ROOT "output\vr_room_indoor_sensor_depth"
        if (Test-Path $unlimited) { $names += $unlimited }
        $ModelList = ($names -join ",")
    }
    if (-not $ModelList) {
        throw "No models resolved for final evaluation."
    }
    Write-Host "[post-wave4] per-view + FPS trials for: $ModelList"
    foreach ($model in ($ModelList -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $dir = if ([System.IO.Path]::IsPathRooted($model)) { $model } else { Join-Path $ROOT $model }
        if (-not (Test-Path $dir)) {
            Write-Host "[warn] skip missing $dir"
            continue
        }
        & $python scripts\evaluate_nvs_per_view.py -m $dir --iteration 30000
    }
    & $python scripts\benchmark_render_fps_trials.py `
        --models $ModelList `
        --trials $Trials `
        --warmup 10 `
        --repeats 30
    Invoke-Summarize
}

switch ($Stage) {
    "summarize" { Invoke-Summarize }
    "cap" { Invoke-CapIsolation }
    "multiseed" { Invoke-Multiseed -Configs $TopConfigs }
    "final" { Invoke-FinalEval -ModelList $Models -Trials $FpsTrials }
    "all" {
        Invoke-Summarize
        Invoke-CapIsolation
        $configs = if ($TopConfigs) { $TopConfigs } else { Get-TopConfigsFromSummary }
        Invoke-Multiseed -Configs $configs
        Invoke-FinalEval -ModelList $Models -Trials $FpsTrials
    }
}

Write-Host "[post-wave4] stage '$Stage' complete."
