# Idempotent end-to-end post-pipeline for the bicycle E4 vs DN-Splatter comparison.
#
# What it does (in order):
#   1. Sanity check: every discovered TAGS seed has a final ply AND a
#      metrics-scored test/{ours_30000,dn-splatter}/ directory.
#   2. For any seed whose test results are missing, run metrics.py.
#      (Skipped if no fresh render assets exist; in that case print
#      "missing-assets" and move on.)
#   3. Run summarize_bicycle_e4_vs_dn.py for seed 0 (TAGS vs DN) to
#      produce the per-seed summary CSV/JSON.
#   4. Run aggregate_bicycle_3seed.py to combine N seeds against DN;
#      output files use *_{N}seed.* based on the discovered seed count.
#   5. Print the headline numbers (TAGS 4-seed mean+-std, DN single,
#      delta) to stdout.
#
# This script is read-only with respect to training and rendering;
# training lives in run_bicycle_e4_winner.ps1 and rendering in
# run_bicycle_dn_other_env.ps1. Re-runs are safe: already-scored seeds
# are not re-scored (the aggregate picks them up from results.json).
#
# Usage:
#   conda activate gaussian_splatting   # needed because metrics.py uses VGG LPIPS
#   cd D:\TAGSs\TAGSs_core
#   .\scripts\run_bicycle_vs_dn_final.ps1
#
# Dry-run (skip metrics re-scoring, still refresh summaries):
#   $env:DRY_RUN = "1"; .\scripts\run_bicycle_vs_dn_final.ps1
#
# Skip per-view aggregate:
#   $env:SKIP_PER_VIEW = "1"; .\scripts\run_bicycle_vs_dn_final.ps1
#
# Re-score everything even if a results.json exists:
#   $env:FORCE = "1"; .\scripts\run_bicycle_vs_dn_final.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$python = if ($env:PYTHON) { $env:PYTHON } elseif ($env:BICYCLE_PYTHON) { $env:BICYCLE_PYTHON } else { "python" }
$dryRun = [bool]($env:DRY_RUN)
$force = [bool]($env:FORCE)
$skipPerView = [bool]($env:SKIP_PER_VIEW)

$OUTPUT = Join-Path $ROOT "output"
$TAGS_GLOB = "bicycle_e4_lpips_ms_w005*"
$DN_DIR = Join-Path $OUTPUT "bicycle_dn_splatter"
$DN_RENDERS = Join-Path $OUTPUT "bicycle_dn_renders\eval"

function Get-TagsSeedDirs {
    return Get-ChildItem -Path $OUTPUT -Directory -Filter $TAGS_GLOB |
        Where-Object { Test-Path (Join-Path $_.FullName "point_cloud\iteration_30000\point_cloud.ply") } |
        Sort-Object Name
}

function Get-SeedLabel {
    param([System.IO.DirectoryInfo]$Dir)
    $name = $Dir.Name
    if ($name -eq "bicycle_e4_lpips_ms_w005") { return 0 }
    if ($name -match "_seed(?<n>\d+)$") { return [int]$Matches.n }
    return $null
}

function Test-SeedScored {
    param([System.IO.DirectoryInfo]$SeedDir)
    # Accept either test/{ours_30000}/results.json or the legacy top-level results.json.
    $candidates = @(
        (Join-Path $SeedDir.FullName "test\ours_30000\results.json"),
        (Join-Path $SeedDir.FullName "test\results.json"),
        (Join-Path $SeedDir.FullName "results.json")
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $true } }
    return $false
}

function Test-DnScored {
    $candidates = @(
        (Join-Path $DN_DIR "test\dn-splatter\results.json"),
        (Join-Path $DN_DIR "test\results.json"),
        (Join-Path $DN_DIR "results.json")
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $true } }
    return $false
}

function Test-DnRendersPresent {
    $pred = Join-Path $DN_RENDERS "pred\rgb"
    $gt = Join-Path $DN_RENDERS "gt\rgb"
    return ((Test-Path $pred) -and (Test-Path $gt))
}

function Get-MissingAssetsMessage {
    $parts = @()
    if (-not (Test-Path $DN_DIR)) { $parts += "DN output dir missing ($DN_DIR)" }
    elseif (-not (Test-DnRendersPresent)) { $parts += "DN renders missing (run run_bicycle_dn_other_env.ps1)" }
    if (-not (Test-Path $OUTPUT)) { $parts += "TAGS output root missing" }
    return ($parts -join "; ")
}

Write-Host "[bicycle-final] python: $python"
Write-Host "[bicycle-final] dry-run: $dryRun  force: $force"

$seedDirs = @(Get-TagsSeedDirs)
if ($seedDirs.Count -eq 0) {
    Write-Error "No TAGS seed dirs found (output/$TAGS_GLOB with iteration_30000/point_cloud.ply). Train first via run_bicycle_e4_winner.ps1."
    exit 2
}
Write-Host "[bicycle-final] Discovered TAGS seeds: $($seedDirs.Count)"

# Step 1: Sanity check and figure out which seeds need re-scoring.
$needsScore = New-Object System.Collections.Generic.List[object]
foreach ($sd in $seedDirs) {
    $label = Get-SeedLabel $sd
    $scored = Test-SeedScored $sd
    Write-Host ("[bicycle-final]   seed {0,-3}  ply=ok  scored={1}  path={2}" -f $label, $scored, $sd.Name)
    if ($force -or -not $scored) {
        $needsScore.Add(@{ label = $label; dir = $sd.FullName })
    }
}

# Step 2: Run metrics.py only for seeds that need it. Skip if no fresh
# assets are available for any given seed (a re-score without new
# renders would just rewrite the same results.json).
if (-not $dryRun -and $needsScore.Count -gt 0) {
    $missing = Get-MissingAssetsMessage
    if ($missing) {
        Write-Warning "[bicycle-final] Cannot re-score: $missing"
        Write-Warning "[bicycle-final] Will only refresh summaries; existing results.json files (if any) will be used."
    } else {
        $dirs = @($needsScore | ForEach-Object { $_.dir }) + $DN_DIR
        Write-Host "[bicycle-final] Running metrics.py on: $($dirs -join ', ')"
        & $python metrics.py -m @dirs
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[bicycle-final] metrics.py exited with $LASTEXITCODE; continuing to summary."
        }
    }
} elseif ($needsScore.Count -gt 0) {
    Write-Host "[bicycle-final] dry-run: skipping metrics.py re-score."
} else {
    Write-Host "[bicycle-final] All seeds already scored; nothing to re-score."
}

# Step 3: Per-seed summary (TAGS seed0 vs DN; pairwise details).
$tags0 = $seedDirs | Where-Object { $_.Name -eq "bicycle_e4_lpips_ms_w005" } | Select-Object -First 1
if ($tags0) {
    Write-Host "[bicycle-final] Step 3: per-seed summary (seed0 vs DN)"
    if (Test-DnScored -or (Test-Path (Join-Path $DN_DIR "results.json"))) {
        & $python scripts\summarize_bicycle_e4_vs_dn.py --tags-dir $tags0.FullName
    } else {
        Write-Warning "[bicycle-final] No DN results.json; skipping per-seed summary."
    }
} else {
    Write-Warning "[bicycle-final] No seed0 dir found (bicycle_e4_lpips_ms_w005). Multi-seed aggregate below will still work."
}

# Step 4: N-seed aggregate.
Write-Host "[bicycle-final] Step 4: N-seed aggregate"
& $python scripts\aggregate_bicycle_3seed.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "[bicycle-final] aggregate_bicycle_3seed.py exited with $LASTEXITCODE."
}

if ($skipPerView) {
    Write-Host "[bicycle-final] SKIP_PER_VIEW=1, skipping optional per-view detail."
} else {
    Write-Host "[bicycle-final] Step 4b: per-view CSV was already produced by aggregate_bicycle_3seed.py"
}

# Step 5: Headline printout.
$summary = Get-ChildItem -Path $OUTPUT -Filter "bicycle_e4_vs_dn_summary_*seed.json" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($summary) {
    Write-Host ""
    Write-Host "[bicycle-final] Headline from $($summary.Name):"
    & $python -c "
import json, sys
p = r'$($summary.FullName)'
d = json.load(open(p, encoding='utf-8-sig'))
prot = d.get('protocol', {})
print('  seeds:', prot.get('seeds'))
print('  dn:   PSNR=%.4f  SSIM=%.4f  LPIPS=%.4f' % (d['dn']['psnr'], d['dn']['ssim'], d['dn']['lpips']))
ts = d.get('tags_seeds', {})
if ts:
    import statistics
    psnrs = [v['psnr'] for v in ts.values()]
    ssims = [v['ssim'] for v in ts.values()]
    lpipss = [v['lpips'] for v in ts.values()]
    print('  TAGS: PSNR=%.4f +- %.4f  SSIM=%.5f +- %.5f  LPIPS=%.5f +- %.5f' % (
        statistics.mean(psnrs), statistics.stdev(psnrs) if len(psnrs) > 1 else 0.0,
        statistics.mean(ssims), statistics.stdev(ssims) if len(ssims) > 1 else 0.0,
        statistics.mean(lpipss), statistics.stdev(lpipss) if len(lpipss) > 1 else 0.0,
    ))
    dlt = d.get('delta_tags_minus_dn', {})
    if dlt:
        print('  delta: dPSNR=%+.3f  dSSIM=%+.4f  dLPIPS=%+.4f' % (
            dlt.get('psnr', 0.0), dlt.get('ssim', 0.0), dlt.get('lpips', 0.0)))
pv = d.get('per_view_tags_better', {})
if pv:
    print('  per_view: PSNR %d/%d  SSIM %d/%d  LPIPS %d/%d' % (
        pv.get('psnr_tags_better', 0), pv.get('num_views', 0),
        pv.get('ssim_tags_better', 0), pv.get('num_views', 0),
        pv.get('lpips_tags_better', 0), pv.get('num_views', 0),
    ))
" -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "[bicycle-final] Done."
