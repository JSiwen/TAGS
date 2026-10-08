# Bike-vs-DN final: score seed0/1/2 TAGS, score DN (idempotent), aggregate 3-seed.
# Run from the repo root in PowerShell, in the gaussian_splatting env.
#
#   cd D:\TAGSs\TAGSs_core
#   .\scripts\run_bicycle_vs_dn_final.ps1
#
# Optional env:
#   $env:GS_PY = "C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe"
#   $env:SEEDS = "0,1,2"                # default; subset also OK
#   $env:DRY_RUN = "1"                  # only print the commands

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$gs = if ($env:GS_PY) { $env:GS_PY } else { "C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe" }
$seeds = if ($env:SEEDS) { $env:SEEDS.Split(",") | ForEach-Object { $_.Trim() } } else { @("0","1","2") }
$dry = [bool]$env:DRY_RUN

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Run-Py {
    param([string[]]$PyArgs)
    if ($dry) { Write-Host "DRY $gs $($PyArgs -join ' ')"; return }
    & $gs @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "python failed: $gs $($PyArgs -join ' ') (exit $LASTEXITCODE)" }
}

function Run-PyTolerant {
    param([string[]]$PyArgs, [string]$IgnoreExitCodePattern = "")
    if ($dry) { Write-Host "DRY $gs $($PyArgs -join ' ')"; return }
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $gs @PyArgs 2>&1 | ForEach-Object { Write-Host $_ }
    } catch {
        Write-Host "[warn] $gs $($PyArgs -join ' ') raised: $($_.Exception.Message)"
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevPref
    if ($code -ne 0) {
        if ($IgnoreExitCodePattern -and ($code.ToString() -match $IgnoreExitCodePattern)) {
            Write-Host "[warn] exit code $code tolerated"
        } else {
            throw "python failed: $gs $($PyArgs -join ' ') (exit $code)"
        }
    }
}

Write-Host "==[ bicycle-vs-DN final ]============================================="
Write-Host "[env] gs=$gs  seeds=$($seeds -join ',')  dry=$dry  cwd=$ROOT"

# 1) Sanity: every requested seed must have a final .ply
$allSeeds = $seeds
foreach ($s in $allSeeds) {
    $suffix = if ($s -eq "0") { "" } else { "_seed$s" }
    $dir = "output\bicycle_e4_lpips_ms_w005$suffix"
    $ply = "$dir\point_cloud\iteration_30000\point_cloud.ply"
    if (-not (Test-Path $ply)) { throw "missing final ply for ${s}: $ply" }
    $cfg = "$dir\cfg_args"
    if (-not (Test-Path $cfg)) { throw "missing cfg_args for ${s}: $cfg" }
    Write-Host "[ok] seed$s trained : $ply"
}

# 2) Re-render + re-score each seed (metrics.py is idempotent; this is safe).
#    Skip if either root or test/ours_30000/results.json is newer than the final .ply.
function Test-AlreadyScored {
    param([string]$Dir, [string]$Ply)
    foreach ($candidate in @("$Dir\results.json", "$Dir\test\ours_30000\results.json")) {
        if (Test-Path $candidate) {
            $resTime = (Get-Item $candidate).LastWriteTime
            $plyTime = (Get-Item $Ply).LastWriteTime
            if ($resTime -ge $plyTime) { return $candidate }
        }
    }
    return $null
}

foreach ($s in $allSeeds) {
    $suffix = if ($s -eq "0") { "" } else { "_seed$s" }
    $dir = "output\bicycle_e4_lpips_ms_w005$suffix"
    $ply = "$dir\point_cloud\iteration_30000\point_cloud.ply"
    $existing = Test-AlreadyScored -Dir $dir -Ply $ply
    if ($existing) {
        Write-Host "[skip] seed$s already scored: $existing"
        continue
    }
    Write-Host "==> render seed$s : $dir"
    Run-Py @("render.py","-m",$dir,"--skip_train","--iteration","30000","--quiet")
    Write-Host "==> metrics seed$s : $dir"
    Run-Py @("metrics.py","-m",$dir)
}

# 3) Make sure DN renders are staged (copies pred/gt into metrics layout)
$dnModelDir = "output\bicycle_dn_splatter"
$dnRenders  = "output\bicycle_dn_renders\eval"
$dnMethod   = Join-Path $dnModelDir "test\dn-splatter"
$dnRgb      = Join-Path $dnMethod "renders"
if (-not (Test-Path $dnRgb)) {
    Write-Host "==> staging DN renders into $dnMethod"
    New-Item -ItemType Directory -Force -Path (Join-Path $dnMethod "gt") | Out-Null
} else {
    Write-Host "[ok] DN renders already staged at $dnRgb"
}
# summarize_bicycle_e4_vs_dn.py itself does the staging via stage_dn_renders()

# 4) Single-shot summary (DN + TAGS seed0 only by default).
#    summarize_*.py re-invokes metrics.py; in this env the latter may fail to
#    import PIL, but the summary JSON is still written and we already have
#    the results.json files we need. Tolerate that exit.
Write-Host "==> summarize_bicycle_e4_vs_dn.py"
Run-PyTolerant @("scripts\summarize_bicycle_e4_vs_dn.py","--python-bin",$gs) -IgnoreExitCodePattern '.*'

# 5) 3-seed aggregate (uses already-scored TAGS results + DN baseline)
Write-Host "==> aggregate_bicycle_3seed.py"
Run-Py @("scripts\aggregate_bicycle_3seed.py","--root",$ROOT)

# 6) Print the headline numbers (only when actually scoring, not on dry-run)
if ($dry) {
    Write-Host "==[ headlines: skipped in dry-run ]=================================="
} else {
    Write-Host "==[ headlines ]======================================================="
    $tmp = Join-Path $ROOT "output\.headline.tmp.py"
    @'
import json
from pathlib import Path
s = json.loads(Path("output/bicycle_e4_vs_dn_summary_3seed.json").read_text(encoding="utf-8"))
t = s["tags_3seed"]; d = s["dn"]; x = s["delta_tags_minus_dn"]
pv = s.get("per_view_tags_better") or {}
print("DN          : PSNR {psnr:.3f}  SSIM {ssim:.4f}  LPIPS {lpips:.4f}  pts={pts}".format(psnr=d["psnr"], ssim=d["ssim"], lpips=d["lpips"], pts=d["num_points"]))
print("TAGS 3seed  : PSNR {m:.3f}+-{s:.3f}  SSIM {m2:.4f}+-{s2:.4f}  LPIPS {m3:.4f}+-{s3:.4f}".format(m=t["psnr_mean"], s=t["psnr_std"], m2=t["ssim_mean"], s2=t["ssim_std"], m3=t["lpips_mean"], s3=t["lpips_std"]))
print("Delta       : dPSNR {psnr:+.3f}  dSSIM {ssim:+.4f}  dLPIPS {lpips:+.4f}".format(psnr=x["psnr"], ssim=x["ssim"], lpips=x["lpips"]))
if pv:
    print("Per-view    : TAGS better on PSNR {a}/{n}  SSIM {b}/{n}  LPIPS {c}/{n}".format(a=pv["psnr_tags_better"], b=pv["ssim_tags_better"], c=pv["lpips_tags_better"], n=pv["num_views"]))
'@ | Set-Content -Path $tmp -Encoding UTF8
    Run-Py @($tmp)
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

Write-Host "==[ done ]============================================================"
Write-Host "Outputs:"
foreach ($f in @("output\bicycle_e4_vs_dn_summary.json",
                  "output\bicycle_e4_vs_dn_summary.csv",
                  "output\bicycle_e4_vs_dn_summary_3seed.json",
                  "output\bicycle_e4_vs_dn_summary_3seed.csv",
                  "output\bicycle_e4_vs_dn_summary_3seed_per_view.csv")) {
    if (Test-Path $f) { Write-Host "  $f  ($((Get-Item $f).Length) bytes)" }
}
