# Prepare bicycle depths for DN-Splatter, using the gaussian_splatting interpreter.
#
# Export images_4 mono_depth and rewrite transforms.json:
#   .\scripts\run_bicycle_dn_splatter.ps1 -Stage export
#
# Check split and depth files only:
#   .\scripts\run_bicycle_dn_splatter.ps1 -CheckOnly
#
# Train and render in the other conda env with scripts/run_bicycle_dn_other_env.ps1.
# Depth export uses this env (cv2). Do not point DN_PYTHON at that step.

param(
    [ValidateSet("export", "train", "render", "all")]
    [string]$Stage = "export",
    [switch]$CheckOnly,
    [switch]$SkipExport
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$tagsPython = if ($env:E4_PYTHON) { $env:E4_PYTHON } else { "python" }
$data = if ($env:BICYCLE_SOURCE) { $env:BICYCLE_SOURCE } else { Join-Path $ROOT "data\bicycle" }

function Invoke-Export {
    param([switch]$Overwrite, [switch]$RewriteTransforms, [switch]$Check)
    $exportArgs = @(
        "scripts\export_bicycle_mono_depth.py",
        "--data", $data,
        "--images", "images_4",
        "--depths", "depth_any_out",
        "--depth-params", "sparse/0/depth_params.json",
        "--out", "mono_depth",
        "--hold", "8"
    )
    if ($Check) {
        $exportArgs += "--check-only"
    } else {
        if ($Overwrite) { $exportArgs += "--overwrite" }
        if ($RewriteTransforms) { $exportArgs += "--rewrite-transforms" }
    }
    & $tagsPython @exportArgs
    if ($LASTEXITCODE -ne 0) {
        throw "depth export/check failed with exit code $LASTEXITCODE"
    }
}

if ($CheckOnly) {
    Invoke-Export -Check
    Write-Host "[bicycle DN] Hold-8 and aligned-depth check passed."
    return
}

if (-not $SkipExport -and ($Stage -eq "export" -or $Stage -eq "all" -or $Stage -eq "train")) {
    Write-Host "[bicycle DN] Exporting Depth Anything maps to mono_depth and rewriting transforms.json"
    Invoke-Export -Overwrite -RewriteTransforms
}

if ($Stage -eq "train" -or $Stage -eq "render" -or $Stage -eq "all") {
    Write-Host "[bicycle DN] Export is done in this env. Train and render in the dn-splatter env:"
    Write-Host "  .\scripts\run_bicycle_dn_other_env.ps1 -CheckOnly"
    Write-Host "  .\scripts\run_bicycle_dn_other_env.ps1"
    return
}

Write-Host "[bicycle DN] Export done. Next, in the dn-splatter env: .\scripts\run_bicycle_dn_other_env.ps1 -CheckOnly"
