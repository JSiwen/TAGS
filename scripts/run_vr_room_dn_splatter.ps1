# Prepare vr_room depths and normals for DN-Splatter, using the gaussian_splatting interpreter.
#
# Export mono_depth from depth/ and rebuild transforms.json from sparse/0/:
#   .\scripts\run_vr_room_dn_splatter.ps1 -Stage export
#
# Generate normals_from_pretrain (omnidata, runs in dn-splatter env):
#   .\scripts\run_vr_room_dn_splatter.ps1 -Stage normals
#
# Check split and depth files only:
#   .\scripts\run_vr_room_dn_splatter.ps1 -CheckOnly
#
# Train and render in the other conda env with scripts/run_vr_room_dn_other_env.ps1.
# Depth export uses this env (cv2). Normals export uses the dn-splatter env
# (omnidata-tools). Do not point DN_PYTHON at those steps.

param(
    [ValidateSet("export", "normals", "train", "render", "all")]
    [string]$Stage = "export",
    [switch]$CheckOnly,
    [switch]$SkipExport
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$tagsPython = if ($env:E4_PYTHON) { $env:E4_PYTHON } else { "python" }
# Fall back to a known interpreter in the gaussian_splatting env if the
# bare `python` on PATH is from base conda. The known path mirrors what
# dn-splatter env uses (C:\Users\yori\miniconda3\envs\...) so a future
# reader can easily extend it.
if ($tagsPython -eq "python") {
    $gsCandidate = "C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe"
    if (Test-Path $gsCandidate) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $cmd -or $cmd.Source -notlike "*gaussian_splatting*") {
            Write-Host "[vr_room DN] using gaussian_splatting env python at $gsCandidate (no active conda env detected)"
            $tagsPython = $gsCandidate
        }
    }
}
$data = if ($env:VR_ROOM_SOURCE) { $env:VR_ROOM_SOURCE } else { Join-Path $ROOT "data\vr_room" }

function Invoke-Export {
    param([switch]$Overwrite, [switch]$RewriteTransforms, [switch]$Check)
    $exportArgs = @(
        "scripts\export_vr_room_mono_depth.py",
        "--data", $data,
        "--images", "images",
        "--depths", "depth",
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

function Invoke-Normals {
    # omnidata normals_from_pretrain: must run in the dn-splatter conda
    # env. The script resolves omnidata_dpt_normal_v2.ckpt via a path
    # relative to its current working directory (it does NOT look
    # inside its own package), so we cd into the dn-splatter checkout
    # before invoking it. The checkpoint is expected at
    # <dn-splatter-root>/omnidata_ckpt/omnidata_dpt_normal_v2.ckpt
    # (D:\DN-Splatter\dn-splatter-main\dn-splatter-main\omnidata_ckpt).
    $dnPython = if ($env:DN_PYTHON) { $env:DN_PYTHON } else { "python" }
    $dnRoot = if ($env:DN_ROOT) { $env:DN_ROOT } else { "D:\DN-Splatter\dn-splatter-main\dn-splatter-main" }
    if (-not (Test-Path $dnRoot)) {
        throw "dn-splatter checkout not found at $dnRoot. Set `$env:DN_ROOT to its location."
    }
    $ckpt = Join-Path $dnRoot "omnidata_ckpt\omnidata_dpt_normal_v2.ckpt"
    if (-not (Test-Path $ckpt)) {
        throw "Omnidata checkpoint missing at $ckpt. Run: python dn_splatter/data/download_scripts/download_omnidata.py"
    }
    $prevCwd = (Get-Location).ProviderPath
    try {
        Set-Location $dnRoot
        $normalsArgs = @(
            "-m", "dn_splatter.scripts.normals_from_pretrain",
            "--data-dir", $data,
            "--img-dir-name", "images",
            "--save-path", (Join-Path $data "normals_from_pretrain"),
            "--normal-format", "omnidata",
            "--resolution", "low"
        )
        & $dnPython @normalsArgs
    } finally {
        Set-Location $prevCwd
    }
    if ($LASTEXITCODE -ne 0) {
        throw "normals_from_pretrain failed with exit code $LASTEXITCODE. Make sure you activated the dn-splatter conda env and that omnidata_ckpt is under the dn-splatter checkout."
    }
}

if ($CheckOnly) {
    Invoke-Export -Check
    $exportExit = $LASTEXITCODE
    # Surface a friendly summary so the caller knows what is missing
    # without having to read the JSON.
    $monoDepthOk = ($exportExit -eq 0)
    $normalsPath = Join-Path $data "normals_from_pretrain"
    $normalsCount = 0
    if (Test-Path $normalsPath) {
        $normalsCount = (Get-ChildItem -Path $normalsPath -File | Where-Object {
            $_.Extension -match '^\.(png|jpg|jpeg)$'
        }).Count
    }
    if ($monoDepthOk -and $normalsCount -ge 418) {
        Write-Host "[vr_room DN] Check passed: mono_depth and normals_from_pretrain both ready."
        exit 0
    } elseif ($monoDepthOk) {
        Write-Host "[vr_room DN] Mono_depth OK but normals_from_pretrain has only $normalsCount files (need 418). Run: .\scripts\run_vr_room_dn_splatter.ps1 -Stage normals"
        exit 0
    } else {
        Write-Host "[vr_room DN] Check failed: mono_depth or transforms.json not ready. Run: .\scripts\run_vr_room_dn_splatter.ps1 -Stage export"
        exit $exportExit
    }
}

if (-not $SkipExport -and ($Stage -eq "export" -or $Stage -eq "all" -or $Stage -eq "normals")) {
    Write-Host "[vr_room DN] Exporting depth maps to mono_depth and rewriting transforms.json"
    Invoke-Export -Overwrite -RewriteTransforms
}

if ($Stage -eq "normals" -or $Stage -eq "all") {
    Write-Host "[vr_room DN] Generating normals_from_pretrain via dn-splatter (omnidata)"
    Invoke-Normals
}

if ($Stage -eq "export") {
    Write-Host "[vr_room DN] Export done. Generate normals: .\scripts\run_vr_room_dn_splatter.ps1 -Stage normals"
    return
}

if ($Stage -eq "all" -or $Stage -eq "normals") {
    Write-Host "[vr_room DN] Export and normals done. Train and render in the dn-splatter env:"
    Write-Host "  .\scripts\run_vr_room_dn_other_env.ps1 -CheckOnly"
    Write-Host "  .\scripts\run_vr_room_dn_other_env.ps1"
}
