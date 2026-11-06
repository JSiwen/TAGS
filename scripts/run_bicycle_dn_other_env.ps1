# DN-Splatter side of the bicycle E4 comparison.
# Run this in the conda env that has nerfstudio and dn_splatter.
# It does not export depths and it does not call TAGS metrics.py.
#
# Data must already be prepared by the gaussian_splatting env:
#   .\scripts\run_bicycle_dn_splatter.ps1 -Stage export
#   .\scripts\run_bicycle_dn_splatter.ps1 -CheckOnly
#
# Same machine, script left in this repo:
#   conda activate dn-splatter
#   cd D:\TAGSs\TAGSs_core
#   .\scripts\run_bicycle_dn_other_env.ps1 -CheckOnly
#   .\scripts\run_bicycle_dn_other_env.ps1
#
# Script copied into the other environment:
#   .\run_bicycle_dn_other_env.ps1 `
#     -Data D:\TAGSs\TAGSs_core\data\bicycle `
#     -OutputRoot D:\TAGSs\TAGSs_core\output
#
# Smoke (1000 steps, separate output, not the comparison run):
#   .\scripts\run_bicycle_dn_other_env.ps1 -Smoke
#
# After render, score back in the gaussian_splatting env:
#   python scripts\summarize_bicycle_e4_vs_dn.py

param(
    [string]$Data = "",
    [string]$OutputRoot = "",
    [ValidateSet("train", "render", "all")]
    [string]$Stage = "all",
    [switch]$CheckOnly,
    [switch]$Smoke,
    [switch]$Force,
    # ------------------------------------------------------------------
    # Ablation support. Defaults ("none" / "") reproduce the main
    # comparison run byte-for-byte, so existing invocations are unaffected.
    #
    # -Ablate depth  : disable the mono-depth supervision loss
    # -Ablate normal : disable both the normal L1 loss and the normal TV loss
    # -Ablate both   : disable depth + normal supervision together
    #
    # -Tag <suffix> appends the suffix to the experiment / render / ply
    # directory names AND to the handoff file, so an ablation never
    # overwrites the main run's outputs.
    # ------------------------------------------------------------------
    [ValidateSet("none", "depth", "normal", "both")]
    [string]$Ablate = "none",
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"

function Resolve-DefaultRoot {
    $parent = Split-Path -Parent $PSScriptRoot
    if (Test-Path (Join-Path $parent "data\bicycle")) {
        return $parent
    }
    return $null
}

$repoRoot = Resolve-DefaultRoot
if (-not $Data) {
    if ($env:BICYCLE_SOURCE) {
        $Data = $env:BICYCLE_SOURCE
    } elseif ($null -ne $repoRoot) {
        $Data = Join-Path $repoRoot "data\bicycle"
    } else {
        throw "Pass -Data. This copy of the script is not next to data\bicycle."
    }
}
if (-not $OutputRoot) {
    if ($null -ne $repoRoot) {
        $OutputRoot = Join-Path $repoRoot "output"
    } else {
        throw "Pass -OutputRoot. This copy of the script is not inside the TAGS repo."
    }
}

$Data = (Resolve-Path $Data).Path
if (-not (Test-Path $OutputRoot)) {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path $OutputRoot).Path

$iterations = if ($env:BICYCLE_DN_ITERS) { $env:BICYCLE_DN_ITERS } elseif ($Smoke) { "1000" } else { "30000" }
$experiment = if ($Smoke) { "bicycle_e4_vs_dn_smoke" } else { "bicycle_e4_vs_dn" }
# Ablation runs get their own experiment name so ns-train writes to a
# separate nerfstudio_models/ tree and never collides with the main run.
if ($Tag -and $Tag -ne "none") {
    $experiment = "{0}_{1}" -f $experiment, $Tag
}
$timestamp = "seed0"
$trainRoot = Join-Path $OutputRoot "bicycle_dn_train"
$config = Join-Path $trainRoot "$experiment\dn-splatter\$timestamp\config.yml"
$renderDir = Join-Path $OutputRoot $(if ($Smoke) { "bicycle_dn_renders_smoke" } else { "bicycle_dn_renders" })
$plyDir = Join-Path $OutputRoot $(if ($Smoke) { "bicycle_dn_splatter_smoke" } else { "bicycle_dn_splatter" })
if ($Tag -and $Tag -ne "none") {
    $renderDir = "{0}_{1}" -f $renderDir, $Tag
    $plyDir = "{0}_{1}" -f $plyDir, $Tag
}
$skipCompleted = if ($Force) { $false } elseif ($env:DN_SKIP_COMPLETED) { $env:DN_SKIP_COMPLETED -ne "0" } else { $true }

# ------------------------------------------------------------------
# Tool resolution: print exactly which ns-train / ns-export / python /
# dn-splatter checkout this run will use, so a future reader does not
# have to guess. Env-var overrides (DN_NS_TRAIN / DN_NS_EXPORT /
# DN_PYTHON) win, matching the existing behaviour.
# ------------------------------------------------------------------
$nsTrain  = if ($env:DN_NS_TRAIN)  { $env:DN_NS_TRAIN  } else { "ns-train" }
$nsExport = if ($env:DN_NS_EXPORT) { $env:DN_NS_EXPORT } else { "ns-export" }
$python   = if ($env:DN_PYTHON)    { $env:DN_PYTHON    } else { "python" }

function Resolve-ToolPath {
    param([string]$Tool)
    $cmd = Get-Command $Tool -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return "<not-on-PATH: $Tool>"
}

function Resolve-DnSplatterLocation {
    # `pip show dn-splatter` is the authoritative source for the editable
    # project location. We invoke it through the dn-splatter env's `pip`
    # (not `python -m pip`) because on this machine the latter was observed
    # to drop the "Editable project location:" line and only print
    # "Location:". That subtle difference was the reason for adding this
    # resolver in the first place.
    $pip = if ($env:DN_PIP) { $env:DN_PIP } else { "pip" }
    $lines = & $pip show dn-splatter 2>$null
    $editable = $lines | Where-Object { $_ -match '^Editable project location:\s*(.+)$' } |
        ForEach-Object { ($_ -split ':\s*', 2)[1].Trim() } | Select-Object -First 1
    if ($editable) { return $editable }
    $loc = $lines | Where-Object { $_ -match '^Location:\s*(.+)$' } |
        ForEach-Object { ($_ -split ':\s*', 2)[1].Trim() } | Select-Object -First 1
    if ($loc) { return $loc }
    return "<unknown>"
}

Write-Host "[bicycle DN other-env] tool resolution:"
Write-Host ("  ns-train   -> {0}" -f (Resolve-ToolPath $nsTrain))
Write-Host ("  ns-export  -> {0}" -f (Resolve-ToolPath $nsExport))
Write-Host ("  python     -> {0}" -f (Resolve-ToolPath $python))
Write-Host ("  dn-splatter editable/location -> {0}" -f (Resolve-DnSplatterLocation))

function Get-ImageNames {
    param([string]$ImagesDir)
    Get-ChildItem -Path $ImagesDir -File | Where-Object {
        $_.Extension -match '^\.(jpg|jpeg|png)$'
    } | Sort-Object Name | ForEach-Object { $_.Name }
}

function Test-DnInputs {
    $imagesDir = Join-Path $Data "images_4"
    $depthDir = Join-Path $Data "mono_depth"
    $normalDir = Join-Path $Data "normals_from_pretrain"
    $colmap = Join-Path $Data "sparse\0\cameras.bin"
    $transformsPath = Join-Path $Data "transforms.json"

    foreach ($path in @($imagesDir, $depthDir, $normalDir, $colmap, $transformsPath)) {
        if (-not (Test-Path $path)) {
            throw "Missing $path. In the gaussian_splatting env run: .\scripts\run_bicycle_dn_splatter.ps1 -Stage export"
        }
    }

    $imageNames = @(Get-ImageNames $imagesDir)
    if ($imageNames.Count -eq 0) {
        throw "No images in $imagesDir"
    }
    $imageStems = @($imageNames | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_) })
    $npyStems = @(Get-ChildItem -Path $depthDir -Filter "*_aligned.npy" -File | ForEach-Object {
        $_.BaseName.Substring(0, $_.BaseName.Length - "_aligned".Length)
    })
    $missing = @($imageStems | Where-Object { $npyStems -notcontains $_ })
    $extra = @($npyStems | Where-Object { $imageStems -notcontains $_ })
    if ($missing.Count -gt 0 -or $extra.Count -gt 0 -or $npyStems.Count -ne $imageStems.Count) {
        throw "mono_depth/*_aligned.npy does not match images_4 (missing=$($missing.Count) extra=$($extra.Count)). Re-export with -Stage export."
    }

    $normalCount = @(Get-ChildItem -Path $normalDir -File | Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' }).Count
    if ($normalCount -eq 0) {
        throw "normals_from_pretrain has no images."
    }

    $transforms = Get-Content -Raw -Path $transformsPath | ConvertFrom-Json
    $frames = @($transforms.frames)
    if ($frames.Count -eq 0) {
        throw "transforms.json has no frames."
    }
    # If transforms.json still points to the wrong directory, try to fix it
    # silently rather than failing.  This guards against the transforms being
    # in an unexpected state (e.g. after a partial re-export).
    $badPaths = @($frames | Where-Object {
        $rel = [string]$_.file_path
        $rel = $rel.Replace("\", "/")
        -not $rel.StartsWith("images_4/")
    })
    if ($badPaths.Count -gt 0) {
        Write-Host "[bicycle DN other-env] WARNING: transforms.json has $($badPaths.Count) bad paths; auto-fixing with Python"
        $fixScript = @"
import json, sys
from pathlib import Path
p = Path(r'$transformsPath')
d = json.load(open(p, encoding='utf-8'))
w0, h0 = int(d['w']), int(d['h'])
# find a real image in images_4 to get true dimensions
import os, glob
imgs = sorted(glob.glob(r'$imagesDir\*.jpg')) or sorted(glob.glob(r'$imagesDir\*.png')) or sorted(glob.glob(r'$imagesDir\*.jpeg'))
if not imgs:
    print('ERROR: no images found in $imagesDir'); sys.exit(1)
import cv2
img = cv2.imread(imgs[0])
tw, th = int(img.shape[1]), int(img.shape[0])
print(f'  target size: {tw}x{th}, from: {imgs[0]}')
if w0 == tw and h0 == th:
    print('  size already matches; just fixing file_paths')
    sx = sy = 1.0
else:
    sx = tw / w0; sy = th / h0
    d['w'] = tw; d['h'] = th
    for k in ('fl_x','fl_y','cx','cy'):
        if k in d: d[k] = float(d[k]) * sx if k in ('fl_x','cx') else float(d[k]) * sy
for f in d['frames']:
    fp = str(f.get('file_path',''))
    if fp.startswith('images/') or fp.startswith('images\\'):
        f['file_path'] = 'images_4/' + fp.replace('images/','').replace('images\\','')
    f['w'] = tw; f['h'] = th
    for k in ('fl_x','fl_y','cx','cy'):
        if k in f: f[k] = float(f[k]) * sx if k in ('fl_x','cx') else float(f[k]) * sy
open(p, 'w', encoding='utf-8').write(json.dumps(d, indent=2))
print(f'  written {p}')
"@
        $fixOut = & $python -c $fixScript 2>&1
        Write-Host $fixOut
    }

    $holdout = @()
    for ($i = 0; $i -lt $imageNames.Count; $i++) {
        if (($i % 8) -eq 0) { $holdout += $imageNames[$i] }
    }
    Write-Host "[bicycle DN other-env] images=$($imageNames.Count) aligned_npy=$($npyStems.Count) normals=$normalCount hold8=$($holdout.Count) transforms_wh=$($transforms.w)x$($transforms.h)"
    Write-Host "[bicycle DN other-env] data=$Data"
    Write-Host "[bicycle DN other-env] output=$OutputRoot experiment=$experiment iterations=$iterations"
}

Test-DnInputs
if ($CheckOnly) {
    Write-Host "[bicycle DN other-env] Check passed."
    return
}

# ------------------------------------------------------------------
# Ablation switches for depth / normal losses. Defaults reproduce the
# main comparison run byte-for-byte.
# ------------------------------------------------------------------
$useDepthLoss = $true
$useNormalLoss = $true
$useNormalTv   = $true
switch ($Ablate) {
    "depth"  { $useDepthLoss = $false; Write-Host "[bicycle DN other-env] ABLATION: depth loss DISABLED" }
    "normal" { $useNormalLoss = $false; $useNormalTv = $false; Write-Host "[bicycle DN other-env] ABLATION: normal loss + normal TV loss DISABLED" }
    "both"   { $useDepthLoss = $false; $useNormalLoss = $false; $useNormalTv = $false; Write-Host "[bicycle DN other-env] ABLATION: depth + normal supervision DISABLED (RGB/SSIM only)" }
    default  { }
}
if ($Tag -and $Tag -ne "none") {
    Write-Host "[bicycle DN other-env] tag=$Tag experiment=$experiment render=$renderDir ply=$plyDir"
}
# BoolArg is needed because ns-train on this machine occasionally rejects
# bare "False" passed as a positional flag; lowercase "false" is fine.
function BoolArg { param([bool]$v) if ($v) { "True" } else { "False" } }

# $nsTrain / $nsExport / $python were resolved at the top of the script and
# printed alongside the dn-splatter editable location. Do not redefine here.

if ($Stage -eq "train" -or $Stage -eq "all") {
    if ($skipCompleted -and (Test-Path $config)) {
        Write-Host "[bicycle DN other-env] Skip train; config exists at $config"
    } else {
        New-Item -ItemType Directory -Force -Path $trainRoot | Out-Null
        Write-Host "[bicycle DN other-env] ns-train dn-splatter for $iterations steps"
        $trainArgs = @(
            $nsTrain, "dn-splatter",
            "--vis", "tensorboard",
            "--output-dir", $trainRoot,
            "--experiment-name", $experiment,
            "--timestamp", $timestamp,
            "--max-num-iterations", $iterations,
            "--pipeline.model.use-depth-loss", (BoolArg $useDepthLoss),
            "--pipeline.model.depth-lambda", "0.2",
            "--pipeline.model.depth-loss-type", "PearsonDepth",
            "--pipeline.model.use-normal-loss", (BoolArg $useNormalLoss),
            "--pipeline.model.use-normal-tv-loss", (BoolArg $useNormalTv),
            "coolermap",
            "--data", $Data,
            "--colmap-path", "sparse/0",
            "--images-path", "images_4",
            "--eval-mode", "interval",
            "--eval-interval", "8",
            "--downscale-factor", "1"
        )
        # Always strip any pre-existing --pipeline.model.normal-supervision
        # (the main-run template below no longer carries it, but be safe).
        $trainArgs = @($trainArgs | Where-Object { $_ -ne "normal-supervision" })
        # Re-inject only when useNormalLoss is on.
        if ($useNormalLoss) {
            $idx = $trainArgs.IndexOf("coolermap")
            if ($idx -gt 0) {
                $trainArgs = $trainArgs[0..($idx-1)] + @("--pipeline.model.normal-supervision", "mono") + $trainArgs[$idx..($trainArgs.Count-1)]
            }
        }
        # Note: dn-splatter 2025+ no longer exposes
        # --pipeline.datamanager.dataparser.load-normals / load-depths;
        # the coolermap dataparser auto-detects mono_depth/ and
        # normals_from_pretrain/ directories.
        # PowerShell will not splat an array via & $arr. Write the command
        # to a temp .cmd so cmd.exe handles the quoting for us.
        # ns-train output is piped through cmd.exe to a persistent log so we can
        # inspect it after the script exits (even on failure).  The temp file
        # is still created so we can reference it from the .cmd file itself.
        $ns = $trainArgs[0]
        $argList = ($trainArgs[1..($trainArgs.Count - 1)] | ForEach-Object {
            if ($_ -match '\s') { '"' + ($_ -replace '"', '""') + '"' } else { $_ }
        }) -join ' '
        $tmpCmd = Join-Path $env:TEMP ("bicycle_abl_train_{0}.cmd" -f ([guid]::NewGuid().Guid.Substring(0,8)))
        $persistentLog = Join-Path $OutputRoot ("_train_abl_{0}.log" -f $Tag)
        Set-Content -Path $tmpCmd -Value "@echo off`r`nset PYTHONIOENCODING=utf-8`r`nset PYTHONLEGACYWINDOWSSTDIO=utf-8`r`n`"$ns`" $argList > `"$persistentLog`" 2>&1" -Encoding ASCII
        $cmdExe = Join-Path $env:WINDIR "System32\cmd.exe"
        & $cmdExe /d /c $tmpCmd | Out-Null
        Remove-Item $tmpCmd -ErrorAction SilentlyContinue
        # Show last lines of the log for debugging even on success.
        if (Test-Path $persistentLog) {
            Get-Content $persistentLog -Tail 15
        }
        if ($LASTEXITCODE -ne 0) {
            throw "ns-train failed with exit code $LASTEXITCODE. See $persistentLog. Activate the env that installed nerfstudio and dn_splatter, then confirm ns-train is on PATH."
        }
    }
}

if ($Stage -eq "train") {
    Write-Host "[bicycle DN other-env] Train only. Render with -Stage render."
    return
}

if (-not (Test-Path $config)) {
    throw "DN config not found: $config. Train first."
}

$predRgb = Join-Path $renderDir "eval\pred\rgb"
$predCount = 0
if (Test-Path $predRgb) {
    $predCount = @(Get-ChildItem -Path $predRgb -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Extension -match '^\.(png|jpg|jpeg)$'
    }).Count
}
if ($skipCompleted -and $predCount -gt 0) {
    Write-Host "[bicycle DN other-env] Skip render; $predCount images already in $predRgb"
} else {
    Write-Host "[bicycle DN other-env] Rendering eval split into $renderDir"
    $renderLog = Join-Path $OutputRoot ("_render_abl_{0}.log" -f $Tag)
    $renderCmd = Join-Path $env:TEMP ("bicycle_abl_render_{0}.cmd" -f ([guid]::NewGuid().Guid.Substring(0,8)))
    $renderArgList = "-m dn_splatter.scripts.render_model --load-config `"$config`" --output-dir `"$renderDir`" --split eval --render-rgb --no-render-depth --no-render-normal"
    Set-Content -Path $renderCmd -Value "@echo off`r`nset PYTHONIOENCODING=utf-8`r`nset PYTHONLEGACYWINDOWSSTDIO=utf-8`r`n`"$python`" $renderArgList > `"$renderLog`" 2>&1" -Encoding ASCII
    $cmdExe2 = Join-Path $env:WINDIR "System32\cmd.exe"
    & $cmdExe2 /d /c $renderCmd | Out-Null
    Remove-Item $renderCmd -ErrorAction SilentlyContinue
    if (Test-Path $renderLog) { Get-Content $renderLog -Tail 10 }
    if ($LASTEXITCODE -ne 0) {
        throw "DN render failed with exit code $LASTEXITCODE. See $renderLog."
    }
}

New-Item -ItemType Directory -Force -Path $plyDir | Out-Null
$ply = Join-Path $plyDir "splat.ply"
if ($skipCompleted -and (Test-Path $ply)) {
    Write-Host "[bicycle DN other-env] Skip ply export; $ply exists"
} else {
    Write-Host "[bicycle DN other-env] Exporting Gaussian ply"
    $plyLog = Join-Path $OutputRoot ("_ply_abl_{0}.log" -f $Tag)
    $plyCmd = Join-Path $env:TEMP ("bicycle_abl_ply_{0}.cmd" -f ([guid]::NewGuid().Guid.Substring(0,8)))
    Set-Content -Path $plyCmd -Value "@echo off`r`nset PYTHONIOENCODING=utf-8`r`nset PYTHONLEGACYWINDOWSSTDIO=utf-8`r`n`"$nsExport`" gaussian-splat --load-config `"$config`" --output-dir `"$plyDir`" > `"$plyLog`" 2>&1" -Encoding ASCII
    $cmdExe3 = Join-Path $env:WINDIR "System32\cmd.exe"
    & $cmdExe3 /d /c $plyCmd | Out-Null
    Remove-Item $plyCmd -ErrorAction SilentlyContinue
    if (Test-Path $plyLog) { Get-Content $plyLog -Tail 10 }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[bicycle DN other-env] ns-export gaussian-splat failed; point count may be missing from the summary."
    }
}

$handoffName = if ($Tag -and $Tag -ne "none") { "bicycle_dn_other_env_done_{0}.txt" -f $Tag } else { "bicycle_dn_other_env_done.txt" }
$handoff = Join-Path $OutputRoot $handoffName
@(
    "seeds=$($timestamp)"
    "iterations=$iterations"
    "ablate=$Ablate"
    "tag=$Tag"
    "use_depth_loss=$useDepthLoss"
    "use_normal_loss=$useNormalLoss"
    "use_normal_tv_loss=$useNormalTv"
    "config=$config"
    "renders=$renderDir\eval"
    "ply_dir=$plyDir"
    "score_in_gaussian_splatting=python scripts\summarize_bicycle_e4_vs_dn.py"
) | Set-Content -Path $handoff -Encoding UTF8

Write-Host "[bicycle DN other-env] Done. Score in the gaussian_splatting env: python scripts\summarize_bicycle_e4_vs_dn.py"
Write-Host "[bicycle DN other-env] Handoff: $handoff"
