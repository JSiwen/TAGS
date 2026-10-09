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
    [switch]$Force
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
$timestamp = "seed0"
$trainRoot = Join-Path $OutputRoot "bicycle_dn_train"
$config = Join-Path $trainRoot "$experiment\dn-splatter\$timestamp\config.yml"
$renderDir = Join-Path $OutputRoot $(if ($Smoke) { "bicycle_dn_renders_smoke" } else { "bicycle_dn_renders" })
$plyDir = Join-Path $OutputRoot $(if ($Smoke) { "bicycle_dn_splatter_smoke" } else { "bicycle_dn_splatter" })
$skipCompleted = if ($Force) { $false } elseif ($env:DN_SKIP_COMPLETED) { $env:DN_SKIP_COMPLETED -ne "0" } else { $true }

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
    $badPaths = @($frames | Where-Object {
        $rel = [string]$_.file_path
        $rel = $rel.Replace("\", "/")
        -not $rel.StartsWith("images_4/")
    })
    if ($badPaths.Count -gt 0) {
        throw "transforms.json still points outside images_4. In the gaussian_splatting env run -Stage export."
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

$nsTrain = if ($env:DN_NS_TRAIN) { $env:DN_NS_TRAIN } else { "ns-train" }
$nsExport = if ($env:DN_NS_EXPORT) { $env:DN_NS_EXPORT } else { "ns-export" }
$python = if ($env:DN_PYTHON) { $env:DN_PYTHON } else { "python" }

if ($Stage -eq "train" -or $Stage -eq "all") {
    if ($skipCompleted -and (Test-Path $config)) {
        Write-Host "[bicycle DN other-env] Skip train; config exists at $config"
    } else {
        New-Item -ItemType Directory -Force -Path $trainRoot | Out-Null
        Write-Host "[bicycle DN other-env] ns-train dn-splatter for $iterations steps"
        & $nsTrain dn-splatter `
            --vis tensorboard `
            --output-dir $trainRoot `
            --experiment-name $experiment `
            --timestamp $timestamp `
            --max-num-iterations $iterations `
            --pipeline.model.use-depth-loss True `
            --pipeline.model.depth-lambda 0.2 `
            --pipeline.model.depth-loss-type PearsonDepth `
            --pipeline.model.use-normal-loss True `
            --pipeline.model.use-normal-tv-loss True `
            --pipeline.model.normal-supervision mono `
            --pipeline.datamanager.dataparser.load-normals True `
            --pipeline.datamanager.dataparser.load-depths True `
            coolermap `
            --data $Data `
            --colmap-path sparse/0 `
            --images-path images_4 `
            --eval-mode interval `
            --eval-interval 8 `
            --downscale-factor 1
        if ($LASTEXITCODE -ne 0) {
            throw "ns-train failed with exit code $LASTEXITCODE. Activate the env that installed nerfstudio and dn_splatter, then confirm ns-train is on PATH."
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
    & $python -m dn_splatter.scripts.render_model `
        --load-config $config `
        --output-dir $renderDir `
        --split eval `
        --render-rgb `
        --no-render-depth `
        --no-render-normal
    if ($LASTEXITCODE -ne 0) {
        throw "DN render failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path $plyDir | Out-Null
$ply = Join-Path $plyDir "splat.ply"
if ($skipCompleted -and (Test-Path $ply)) {
    Write-Host "[bicycle DN other-env] Skip ply export; $ply exists"
} else {
    Write-Host "[bicycle DN other-env] Exporting Gaussian ply"
    & $nsExport gaussian-splat --load-config $config --output-dir $plyDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[bicycle DN other-env] ns-export gaussian-splat failed; point count may be missing from the summary."
    }
}

$handoff = Join-Path $OutputRoot "bicycle_dn_other_env_done.txt"
@(
    "config=$config"
    "renders=$renderDir\eval"
    "ply_dir=$plyDir"
    "score_in_gaussian_splatting=python scripts\summarize_bicycle_e4_vs_dn.py"
) | Set-Content -Path $handoff -Encoding UTF8

Write-Host "[bicycle DN other-env] Done. Score in the gaussian_splatting env: python scripts\summarize_bicycle_e4_vs_dn.py"
Write-Host "[bicycle DN other-env] Handoff: $handoff"
