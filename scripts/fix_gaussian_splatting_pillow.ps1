# Fix Pillow "DLL load failed" on Windows when libtiff expects libdeflate.dll.
# Usage:
#   conda activate gaussian_splatting
#   .\scripts\fix_gaussian_splatting_pillow.ps1

$ErrorActionPreference = "Stop"

if (-not $env:CONDA_PREFIX) {
    Write-Error "Activate the target conda env first, e.g. conda activate gaussian_splatting"
}

$bin = Join-Path $env:CONDA_PREFIX "Library\bin"
$deflate = Join-Path $bin "deflate.dll"
$libdeflate = Join-Path $bin "libdeflate.dll"

if (-not (Test-Path $deflate)) {
    Write-Error "Missing $deflate. Try: conda install -n gaussian_splatting -c conda-forge libdeflate libtiff pillow"
}

if (-not (Test-Path $libdeflate)) {
    Copy-Item $deflate $libdeflate
    Write-Host "Created $libdeflate from deflate.dll"
} else {
    Write-Host "Already present: $libdeflate"
}

python -c "from PIL import Image; print('Pillow OK:', Image.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pillow still failed to import. Reinstall with: conda install -c conda-forge libtiff libdeflate pillow --force-reinstall"
}

python -c "import urllib3; import sys; sys.exit(0 if urllib3.__version__.startswith('1.') else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "urllib3 v2 is incompatible with Python 3.7 OpenSSL 1.1.0 in this env."
    Write-Host "Run: conda install -c conda-forge `"urllib3=1.26.18`" --force-reinstall -y"
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        conda install -c conda-forge "urllib3=1.26.18" --force-reinstall -y
    }
}

python -c "import urllib3; from torchvision import models; print('urllib3 OK:', urllib3.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "urllib3/torchvision import still failed."
}

Write-Host "Done."
