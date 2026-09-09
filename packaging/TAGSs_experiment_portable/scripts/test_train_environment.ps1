# Preflight checks before TAGSs training on Windows.
# Usage:
#   conda activate gaussian_splatting
#   .\scripts\test_train_environment.ps1

$ErrorActionPreference = "Stop"

function Write-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    $mark = if ($Ok) { "[OK]" } else { "[FAIL]" }
    Write-Host "$mark $Name"
    if ($Detail) { Write-Host "      $Detail" }
}

function Invoke-PythonCheck {
    param([string]$Code)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & python -c $Code 2>&1
        $lines = foreach ($item in $raw) {
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                $item.ToString()
            } else {
                "$item"
            }
        }
        return @{
            ExitCode = $LASTEXITCODE
            Output = ($lines -join "`n").Trim()
        }
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

$allOk = $true

Write-Host "=== TAGSs training environment check ==="
Write-Host ""

if (-not $env:CONDA_PREFIX) {
    Write-Check "Conda env active" $false "Run: conda activate gaussian_splatting"
    $allOk = $false
} else {
    Write-Check "Conda env active" $true $env:CONDA_PREFIX
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Check "python in PATH" $false "Activate conda env first"
    $allOk = $false
} else {
    Write-Check "python in PATH" $true $python.Source
}

if ($python) {
    $torch = Invoke-PythonCheck "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
    if ($torch.ExitCode -eq 0) {
        Write-Check "PyTorch import" $true ($torch.Output -replace "`n", " | ")
    } else {
        Write-Check "PyTorch import" $false $torch.Output
        $allOk = $false
        if ($torch.Output -match "4551|AppControl|应用程序控制|fbgemm") {
            Write-Host ""
            Write-Host "Windows blocked a PyTorch DLL (fbgemm.dll)."
            Write-Host "Try one of:"
            Write-Host "  1. Windows Security -> App & browser control -> disable Smart App Control"
            Write-Host "  2. Ask IT to allowlist: $env:CONDA_PREFIX\Lib\site-packages\torch\lib\"
            Write-Host "  3. Run training in WSL2 / Linux instead of native Windows"
            Write-Host "  4. Reinstall PyTorch in a fresh conda env under a user-writable path"
        }
    }

    $pillow = Invoke-PythonCheck "from PIL import Image; print(Image.__version__)"
    if ($pillow.ExitCode -eq 0) {
        Write-Check "Pillow import" $true $pillow.Output
    } else {
        Write-Check "Pillow import" $false $pillow.Output
        Write-Host "      Fix: .\scripts\fix_gaussian_splatting_pillow.ps1"
        $allOk = $false
    }
}

Write-Host ""
if ($allOk) {
    Write-Host "Environment OK. You can run training scripts."
    exit 0
}

Write-Host "Environment NOT ready. Fix the items above before training."
exit 1
