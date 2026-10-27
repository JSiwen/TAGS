# Watch TAGSs training progress in another terminal while training runs.
#
# Usage:
#   .\scripts\watch_train_progress.ps1
#   .\scripts\watch_train_progress.ps1 -ModelDir "output\v16_local_3080\bicycle_v16_3080_resolution_aware"
#   .\scripts\watch_train_progress.ps1 -LogPath "output\v16_local_3080\logs\bicycle_v16_3080_resolution_aware.log"
#
# Options:
#   -IntervalSec 5
#   -TargetIter 6000
#   -FollowLog     also tail training log if present

param(
    [string]$ModelDir = "",
    [string]$LogPath = "",
    [int]$IntervalSec = 5,
    [int]$TargetIter = 0,
    [switch]$FollowLog
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

function Get-TrainProcess {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "train_optimized\.py" }
}

function Invoke-ProgressPython {
    param([string[]]$CommandArgs)
    $condaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "gaussian_splatting" }
    if ($env:CONDA_DEFAULT_ENV -eq $condaEnv) {
        & python @CommandArgs 2>$null
        return $LASTEXITCODE
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        & conda run -n $condaEnv --no-capture-output python @CommandArgs 2>$null
        return $LASTEXITCODE
    }
    & python @CommandArgs 2>$null
    return $LASTEXITCODE
}

function Read-ProgressSnapshot {
    param([string]$Dir, [int]$Target)

    $savedIters = @()
    $pcRoot = Join-Path $Dir "point_cloud"
    if (Test-Path $pcRoot) {
        $savedIters = Get-ChildItem $pcRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "^iteration_\d+$" } |
            ForEach-Object {
                if ($_.Name -match "iteration_(\d+)") { [int]$Matches[1] }
            }
    }

    $progress = @{
        CurrentIter = ($savedIters | Measure-Object -Maximum).Maximum
        SavedIters = ($savedIters | Sort-Object)
        Loss = $null
        Points = $null
        TbStep = $null
    }
    if (-not $progress.CurrentIter) { $progress.CurrentIter = 0 }

    $pyArgs = @("scripts/read_train_progress.py", "--model_dir", $Dir, "--target_iter", "$Target")
    $raw = Invoke-ProgressPython -CommandArgs $pyArgs
    if ($LASTEXITCODE -eq 0 -and $raw) {
        foreach ($line in $raw) {
            if ($line -match "^current_iter=(\d+)") { $progress.CurrentIter = [int]$Matches[1] }
            if ($line -match "^train_total_loss=([0-9.eE+-]+)") { $progress.Loss = [double]$Matches[1] }
            if ($line -match "^scene_total_points=(\d+)") { $progress.Points = [int]$Matches[1] }
            if ($line -match "^tensorboard_step=(\d+)") { $progress.TbStep = [int]$Matches[1] }
            if ($line -match "^saved_checkpoint_iter=(\d+)") {
                $saved = [int]$Matches[1]
                if ($saved -gt 0 -and ($progress.SavedIters -notcontains $saved)) {
                    $progress.SavedIters = @($progress.SavedIters + $saved | Sort-Object -Unique)
                }
            }
        }
    }
    return $progress
}

function Get-GpuSnapshot {
    $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    if (-not $line) { return $null }
    $parts = $line -split ","
    if ($parts.Count -lt 3) { return $null }
    return @{
        Util = [int]$parts[0].Trim()
        MemUsed = [int]$parts[1].Trim()
        MemTotal = [int]$parts[2].Trim()
    }
}

$proc = Get-TrainProcess | Select-Object -First 1
if (-not $ModelDir) {
    if ($proc -and $proc.CommandLine -match "-m\s+(""[^""]+""|\S+)") {
        $ModelDir = $Matches[1].Trim('"')
    } else {
        $ModelDir = Join-Path $ROOT "output\v16_local_3080\bicycle_v16_3080_resolution_aware"
    }
}
if (-not [System.IO.Path]::IsPathRooted($ModelDir)) {
    $ModelDir = Join-Path $ROOT $ModelDir
}
if (-not $LogPath) {
    $caseName = Split-Path $ModelDir -Leaf
    $LogPath = Join-Path $ROOT "output\v16_local_3080\logs\$caseName.log"
}
if ($TargetIter -le 0) {
    if ($proc -and $proc.CommandLine -match "--iterations\s+(\d+)") {
        $TargetIter = [int]$Matches[1]
    } else {
        $TargetIter = 6000
    }
}

Write-Host "Watching training progress"
Write-Host "  Model:  $ModelDir"
Write-Host "  Target: $TargetIter iterations"
Write-Host "  Log:    $LogPath"
Write-Host "  Every:  ${IntervalSec}s  (Ctrl+C to stop)"
Write-Host ""

$lastLogLine = ""
while ($true) {
    $proc = Get-TrainProcess | Select-Object -First 1
    $now = Get-Date -Format "HH:mm:ss"
    Clear-Host
    Write-Host "=== TAGSs training watch [$now] ==="
    Write-Host ""

    if ($proc) {
        $started = [Management.ManagementDateTimeConverter]::ToDateTime($proc.CreationDate)
        $elapsed = (Get-Date) - $started
        $elapsedText = "{0:hh\:mm\:ss}" -f $elapsed
        Write-Host "[RUNNING] PID $($proc.ProcessId)  elapsed $elapsedText"
    } else {
        Write-Host "[STOPPED] train_optimized.py not found"
    }

    $gpu = Get-GpuSnapshot
    if ($gpu) {
        Write-Host ("[GPU] util {0}%  mem {1}/{2} MiB" -f $gpu.Util, $gpu.MemUsed, $gpu.MemTotal)
    }

    $snap = Read-ProgressSnapshot -Dir $ModelDir -Target $TargetIter
    $pct = if ($TargetIter -gt 0) { [math]::Min(100, 100.0 * $snap.CurrentIter / $TargetIter) } else { 0 }
    Write-Host ("[PROGRESS] iter {0}/{1} ({2:N1}%)" -f $snap.CurrentIter, $TargetIter, $pct)
    if ($snap.Loss -ne $null) {
        Write-Host ("[LOSS] train/total_loss = {0:N6}" -f $snap.Loss)
    }
    if ($snap.Points -ne $null) {
        Write-Host ("[SCENE] gaussians = {0}" -f $snap.Points)
    }
    if ($snap.SavedIters -and $snap.SavedIters.Count -gt 0) {
        Write-Host ("[SAVED] " + ($snap.SavedIters -join ", "))
    } else {
        Write-Host "[SAVED] none yet (first checkpoint may take a while during CPU init)"
    }

    if ($FollowLog -and (Test-Path $LogPath)) {
        Write-Host ""
        Write-Host "--- log tail ---"
        Get-Content $LogPath -Tail 8 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
    } elseif (Test-Path $LogPath) {
        $lines = Get-Content $LogPath -Tail 1 -ErrorAction SilentlyContinue
        if ($lines) {
            Write-Host ""
            Write-Host "--- latest log ---"
            Write-Host $lines[-1]
        }
    } else {
        Write-Host ""
        Write-Host "Tip: current run may not stream log until script update; iter/loss come from TensorBoard events."
    }

    if (-not $proc) {
        Write-Host ""
        Write-Host "Training process ended."
        break
    }
    Start-Sleep -Seconds $IntervalSec
}
