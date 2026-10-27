# Detached launcher for the bicycle 1237x822 re-run.
#
# Runs the three cases sequentially and is intended to be started detached so
# it survives the calling shell being reaped:
#   powershell -ExecutionPolicy Bypass -File .\scripts\_launch_bicycle_rerun_detached.ps1
#
# Each case writes its own log under output\ (see run_bicycle_rerun_1237.ps1).
# The launcher log (output\bicycle_rerun_1237_master.log) only records the
# case boundaries and exit codes, not the training stream, so it stays small.

$ErrorActionPreference = "Continue"

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$master = Join-Path $ROOT "output\bicycle_rerun_1237_master.log"
$runner = Join-Path $ROOT "scripts\run_bicycle_rerun_1237.ps1"

$env:RERUN_ITERATIONS = "30000"
$env:RERUN_CASE = ""
$env:RERUN_OFFICIAL = "1"          # Case A is opt-in; we do want it.
$env:RERUN_MIN_FREE_MB = "1500"    # OOM guard for the uncapped official run.

function Say($msg) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" | Out-File -FilePath $master -Append -Encoding utf8
}

Say "detached launcher start (pid $PID)"

foreach ($case in @("e4", "v14", "official")) {
    $env:RERUN_CASE = $case
    Say "=== case $case start ==="
    try {
        & powershell -ExecutionPolicy Bypass -NoProfile -File $runner *>&1 |
            Out-File -FilePath $master -Append -Encoding utf8
        Say "=== case $case exit=$LASTEXITCODE ==="
    } catch {
        Say "=== case $case FAILED: $_ ==="
    }
}

Say "detached launcher done"
