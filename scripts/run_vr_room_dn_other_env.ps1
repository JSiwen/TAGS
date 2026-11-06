# DN-Splatter side of the vr_room e4b / e4 vs-DN comparison.
# Run this in the conda env that has nerfstudio and dn_splatter installed
# (editable install of D:\DN-Splatter\dn-splatter-main\dn-splatter-main, by
# the looks of `pip show dn-splatter` on this machine). It does not export
# depths and it does not call TAGS metrics.py.
#
# Data must already be prepared by the gaussian_splatting env:
#   .\scripts\run_vr_room_dn_splatter.ps1 -Stage export
#   .\scripts\run_vr_room_dn_splatter.ps1 -CheckOnly
#
# Same machine, script left in this repo:
#   conda activate dn-splatter
#   cd D:\TAGSs\TAGSs_core
#   .\scripts\run_vr_room_dn_other_env.ps1 -CheckOnly
#   .\scripts\run_vr_room_dn_other_env.ps1
#
# Script copied into the other environment:
#   .\run_vr_room_dn_other_env.ps1 `
#     -Data D:\TAGSs\TAGSs_core\data\vr_room `
#     -OutputRoot D:\TAGSs\TAGSs_core\output
#
# Smoke (1000 steps, separate output, not the comparison run):
#   .\scripts\run_vr_room_dn_other_env.ps1 -Smoke
#
# Run only one specific seed (e.g. add a missing seed=2 on top of an existing
# 0,1 run):  .\scripts\run_vr_room_dn_other_env.ps1 -Seeds @(2)
#
# Resume from a previous partial run (auto-detect latest per-seed checkpoint
# and continue from that step):
#   .\scripts\run_vr_room_dn_other_env.ps1 -Resume
#
# Force a specific resume step (e.g. from step 12000 instead of the latest):
#   .\scripts\run_vr_room_dn_other_env.ps1 -Resume -LoadStep 12000
#
# After render, score back in the gaussian_splatting env:
#   python scripts\summarize_vr_room_e4b_vs_dn.py

param(
    [string]$Data = "",
    [string]$OutputRoot = "",
    [int[]]$Seeds = @(0, 1, 2),
    [ValidateSet("train", "render", "all")]
    [string]$Stage = "all",
    [switch]$CheckOnly,
    [switch]$Smoke,
    [switch]$Force,
    [switch]$Resume,
    [int]$LoadStep = -1,
    [switch]$VerbosePaths,
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
    if (Test-Path (Join-Path $parent "data\vr_room")) {
        return $parent
    }
    return $null
}

$repoRoot = Resolve-DefaultRoot
if (-not $Data) {
    if ($env:VR_ROOM_SOURCE) {
        $Data = $env:VR_ROOM_SOURCE
    } elseif ($env:VR_ROOM_DATA) {
        $Data = $env:VR_ROOM_DATA
    } elseif ($null -ne $repoRoot) {
        $Data = Join-Path $repoRoot "data\vr_room"
    } else {
        throw "Pass -Data. This copy of the script is not next to data\vr_room."
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

$iterations = if ($env:VR_ROOM_DN_ITERS) { $env:VR_ROOM_DN_ITERS } elseif ($env:DN_ITERS) { $env:DN_ITERS } elseif ($Smoke) { "1000" } else { "30000" }
$experiment = if ($Smoke) { "vr_room_e4_vs_dn_smoke" } else { "vr_room_e4_vs_dn" }
# Ablation runs get their own experiment name so ns-train writes to a
# separate nerfstudio_models/ tree and never collides with the main run.
if ($Tag -and $Tag -ne "none") {
    $experiment = "{0}_{1}" -f $experiment, $Tag
}
$trainRoot = Join-Path $OutputRoot "vr_room_dn_train"
$renderBase = Join-Path $OutputRoot $(if ($Smoke) { "vr_room_dn_renders_smoke" } else { "vr_room_dn_renders" })
$plyBase = Join-Path $OutputRoot $(if ($Smoke) { "vr_room_dn_splatter_smoke" } else { "vr_room_dn_splatter" })
if ($Tag -and $Tag -ne "none") {
    $renderBase = "{0}_{1}" -f $renderBase, $Tag
    $plyBase = "{0}_{1}" -f $plyBase, $Tag
}
$skipCompleted = if ($Force) { $false } elseif ($env:DN_SKIP_COMPLETED) { $env:DN_SKIP_COMPLETED -ne "0" } else { $true }

# ------------------------------------------------------------------
# Tool resolution (this is what the previous question was about: what
# does this script actually call?). Print at the top of every run
# unless -VerbosePaths:$false, so a future reader can see exactly
# which DN-Splatter checkout / Python interpreter / ns-train binary
# is in use without having to guess.
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
    # `python -m pip show dn-splatter` is the authoritative source for the
    # editable project location. We deliberately use `python -m pip`
    # (instead of bare `pip`) so we query the same interpreter that the
    # rest of this script will actually run ns-train under; on this
    # machine bare `pip` resolves to base conda and prints
    # "Package(s) not found" even when dn-splatter is installed in the
    # active env, which is exactly the trap the original version fell
    # into.
    $py = if ($env:DN_PYTHON) { $env:DN_PYTHON } else { "python" }
    $lines = & $py -m pip show dn-splatter 2>$null
    $editable = $lines | Where-Object { $_ -match '^Editable project location:\s*(.+)$' } |
        ForEach-Object { ($_ -split ':\s*', 2)[1].Trim() } | Select-Object -First 1
    if ($editable) { return $editable }
    $loc = $lines | Where-Object { $_ -match '^Location:\s*(.+)$' } |
        ForEach-Object { ($_ -split ':\s*', 2)[1].Trim() } | Select-Object -First 1
    if ($loc) { return $loc }
    return "<unknown>"
}

# ------------------------------------------------------------------
# Resume helpers. nerfstudio writes checkpoints under
#   <seedDir>/nerfstudio_models/step-NNNNNNN.ckpt
# We pick the highest-step file in that directory and pass its parent
# dir as --load-dir and the step number as --load-step to ns-train.
# ------------------------------------------------------------------
function Find-LatestDnCheckpoint {
    param([string]$SeedDir)
    $ckptRoot = Join-Path $SeedDir "nerfstudio_models"
    if (-not (Test-Path $ckptRoot)) { return $null }
    $latest = Get-ChildItem -Path $ckptRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($latest) { return $latest.FullName }
    return $null
}

function Read-LatestCkptStep {
    param([string]$CkptDir)
    # nerfstudio convention: the checkpoint directory is named "step-NNNNNNN"
    $leaf = Split-Path -Leaf $CkptDir
    if ($leaf -match '^step-(\d+)$') { return [int]$Matches[1] }
    # Fallback: scan inside for step-*.ckpt files
    $files = Get-ChildItem -Path $CkptDir -Filter "step-*.ckpt" -File -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    if ($files.Count -gt 0 -and $files[0].Name -match '^step-(\d+)\.ckpt$') {
        return [int]$Matches[1]
    }
    return -1
}

Write-Host "[vr_room DN other-env] tool resolution:"
Write-Host ("  ns-train   -> {0}" -f (Resolve-ToolPath $nsTrain))
Write-Host ("  ns-export  -> {0}" -f (Resolve-ToolPath $nsExport))
Write-Host ("  python     -> {0}" -f (Resolve-ToolPath $python))
Write-Host ("  dn-splatter editable/location -> {0}" -f (Resolve-DnSplatterLocation))
Write-Host "[vr_room DN other-env] seeds=$($Seeds -join ',') iterations=$iterations experiment=$experiment data=$Data output=$OutputRoot"

# ------------------------------------------------------------------
# Input validation: images / mono_depth / normals / transforms.json all
# exist, align one-to-one, and transforms.json points at images/ .
# Note: images/ may contain extra jpg that have no COLMAP pose and were
# therefore skipped by export_vr_room_mono_depth.py (those frames end up
# neither in mono_depth nor in transforms.json). We mirror that filter
# here so the comparison is image set == mono_depth set == transforms
# set, all measured against COLMAP-registered frames.
# ------------------------------------------------------------------
function Get-RegisteredImageNames {
    # Use the dedicated helper instead of embedding Python here; an earlier
    # version tried to inline Python with literal quotes and the quoting
    # was mangled by PowerShell, producing invalid Python. The helper
    # wraps read_write_model.read_model and prints one basename per line.
    $py = if ($env:DN_PYTHON) { $env:DN_PYTHON } else { "python" }
    $helper = Join-Path $repoRoot "scripts\list_vr_room_registered_images.py"
    if (-not (Test-Path $helper)) {
        throw "Missing helper: $helper"
    }
    $output = & $py $helper --data $Data 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $output) {
        throw "Could not read registered images from sparse/0/images.bin via $helper. Run: .\scripts\run_vr_room_dn_splatter.ps1 -Stage export"
    }
    return @($output | Where-Object { $_ -match '\.(jpg|jpeg|png)$' })
}

function Get-ImageNames {
    param([string]$ImagesDir)
    Get-ChildItem -Path $ImagesDir -File | Where-Object {
        $_.Extension -match '^\.(jpg|jpeg|png)$'
    } | Sort-Object Name | ForEach-Object { $_.Name }
}

function Test-DnInputs {
    $imagesDir = Join-Path $Data "images"
    $depthDir  = Join-Path $Data "mono_depth"
    $normalDir = Join-Path $Data "normals_from_pretrain"
    $colmap    = Join-Path $Data "sparse\0\cameras.bin"
    $transformsPath = Join-Path $Data "transforms.json"

    foreach ($path in @($imagesDir, $depthDir, $colmap, $transformsPath)) {
        if (-not (Test-Path $path)) {
            throw "Missing $path. In the gaussian_splatting env run: .\scripts\run_vr_room_dn_splatter.ps1 -Stage export"
        }
    }

    $registered = @(Get-RegisteredImageNames)
    if ($registered.Count -eq 0) {
        throw "Could not read registered images from sparse/0/images.bin. Run: .\scripts\run_vr_room_dn_splatter.ps1 -Stage export"
    }
    $registeredStems = @($registered | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_) })
    $npyStems = @(Get-ChildItem -Path $depthDir -Filter "*_aligned.npy" -File | ForEach-Object {
        $_.BaseName.Substring(0, $_.BaseName.Length - "_aligned".Length)
    })
    $missing = @($registeredStems | Where-Object { $npyStems -notcontains $_ })
    $extra   = @($npyStems   | Where-Object { $registeredStems -notcontains $_ })
    if ($missing.Count -gt 0 -or $extra.Count -gt 0 -or $npyStems.Count -ne $registeredStems.Count) {
        throw "mono_depth/*_aligned.npy does not match COLMAP-registered images (registered=$($registeredStems.Count) missing=$($missing.Count) extra=$($extra.Count)). Re-export with -Stage export."
    }

    $normalCount = -1
    if (Test-Path $normalDir) {
        $normalCount = @(Get-ChildItem -Path $normalDir -File | Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' }).Count
    }

    $transforms = Get-Content -Raw -Path $transformsPath | ConvertFrom-Json
    $frames = @($transforms.frames)
    if ($frames.Count -eq 0) {
        throw "transforms.json has no frames."
    }
    $transformsStems = @($frames | ForEach-Object {
        $rel = ([string]$_.file_path).Replace("\", "/")
        [System.IO.Path]::GetFileNameWithoutExtension(($rel -split '/')[-1])
    })
    $missingFromTransforms = @($registeredStems | Where-Object { $transformsStems -notcontains $_ })
    if ($missingFromTransforms.Count -gt 0) {
        throw "transforms.json is missing $(($missingFromTransforms | Select-Object -First 3) -join ',')... (count=$($missingFromTransforms.Count)). In the gaussian_splatting env run: .\scripts\run_vr_room_dn_splatter.ps1 -Stage export"
    }
    $badPaths = @($frames | Where-Object {
        $rel = ([string]$_.file_path).Replace("\", "/")
        -not $rel.StartsWith("images/")
    })
    if ($badPaths.Count -gt 0) {
        throw "transforms.json still points outside images/. In the gaussian_splatting env run -Stage export."
    }

    $imageNames = @(Get-ImageNames $imagesDir)
    $holdout = @()
    for ($i = 0; $i -lt $registered.Count; $i++) {
        if (($i % 8) -eq 0) { $holdout += $registered[$i] }
    }
    Write-Host "[vr_room DN other-env] images_on_disk=$($imageNames.Count) registered=$($registered.Count) aligned_npy=$($npyStems.Count) normals=$normalCount hold8=$($holdout.Count) transforms_wh=$($transforms.w)x$($transforms.h)"
    Write-Host "[vr_room DN other-env] data=$Data"
    Write-Host "[vr_room DN other-env] output=$OutputRoot experiment=$experiment iterations=$iterations"
}

Test-DnInputs
if ($CheckOnly) {
    Write-Host "[vr_room DN other-env] Check passed."
    return
}

# ------------------------------------------------------------------
# Per-seed train + render + ply. Default 3 seeds; pass -Seeds @(0)
# for a single-seed run, or -Seeds @(2) to add a missing seed.
# ------------------------------------------------------------------
$useNormalLoss = $true
$useNormalTv   = $true
$normalDir = Join-Path $Data "normals_from_pretrain"
if (-not (Test-Path $normalDir)) {
    Write-Warning "[vr_room DN other-env] $normalDir missing; will run DN with --pipeline.model.use-normal-loss False"
    $useNormalLoss = $false
    $useNormalTv   = $false
} else {
    $normalCount = @(Get-ChildItem -Path $normalDir -File | Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' }).Count
    if ($normalCount -eq 0) {
        Write-Warning "[vr_room DN other-env] $normalDir has no images; will run DN with --pipeline.model.use-normal-loss False"
        $useNormalLoss = $false
        $useNormalTv   = $false
    }
}

# ------------------------------------------------------------------
# Ablation switches. depth loss was previously hardcoded True; it is
# now driven by -Ablate so we can isolate whether the (unscaled)
# mono_depth prior is what is breaking reconstruction quality.
# ------------------------------------------------------------------
$useDepthLoss = $true
switch ($Ablate) {
    "depth"  { $useDepthLoss = $false; Write-Host "[vr_room DN other-env] ABLATION: depth loss DISABLED" }
    "normal" { $useNormalLoss = $false; $useNormalTv = $false; Write-Host "[vr_room DN other-env] ABLATION: normal loss + normal TV loss DISABLED" }
    "both"   { $useDepthLoss = $false; $useNormalLoss = $false; $useNormalTv = $false; Write-Host "[vr_room DN other-env] ABLATION: depth + normal supervision DISABLED (RGB/SSIM only)" }
    default  { }
}
if ($Tag -and $Tag -ne "none") {
    Write-Host "[vr_room DN other-env] tag=$Tag experiment=$experiment render=$renderBase ply=$plyBase"
}

foreach ($s in $Seeds) {
    $timestamp  = "seed$s"
    $seedDir    = Join-Path $trainRoot "$experiment\dn-splatter\$timestamp"
    $config     = Join-Path $seedDir "config.yml"
    $renderDir  = Join-Path $renderBase ("eval_" + $timestamp)
    $plyDir     = Join-Path $plyBase    ("splat_" + $timestamp)

    Write-Host ""
    Write-Host "[vr_room DN other-env] === seed $s ==="
    Write-Host "  train  -> $seedDir"
    Write-Host "  render -> $renderDir"
    Write-Host "  ply    -> $plyDir"

    if ($Stage -eq "train" -or $Stage -eq "all") {
        if ($skipCompleted -and (Test-Path $config) -and -not $Resume) {
            Write-Host "[vr_room DN other-env] Skip train; config exists at $config"
        } else {
            New-Item -ItemType Directory -Force -Path $trainRoot | Out-Null
            # ns-train / tyro serializes bool arguments as the literal
            # strings "True" / "False" (Python repr), not "true" / "false".
            # Passing lowercased JSON-style booleans triggers:
            #   "invalid choice 'true' for argument ... Expected one of
            #   ('True', 'False')"
            # and ns-train exits 2 before any training starts. Convert
            # every PowerShell bool here with the same helper so this
            # bug can't regress on one of the four flags.
            #
            # Note: the depth and normal *file* loading is wired into
            # DN-Splatter's custom dataparser via `load_depths` and
            # `load_normals`, which are NOT exposed as CLI flags on
            # `ns-train dn-splatter`. We therefore set them through the
            # dn-splatter source default (load_normals=True) and through
            # the saved config.yml after ns-train writes it. Earlier
            # versions of this wrapper passed
            # `--pipeline.datamanager.dataparser.{load-normals,load-depths}`
            # which are not exposed by ns-train dn-splatter in this
            # build and triggered "Unrecognized options: ... True"
            # (tyro treats those as nargs="+"). They are intentionally
            # omitted here.
            function Script:BoolArg([bool]$v) { if ($v) { "True" } else { "False" } }

            $trainArgs = @(
                "dn-splatter",
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
                "--pipeline.model.normal-supervision", "mono"
            )
            if (-not $useNormalLoss) {
                # normal-supervision is meaningless with both normal losses
                # off; drop the flag so ns-train does not build a normal
                # datamanager field it will never consume.
                $trainArgs = @($trainArgs | Where-Object { $_ -ne "normal-supervision" })
            }

            # ------------------------------------------------------------------
            # Resume logic. ns-train supports --load-dir <ckpt_dir> and
            # --load-step <N> to continue a previous run instead of starting
            # from scratch. We auto-detect the highest-step checkpoint in
            # nerfstudio_models/ when -Resume is set, unless -LoadStep is
            # given (which forces a specific step).
            # ------------------------------------------------------------------
            $ckptDir = $null
            $resumeFromStep = -1
            if ($Resume) {
                $ckptDir = Find-LatestDnCheckpoint -SeedDir $seedDir
                if ($ckptDir) {
                    $resumeFromStep = if ($LoadStep -gt 0) { $LoadStep } else { Read-LatestCkptStep -CkptDir $ckptDir }
                    if ($resumeFromStep -gt 0) {
                        $trainArgs += @("--load-dir", $ckptDir, "--load-step", "$resumeFromStep")
                        Write-Host "[vr_room DN other-env] Resuming seed $s from step $resumeFromStep (ckpt=$ckptDir)"
                    } else {
                        Write-Host "[vr_room DN other-env] -Resume set but no checkpoint files in $ckptDir; starting fresh"
                    }
                } else {
                    Write-Host "[vr_room DN other-env] -Resume set but no $seedDir/nerfstudio_models/ found; starting fresh"
                }
            }
            if (-not $Resume -or -not $ckptDir -or $resumeFromStep -le 0) {
                Write-Host "[vr_room DN other-env] ns-train dn-splatter for $iterations steps (seed $s, from scratch)"
            }
            $trainArgs += @("coolermap", "--data", $Data, "--colmap-path", "sparse/0",
                            "--images-path", "images", "--eval-mode", "interval",
                            "--eval-interval", "8", "--downscale-factor", "1")
            # `$ErrorActionPreference = "Stop"` (set at the top of this
            # script) plus PowerShell's default behaviour of treating any
            # non-empty stderr from a native command as a
            # NativeCommandError causes `& $nsTrain` to throw on
            # innocuous Python RuntimeWarnings (e.g.
            # "Windows does not yet support torch.compile") before
            # ns-train has produced anything useful. We handle this in
            # two parts:
            #   1. Temporarily relax `$ErrorActionPreference` to
            #      "Continue" so the `NativeCommandError` does not
            #      abort the pipeline (we still check $LASTEXITCODE
            #      ourselves below, so a real failure is still caught).
            #   2. Tee merged stdout+stderr through a temp file so we
            #      can guarantee the file exists if we need to inspect
            #      it later. We do NOT remove the file in `finally`
            #      because the whole point is to keep ns-train's
            #      output around for post-mortem if it fails.
            #
            # PYTHONIOENCODING=utf-8 / PYTHONUTF8=1 are set because the
            # Windows console codepage on this machine is GBK (CP936).
            # When wandb's stdio capture worker thread tries to print a
            # non-ASCII character (a 🎉 from the "Training Finished"
            # banner) through a GBK console, Python raises
            # UnicodeEncodeError and crashes the worker thread mid-train.
            # Forcing UTF-8 stdio avoids that.
            $prevPyIo = $env:PYTHONIOENCODING
            $prevPyUtf8 = $env:PYTHONUTF8
            $prevEap = $ErrorActionPreference
            $env:PYTHONIOENCODING = "utf-8"
            $env:PYTHONUTF8 = "1"
            $nsTrainOut = [System.IO.Path]::GetTempFileName()
            try {
                $ErrorActionPreference = "Continue"
                & $nsTrain @trainArgs *>&1 | Tee-Object -FilePath $nsTrainOut -ErrorAction Continue | Out-Host
            } finally {
                $env:PYTHONIOENCODING = $prevPyIo
                $env:PYTHONUTF8 = $prevPyUtf8
                $ErrorActionPreference = $prevEap
                if ($LASTEXITCODE -ne 0 -and (Test-Path $nsTrainOut)) {
                    Write-Host "[vr_room DN other-env] ns-train output retained at: $nsTrainOut"
                } else {
                    Remove-Item $nsTrainOut -ErrorAction SilentlyContinue
                }
            }
            if ($LASTEXITCODE -ne 0) {
                throw "ns-train failed (seed $s) with exit code $LASTEXITCODE. Confirm dn-splatter env is active and ns-train is on PATH. If this is a resume, the checkpoint at $ckptDir step $resumeFromStep may be corrupt."
            }

            # ------------------------------------------------------------------
            # Sanity-check the saved config.yml for the same kind of mismatch
            # that just bit us: dn-splatter's CoolerMapDataParser has a
            # `load_normals` flag (defaulted in the third-party source) that
            # is NOT exposed via the `ns-train dn-splatter` CLI. If it's
            # False, the datamanager will silently hand `gt_normal = None`
            # to the model, which then crashes inside NormalLoss on the
            # first iteration:
            #     TypeError: unsupported operand type(s) for -:
            #                'Tensor' and 'NoneType'
            # because the loss does `torch.abs(pred - gt)`. We therefore
            # inspect the saved YAML after ns-train writes it and, if
            # `load_normals` is False while `normals_from_pretrain/`
            # actually exists with matching PNGs, we flip it to True in
            # place. ns-train constructs the pipeline from this same YAML
            # in a later phase (eval/render), so correcting the file now
            # means render-stage metrics are computed against the same
            # normal supervision that training used.
            # ------------------------------------------------------------------
            if (Test-Path $config) {
                $yamlText = Get-Content -Raw -Path $config
                $normalDir = Join-Path $Data "normals_from_pretrain"
                $hasNormalsDir = (Test-Path $normalDir) -and `
                    @(Get-ChildItem -Path $normalDir -Filter "*.png" -ErrorAction SilentlyContinue).Count -gt 0
                if ($hasNormalsDir -and ($yamlText -match '(?m)^(\s+)load_normals:\s*False\b')) {
                    Write-Host "[vr_room DN other-env] Patching saved config: load_normals: False -> True (normals_from_pretrain present)"
                    $yamlText = [regex]::Replace(
                        $yamlText,
                        '(?m)^(\s+)load_normals:\s*False\b',
                        '${1}load_normals: True'
                    )
                    Set-Content -Path $config -Value $yamlText -Encoding UTF8
                }
            }
        }
    }

    if ($Stage -eq "train") {
        continue
    }

    if (-not (Test-Path $config)) {
        throw "DN config not found: $config. Train first (-Stage train) or include this seed in the prior train run."
    }

    $predRgb = Join-Path $renderDir "pred\rgb"
    $predCount = 0
    if (Test-Path $predRgb) {
        $predCount = @(Get-ChildItem -Path $predRgb -File -ErrorAction SilentlyContinue | Where-Object {
            $_.Extension -match '^\.(png|jpg|jpeg)$'
        }).Count
    }
    if ($skipCompleted -and $predCount -gt 0) {
        Write-Host "[vr_room DN other-env] Skip render; $predCount images already in $predRgb"
    } else {
        Write-Host "[vr_room DN other-env] Rendering eval split into $renderDir"
        & $python -m dn_splatter.scripts.render_model `
            --load-config $config `
            --output-dir $renderDir `
            --split eval `
            --render-rgb `
            --no-render-depth `
            --no-render-normal
        if ($LASTEXITCODE -ne 0) {
            throw "DN render failed (seed $s) with exit code $LASTEXITCODE"
        }
    }

    New-Item -ItemType Directory -Force -Path $plyDir | Out-Null
    $ply = Join-Path $plyDir "splat.ply"
    if ($skipCompleted -and (Test-Path $ply)) {
        Write-Host "[vr_room DN other-env] Skip ply export; $ply exists"
    } else {
        Write-Host "[vr_room DN other-env] Exporting Gaussian ply (seed $s)"
        & $nsExport gaussian-splat --load-config $config --output-dir $plyDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[vr_room DN other-env] ns-export gaussian-splat failed (seed $s); point count will be missing from the summary."
        }
    }
}

$handoffName = if ($Tag -and $Tag -ne "none") { "vr_room_dn_other_env_done_{0}.txt" -f $Tag } else { "vr_room_dn_other_env_done.txt" }
$handoff = Join-Path $OutputRoot $handoffName
$handoffLines = @(
    "seeds=$($Seeds -join ',')",
    "iterations=$iterations",
    "ablate=$Ablate",
    "tag=$Tag",
    "use_depth_loss=$useDepthLoss",
    "use_normal_loss=$useNormalLoss",
    "use_normal_tv_loss=$useNormalTv",
    "config_dir=$trainRoot\$experiment\dn-splatter",
    "renders_dir=$renderBase",
    "ply_dir=$plyBase",
    "score_in_gaussian_splatting=python scripts\summarize_vr_room_e4b_vs_dn.py"
)
$handoffLines | Set-Content -Path $handoff -Encoding UTF8

Write-Host ""
Write-Host "[vr_room DN other-env] Done. Score in the gaussian_splatting env: python scripts\summarize_vr_room_e4b_vs_dn.py"
Write-Host "[vr_room DN other-env] Config: ablate=$Ablate use_depth_loss=$useDepthLoss use_normal_loss=$useNormalLoss use_normal_tv=$useNormalTv"
Write-Host "[vr_room DN other-env] Handoff: $handoff"
