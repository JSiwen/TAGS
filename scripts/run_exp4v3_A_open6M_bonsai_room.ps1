# 启动 v3 Group A: bonsai/room × 4 方法 × 3 seed = 24 runs
# A 改动: --max-gaussians 6000000 (放开, 让 SPA 真正稀疏化)
# B 改动: 已有 (train_optimized.py + exp4_runner.py 写 runtime.json)
# E4_SKIP_COMPLETED=1: 已有 partial 的 bonsai/room 不会被重跑

$env:E4_SCENES = "bonsai,room"
$env:E4_GROUPS = "A1,A2,A3,A4"
$env:E4_SEEDS  = "0,1,2"
$env:E4_ITERATIONS  = "30000"
$env:E4_RESOLUTION  = "2"
$env:E4_SKIP_COMPLETED = "1"
$env:TAGGS_CPU_KNN_INIT  = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# wandb (沿用 terminal 1 的 project/group 区分)
$env:WANDB_ENABLED   = "1"
$env:WANDB_PROJECT   = "TAGSs_exp4v3_A_method_comparison"
$env:WANDB_ENTITY    = "jsweni-neu"
$env:WANDB_RUN_GROUP = "exp4v3_A_open6M"  # 区分于 "exp4v3_A_full" (2.5M)
$env:WANDB_TAGS      = "bonsai,room,exp4v3,method_comparison,open6M"
$env:NO_PROXY        = "api.wandb.ai,api.wandb.ai:443,wandb.ai,localhost,127.0.0.1"
$env:no_proxy        = $env:NO_PROXY
$env:WANDB_IGNORE_PROXY = "1"

# 输出到独立 log
$log = "output\exp4v3_A_open6M_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host "=== bonsai/room × 4 方法 × 3 seed = 24 runs (max_gaussians=6M, 放开) ===" -ForegroundColor Green
Write-Host "log: $log"
"---"
& "C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe" -u `
    scripts\run_exp4v3_A_method_comparison.py `
    --scenes bonsai,room `
    --groups A1,A2,A3,A4 `
    --seeds 0,1,2 `
    --iterations 30000 `
    --max-gaussians 6000000 `
    --skip-existing `
    *>&1 | Tee-Object -FilePath $log
