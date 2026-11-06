# TAGSs 续跑 (Resume) Skill — 全仓库总览

> **建立时间**: 2026-10-14 (配套 exp4 v2 wandb 集成)
> **最后更新**: 2026-10-15 (加 v3 6 个 PS1 + 1/4 用户决策备选)
> **目标**: 一份文档说清"每个脚本的续跑怎么开 / 怎么关 / 边界 case 怎么解"
> **配套**: `EXP4_RUN_BOOK.md` (操作手册) + `SECTION_4_6_FULL_COMPARISON.md` (实验矩阵)

---

## 0) TL;DR — 3 个开关

| 开关 | 含义 | 默认 | 关掉怎么设 |
|---|---|:---:|---|
| `E4_SKIP_COMPLETED` | final ply 在就跳过整个 case (不重训/不重渲染/不重评分) | **1** (开) | `$env:E4_SKIP_COMPLETED = "0"` |
| `E4_AUTO_RESUME` | 找到 chkpnt*.pth 就 inject `--start_checkpoint` 续跑 | **1** (开) | `$env:E4_AUTO_RESUME = "0"` |
| `EXP4_FORCE_RETRAIN` | 强制从头跑 (删 final ply + 删 chkpnt) | **0** (关) | `$env:EXP4_FORCE_RETRAIN = "1"` (只对 `exp4_runner.py` 有效) |

**实战默认值**: 两个都开, **断电/kill 之后重跑同一个脚本, 自动从最新 chkpnt 续, 已完成的自动跳过**.

---

## 1) 当前 17 个 .ps1 脚本的续跑状态

| 脚本 | 类型 | SKIP_COMPLETED | AUTO_RESUME | 你用吗 |
|---|---|:---:|:---:|---|
| `run_bicycle_e4_winner.ps1` | 4.5 主表 (bicycle) | ✅ | ✅ | **现在用** |
| `run_vr_room_e4_perceptual_ablation.ps1` | 4.5 主表 (vr_room) | ✅ | ✅ + **额外 chkpnt 兼容性检查** | **现在用** |
| `run_e4_multiseed_finals.ps1` | 4.5 编排器 | ✅ (env 透传) | ✅ (env 透传) | **现在用** |
| `run_bicycle_v15_30k_densify_fix.ps1` | v15 修复 | ✅ | ✅ | 以后用 |
| `run_bicycle_v15_fast_test_3080.ps1` | v15 smoke | ✅ | ✅ | 调试用 |
| `run_vr_room_e2_gaussian_budget.ps1` | E2 (规模扫描) | ✅ | ✅ | 复盘用 |
| `run_vr_room_e3_prune_ablation.ps1` | E3 (剪枝) | ✅ | ✅ | 复盘用 |
| `run_vr_room_e4b_geo_overlap.ps1` | E4b (几何重叠) | ✅ | ✅ | 复盘用 |
| `run_vr_room_dn_other_env.ps1` | DN baseline (vr_room) | ✅ | ✅ | 复盘用 |
| `run_bicycle_dn_other_env.ps1` | DN baseline (bicycle) | ✅ | ✅ | 复盘用 |
| `run_vr_room_dn_splatter.ps1` | DN baseline | ✅ | — | 复盘用 |
| `run_bicycle_dn_splatter.ps1` | DN baseline | — | — | 复盘用 |
| `run_bicycle_v4_restore.ps1` | legacy v4 | — | — | 不用 |
| `run_bicycle_vs_dn_final.ps1` | legacy vs DN | — | — | 不用 |
| `run_e3_post_wave4.ps1` | legacy post | — | — | 不用 |
| `run_e4_post.ps1` | legacy post | — | — | 不用 |
| `run_e4b_post.ps1` | legacy post | — | — | 不用 |

### v3 新增 6 个 PS1 (2026-10-15, Stage 1/2/4 编排)

| 脚本 | 类型 | SKIP_COMPLETED | AUTO_RESUME | wandb 集成 | 你用吗 |
|---|---|:---:|:---:|:---:|---|
| `run_prepare_extra_scenes.ps1` | Stage 1 预处理 (depth/normal/params/export) | ❌ (不跳过, 重新跑) | ❌ | ❌ (不上 wandb) | **现在用** |
| `run_qa_baseline.ps1` | Stage 2 QA baseline (1/2 或 1/4 no-pretrain) | ❌ (--skip-existing 走 `qa_baseline_results.csv`) | ❌ | ✅ (env 透传) | **现在用** |
| `run_exp4v3_A_method_comparison.ps1` | Stage 4 Group A (4 method × 4 scene × 3 seed) | ✅ (summary.json 在就跳过) | ✅ (env 透传到 Python) | ✅ (env 透传) | **现在用** |
| `run_exp4v3_B_module_ablation.ps1` | Stage 4 Group B (5 组 × 4 scene × 3 seed) | ✅ | ✅ | ✅ | **现在用** |
| `run_exp4v3_C_resolution_sensitivity.ps1` | Stage 4 Group C (3 组 × 4 scene × 3 seed, 1/4/1/2/std) | ✅ | ✅ | ✅ | **现在用** |
| `run_exp4v3_std_resolution_validation.ps1` | Stage 4 std 验证 (3 scene × 1 method × 3 seed, 1/1) | ✅ | ✅ | ✅ | **现在用** |

**v3 关键设计**:
- v3 PS1 都通过 `exp4_runner.run_full()` 跑, 续跑机制与 v2 一致 (chkpnt*.pth → `--start_checkpoint`, final ply → 跳过)
- v3 默认 1/2 (`images_2`), 1/4 (`images_4`) 作赶时间/低显存备选 — 切换只改 `E4_IMAGES_DIR`, **不需重训即可重跑 (新 outputs 在新路径)**
- v3 wandb 钩子在 PS1 头部, 默认 OFF (`$env:WANDB_ENABLED = "1"` 启用); 同一个 (scene, group, seed) 多次跑会写多个 wandb run (不覆盖)

`scripts/exp4_runner.py` (Python 编排器, 4.2/4.3/4.4 以后会用) — **v2 (2026-10-14) 补齐, 全部支持**.

---

## 2) 续跑的工作机制 (4 步)

以 `run_bicycle_e4_winner.ps1` 为例, 其它脚本同款.

### Step 1: 检测 final ply

```powershell
$finalPly = "output\bicycle_e4_lpips_ms_w005\point_cloud\iteration_30000\point_cloud.ply"
if ($skipCompleted -and (Test-Path $finalPly)) {
    Write-Host "[bicycle E4] Skip completed"
    continue
}
```

如果 final ply 在, **直接跳到下一个 case** (不重训/不重渲染/不重评分).

### Step 2: 找最新 chkpnt

```powershell
function Get-LatestCheckpoint {
    param([string]$ModelDir)
    # 找 chkpnt<N>.pth, 取最大的 N
}
```

### Step 3: 注入 --start_checkpoint

```powershell
if ($null -ne $resumeCkpt) {
    $trainArgs += @("--start_checkpoint", "$($resumeCkpt.Path)")
}
```

### Step 4: 续跑时 log 追加 RESUME 标记

```
==== RESUME 2026-10-14 14:00:00 from chkpnt24000 ====
```

方便审计 / 排查"这次跑的 iter 段".

### Step 5 (仅 vr_room): 兼容性检查

```powershell
# 跑前检查 chkpnt 的 (profile, seed, perceptual) 是否和当前 run 匹配
# 不匹配直接 throw, 不会用错的 chkpnt 续
```

bicycle 没有这步 (假设同一 model_dir 不会换 profile), vr_room 有 (因为 E4 ablation 跨 profile 共享 train_dir).

---

## 3) `exp4_runner.py` (Python 编排) 的续跑 API

```python
from scripts.exp4_runner import run_full, should_skip_run, find_latest_checkpoint

# 1) 预检查
train_dir = Path("output/exp4_bicycle/4_2_C_Reference/train/seed0")
if should_skip_run(train_dir, iterations=30000):
    print("已跑完, 跳过")
    sys.exit(0)

ckpt = find_latest_checkpoint(train_dir)
if ckpt:
    print(f"续跑 from chkpnt{ckpt['iteration']} ({ckpt['size_mb']:.1f} MB)")

# 2) 跑 (内部已含 skip + resume)
info = run_full(
    scene="bicycle",
    group="4_2_C_Reference",
    seed=0,
    extra_args=["--prune_ratio2", "0.04"],
    output_root=Path("output/exp4_bicycle"),
    skip_completed=True,   # 默认 True
    auto_resume=True,      # 默认 True
)
print(info)
```

返回的 `info` dict 新增两个字段:
- `skipped: bool` — 这次被跳过了 (因为 final ply 在)
- `resumed_from: int | None` — 从哪个 chkpnt iter 续跑的, None 表示从头

---

## 4) 实战场景速查

| 场景 | 命令 | 行为 |
|---|---|---|
| **第一次跑 5-seed** | `$env:E4_SEEDS = "0,1,2,3,4"; .\scripts\run_bicycle_e4_winner.ps1` | 全 5 个 seed 从头跑 |
| **跑到 seed=2 失败, 重跑** | 同上 | seed 0/1 完成 → skip; seed 2 找到 chkpnt21000 → 续跑; seed 3/4 从头 |
| **断电了, 明天重跑** | 重开 PowerShell, 重新设 `$env:E4_SEEDS = "0,1,2,3,4"`, 重跑 | 同上, 全部从最新 chkpnt 续 |
| **想从头跑 seed=2** | 手动删 `output\bicycle_e4_lpips_ms_w005_seed2\point_cloud` + `chkpnt*.pth` | 不删的话会被 skip 或 resume |
| **强制全 5-seed 从头** | `$env:E4_SKIP_COMPLETED = "0"; $env:E4_AUTO_RESUME = "0"; .\scripts\run_bicycle_e4_winner.ps1` | 关闭 2 个开关, 全 5 个从头 |
| **跑 5-seed 但想盯某个** | `$env:E4_SEEDS = "0"; .\scripts\run_bicycle_e4_winner.ps1` | 只跑 seed 0, 其它不动 |

---

## 5) 续跑的安全网 (出了事怎么救)

### 5.1 续跑用了错 chkpnt 怎么办

| 原因 | 解决 |
|---|---|
| 跨 profile 用了错 chkpnt (仅 vr_room) | **不会发生** — vr_room 跑前自动 check profile 兼容性, 不匹配直接 throw |
| 手贱改了 `--iterations` 但忘了 chkpnt 是 30k 跑的, 现在改 60k | 续跑会从 chkpnt30000 → 60k, **正常行为**, 但中间有 30k 的"学习率/调度"断层, 可能效果差. 解决: 删 chkpnt 从头 |
| 改 `--seed` 跑了同一 model_dir | seed 0 chkpnt + seed 1 跑 → seed 0 chkpnt 没用, 因为 train_optimized 内部用 seed 控制 init 噪声, 不从 chkpnt 继承. **不会污染结果**, 但浪费一次续跑. 解决: seed 0 的 model_dir 用 `seed0` 后缀 |

### 5.2 chkpnt 损坏怎么办

PLY/chkpnt 文件可能因为断电半写. 检测:
- 跑的时候 `train_console.log` 报错: `RuntimeError: Error(s) in loading state_dict`
- 解决: 删掉这个 chkpnt, 跑下一个 (脚本会从更早的 chkpnt 续, 没有就从头)

```powershell
# 例: 删 chkpnt30000.pth (假设它坏了), 让脚本从 chkpnt27000 续
Remove-Item output\bicycle_e4_lpips_ms_w005\chkpnt30000.pth
.\scripts\run_bicycle_e4_winner.ps1
```

### 5.3 续跑后 PSNR 暴跌怎么办

可能原因:
1. `--iterations` 改了, 但 chkpnt 是老 iter 跑的 (调度不匹配)
2. `extra_args` 改了 (LPIPS 介入时间等关键超参改了)
3. 数据/路径改了 (但 chkpnt 的 state_dict 还在旧数据上训过)

解决: 删 chkpnt + final ply 从头跑. `exp4_runner.py` 提供了 `--force-retrain`:
```bash
python scripts/exp4_runner.py --scene X --group Y --seed 0 \
    --output-root output/exp4_X --force-retrain
```

---

## 6) 4.2/4.3/4.4 以后建脚本时的模板 (新脚本必须带续跑)

```powershell
# 模板开头 (复制粘贴)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$skipCompleted = if ($env:E4_SKIP_COMPLETED) { $env:E4_SKIP_COMPLETED -ne "0" } else { $true }
$autoResume = if ($env:E4_AUTO_RESUME) { $env:E4_AUTO_RESUME -ne "0" } else { $true }

# (复制 Get-LatestCheckpoint 函数)

foreach ($case in $cases) {
    $modelDir = Join-Path $ROOT "output\$case"
    $finalPly = Join-Path $modelDir "point_cloud\iteration_30000\point_cloud.ply"

    if ($skipCompleted -and (Test-Path $finalPly)) {
        Write-Host "[exp4] Skip $case"
        continue
    }

    $resumeCkpt = $null
    if ($autoResume) {
        $resumeCkpt = Get-LatestCheckpoint -ModelDir $modelDir
    }

    $args = @("train_optimized.py", ...) + $extraArgs
    if ($null -ne $resumeCkpt) {
        $args += @("--start_checkpoint", "$($resumeCkpt.Path)")
    }

    & python @args
}
```

---

## 7) 关键文件清单 (v2, 2026-10-14)

| 文件 | 状态 | 说明 |
|---|:---:|---|
| `scripts/exp4_runner.py` | ✅ 补齐 | 加 `find_latest_checkpoint` / `should_skip_run` / `run_full` 内 4 处续跑 + `--no-skip-completed` / `--no-auto-resume` / `--force-retrain` CLI |
| `scripts/run_bicycle_e4_winner.ps1` | ✅ 原有 | 5 层续跑 |
| `scripts/run_vr_room_e4_perceptual_ablation.ps1` | ✅ 原有 | 6 层 (含 chkpnt 兼容性 check) |
| `scripts/run_e4_multiseed_finals.ps1` | ✅ 原有 (编排器) | env 透传 |
| `docs/EXP4_RESUME_SKILL.md` | ✅ 新建 | 本文档 |

---

## 8) 验证 (你已经做的)

```powershell
# 1) 单元测试 (刚才跑过)
python -c "from scripts.exp4_runner import should_skip_run, find_latest_checkpoint, is_run_completed; print(should_skip_run('output/bicycle_e4_lpips_ms_w005', 30000))"
# 应输出: True (因为 final ply 在)

# 2) CLI 注册 (刚才跑过)
python scripts\exp4_runner.py --help
# 应看到: --no-skip-completed / --no-auto-resume / --force-retrain
```

---

**owner**: 谁改 exp4 相关脚本, 谁负责保持这文档同步
**version**: v3 (2026-10-15, 加 v3 6 PS1 续跑状态 + 1/4 用户决策备选, wandb 集成)
