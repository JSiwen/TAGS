# vr_room E4b：几何—感知时间表重叠

## 目标

在冻结 E4 胜出配方 `lpips_ms_w005` 的前提下，**只延长 geo prior 衰减终点**，让 depth/normal 与 21k 阶跃启用的 LPIPS 产生重叠，检验结构和单目深度一致性是否能跟上感知收益。

E4 已证明：126/126 配对视角 LPIPS 均改善，但 SSIM 仅约 41/126 改善，且 geo 在 20k 结束、LPIPS 在 21k 才启动。E4b 不再调 LPIPS 权重。

## 冻结配方（与 E4 winner 相同）

| 参数 | 值 |
|------|-----|
| 感知损失 | 多尺度 LPIPS 0.05，21k step |
| `max_gaussians` | 2,500,000 |
| `prune_opacity_threshold` | 0.02 |
| `densify_until_fraction` / `prune_only_from_fraction` | 0.50 |
| geo warmup / decay start | 4k / 10k |
| 迭代 | 30,000 |

`decay20k` 默认**不重训**，直接复用 `output/vr_room_e4_lpips_ms_w005[_seedN]`。

## 实验矩阵

| Profile | geo decay until | depth override | normal override | 默认是否跑 |
|---------|----------------:|---------------:|----------------:|:----------:|
| `decay20k` | 20,000 | 继承 | 继承 | 否（复用 E4 freeze） |
| `decay24k` | 24,000 | 继承 | 继承 | 是（Wave 1） |
| `decay27k` | 27,000 | 继承 | 继承 | 是（Wave 1） |
| `depth_tail27k` | 27,000 | 27,000 | 20,000 | 否（仅当 B/C 过平滑） |

第一轮只改 `decay_until`。`depth_tail27k` 依赖 `--geo_prior_depth_decay_until` / `--geo_prior_normal_decay_until`（0 表示继承共享日程）。

## 通过标准

对照分两类：

- **感知**：相对同 seed 的 E4 `rgb_ssim`，平均 LPIPS 改善仍 ≥ **12%**，且每 seed 不得恶化。
- **结构 / 几何**：相对同 seed 的 E4 winner（`decay20k`）
  1. 平均 PSNR drop ≤ **0.05 dB**，最差 seed ≤ **0.08 dB**
  2. 平均 SSIM drop ≤ **0.0002**，最差 seed ≤ **0.001**
  3. 深度 log-L1 平均改善 ≥ **3%**，Spearman 不下降
  4. 每 seed 点数 ≤ **2M**
  5. FPS ≥ unlimited 的 **1.5×**

seed=0 预筛另看逐视角 SSIM 改善数（相对 `rgb_ssim`）≥ **18 / 42**；缺 `per_view.json` 时跳过该项。三 seed 上应明显高于 E4 winner 的 41/126。

单目 inverse-depth 一致性只作相对诊断，不是绝对几何精度。

## 运行命令

```powershell
conda activate gaussian_splatting
cd D:\TAGSs\TAGSs_core
```

### Smoke test

```powershell
$env:E4B_SMOKE = "1"
$env:E4B_PROFILES = "decay24k"
.\scripts\run_vr_room_e4b_geo_overlap.ps1
```

检查：iter 700 后 `train/lpips_loss` 有限；21k 前 `geo/depth_weight` 仍高于 E4 的 20k 尾值。

### Wave 1 — seed=0（decay24k / decay27k）

```powershell
Remove-Item Env:E4B_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:E4B_PROFILES -ErrorAction SilentlyContinue
.\scripts\run_vr_room_e4b_geo_overlap.ps1
.\scripts\run_e4b_post.ps1 -Stage final
```

### Wave 2 — 三 seed 决赛圈

从 `output/vr_room_e4b_summary.json` 的 `recommended_multiseed_set` 取最优非 freeze 配置：

```powershell
$env:E4B_PHASE = "multiseed"
$env:E4B_TOP_CONFIGS = "decay24k"
$env:E4B_SEEDS = "0,1,2"
.\scripts\run_vr_room_e4b_geo_overlap.ps1
.\scripts\run_e4b_post.ps1 -Stage final
```

### 条件配置 D（depth-only tail）

仅当 `decay24k` / `decay27k` 出现过平滑（SSIM/PSNR 明显低于 freeze）时：

```powershell
Remove-Item Env:E4B_PHASE -ErrorAction SilentlyContinue
$env:E4B_PROFILES = "depth_tail27k"
$env:E4B_SEEDS = "0"
.\scripts\run_vr_room_e4b_geo_overlap.ps1
```

若必须重训 freeze 以排查代码漂移：

```powershell
$env:E4B_RETRAIN_FREEZE = "1"
$env:E4B_PROFILES = "decay20k"
.\scripts\run_vr_room_e4b_geo_overlap.ps1
```

## 输出

- `output/vr_room_e4b_<profile>[_seedN]/`
- `output/vr_room_e4b_summary.json`
- `output/vr_room_e4b_summary.csv`

## 判读流程

1. 确认 freeze `decay20k` 三 seed 与 E4 winner 一致。
2. seed=0：看 21k–27k 的 `geo/depth_weight` 是否仍高于 E4 尾值。
3. 预筛通过者补 seed 1/2，检查 `multiseed_gate`。
4. 优先选 SSIM 不低于 freeze、深度 log-L1 改善最大、且 LPIPS 仍 ≥12% 的配置。
5. 若延长 normal 伤害细节，再跑 `depth_tail27k`。

## 单元测试

```powershell
python -m unittest tests.test_geo_prior_schedule tests.test_summarize_vr_room_e4b -v
```

## 代码入口

| 组件 | 路径 |
|------|------|
| 训练 runner | `scripts/run_vr_room_e4b_geo_overlap.ps1` |
| 后处理 | `scripts/run_e4b_post.ps1` |
| 汇总 | `scripts/summarize_vr_room_e4b.py` |
| 分项衰减 | `experiments/geo_prior.py` |
| CLI | `--geo_prior_depth_decay_until` / `--geo_prior_normal_decay_until` |
