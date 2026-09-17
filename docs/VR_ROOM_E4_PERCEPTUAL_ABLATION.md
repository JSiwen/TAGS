# vr_room E4：感知损失消融

## 目标

在 E3 已验证的 **2.5M-cap 剪枝日程** 上，仅改变感知损失，解决 `vr_room` 上 LPIPS 落后以及深度先验利用不足的问题。

E4 固定结构基线（与 E3 `combo_opacity0.02_densify0.50_cap2.5m` 相同）：

| 参数 | 值 |
|------|-----|
| `max_gaussians` | 2,500,000 |
| `prune_opacity_threshold` | 0.02 |
| `densify_until_fraction` | 0.50 |
| `prune_only_from_fraction` | 0.50 |
| `prune_until_fraction` | 1.0 |
| 迭代 | 30,000 |
| geo prior | 与 E3 相同（depth+normal，0→20k decay） |

## E3 交叉参考（三 seed 均值）

| 指标 | E3 2.5M 路线 |
|------|-------------|
| Points | 1.916M |
| PSNR | 27.470 dB |
| SSIM | 0.87913 |
| LPIPS | 0.22219 |
| FPS | 167.2（2.10× unlimited） |

E4 内部会重跑 `rgb_ssim` 三 seed 作为配对基线；若与 E3 偏差过大，应先排查代码/环境漂移。

## 实验矩阵（seed=0 筛选，8 组）

| Profile | Photo loss | LPIPS | 多尺度 |
|---------|------------|-------|--------|
| `rgb_l1` | L1 only (`lambda_dssim=0`) | 关 | 否 |
| `rgb_ssim` | L1+SSIM (`lambda_dssim=0.2`) | 关 | 否 |
| `lpips_w001` | L1+SSIM | 0.01，21k 阶跃 | 否 |
| `lpips_w003` | L1+SSIM | 0.03，21k 阶跃 | 否 |
| `lpips_w005` | L1+SSIM | 0.05，21k 阶跃 | 否 |
| `lpips_ms_w001` | L1+SSIM | 0.01，21k 阶跃 | 是（1×/0.5×/0.25×，权重 0.5/0.3/0.2） |
| `lpips_ms_w003` | L1+SSIM | 0.03，21k 阶跃 | 是 |
| `lpips_ms_w005` | L1+SSIM | 0.05，21k 阶跃 | 是 |

LPIPS 配置：

- `--lpips_loss_schedule step`：iter `< 21000` 权重为 0，iter `≥ 21000` 直接跳到目标权重
- `--lpips_loss_interval 1`
- VGG backbone，训练期缓存 LPIPS 网络

其余晚期损失（detail / PSNR / color-align / balanced-late / hard-view）全部关闭。

## 通过标准

### seed=0 预筛（每类保留 1 名）

相对同 seed 的 `rgb_ssim`：

1. LPIPS 改善 ≥ **3%**
2. PSNR drop ≤ **0.25 dB**
3. SSIM drop ≤ **0.003**

### 最终硬门槛（三 seed）

相对同 seed 的 `rgb_ssim` 配对比较：

1. 平均 LPIPS 降低 ≥ **5%**，且每个 seed 均不得恶化
2. 平均 PSNR drop ≤ **0.2 dB**，最差 seed ≤ **0.25 dB**
3. 平均 SSIM drop ≤ **0.002**，最差 seed ≤ **0.003**
4. 每 seed 最终点数 ≤ **2M**
5. FPS 相对 unlimited ≥ **1.5×**；相对 `rgb_ssim` 下降 >10% 记软警告

绝对交叉线（相对 E3 三 seed 均值）：LPIPS ≤ **0.21108**，PSNR ≥ **27.2697 dB**。

## 运行命令

```powershell
conda activate gaussian_splatting
cd D:\TAGSs\TAGSs_core
```

### Smoke test（1k iter，验证 LPIPS/显存）

```powershell
$env:E4_SMOKE = "1"
$env:E4_PROFILES = "lpips_w003"
.\scripts\run_vr_room_e4_perceptual_ablation.ps1
```

检查：

- iter 700 前 `train/lpips_weight=0`
- iter 700 后 `train/lpips_loss` 出现且有限
- 3080 峰值显存 <90%

### Wave 1 — seed=0 全矩阵

```powershell
Remove-Item Env:E4_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:E4_PROFILES -ErrorAction SilentlyContinue
.\scripts\run_vr_room_e4_perceptual_ablation.ps1
python scripts\summarize_vr_room_e4.py
```

### Wave 2 — 三 seed 决赛圈

从汇总 JSON 的 `recommended_multiseed_set` 取 top 单尺度 + top 多尺度，连同 `rgb_l1`、`rgb_ssim`：

```powershell
$env:E4_PHASE = "multiseed"
$env:E4_TOP_CONFIGS = "rgb_l1,rgb_ssim,lpips_w003,lpips_ms_w003"
$env:E4_SEEDS = "0,1,2"
.\scripts\run_vr_room_e4_perceptual_ablation.ps1
.\scripts\run_e4_post.ps1 -Stage final
```

### 汇总与诊断

```powershell
python scripts\summarize_vr_room_e4.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
python scripts\evaluate_nvs_per_view.py -m output\vr_room_e4_<profile> --iteration 30000
python scripts\evaluate_depth_per_view.py -m output\vr_room_e4_<profile> --iteration 30000
```

输出：

- `output/vr_room_e4_summary.json`
- `output/vr_room_e4_summary.csv`
- 各 case 内 `per_view.json`、`depth_per_view.json`、`live_logs/*.jsonl`

## 深度利用诊断

E4 **不修改** geo prior 时间表（20k 结束），LPIPS 21k 才启用。因此本轮只能回答：

- 晚期感知优化是否破坏已有几何
- 训练期 `geo/depth_*` 轨迹是否异常
- 训练后与单目 inverse-depth 的一致性是否下降

若深度指标仍不足，应另开 **E4b** 调整 geo 时间表，不能在本轮同时改动。

## 判读流程

1. 先看 `rgb_ssim` 三 seed 是否与 E3 2.5M 路线一致（防基线漂移）
2. seed=0 预筛：每类（单尺度 / 多尺度）保留 1 名
3. 决赛圈跑三 seed，检查 `multiseed_gate`
4. 对通过者做 per-view + depth consistency
5. 优先选择 LPIPS 改善最大且 PSNR/SSIM/FPS 综合最优者

## 单元测试

```powershell
python -m unittest tests.test_lpips_loss_builder -v
```

## 代码入口

| 组件 | 路径 |
|------|------|
| 训练 runner | `scripts/run_vr_room_e4_perceptual_ablation.ps1` |
| 后处理 | `scripts/run_e4_post.ps1` |
| 汇总 | `scripts/summarize_vr_room_e4.py` |
| 深度一致性 | `scripts/evaluate_depth_per_view.py` |
| 损失实现 | `experiments/loss_builder.py` |
| LPIPS 缓存 | `lpipsPyTorch/__init__.py` |
