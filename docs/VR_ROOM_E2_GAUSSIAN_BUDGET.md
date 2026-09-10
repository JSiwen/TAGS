# vr_room E2：固定 Gaussian 预算

## 目标

在保持 TAGS 重建质量优势的前提下，将 `vr_room` 的模型规模从
5,271,565 Gaussians 降至 1–2M，并建立质量—规模曲线。

当前无限制基线（30k、test 42 views）：

- Gaussians：5,271,565
- PSNR：27.5355
- SSIM：0.883112
- LPIPS：0.206957（越低越好）

E2 只改变 Gaussian 总量上限，其他训练参数、数据划分、随机种子和评估视角保持一致。

## 预算组

- 1.0M
- 1.5M
- 2.0M
- 3.0M
- 0（无限制对照；默认复用当前 5.27M 完整结果）

DN-Splatter 约 1.12M，因此 1.0M 是更严格且最关键的对照。如果 1.0M 已优于
DN，则可直接证明 TAGS 在更小规模下仍领先；如果 1.0M 未领先而 1.5M 领先，
再补跑精确的 1.12M：

```powershell
$env:E2_BUDGETS = "1120000"
.\scripts\run_vr_room_e2_gaussian_budget.ps1
```

不要从当前 5.27M checkpoint 裁剪后继续训练作为主结果。该做法改变了优化路径，
不能与从训练开始即受预算约束的结果公平比较。代码会拒绝载入超过当前预算的 checkpoint。

## 固定预算实现

参数 `--max_gaussians` 控制 densification 后存续 Gaussian 的硬上限：

- `0`：保持原始无限制行为。
- clone 和 split 共用剩余预算。
- 候选超出预算时按位置梯度保留最强候选，不做随机截断。
- split 的预算按“删除父 Gaussian 后的净增长”计算。
- opacity 和 oversized pruning 在达到预算后仍正常运行。
- 每轮日志记录 `scene/total_points`、`scene/gaussian_budget` 和
  `scene/budget_utilization`。

注意：split 的现有实现先追加子 Gaussian，再删除父 Gaussian，因此操作内部存在短暂的
临时峰值；操作完成后的存续数量不会超过预算。

## 运行

先激活现有环境，再从仓库根目录运行：

```powershell
conda activate gaussian_splatting
.\scripts\run_vr_room_e2_gaussian_budget.ps1
```

可只跑部分预算：

```powershell
$env:E2_BUDGETS = "1000000,1500000,2000000"
.\scripts\run_vr_room_e2_gaussian_budget.ps1
```

默认跳过已有 30k point cloud 的实验。强制重跑：

```powershell
$env:E2_SKIP_COMPLETED = "0"
.\scripts\run_vr_room_e2_gaussian_budget.ps1
```

默认命令只训练四个固定预算组，汇总脚本会自动读取
`output/vr_room_indoor_sensor_depth` 作为无限制对照。如需从头重跑无限制组：

```powershell
$env:E2_BUDGETS = "0"
.\scripts\run_vr_room_e2_gaussian_budget.ps1
```

输出目录使用 `output/vr_room_e2_budget_<预算标签>`。每组在 7k、10k、15k、
18k、24k、30k 评估，在 15k、24k、30k 保存 checkpoint，只在 30k 保存正式
point cloud。

## 汇总与 DN 对比

先准备 DN 参考 JSON，例如：

```json
{
  "num_points": 1120000,
  "psnr": 0.0,
  "ssim": 0.0,
  "lpips": 0.0
}
```

将占位指标替换为同一 `vr_room` test split、同一分辨率和同一 LPIPS backbone
下的真实结果，然后运行：

```powershell
python scripts\summarize_vr_room_e2.py `
  --dn-reference path\to\vr_room_dn_reference.json
```

汇总结果写入：

- `output/vr_room_e2_summary.json`
- `output/vr_room_e2_summary.csv`

默认把相邻预算 PSNR 增益低于 0.05 dB 的第一个预算标为饱和起点。可调整阈值：

```powershell
python scripts\summarize_vr_room_e2.py --saturation-threshold-db 0.03
```

## 判读规则

1. 先检查每组 `num_points <= budget`；无限制组不做此检查。
2. 重点报告 1.0M 相对 DN 的 PSNR、SSIM 和 LPIPS，而不是只报告单一综合结论。
3. PSNR 饱和点按相邻预算增益判断，同时检查 SSIM 是否仍有实质变化。
4. LPIPS 下降表示感知质量改善；汇总脚本会标记其随规模单调改善、单调恶化或混合。
5. 如果 1.0M 到 1.5M 跨过 DN 基线，补跑 1.12M，避免用线性插值代替实验。
6. 最终推荐预算应优先选择仍领先 DN 的最小规模；若 1.5M 与 2.0M 差异低于
   0.05 dB 且 LPIPS 无明显改善，优先选择 1.5M。

## 版本回溯

E2 修改前的完整源码已保存在本地 Git 提交：

```text
snapshot: preserve pre-E2 TAGSs codebase
```

## 2026-09-09 实验结果

四个固定预算组均完成 30k 训练，且实际 Gaussian 数未超过预算：

- 1.0M：999,874 points；PSNR 27.1490；SSIM 0.867909；LPIPS 0.246145。
- 1.5M：1,499,692 points；PSNR 27.2906；SSIM 0.872869；LPIPS 0.236089。
- 2.0M：1,999,465 points；PSNR 27.3212；SSIM 0.875730；LPIPS 0.227785。
- 3.0M：2,998,868 points；PSNR 27.4016；SSIM 0.879793；LPIPS 0.217978。
- 无限制：5,271,565 points；PSNR 27.5355；SSIM 0.883112；LPIPS 0.206957。

初步结论：

- 1.5M→2.0M 的 PSNR 仅增加 0.0306 dB，出现局部饱和；但 2.0M→3.0M
  又增加 0.0804 dB，因此不能断言 PSNR 从 2.0M 起持续饱和。
- LPIPS 随 Gaussian 数增加单调改善，没有发现规模增加导致感知质量恶化。
- 1.5M 是 PSNR/规模的明显拐点；若 LPIPS 更重要，2.0M 比 1.5M 仍有
  0.00830 的有效改善。
- 在目标范围内，1.0M 相对无限制损失 0.3865 dB，1.5M 损失 0.2450 dB，
  2.0M 损失 0.2144 dB。
## DN-Splatter（vr_room Within）对照

来源：matched 42-view holdout scorecard / comparison notes。

| Method | Gaussians | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|----------:|------:|------:|-------:|
| DN-Splatter Within | 1.12M | 26.4 | 0.86 | 0.134 |
| TAGS 1.0M | 0.999M | 27.149 | 0.868 | 0.246 |
| TAGS 1.5M | 1.500M | 27.291 | 0.873 | 0.236 |
| TAGS 2.0M | 2.000M | 27.321 | 0.876 | 0.228 |
| TAGS 3.0M | 2.999M | 27.402 | 0.880 | 0.218 |
| TAGS unlimited | 5.272M | 27.536 | 0.883 | 0.207 |

相对 DN 的差值（TAGS − DN）：

| TAGS budget | ΔPSNR | ΔSSIM | ΔLPIPS | 规模比 |
|-------------|------:|------:|-------:|-------:|
| 1.0M | +0.75 | +0.008 | +0.112 | 0.89× |
| 1.5M | +0.89 | +0.013 | +0.102 | 1.34× |
| 2.0M | +0.92 | +0.016 | +0.094 | 1.79× |
| 3.0M | +1.00 | +0.020 | +0.084 | 2.68× |
| unlimited | +1.14 | +0.023 | +0.073 | 4.71× |

### 结论

1. **NVS 表格指标（PSNR/SSIM）**：TAGS 在全部 E2 预算下均领先 DN。即便压缩到约 1.0M（略小于 DN 的 1.12M），仍有约 **+0.75 dB PSNR / +0.008 SSIM**。约 1.12M 不必补跑也能判定 PSNR/SSIM 仍优于 DN。
2. **规模效率**：DN 更紧凑；TAGS 无限制模型约为 DN 的 4.7×，但质量优势并非单纯靠点数堆出来——1.0M TAGS 已超过 DN。
3. **LPIPS**：当前数值上 DN 更好（0.134 vs TAGS 0.21–0.25），但双方 LPIPS 实现不同（torchmetrics vs TAGS 自带包），且 bicycle 上曾出现反向结果，**不宜把 vr_room LPIPS 绝对差值当作稳健结论**。在 TAGS 内部，LPIPS 随点数增加单调改善。
4. **推荐预算**：若目标是「压到 1–2M 且保持相对 DN 的 NVS 优势」，**1.5M** 是性价比拐点（相对 DN +0.89 dB，相对无限制仅 −0.25 dB）。若更看重 TAGS 内部感知指标继续下降，可取 **2.0M**。
5. **协议注意**：poses（mushroom json vs COLMAP）、rasterizer（gsplat vs Inria）不完全一致；图中 DN AbsRel/RMSE 几何指标 TAGS 尚未按同协议评测，几何优劣暂不排名。

下一阶段见 [`docs/VR_ROOM_E3_PRUNE_ABLATION.md`](VR_ROOM_E3_PRUNE_ABLATION.md)：在保持相对 DN 的 NVS 优势前提下，用剪枝策略把规模压到 ≤2M，并要求 FPS ≥1.5× 无限制基线。
