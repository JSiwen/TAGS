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
