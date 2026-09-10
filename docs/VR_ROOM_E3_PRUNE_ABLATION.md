# vr_room E3：剪枝策略消融

## 目标

在 E2 质量—规模结论之上，用**剪枝/增殖日程**而不是单纯硬预算，把 `vr_room` 压到 ≤2M，同时尽量保住相对无限制基线与 DN-Splatter 的 NVS 优势，并提升渲染 FPS。

参考基线：

| Method | Points | PSNR | SSIM | LPIPS |
|--------|-------:|-----:|-----:|------:|
| TAGS unlimited | 5.272M | 27.536 | 0.883 | 0.207 |
| TAGS E2 1.5M | 1.500M | 27.291 | 0.873 | 0.236 |
| TAGS E2 2.0M | 2.000M | 27.321 | 0.876 | 0.228 |
| DN-Splatter Within | 1.12M | 26.4 | 0.86 | 0.134 |

E2 表明：约 1.0–1.5M 硬预算已能在 PSNR/SSIM 上超过 DN。E3 验证哪些剪枝旋钮能在**不依赖硬预算**（或仅作安全帽）时达到类似规模与通过标准。

## 通过标准

1. Gaussian 数量 ≤ **2,000,000**
2. 相对无限制 PSNR 下降 ≤ **0.3 dB**（即 PSNR ≥ 约 27.236）
3. PSNR 仍高于 DN（> **26.4**）
4. 渲染 FPS 相对无限制至少提升 **50%**（≥ 1.5×）

## 代码回溯

E3 修改前完整源码标签：

```text
pre-e3
```

对应提交说明：`snapshot: preserve pre-E3 TAGSs codebase (post-E2)`。

## 新增训练旋钮

| 参数 | 含义 | 默认 |
|------|------|------|
| `--prune_opacity_threshold` | opacity 剪枝阈值 | 0.005 |
| `--prune_min_visibility` | 终身可见次数下限；0=关闭 | 0 |
| `--prune_max_screen_size` | 屏幕投影半径阈值；`<0` 旧行为，`0` 关闭，`>0` 固定阈值 | -1 |
| `--prune_grad_threshold` | 平均位置梯度贡献下限；0=关闭 | 0 |
| `--prune_grad_min_denom` | 梯度剪枝前最少观测次数 | 1 |
| `--densify_until_fraction` | 增殖停止点=iterations×fraction；0 用绝对 iter | 0 |
| `--prune_only_from_fraction` | 此进度后只剪不增 | 0→等同 densify 结束 |
| `--prune_until_fraction` | 剪枝持续到此进度；0=旧行为（随 densify 结束） | 0 |
| `--seed` | RNG 种子 | 0 |

实现要点：

- densify 结束后若 `prune_until_fraction=1.0`，继续周期性剪枝（E3 默认开启）。
- 可见性/梯度剪枝在 clone/split **之前**执行，避免 densify 清零统计后失效。
- `visibility_count` 为终身计数，不随 densify 重置。

## 消融矩阵（单种子筛选，seed=0）

基准配方：vr_room indoor sensor depth（与 E2 相同 geo prior / LR），再逐项改一个因素。

1. **opacity**：0.005 / 0.01 / 0.02  
2. **最低可见次数**：1 / 3 / 5  
3. **屏幕投影面积**：0（关）/ 10 / 20 / 40  
4. **梯度贡献**：1e-7 / 5e-7 / 1e-6  
5. **densify 停止 + 其后只剪不增**：50% / 65% / 80% / 90%（最后 10%–20% 对应 0.90 / 0.80）  
6. **组合候选**  
   - `combo_opacity0.01_densify0.50`  
   - `combo_opacity0.01_densify0.65_cap2m`（2M 安全帽）

## 运行

```powershell
conda activate gaussian_splatting

# Phase A: 单种子全矩阵
.\scripts\run_vr_room_e3_prune_ablation.ps1

# 只跑部分
$env:E3_PROFILES = "opacity_0.01,densify_0.50,combo_opacity0.01_densify0.50"
.\scripts\run_vr_room_e3_prune_ablation.ps1

# 汇总 + FPS（会测 unlimited 基线与各 E3 case）
python scripts\summarize_vr_room_e3.py --measure-fps 1
```

输出目录：`output/vr_room_e3_<profile>/`。

### 断点续训与 checkpoint

- 若已有 `point_cloud/iteration_30000/point_cloud.ply`，默认跳过该 case。
- 否则自动选取目录中**迭代号最大**的 `chkpnt*.pth`，传入 `--start_checkpoint` 续训。
- 默认每 **3k** 保存一次 checkpoint：`3000…30000`（可用 `E3_CHECKPOINT_ITERS` 覆盖）。
- 关闭自动续训：`$env:E3_AUTO_RESUME = "0"`。
- 强制重跑已完成 case：`$env:E3_SKIP_COMPLETED = "0"`。

## 多种子复现（最优 2 个配置 × 3 种子）

单种子筛选后，从 `output/vr_room_e3_summary.json` 的 `recommended_top2_for_multiseed` 取值：

```powershell
$env:E3_PHASE = "multiseed"
$env:E3_TOP_CONFIGS = "densify_0.50,combo_opacity0.01_densify0.50"  # 替换为实际 top2
$env:E3_SEEDS = "0,1,2"
.\scripts\run_vr_room_e3_prune_ablation.ps1
python scripts\summarize_vr_room_e3.py --measure-fps 1
```

## 判读流程

1. 先看 `num_points ≤ 2M` 与 `psnr_drop ≤ 0.3`。  
2. 确认 `beats_dn_psnr`。  
3. 用 `scripts/benchmark_render_fps.py` / summarizer 的 `fps_speedup_vs_unlimited ≥ 1.5`。  
4. 在通过门控的配置中，优先选 **点数更低且 PSNR 下降更小** 者进入多种子。  
5. 若纯剪枝难以同时满足 ≤2M 与 ≤0.3 dB，记录失败原因，并对比带 `cap2m` 的组合；E2 硬预算仍可作为对照，不替代 E3 结论。

## 结果记录

实验完成后将表格追加到本节，并更新：

- `output/vr_room_e3_summary.json`
- `output/vr_room_e3_summary.csv`
