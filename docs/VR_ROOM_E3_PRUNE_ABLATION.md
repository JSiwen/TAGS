# vr_room E3：剪枝策略消融

## 目标

在 E2 质量—规模结论之上，用**剪枝/增殖日程**把 `vr_room` 压到 ≤2M，同时尽量保住相对无限制基线与 DN-Splatter 的 NVS 优势，并提升渲染 FPS。

参考基线：

| Method | Points | PSNR | SSIM | LPIPS |
|--------|-------:|-----:|-----:|------:|
| TAGS unlimited | 5.272M | 27.536 | 0.883 | 0.207 |
| TAGS E2 1.5M | 1.500M | 27.291 | 0.873 | 0.236 |
| TAGS E2 2.0M | 2.000M | 27.321 | 0.876 | 0.228 |
| DN-Splatter Within | 1.12M | 26.4 | 0.86 | 0.134 |

E2 表明：约 1.0–1.5M 硬预算已能在 PSNR/SSIM 上超过 DN。E3 验证哪些剪枝旋钮能在 2M 安全帽下达到类似规模与通过标准。v2 起默认 `--max_gaussians 2000000`，与通过标准第 1 条对齐，并避免 3080 上出现 4M+ 爆显存。

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

## v1 失败原因（已在 v2 修复）

cap2m 三组在 densify 阶段顶着 2M、PSNR ~26.5–26.9，一进入 prune-only 就崩盘（最终 PSNR ~3.74，点数几千到 2 万）：

- `prune_until_fraction=1.0` 在停止增殖后仍每 100 iter 跑完整 `densify_and_prune`。
- 默认 `prune_max_screen_size=-1` 会沿用半径 20 + 世界尺度 oversized 剪枝；没有 clone/split 补回时，一轮就从 2.00M 砍到 ~62 万，随后持续失血。
- `--test_iterations` 缺少 10k/18k/19.5k；且评估写在剪枝之前，15k/24k 的 PSNR 是剪枝前快照。

v2 修复：

- prune-only 阶段关闭 oversized 屏幕/世界剪枝，只保留 opacity / visibility / gradient 剪枝。
- 评估与 stage snapshot 改到 densify/prune **之后**。
- 评估点：7000, 10000, 15000, 18000, 19500, 24000, 30000。
- 输出目录改为 `output/vr_room_e3_v2_<profile>/`，不与已崩盘的 v1 结果混用。

## 新增训练旋钮

| 参数 | 含义 | 默认 |
|------|------|------|
| `--prune_opacity_threshold` | opacity 剪枝阈值 | 0.005 |
| `--prune_min_visibility` | 终身可见次数下限；0=关闭 | 0 |
| `--prune_max_screen_size` | 屏幕投影半径；`<0` 仅 densify 期用旧阈值 20，`0` 关闭，`>0` 固定阈值。prune-only 一律关闭 oversized | -1 |
| `--prune_grad_threshold` | 平均位置梯度贡献下限；0=关闭 | 0 |
| `--prune_grad_min_denom` | 梯度剪枝前最少观测次数 | 1 |
| `--densify_until_fraction` | 增殖停止点=iterations×fraction | 0 |
| `--prune_only_from_fraction` | 此进度后只剪不增；若晚于 densify 停止点，中间为纯优化 | 0→等同 densify 结束 |
| `--prune_until_fraction` | 剪枝持续到此进度；0=随 densify 结束 | 0 |
| `--max_gaussians` | E2 风格硬上限；E3 v2 默认 2M | 2000000 |
| `--seed` | RNG 种子 | 0 |

实现要点：

- densify 结束后若 `prune_until_fraction=1.0`，继续周期性 **opacity/可见性/梯度** 剪枝，不再砍大高斯。
- 可见性/梯度剪枝在 clone/split **之前**执行，避免 densify 清零统计后失效。
- `visibility_count` 为终身计数，不随 densify 重置。
- `opacity_reset` 仍只在 densify 期内执行。

## 消融矩阵（单种子筛选，seed=0）

基准配方：与 E2 相同的 geo prior / LR；densify 到 60%（18k）；**最后 20%（24k–30k）只剪不增**；2M 安全帽。逐项只改一个因素。

1. **opacity**：`opacity_0.005` / `opacity_0.01` / `opacity_0.02`
2. **最低可见次数**：`minvis_1` / `minvis_3` / `minvis_5`
3. **屏幕投影面积**（仅 densify 期生效）：`screen_0` / `screen_10` / `screen_20` / `screen_40`
4. **梯度贡献**：`grad_1e-7` / `grad_5e-7` / `grad_1e-6`
5. **densify 停止 + 其后只剪不增**：`densify_0.50` / `densify_0.65` / `densify_0.80`（80% = 最后 20%）
6. **最后 10% 只剪不增**：`lateprune_0.90`（densify 仍到 60%）
7. **组合**：`combo_opacity0.01_densify0.50`、`combo_opacity0.01_densify0.80`

共 19 个单种子 case。

## 推荐训练方案（3080）

先激活环境。v1 崩盘目录可留作对照，v2 不会覆盖它们。

```powershell
conda activate gaussian_splatting
cd D:\TAGSs\TAGSs_core
```

### Wave 1 — 日程与基线（先确认不再崩盘）

```powershell
$env:E3_PROFILES = "opacity_0.005,densify_0.50,densify_0.65,densify_0.80,lateprune_0.90"
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

看 15k / 18k / 19.5k / 24k / 30k 的点数应仍在 2M 附近，PSNR 不应掉到个位数。若 `densify_0.80` 的 24k→30k 仍稳定，再开全矩阵。

### Wave 2 — 其余单因素

```powershell
$env:E3_PROFILES = "opacity_0.01,opacity_0.02,minvis_1,minvis_3,minvis_5,screen_0,screen_10,screen_20,screen_40,grad_1e-7,grad_5e-7,grad_1e-6"
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

### Wave 3 — 组合

```powershell
$env:E3_PROFILES = "combo_opacity0.01_densify0.50,combo_opacity0.01_densify0.80"
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

也可一次跑完全矩阵（已完成的会自动跳过）：

```powershell
Remove-Item Env:E3_PROFILES -ErrorAction SilentlyContinue
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

### 汇总

```powershell
python scripts\summarize_vr_room_e3.py --measure-fps 1
```

输出：`output/vr_room_e3_v2_<profile>/`。

### 断点续训

- 已有 `point_cloud/iteration_30000/point_cloud.ply` 则跳过。
- 否则从最大 `chkpnt*.pth` 续训。
- 默认每 3k 存盘。强制重跑：`$env:E3_SKIP_COMPLETED = "0"`。

## 多种子复现（最优 2 个配置 × 3 种子）

从 `output/vr_room_e3_summary.json` 的 `recommended_top2_for_multiseed` 取值。通过门控的优先；否则取 PSNR 下降最小且点数 ≤2M 的两个。

```powershell
$env:E3_PHASE = "multiseed"
$env:E3_TOP_CONFIGS = "densify_0.80,opacity_0.01"  # 替换为实际 top2
$env:E3_SEEDS = "0,1,2"
.\scripts\run_vr_room_e3_prune_ablation.ps1
python scripts\summarize_vr_room_e3.py --measure-fps 1
```

## 判读流程

1. 先看 `num_points ≤ 2M` 与 `psnr_drop ≤ 0.3`。
2. 确认 `beats_dn_psnr`。
3. `fps_speedup_vs_unlimited ≥ 1.5`。
4. 通过门控的配置中，优先选 **点数更低且 PSNR 下降更小** 者进入多种子。
5. 与 E2 2.0M（PSNR 27.321）对照：E3 若点数更低且 PSNR 不更差，才说明剪枝日程优于单纯硬预算。

## 结果记录

实验完成后将表格追加到本节，并更新：

- `output/vr_room_e3_summary.json`
- `output/vr_room_e3_summary.csv`
