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

质量门控 `quality_pass`（可在未测 FPS 时判定）：

1. Gaussian 数量 ≤ **2,000,000**
2. 相对无限制 PSNR 下降 ≤ **0.3 dB**（即 PSNR ≥ 约 27.236）
3. PSNR 仍高于 DN（> **26.4**）

完整门控 `full_pass` 另需：

4. 渲染吞吐相对无限制至少提升 **50%**（test split 遍历 FPS ≥ 1.5×）。未测 FPS 记为 `fps_pending`，不把质量项判失败。

DN 的 LPIPS（0.134）实现/评测条件不同，只作外部参考，不作为硬门控。

## 代码回溯

| 标签 | 含义 |
|------|------|
| `pre-e3` | E3 修改前（post-E2） |
| `e3-v2-wave3` | v2 协议 + Wave 3 筛选完成时的代码；正交日程/评测硬化之前 |

对应提交：`snapshot: preserve E3 v2 Wave 3 screening codebase`。回滚：`git checkout e3-v2-wave3`。

## v1 失败原因（已在 v2 修复）

cap2m 三组在 densify 阶段顶着 2M、PSNR ~26.5–26.9，一进入 prune-only 就崩盘（最终 PSNR ~3.74，点数几千到 2 万）：

- `prune_until_fraction=1.0` 在停止增殖后仍每 100 iter 跑完整 `densify_and_prune`。
- 默认 `prune_max_screen_size=-1` 会沿用半径 20 + 世界尺度 oversized 剪枝；没有 clone/split 补回时，一轮就从 2.00M 砍到 ~62 万，随后持续失血。
- `--test_iterations` 缺少 10k/18k/19.5k；且评估写在剪枝之前，15k/24k 的 PSNR 是剪枝前快照。

v2 修复：

- prune-only 阶段关闭 oversized 屏幕/世界剪枝，只保留 opacity / visibility / gradient 剪枝。
- 评估与 stage snapshot 改到 densify/prune **之后**。
- 评估点（v2）：7000, 10000, 15000, 18000, 19500, 24000, 30000。
- 输出目录改为 `output/vr_room_e3_v2_<profile>/`，不与已崩盘的 v1 结果混用。

v2 中间评估仍会撞上 densify 期内的 `opacity_reset`（15000 / 18000），test PSNR 可短暂掉到 ~7 dB；**30k 最终指标可信**。Wave 4 起评估改为 15100 / 18100 / 21100，并在 reset 时打日志。

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
5. **densify 停止 + prune-only 起点耦合**（历史命名，同时改两个旋钮）：`densify_0.50` / `densify_0.65` / `densify_0.80`
6. **最后 10% 只剪不增**：`lateprune_0.90`（densify 仍到 60%）
7. **组合**：`combo_opacity0.01_densify0.50`、`combo_opacity0.01_densify0.80`、`combo_opacity0.02_densify0.50`、`combo_opacity0.02_densify0.65`

Wave 3 实际完成 3/4 combo（缺 `combo_opacity0.02_densify0.65`）。`densify_*` 不是单因素：它把 densify 停止点与 prune-only 起点设成同一值。

### Wave 4 正交日程（seed=0，补点）

把 densify 停止与 prune-only 起点拆开。opacity=0.005 四角：

| | prune-only = densify 结束（立即剪） | prune-only = 0.80（纯优化窗口） |
|---|---|---|
| densify 0.50 | 已有 `densify_0.50` | 新增 `densify0.50_pruneonly0.80` |
| densify 0.60 | 新增 `densify0.60_pruneonly0.60` | 已有 `opacity_0.005`（基准 0.60/0.80） |

opacity=0.02 四角：已有 `combo_opacity0.02_densify0.50`（0.50/0.50）；补 `combo_opacity0.02_densify0.50_pruneonly0.80`、`combo_opacity0.02_densify0.60_pruneonly0.60`，并以 `opacity_0.02` 作为 0.60/0.80 对照。另补缺失 combo `combo_opacity0.02_densify0.65`。

E2 硬帽对照 profile：`e2_style_cap2m`（densify/prune 均在 60% 结束，2M cap，无后期 prune-only）。

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
$env:E3_PROFILES = "combo_opacity0.01_densify0.50,combo_opacity0.01_densify0.80,combo_opacity0.02_densify0.50"
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

### Wave 4 — 正交日程筛选（seed=0）

新 profile 名称不覆盖已有 v2 目录。评估点改为 7k / 10k / 15.1k / 18.1k / 19.5k / 21.1k / 24k / 30k。已完成的 case 会跳过。

```powershell
$env:E3_PROFILES = "combo_opacity0.02_densify0.65,densify0.50_pruneonly0.80,densify0.60_pruneonly0.60,combo_opacity0.02_densify0.50_pruneonly0.80,combo_opacity0.02_densify0.60_pruneonly0.60"
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

第一闸门：淘汰 PSNR drop >0.3 dB，或在点数、PSNR、FPS 上被支配的配置。进入 cap / 多种子前只保留 Pareto 候选。

### Wave 4b — 硬帽归因（第一闸门之后）

对筛选出的最佳日程与 `e2_style_cap2m` 做同 seed=0 对照。3080 先 2.0M，再 2.5M；仅当 2.5M 显存安全且点数仍顶帽时才尝试 `$env:E3_MAX_GAUSSIANS = "0"`。出现 OOM 或峰值显存持续 >90% 则停止。独立 tag，避免覆盖 v2：

```powershell
$env:E3_RUN_TAG = "v3cap"
$env:E3_PROFILES = "combo_opacity0.02_densify0.50,e2_style_cap2m"
$env:E3_MAX_GAUSSIANS = "2500000"
.\scripts\run_vr_room_e3_prune_ablation.ps1
```

### 汇总

```powershell
python scripts\summarize_vr_room_e3.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
```

输出：`output/vr_room_e3_v2_<profile>/`（`E3_RUN_TAG` 可改）。FPS 口径为 **test split 吞吐**（42 相机 × repeats），报告 mean/std 与 p50/p95 帧时，不是单视角交互帧率。

Per-view NVS：

```powershell
python scripts\evaluate_nvs_per_view.py -m output\vr_room_e3_v2_<profile> --iteration 30000
```

### 断点续训

- 已有 `point_cloud/iteration_30000/point_cloud.ply` 则跳过。
- 否则从最大 `chkpnt*.pth` 续训。
- 若目录里已有 `e3_profile.json`，训练旋钮与当前 profile 不一致则拒绝续训。
- 默认每 3k 存盘。强制重跑：`$env:E3_SKIP_COMPLETED = "0"`。

## 多种子复现（最优 2 个配置 × 3 种子）

正交筛选完成后再定 top2，不要预先锁死 `densify_0.50`。当前 Wave 3 单种子领先者是 `combo_opacity0.02_densify0.50`；第二种子候选等 Wave 4 四角结果。

对 top2 与 `e2_style_cap2m` 统一跑 seeds 0/1/2。通过标准建议：每个 seed 点数 ≤2M；平均 PSNR drop ≤0.3 dB；最差 seed 不明显越界；FPS 95% 下界 ≥1.5×。若 top1–top2 平均 PSNR 差小于一个标准差，按点数 / p95 帧时 / 显存选更轻方案。

```powershell
$env:E3_PHASE = "multiseed"
$env:E3_TOP_CONFIGS = "combo_opacity0.02_densify0.50,<wave4_runner_up>"
$env:E3_SEEDS = "0,1,2"
.\scripts\run_vr_room_e3_prune_ablation.ps1
python scripts\summarize_vr_room_e3.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
```

## 判读流程

1. 先看 `quality_pass`：`num_points ≤ 2M` 与 `psnr_drop ≤ 0.3`，以及 `beats_dn_psnr`。
2. 再看 `full_pass` / `fps_status`。未测 FPS 为 pending。
3. 通过质量门控的配置中，优先选 **点数更低且 PSNR 下降更小** 者进入多种子；top2 按唯一 profile 聚合。
4. 与 `e2_style_cap2m` / E2 2.0M 对照：仅当 **相同 cap** 下日程更好，才能归因于剪枝日程；若无帽压不到 ≤2M，结论应写成「2M 安全帽 + 日程」联合方案。
5. 15k/18k 若出现个位数 PSNR，先核对是否紧挨 opacity reset，不要当成永久崩盘。

## 结果记录

### Wave 3（seed=0，v2，2026-09-15）

| Profile | Points | PSNR | ΔPSNR | FPS | × | Pass |
|---------|-------:|-----:|------:|----:|--:|:----:|
| combo_opacity0.02_densify0.50 | 1.587M | 27.422 | 0.114 | 193.6 | 2.36 | yes |
| combo_opacity0.01_densify0.50 | 1.748M | 27.375 | 0.161 | 179.1 | 2.18 | yes |
| combo_opacity0.01_densify0.80 | 1.915M | 27.153 | 0.382 | 137.4 | 1.67 | no |
| combo_opacity0.02_densify0.65 | — | — | — | — | — | missing |

汇总：`output/vr_room_e3_summary.json`、`output/vr_room_e3_summary.csv`。
