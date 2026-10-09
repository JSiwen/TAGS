# TAGSs 全实验汇总：整体效果最好的版本

> **生成时间**：2026-10-09
> **数据来源**：`output/*.csv`、`output/*.json`、`docs/VR_ROOM_*.md`、`docs/BICYCLE_*.md`、`output/bicycle_v14_*.md`、`output/bicycle_v15_experiment_architecture.md`、`docs/THESIS_SYSTEM_DESIGN.md`
> **协议窗口**：30 000 迭代，固定 holdout 视角评估，仓库自有 `metrics.py` 同口径

本文件把所有完成训练的 TAGS 实验拉到同一张表上比较，分别给出 **室外 `bicycle` / Mip-NeRF 360** 和 **室内 `vr_room`** 两条最佳配置，并列出每条协议线下达到前 1 名的候选名单与它们的关键改进点，供论文撰写与系统选型使用。

---

## 一、TL;DR：两条最佳配置

| 场景 | 最佳配置 (TAGS) | 路径 | 指标均值 (3-4 seed) / 一次跑 |
|---|---|---|---|
| **室内 `vr_room`** | `lpips_ms_w005`（多尺度 LPIPS @ 21k step，权重 0.05） + 250 万 cap 剪枝日程 + L1+SSIM 基线 + depth/normal 先验在 20k 衰减 | `output/vr_room_e4_lpips_ms_w005[_seedN]/` | **PSNR 27.473 dB · SSIM 0.87780 · LPIPS 0.19137 · 1.909M 高斯 · 165 FPS (vs unlimited 81.3, 2.03×)** |
| **室外 `bicycle`** | 同 `lpips_ms_w005` 配方原样迁移到 bicycle（一次性首跑） | `output/bicycle_e4_lpips_ms_w005/` | **PSNR 25.030 dB · SSIM 0.7487 · LPIPS 0.2032 · 2.282M 高斯** |
| **自行车 v14 体系内最佳** | `bicycle_v14_v8_monitor_28500_select`（单独 v8 骨架 + 28500 checkpoint，**唯一 gate-pass**） | A100 上 `output/bicycle_v14_v8_monitor_28500_select/` | **PSNR 25.518 dB · SSIM 0.7745 · LPIPS 0.1864**（gate：pass） |

> **论文主推**：室内推荐 `lpips_ms_w005`（E4 胜出配方），论文 vs DN-Splatter 主表用 `vr_room_e4_lpips_ms_w005` 的室内 3-seed 均值；自行车同配方迁移得分（25.061 ± 0.021 / 0.7490 ± 0.0006 / 0.2029 ± 0.0009，**4-seed mean ± std**）同表对比 DN-Splatter (22.898 / 0.679 / 0.315)，**ΔPSNR +2.163 ± 0.021 dB · ΔSSIM +0.070 · ΔLPIPS −0.113（绝对值更有利于 TAGS）**。Per-view 胜率（按 sorted-index 配对；TAGS 视角名 `00000..24`，DN 视角名 `_DSC8679..8872`）：**PSNR 25/25、SSIM 23/25、LPIPS 23/25**。Reproduce: `.\scripts\run_bicycle_vs_dn_final.ps1`。

---

## 二、跨场景横向比较（同口径 `metrics.py`）

### 2.1 室内 vr_room 全实验表（三 seed 均值，除非标注）

| 系列 | Case / 配置 | Points | PSNR↑ | SSIM↑ | LPIPS↓ | FPS | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| E2 预算 | unlimited（5.27M） | 5,271,565 | 27.536 | 0.88311 | 0.20696 | 81.3 | reference 上限 |
| E2 预算 | 3.0M | 2,998,868 | 27.402 | 0.87979 | 0.21798 | — | — |
| E2 预算 | 2.0M | 1,999,465 | 27.321 | 0.87573 | 0.22778 | — | — |
| E2 预算 | 1.5M | 1,499,692 | 27.291 | 0.87287 | 0.23609 | — | 性价比拐点 |
| E2 预算 | 1.0M | 999,874 | 27.149 | 0.86791 | 0.24615 | — | 已超 DN（26.4/0.86） |
| **E3 剪枝基线** | `combo_opacity0.02_densify0.50_cap2.5m`（3-seed mean） | **1,915,994** | **27.470** | **0.87913** | **0.22219** | **167.2** | **2.5M cap + 50% 后只剪不增 + opacity 0.02** —— E4 结构基线 |
| E4 感知 | `rgb_ssim`（baseline for E4，3-seed mean） | 1,915,994 | 27.428 | 0.87857 | 0.22296 | 168.9 | L1+SSIM，无 LPIPS |
| E4 感知 | `lpips_w005`（单尺度 0.05, 3-seed mean） | 1,899,000 | 27.436 | 0.87727 | 0.18864 | 164.2 | LPIPS 最强，但 SSIM 下降 -0.00130 |
| **E4 感知 winner** | **`lpips_ms_w005`（多尺度 0.05, 3-seed mean）** | **1,909,000** | **27.473** | **0.87780** | **0.19137** | **165.1** | **★ 论文推荐**：LPIPS 改善 -14.17%，PSNR +0.045，SSIM -0.00077，126/126 视角 LPIPS 全胜 |
| E4b 重叠 | `decay27k` 深度+法线共同延至 27k（3-seed mean） | 1,901,627 | 27.463 | 0.87748 | 0.19119 | 168.4 | 深度 log-L1 改善 -11.40%；SSIM 多降 0.00032，未过预设 0.0002 gate |
| E4b 重叠 | `depth24k_normal27k` / `depth27k_normal24k` 拆分 | 1,901,000 级 | 27.450/27.462 | 0.87756/0.87759 | 0.19134/0.19164 | 168/170 | 同样 SSIM 未过 gate，但 LPIPS/PSNR 守住 |
| E4b 重叠 | `decay24k`（共同 24k，3 seed mean） | 1,917,000 | 25.298 | 0.83802 | 0.26895 | — | **种子 2 崩溃到 47k 高斯**，不稳定，不推荐 |

### 2.2 室外 bicycle 全实验表（同口径单次跑 / 3 seed）

| 系列 | Case | Points | PSNR↑ | SSIM↑ | LPIPS↓ | 备注 |
|---|---|---:|---:|---:|---:|---|
| **vs DN 主表** | **TAGS `lpips_ms_w005`（室内胜出配方原样迁到 bicycle）** | **2,282,018** | **25.030** | **0.7487** | **0.2032** | **vs DN +2.16dB / +0.070 / −0.113**（4-seed mean vs DN） |
| vs DN 主表 | DN-Splatter | 1,797,280 | 22.898 | 0.6795 | 0.3154 | 同样 images_4 (1237×822)，同 test holdout |
| v4_restore baseline | `bicycle_integrated_v4_restore` | ~2.1M | 25.495 | 0.7740 | 0.1870 | perceptual-front / geo-late 集成 |
| v8 gate ref | `bicycle_v8_color_geo_soft` | ~1.97 GB | 25.513 | 0.7746 | 0.1863 | gate-pass 阈值参考 |
| **v14 winner** | **`bicycle_v14_v8_monitor_28500_select`（checkpoint=28500）** | — | **25.518** | **0.7745** | **0.1864** | **唯一 gate-pass**；28500 vs 30000 选点策略 |
| v14 | `bicycle_v14_v8_guard_recheck`（ours_30000） | — | 25.513 | 0.7745 | 0.1861 | gate: fail (psnr_regressed_views) |
| v14 | `bicycle_v14_v8_guard_polish_soft075` | — | 25.478 | 0.7743 | 0.1860 | gate: fail (global_psnr/ssim) |
| v14 | `bicycle_v14_v8_guard_polish_soft050` | — | 25.495 | 0.7738 | 0.1865 | gate: fail |
| v14 | `bicycle_v14_v8_guard_refine_end30000` | — | 25.467 | 0.7737 | 0.1865 | gate: fail（chronic_psnr_drop） |

> 注：v14 系列在 A100 上独立训练，与 bicycle_e4_lpips_ms_w005 的 3080 + 室内 E4 配方不同协议，**不能直接合表**。`bicycle_v14_v8_monitor_28500_select` 在协议内仍是 v8/v14 体系最佳，但论文主表应使用 `bicycle_e4_lpips_ms_w005` 对照 DN-Splatter。

---

## 三、按场景的「整体最好」结论与依据

### 3.1 室内 `vr_room`：E4 胜出配方 `lpips_ms_w005`

**为什么不是 `lpips_w005`（单尺度 0.05）？**
两者 LPIPS/PSNR 都达标，但多尺度 0.05 三个种子的 SSIM 标准差 (0.00028) 小于单尺度 (0.00071)；PSNR 比基线高 +0.045 vs 单尺度 −0.020；LPIPS 改善 −14.17%（单尺度为 −15.39%），差距 1.2pp 但用结构稳定性换——论文「质量权衡更稳」结论支持选多尺度（见 `output/vr_room_e4_winner.json` 中 `selection_reason: "more_stable_quality_tradeoff"`）。

**为什么不是 `decay27k`（E4b 延几何）？**
`decay27k` 把 LPIPS 改善仍保持 14.25%、深度 log-L1 再改善 11.4%、Spearman 升至 0.9591，但 **平均 SSIM 多降 0.00032**，**超过** 0.0002 的预设门控。论文文本明确写「当前推荐 20k 结束；如强调深度一致性才用 27k」。

**为什么不用 unlimited 5.27M？**
吞吐只有 81 FPS，是 250 万 cap 下的 165 FPS 的 49%；FPS 没达到 1.5× 的完整通过门槛；和 250 万 cap 的 2.0M 加权 PSNR 仅差 0.06 dB，性价比悬殊。

**E2 1.5M 拐点的角度**
如果只追求 PSNR，1.5M 是局部拐点（27.291 dB vs 2.0M 27.321 dB，相邻 +0.030 dB）。E3 后通过剪枝日程得到的 1.92M 是兼顾「PSNR/SSIM/FPS」的工作点（27.470 dB @ 2.10× unlimited FPS）。论文主结果应给出 E3 的「190 万 + 后期 opacity 剪枝 + 2.5M cap」的 3-seed 均值。

### 3.2 室外 `bicycle`：`bicycle_e4_lpips_ms_w005`（E4 配方直接迁移）

**为什么不是 `bicycle_v14_v8_monitor_28500_select`？**
两者不可比——前者用室内 E4 配方（LPIPS @ 21k step，权重 0.05），跨数据集迁移；后者用 v8 骨架 + color align + SPA + chronic guard。**论文主表 vs DN 必须用同一协议脚本（`scripts/run_bicycle_e4_winner.ps1`）**，`bicycle_v14_v8_monitor_28500_select` 不在这张主表里。

**vs DN-Splatter 增量**
- PSNR：+2.132 dB
- SSIM：+0.069
- LPIPS：−0.112（**TAGS 更低，更优**——但 LPIPS 实现不同，论文应注明不可绝对比较的事实）
- 规模：TAGS 2.28M vs DN 1.80M = 1.27×
- 协议：双方 `images_4` 原生分辨率 1237×822，同 holdout 8 划分，同 metrics.py

**v15 路线尚未执行**
v15 建议在 `bicycle_v14_v8_monitor_28500_select` 基础上做 dual-checkpoint / hard-view reweight，本文件生成时未看到 v15 实跑结果。如未来跑出 v15 数据，应以 v15 为 v8 体系内的最新最佳；本文档视 v14 winner 为 v8 体系的当前最佳。

---

## 四、单实验系列内的"第一名"明细

### 4.1 E2（规模预算）：1.0M 仍超 DN，1.5M 是性价比拐点
- 1.0M 27.149 / 0.868 / 0.246（vs DN 26.4 / 0.86 / 0.134，PSNR/SSIM 胜）
- 1.5M 27.291 / 0.873 / 0.236（性价比拐点）
- 2.0M 27.321 / 0.876 / 0.228
- 3.0M 27.402 / 0.880 / 0.218
- unlimited 27.536 / 0.883 / 0.207

### 4.2 E3（剪枝日程）：`combo_opacity0.02_densify0.50_cap2.5m`（3-seed, full_pass）
| seed | points | PSNR | SSIM | LPIPS | FPS |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,918,750 | 27.407 | 0.87877 | 0.22238 | 165.5 |
| 1 | 1,915,352 | 27.500 | 0.87950 | 0.22182 | 166.8 |
| 2 | 1,913,881 | 27.502 | 0.87912 | 0.22238 | 169.3 |
| 均 | 1,915,994 | 27.470 | 0.87913 | 0.22219 | 167.2 |

### 4.3 E4（感知损失）：`lpips_ms_w005`（3-seed, full_pass）
3 seed 均值：PSNR 27.473 ± 0.044 · SSIM 0.87780 ± 0.00028 · LPIPS 0.19137 ± 0.00047 · 1.909M · 165 FPS · 126/126 配对视角 LPIPS 全胜。

### 4.4 E4b（几何—感知重叠）：`decay27k`（LPIPS/PSNR 守住，SSIM 多降 0.00032）
3 seed 均值：PSNR 27.463 · SSIM 0.87748 · LPIPS 0.19119 · 1.902M · 168 FPS · log-L1 0.0493（−11.40%）· Spearman 0.95913。**未过 SSIM 0.0002 gate**。24k 共同衰减在 seed 2 训练崩溃（点数 47k），不稳定。

### 4.5 v8/v14（A100 室内体系）：`bicycle_v14_v8_monitor_28500_select` 唯一 gate-pass
5/5 候选完成训练；只有 `monitor_28500_select` 通过 gate。Score +0.0316。`monitor_28500` 较 `ours_30000` 同时 PSNR/SSIM 更高，与「final polish 牺牲 SSIM/PSNR」一致。结论：不启用 balanced late controller，只用 28500 checkpoint。

### 4.6 室外同配方迁移（vs DN）：`bicycle_e4_lpips_ms_w005`
PSNR 25.030 / SSIM 0.7487 / LPIPS 0.2032 / 2.282M（seed0）；4-seed mean PSNR 25.061 ± 0.021 / SSIM 0.7490 ± 0.0006 / LPIPS 0.2029 ± 0.0009。相对 DN +2.16 dB PSNR、+0.070 SSIM、−0.113 LPIPS。

---

## 五、关键洞察（论文表述建议）

1. **质量不是单纯靠点数堆出来**：vr_room 1.0M（1.0M / 999k / 27.149 dB）已超过 DN 1.12M 参考（26.4 dB），单点性价比极高。E3 的「约 192 万」由 250 万 cap + 后期 opacity 剪枝共同决定，**不是硬卡 192 万**。
2. **多尺度感知比单尺度稳**：E4 同 seed 的 LPIPS `ms_w005` 和 `w005` 都过 gate，但 ms 的 PSNR 不降反升 +0.045 dB，SSIM 标准差减半。论文应优先表 ms。
3. **晚期几何先验有利有弊**：21k 之后 LPIPS 接入，27k 之前结束 depth/normal 可再压 LPIPS 11–14%，继续延长 geo 衰减能换来深度一致性，但 SSIM 平均会再多降 0.0003，必须权衡。
4. **门控没被「绕过」是好事**：E4b 全部不通过预设的双门控（都是 SSIM 差一项），论文就是要把这个 0.0002 的敏感度公开汇报，便于复现者调 gate。
5. **A100 vs 3080 两套产物不能混表**：`bicycle_v14_*` 系列在 A100 单独训练环境，与 3080 的 `bicycle_e4_lpips_ms_w005`/`vr_room_e*` 是两条独立协议。论文 vs DN 主表只用后者；v14 winner 仅作为 v8 体系内的内部最佳记录。

---

## 六、推荐引用清单

| 在论文里写的最一句 | 引用 |
|---|---|
| 室内多规模曲线与 1.0M 已超 DN | `output/vr_room_e2_summary.csv` `docs/VR_ROOM_E2_GAUSSIAN_BUDGET.md` |
| 192 万工作点的剪枝日程 | `output/vr_room_e3_summary.csv` `docs/VR_ROOM_E3_PRUNE_ABLATION.md` |
| E4 胜出配方（ms LPIPS 0.05 @ 21k step） | `output/vr_room_e4_summary.csv` `output/vr_room_e4_winner.json` `docs/VR_ROOM_E4_PERCEPTUAL_ABLATION.md` |
| E4b 27k 重叠与崩溃讨论 | `output/vr_room_e4b_summary.csv` `docs/VR_ROOM_E4B_GEO_OVERLAP.md` |
| bicycle vs DN 主表 | `output/bicycle_e4_vs_dn_summary.csv` `docs/BICYCLE_E4_VS_DN.md` |
| v14 monitor_28500 唯一 gate-pass | `output/bicycle_v14_polish_checkpoint_summary.md` `output/bicycle_v14_polish_checkpoint_summary.json` |

---

## 附录 A：完整汇总脚本（仅 introspect，不改任何代码）

```powershell
# 全套汇总产物在仓库根目录一次性重生成
python scripts\summarize_vr_room_e2.py
python scripts\summarize_vr_room_e3.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
python scripts\summarize_vr_room_e4.py --measure-fps 1 --fps-warmup 10 --fps-repeats 30
python scripts\summarize_vr_room_e4b.py
python scripts\summarize_bicycle_e4_vs_dn.py
```
