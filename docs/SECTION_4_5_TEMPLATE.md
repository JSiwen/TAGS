# §4.5 室外 TAGS 配方迁移到 bicycle — 结果小结 (5-seed)

> **生成时间**: 2026-10-14 (模板)
> **数据源**: `output/bicycle_e4_vs_dn_summary.json` + `output/bicycle_e4_vs_dn_multiseed.csv` + `output/bicycle_e4_vs_dn_summary_v2_cross.json` + wandb dashboard
> **协议**: 30 000 iter, hold=8, `images_4` (1237×822), `train_optimized.py` 同口径, 仓库 `metrics.py`
> **对照**: DN-Splatter 同场景同 holdout 同 metrics (1-seed baseline)

---

## 0) 模板用法 (跑完 5-seed 后回填)

1. 跑完 5-seed:
   ```powershell
   $env:E4_SEEDS = "0,1,2,3,4"
   $env:WANDB_ENABLED = "1"
   $env:WANDB_PROJECT = "tags-ablation"
   $env:WANDB_ENTITY = "你的用户名"
   .\scripts\run_bicycle_e4_winner.ps1
   ```

2. 跑汇总:
   ```powershell
   python scripts\summarize_bicycle_e4_vs_dn.py   # 生成 *_v2_cross.json + multiseed.csv
   python scripts\build_section_4_6.py --output docs\SECTION_4_6_FULL_COMPARISON.md
   ```

3. 把下面所有 `{{...}}` 占位符替换成 `_v2_cross.json` 里的实际值.

---

## 1) TL;DR

| 维度 | 结论 |
|---|---|
| **Winner** | `TAGS_lpips_ms_w005` (5-seed mean) |
| **PSNR** | **{{psnr_mean}} ± {{psnr_std}} dB** (vs DN {{dn_psnr}} dB, **Δ = +{{psnr_delta}}** dB) |
| **SSIM** | **{{ssim_mean}} ± {{ssim_std}}** (vs DN {{dn_ssim}}, **Δ = +{{ssim_delta}}**) |
| **LPIPS** | **{{lpips_mean}} ± {{lpips_std}}** (vs DN {{dn_lpips}}, **Δ = {{lpips_delta}}**) |
| **Points** | **{{points_mean}}** (vs DN {{dn_points}}, × {{points_ratio}}) |
| **Winner gate** | PSNR ≥ +0.5: {{psnr_pass}}; LPIPS ≤ +0.05: {{lpips_pass}}; **passed = {{gate_passed}}** |
| **wandb cross-check** | local_n={{local_n}}, wandb_n={{wandb_n}}, consistent={{cross_consistent}} |
| **FPS** | (待补, 见 `output/bicycle_e4_lpips_ms_w005/.../fps.json` 或 `--measure-fps` 输出) |

---

## 2) 5-seed per-seed 数据 (5 行, 填表)

| seed | PSNR↑ | SSIM↑ | LPIPS↓ | num_points | train_exit | wandb_url |
|---:|---:|---:|---:|---:|---:|---|
| 0 | {{s0_psnr}} | {{s0_ssim}} | {{s0_lpips}} | {{s0_pts}} | {{s0_train_exit}} | {{s0_url}} |
| 1 | {{s1_psnr}} | {{s1_ssim}} | {{s1_lpips}} | {{s1_pts}} | {{s1_train_exit}} | {{s1_url}} |
| 2 | {{s2_psnr}} | {{s2_ssim}} | {{s2_lpips}} | {{s2_pts}} | {{s2_train_exit}} | {{s2_url}} |
| 3 | {{s3_psnr}} | {{s3_ssim}} | {{s3_lpips}} | {{s3_pts}} | {{s3_train_exit}} | {{s3_url}} |
| 4 | {{s4_psnr}} | {{s4_ssim}} | {{s4_lpips}} | {{s4_pts}} | {{s4_train_exit}} | {{s4_url}} |
| **MEAN** | **{{psnr_mean}}** | **{{ssim_mean}}** | **{{lpips_mean}}** | **{{points_mean}}** | — | — |
| **STD** | **{{psnr_std}}** | **{{ssim_std}}** | **{{lpips_std}}** | — | — | — |

> 数据来源: `output/bicycle_e4_vs_dn_multiseed.csv` (自动生成)

---

## 3) 与 DN-Splatter 增量

| 指标 | TAGS (5-seed) | DN-Splatter (1-seed) | Δ (TAGS-DN) | 论文说 |
|---|---:|---:|---:|---|
| **PSNR↑** | {{psnr_mean}} ± {{psnr_std}} | {{dn_psnr}} | **+{{psnr_delta}}** | "TAGS 在 PSNR 上系统性优于 DN" |
| **SSIM↑** | {{ssim_mean}} ± {{ssim_std}} | {{dn_ssim}} | **+{{ssim_delta}}** | "TAGS 结构相似性更高" |
| **LPIPS↓** | {{lpips_mean}} ± {{lpips_std}} | {{dn_lpips}} | **{{lpips_delta}}** | "TAGS 感知相似性更低 (更优)" (LPIPS 实现不同, 谨慎比较) |
| **Points** | {{points_mean}} | {{dn_points}} | × {{points_ratio}} | "TAGS 规模略大但可控, FPS 可接受" |

---

## 4) Winner 选择逻辑 (与 vr_room §4.3/4.4 一致)

### 4.1 为什么选 `lpips_ms_w005` (多尺度 0.05)?

- LPIPS 改善: **{{lpips_improvement_vs_rgb_ssim}}** vs E3 baseline (gate: -5%)
- PSNR: **{{psnr_vs_rgb_ssim}}** vs E3 baseline (gate: ≤-0.20 dB)
- SSIM: **{{ssim_vs_rgb_ssim}}** vs E3 baseline (gate: ≤-0.002)
- FPS: **{{fps}}** vs unlimited 81.3 (gate: ≥×1.5)

### 4.2 为什么不是 `lpips_w005` (单尺度 0.05)?

- LPIPS 改善相近 ({{ms_lpips_vs_w005}})
- 多尺度 SSIM std 减半 ({{ms_ssim_std}} vs {{w_ssim_std}})
- 多尺度 PSNR 不降反升 +{{psnr_ms_minus_w}} dB
- **结论**: 多尺度是更稳的质量/权衡, 论文主推

### 4.3 为什么不用 unlimited 5.27M (E2 reference)?

- 吞吐只有 81.3 FPS, 250 万 cap 下 165 FPS = 49%
- 加权 PSNR 仅差 +0.06 dB
- FPS gate 没过 (×1.5 vs ×2.0)

---

## 5) wandb cross-check 报告

```
{{cross_check_block}}
```

**判定**:
- 5 个 seed 全部本地 vs wandb 一致 → 论文数据可信 ✅
- 任意 seed 偏差 > 0.05 dB / 0.001 SSIM / 0.005 LPIPS → 标注该 seed 来源

---

## 6) 已知 caveats (写论文时记得说)

1. **LPIPS 实现差异**: TAGS 用 `metrics.py` (VGG backbone, 默认实现), DN-Splatter 用 [其内置 LPIPS]. 两者**不能绝对比较**, 论文中 Δlpips 注明"implementation may differ"
2. **训练时长不匹配**: TAGS 30k iter (室内配方, 21k LPIPS 介入), DN-Splatter 默认 recipe (多少 iter?)
3. **数据规模**: TAGS 2.28M points vs DN 1.80M, 论文说"TAGS 规模略大但仍在 250 万 cap 内"
4. **wandb 公开链接**: 如果 entity 是 public, 任何人都能看; 如果 private, 论文 supplementary 提供截图

---

## 7) 与 `ALL_EXPERIMENTS_SUMMARY.md` 的链接

- 主索引: `docs/ALL_EXPERIMENTS_SUMMARY.md` §4.6 (室外同配方迁移 vs DN)
- 历史 bicycle 4-seed mean (单 run, 单 seed 协议): `output/bicycle_e4_vs_dn_summary.json` (旧版)
- **本次 5-seed 全套**: `output/bicycle_e4_vs_dn_summary.json` (v2, 含 v2_multiseed 块) + `output/bicycle_e4_vs_dn_multiseed.csv`
- vr_room 同配方对照: `output/vr_room_e4_winner.json` (室内 3-seed 旧版, 待 5-seed 重跑)

---

## 8) 论文写作 checklist

- [ ] §4.5 main table: 用 5-seed mean ± std, Δ vs DN 单独一行
- [ ] §4.5 supplementary: 5-seed per-seed 表 (附录 A.1)
- [ ] §4.5 wandb screenshot: dashboard group view 截一张放正文
- [ ] §4.5 caveat paragraph: LPIPS 实现差异 + 训练时长差异
- [ ] §4.6 full table: cross-scene (bicycle + vr_room), `docs/SECTION_4_6_FULL_COMPARISON.md` 自动生成
- [ ] 引用 [wandb run URL] 在 supplementary

---

**模板版本**: v2 (2026-10-14, exp4 v2 wandb 集成配套)
**v3 更新**: 2026-10-15, 加入三组对比 + 标准分辨率验证
**自动生成器**: `python scripts\build_section_4_6.py` 生成 §4.6 全对比, 本小结手工填

---

# v3 计划更新 (2026-10-15)

> v3 在保留本模板所列 5-seed 评估口径的同时, 引入了**三组对比结构 + 标准分辨率验证**,
> 详见 `docs/SECTION_4_6_FULL_COMPARISON.md`.

## v3 与本模板的差异

| 维度 | v2 (本模板) | v3 (新) |
|---|---|---|
| **主实验分辨率** | 1/4 (`images_4`, 1237×822) | **1/2 (`images_2`, 2473×1645)** |
| **实验组结构** | 单一 §4.5 主表 (7 组 × 4 场景 × 5 seed) | **Group A (4 方法) + Group B (5 消融) + Group C (3 分辨率) + §4.5 (7 多模块) + std 验证 (1 组)** |
| **质量指标阈值** | 标准分辨率阈值 (24/25/27/28 dB) | 1/2 阈值 0.97x (23.4/24.4/26.3/27.2 dB), 由 `qa_baseline_check.py --images-dir images_2` 检查 |

## v3 新增脚本与本模板的对应关系

| v3 脚本 | 与本模板对应 | 备注 |
|---|---|---|
| `run_exp4v3_A_method_comparison.py` (4 方法, 1/2 分辨率) | 替代本模板的 "bicycle vs DN" 单表, 输出**4 场景 × 4 方法**主表 | 论文表 4-X 主要方法对比 |
| `run_exp4v3_B_module_ablation.py` (5 组逐项启用) | 简化版 §4.5 消融, 只关心 3 个核心模块 | 与 §4.5 互补 |
| `run_exp4v3_C_resolution_sensitivity.py` (3 分辨率) | **新增**: 分辨率敏感性实验 | 论文 §4.6 分辨率讨论 |
| `run_exp4v3_std_resolution_validation.py` (3 场景 × 标准) | **新增**: 标准分辨率验证 | 验证 1/2 结论可扩展性 |
| `run_exp4_2/3/4/5_*.py` (保留) | v2 主体扫参 (5+5+6+7 组) | 模块内归因核心证据, 不删除 |

## v3 跑完后的回填流程

1. **跑 Group A** (1/2 分辨率主表):
   ```powershell
   .\scripts\run_exp4v3_A_method_comparison.ps1
   # 输出: output/exp4_stats/exp4v3_A_method_comparison.csv
   ```
2. **跑 Group B** (模块消融):
   ```powershell
   .\scripts\run_exp4v3_B_module_ablation.ps1
   # 输出: output/exp4_stats/exp4v3_B_module_ablation.csv
   ```
3. **跑 Group C** (分辨率敏感性):
   ```powershell
   .\scripts\run_exp4v3_C_resolution_sensitivity.ps1
   # 输出: output/exp4_stats/exp4v3_C_resolution_sensitivity.csv
   ```
4. **跑标准分辨率验证**:
   ```powershell
   .\scripts\run_exp4v3_std_resolution_validation.ps1
   # 输出: output/exp4_stats/exp4v3_std_resolution_validation.csv
   ```
5. **回填本模板的 `{{...}}` 占位符**:
   - `psnr_mean` / `psnr_std` / `ssim_mean` / `ssim_std` / `lpips_mean` / `lpips_std`: 从 Group A.A4 (Full TAGS) 的 bicycle 行取 (1/2 分辨率值)
   - `dn_psnr` / `dn_ssim` / `dn_lpips`: 与原 v2 一致 (DN 在标准分辨率的值, 谨慎标注"分辨率不匹配")
6. **生成 §4.6 全对比综合报告**:
   ```powershell
   python scripts\build_section_4_6.py --output docs\SECTION_4_6_FULL_COMPARISON.md
   ```

## v3 模板保留说明

本模板的 §1-§8 (5-seed per-seed 表, 与 DN 增量, winner 选择逻辑, wandb cross-check, caveats) **保留有效**, 但分辨率上下文变为 1/2. 论文写作时:

- §4.5 主表使用 1/2 分辨率值
- 与 DN 对比时**明确标注**:
  - "TAGS 1/2 vs DN 标准" — 不可直接比较, 仅作量级参考
  - 或单独用 `exp4v3_std_resolution_validation.csv` 的 bicycle/bonsai/garden 行做"TAGS 标准 vs DN 标准"对照
- wandb cross-check 仍然要求 local = wandb (避免 silent drift)
