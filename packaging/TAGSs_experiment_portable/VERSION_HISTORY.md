# TAGSs 版本更迭说明

本文档记录 TAGSs 项目在官方 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 基础上的实验版本演进。所有指标均在 Mip-NeRF 360 `bicycle` 场景、30k 迭代、`--eval --llffhold 8` 协议下测得。

**生产基线**：`bicycle_integrated_v4_restore`（PSNR 25.495 / SSIM 0.7740 / LPIPS 0.1870）

---

## 版本总览

| 版本 | 代表脚本 | 核心变化 | 最佳 PSNR | 状态 |
|------|----------|----------|-----------|------|
| origin | `train.py` | 原始 3DGS 基线 | 25.151 | 参考基线 |
| v2 | `run_bicycle_integrated_v2_ablation.sh` | geo / resolution / spa / integrated 四组消融 | — | 早期验证 |
| v3 | integrated_v3 | masked depth prior、保守 SPA | — | 已超越 |
| v4 / v4_repro | `bicycle_integrated_v4_repro` | group LR、geometry controller、LPIPS | 25.300 | 复现差异大 |
| **v4_restore** | `run_bicycle_v4_restore.sh` | 当前最佳稳定配置 | **25.495** | **生产基线** |
| v4_restore_baseline | `run_bicycle_v4_restore_baseline.sh` | 去掉 hard-view / color align 对照 | — | 对照组 |
| v5 | `run_bicycle_v5.sh` | hard-view 采样 + 强 color align | 25.412 | 未超 v4_restore |
| v6 | `run_bicycle_v6_conservative.sh` | 回退至 v4_restore 验证 | 25.448 | 确认 v5 无效 |
| v7 | `run_bicycle_v7_v4restore_ablation.sh` | 单变量消融（LPIPS/detail/color/exposure/geo） | 25.484 | 收益递减 |
| v8 | `run_bicycle_v8_v4restore_refine.sh` | color + geo 晚期精修组合 | 25.513 | 全局最优但 40% 视角回退 |
| v9 | `run_bicycle_v9_v4restore_refine.sh` | geo tail 边界搜索 + color 变体 | 25.502 | per-view 更均衡 |
| v10 | `run_bicycle_v10_balanced_fast.sh` | 删除无效 late 正则，加速训练 | 25.506 | Gate 全 fail |
| v11 | `run_bicycle_v11_balanced_guarded.sh` | 弱 LPIPS guard + guarded color | 25.510 | Gate 全 fail，进行中 |

---

## 各版本详细说明

### origin — 原始 3DGS 基线

- **指标**：PSNR 25.151 / SSIM 0.7620 / LPIPS 0.2101
- **配置**：默认 `train.py`，加载深度图但不启用集成模块
- **意义**：所有后续版本的对比起点

---

### v2 — 集成模块消融（2026-06 初）

- **脚本**：`scripts/run_bicycle_integrated_v2_ablation.sh`
- **变化**：将 TAGSs 三大支柱拆分为独立实验组
  - 几何先验（geo prior）
  - 分辨率自适应调度（resolution adaptive）
  - 稀疏优化（SPA）
  - 三者集成（integrated）
- **结论**：集成组显著优于单模块，验证多模块协同有效

---

### v3 — 掩码深度与保守 SPA

- **变化**：
  - 引入 masked depth prior，仅在可靠区域施加深度约束
  - 保守 SPA 配置（`prune_ratio2=0.08`）
- **意义**：为 v4 的稳定配置奠定基础

---

### v4 / v4_repro — 分组学习率与几何控制器

- **脚本**：`bicycle_integrated_v4_repro`
- **新增模块**：
  - `GroupLRScheduler`：三阶段分别调节 geometry / color 参数学习率
  - `GeometryController`：周期性 prune oversized 高斯
  - LPIPS 晚期感知损失
- **指标**：v4_repro PSNR 25.300（明显低于 v4_restore，说明超参细节影响巨大）
- **教训**：复现需严格对齐 `cfg_args`，脚本描述与实际运行配置可能不一致

---

### v4_restore — 当前生产基线 ★

- **脚本**：`scripts/run_bicycle_v4_restore.sh`
- **指标**：PSNR **25.495** / SSIM **0.7740** / LPIPS **0.1870**（相对 origin +0.344 dB）
- **关键配置**：
  - 几何先验：depth final 0.04, normal 0.012, warmup 4k, decay 10k–20k（实际 cfg 中为 decay until 24k）
  - 分辨率：`1200:4,4000:2,30000:1`
  - SPA：13k–25k, prune_ratio2 0.04, hybrid metric
  - Group LR + Geometry Controller 启用
  - LPIPS final 0.01 @ 16k
- **注意**：实际 baseline `cfg_args` 与脚本描述存在差异（如无 color align、无 detail loss、无 hard-view reweight），后续消融均以此实际 cfg 为基准

---

### v5 — 困难视角采样实验

- **脚本**：`scripts/run_bicycle_v5.sh`
- **变化**：
  - hard-view **采样**（非仅损失重加权）
  - 更强 color align
  - geo 低纹理 relax
  - 边缘引导 densify（factor 0.72）
- **指标**：PSNR 25.412（**未超过** v4_restore）
- **结论**：采样扰动过大，不稳定；hard-view 采样路线被证伪

---

### v6 — 保守回退验证

- **脚本**：`scripts/run_bicycle_v6_conservative.sh`
- **目的**：验证 v5 增益是否来自激进策略
- **指标**：PSNR 25.448（等同 v4_restore 量级）
- **结论**：确认 v5 的 hard-view 采样无实质收益

---

### v7 — 单变量消融

- **脚本**：`scripts/run_bicycle_v7_v4restore_ablation.sh`
- **设计**：以 v4_restore 为基准，每次只改一个变量

| Case | 变量 | PSNR | 结论 |
|------|------|------|------|
| `v7_lpips_weak_late` | LPIPS 0.006 @ 20k | 25.484 | 全局 PSNR 略低 |
| `v7_lpips_strong_late` | LPIPS 0.012 @ 20k | 25.484 | LPIPS 略优但 PSNR 仍低 |
| `v7_detail_micro_late` | detail 0.008 @ 18k | 25.478 | LPIPS 25/25 视角全回退 |
| `v7_color_ultralate` | 更晚更弱 color align | 25.504 | +0.010 dB，有效方向 |
| `v7_exposure_mild` | exposure_lr 微调 | — | 全局略负，无效 |
| `v7_geo_decay_long` | geo decay until 26k | 25.499 | +0.005 dB，弱有效 |

- **结论**：晚期微调进入**收益递减区**（< 0.02 dB）；LPIPS/detail 训练损失被证伪；color align 和 geo decay 延长是有效方向

---

### v8 — color + geo 晚期精修

- **脚本**：`scripts/run_bicycle_v8_v4restore_refine.sh`
- **最佳配置**：`bicycle_v8_color_geo_soft`
- **指标**：PSNR **25.513** / SSIM 0.7746 / LPIPS 0.1863（全局最高）
- **问题**：
  - 40% 测试视角 PSNR 回退（10/25）
  - 收益高度集中于 00001/00004/00020 三个视角（占正向增益 56.8%）
  - 慢性问题视角 00008/00011/00016/00017/00018 在全部 7 个 case 中均低于 baseline
- **结论**：全局最优但 per-view trade-off 严重，不能固化为生产配置

---

### v9 — geo tail 边界与 color 变体

- **脚本**：`scripts/run_bicycle_v9_v4restore_refine.sh`
- **已完成最佳**：`bicycle_v9_geo245_color_ultralate`
- **指标**：PSNR 25.502 / LPIPS 0.1864（全局略低于 v8 soft，但慢性视角回退明显缓解）
- **关键发现**：
  - `geo_decay_until=24500` 优于 25000（缓解 00008/00011 回退）
  - 但会牺牲 00001/00004 上的大幅增益
  - `geo250_color_later_weak` 训练失败（PSNR 9.31，无效产物）
- **结论**：不存在单点配置能同时优化全局三项且无视角回退

---

### v10 — balanced-fast 架构重构

- **脚本**：`scripts/run_bicycle_v10_balanced_fast.sh`
- **日期**：2026-06-16
- **架构决策**：

  **保留**：几何先验、分辨率调度、Group LR、GeometryController、SPA、per-view 验收

  **删除/停用**：
  - LPIPS 训练损失（v7 证伪 + VGG 开销）
  - Detail loss（v7 证伪，LPIPS 全视角回退）
  - Hard-view 采样/重加权（v5 证伪）
  - exposure_lr 微调（v7 证伪）
  - geo_decay_until=26000（过平滑）
  - 高 color_align_std_weight（LPIPS 回退）

- **候选结果**（5/5 完成，Gate 全部 fail）：

| Case | PSNR Δ | LPIPS Δ | 慢性视角最大回退 |
|------|--------|---------|------------------|
| `v10_blc_geo24500_guarded_spa24000` | +0.012 | +0.004 | **0.115**（最佳） |
| `v10_blc_geo24500_guarded` | +0.013 | +0.004 | 0.198 |
| `v10_blc_geo24250_guarded` | +0.005 | +0.005 | 0.186 |
| `v10_blc_geo24750_guarded` | −0.021 | +0.004 | 0.201 |
| `v10_no_late_regularizers` | −0.024 | +0.006 | 0.199 |

- **结论**：
  - `spa_end_iter=24000` 对慢性视角有帮助
  - `geo_prior_decay_until=24500` 是当前均衡点
  - 关闭 LPIPS 后感知指标系统性退化（24–25/25 视角 LPIPS 回退）
  - 需要恢复**弱 LPIPS guard**而非强主损失

---

### v11 — 弱 LPIPS guard + guarded color（当前最新）

- **脚本**：`scripts/run_bicycle_v11_balanced_guarded.sh`
- **日期**：2026-06-23
- **设计**：继承 v10 的 `geo24500 + spa24000` 骨架，新增：
  - 弱 LPIPS guard（`lpips_loss_weight_final=0.003`，start 24k–25k）
  - guarded late appearance（mean-only color align）
  - per-view gate 验收（`scripts/check_v11_gate.py`）

- **候选结果**（4/4 完成，Gate 全部 fail）：

| Case | PSNR Δ | LPIPS Δ | 慢性视角最大回退 |
|------|--------|---------|------------------|
| `v11_cand_d_lpips003_color25500_spa0375` | +0.015 | +0.004 | 0.158 |
| `v11_cand_b_lpips005` | +0.024 | +0.004 | 0.204 |
| `v11_cand_c_lpips003_color25500` | −0.003 | +0.004 | **0.112** |
| `v11_cand_a_lpips003` | −0.015 | +0.005 | 0.152 |

- **结论**：
  - PSNR 有进一步提升（cand_b +0.024 dB），但 LPIPS 仍未通过 gate
  - 弱 LPIPS guard 方向正确但权重/时机需继续调优
  - 尚无配置通过全部 gate 验收标准

---

## 上游 3DGS 基座更新

TAGSs 基于 Inria 官方 3DGS 代码，上游重要更新：

| 日期 | 更新内容 |
|------|----------|
| 2023-07 | 原始发布：3D Gaussian Splatting |
| 2024 春 | OpenXR VR 查看支持 |
| 2024-10 | 训练加速（Taming-3DGS sparse Adam）、深度正则、抗锯齿、曝光补偿集成 |

TAGSs 已集成并扩展的功能：深度/法线几何先验、分辨率调度、GaussianSpa 稀疏化、分组 LR、几何控制器、LPIPS/颜色/困难视角晚期损失。

---

## 借鉴的外部方法

| 方法 | 在本项目中的体现 |
|------|------------------|
| 3D Gaussian Splatting (Kerbl et al., 2023) | 基础框架 |
| DN-Splatter (Turkulainen et al., 2024) | `experiments/geo_prior.py` 深度/法线先验 |
| GaussianSpa (Zhang et al., CVPR 2025) | `experiments/sparse_controller.py` ADMM 稀疏化 |
| Hierarchical 3DGS | 深度权重调度、per-view 曝光补偿 |
| Depth Anything V2 | 单目深度先验生成（数据准备阶段） |
| Taming 3DGS | 可选 `--optimizer_type sparse_adam` 加速 |
| Mip Splatting | 可选 `--antialiasing` 抗锯齿 |

---

## 演进规律总结

1. **v2→v4_restore（结构优化阶段）**：集成模块带来 +0.34 dB 实质性提升，模块间协同明显。
2. **v7→v11（晚期微调阶段）**：手动调参在 0.02 dB 量级竞争，接近噪声地板；必须用 per-view / 区域指标验收。
3. **被证伪的方向**：LPIPS 强训练损失、detail micro loss、hard-view 采样、exposure_lr 微调、geo_decay=26000。
4. **有效方向**：color align（超晚期弱权重）、geo decay 轻度延长（24k–25k）、spa_end=24000、弱 LPIPS guard。
5. **核心矛盾**：全局 PSNR/SSIM 与 per-view LPIPS/慢性视角回退之间的 trade-off，尚无配置同时满足所有 gate。

---

## 推荐配置速查

| 用途 | 脚本 | 说明 |
|------|------|------|
| 生产部署 | `scripts/run_bicycle_v4_restore.sh` | 稳定、全局 25.495 dB |
| 研究最优（全局） | v8 `color_geo_soft` | 全局最高但有视角回退 |
| 研究最优（per-view） | v9 `geo245_color_ultralate` | 慢性视角更好 |
| 最新实验 | `scripts/run_bicycle_v11_balanced_guarded.sh` | 弱 LPIPS guard 路线 |

---

*文档生成日期：2026-07-06 | 数据来源：EXPERIMENT_SUMMARY.md 及 v8–v11 实验汇总*
