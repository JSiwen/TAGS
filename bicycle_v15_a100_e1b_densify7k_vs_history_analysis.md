# Bicycle v15 A100 E1b densify7k 实验分析与历史对比

> 生成日期：2026-08-26  
> 当前实验：`bicycle_v15_a100_e1b_densify7k`  
> 对比范围：A100 全分辨率历史实验（v4_restore / v8 / v14）及 3080 E1b 参考

---

## 1. 核心结论

**本次 v15 A100 E1b densify7k 实验未超过先前 A100 全分辨率历史最佳。**

| 结论项 | 说明 |
|--------|------|
| 总体判断 | 历史实验（尤其 v14 monitor）明显更好 |
| 推荐 checkpoint（本次） | `ours_28500`（PSNR/SSIM 峰值） |
| 历史 A100 最佳 | `bicycle_v14_v8_monitor_28500_select` 的 `ours_28500` |
| 3080 参考 | 仅作配置对照，**绝对指标不可与 A100 混排** |

相对 v14 monitor（A100 历史 gate-pass 最佳），本次 `ours_28500`：

- 全局 PSNR：**-0.208 dB**
- 全局 SSIM：**-0.015**
- 全局 LPIPS：**+0.030**（感知质量更差）
- 逐视角：25 张测试图中 **仅 1 张 PSNR 更好，0 张 LPIPS 更好**

---

## 2. 全局指标对比

### 2.1 同协议可直接对比（A100 全分辨率）

评估协议：`images` + `resolution=-1` + `--eval --llffhold 8` + 30k iter

| 实验 | Checkpoint | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 备注 |
|------|------------|--------|--------|---------|------|
| **本次 v15 A100 E1b** | `ours_28500` | **25.340** | **0.760** | **0.217** | densify 止于 7200 |
| 本次 v15 A100 E1b | `ours_30000` | 25.336 | 0.760 | 0.216 | LPIPS 略优，PSNR/SSIM 回落 |
| v4_restore（生产基线） | `ours_30000` | 25.495 | 0.774 | 0.187 | 稳定基线 |
| v8 color_geo_soft | `ours_30000` | 25.513 | 0.775 | 0.186 | v4 系列全局最优 |
| **v14 monitor（A100 最佳）** | `ours_28500` | **25.548** | **0.775** | 0.187 | gate-pass |
| v14 monitor | `ours_30000` | 25.518 | 0.774 | 0.186 | 交付 checkpoint |

### 2.2 不可直接对比（3080 参考）

| 实验 | 协议 | Checkpoint | PSNR | SSIM | LPIPS |
|------|------|------------|------|------|-------|
| v15 3080 E1b densify7k | `images_4`, res=800 | `ours_28500` | 26.693 | 0.829 | 0.150 |

3080 与 A100 分辨率协议不同，**只能看训练趋势，不能写进同一张全分辨率对比表**。

### 2.3 本次实验内部 checkpoint 选择

来源：`bicycle_v15_a100_e1b_densify7k_checkpoint_summary.md`

| Checkpoint | PSNR | SSIM | LPIPS |
|------------|------|------|-------|
| `ours_28500` | 25.340 | 0.760 | 0.217 |
| `ours_30000` | 25.336 | 0.760 | 0.216 |

`28500 → 30000` 变化：dPSNR **-0.004**，dSSIM **-0.0003**，dLPIPS **-0.0004**

- PSNR/SSIM 在末段轻微回落，LPIPS 略改善
- 与 v14/v15 在 3080 上的观察一致
- **质量主指标仍应选 `ours_28500`**

---

## 3. 训练阶段轨迹：差距何时拉开

| 阶段 | v15 PSNR | v14 PSNR | 差距 (v15−v14) |
|------|----------|----------|----------------|
| 7000 | 23.862 | 23.868 | **-0.007**（几乎持平） |
| 18000 | 25.167 | 25.363 | **-0.195** |
| 24000 | 25.305 | 25.473 | -0.167 |
| 28500 | 25.340 | 25.548 | **-0.208** |
| 30000 | 25.336 | 25.518 | -0.182 |

**关键观察：**

1. 早期（7000）两者几乎相同，说明初始化与低分辨率阶段正常。
2. **7200 停止 densify 之后**，v15 相对 v14 迅速落后并在后续阶段维持约 **-0.17 ~ -0.21 dB** 差距。
3. 28500 时高斯数量：**v15 约 439 万 vs v14 约 853 万（少 ~48%）**。

---

## 4. 主要问题诊断

### 4.1 根因：早期停止 densify 导致全分辨率容量不足

| 配置项 | v15 A100 E1b | v14 monitor |
|--------|--------------|-------------|
| `densify_until_iter` | **7200** | **18000** |
| 28500 高斯数 | ~4.39M | ~8.53M |
| 28500 point cloud 大小 | ~1.09 GB | ~2.12 GB |

E1b 在 3080 降分辨率协议下有效，但迁到 A100 全分辨率后，**7200 停 densify 的副作用被放大**：全分辨率细节（辐条、草叶、远景 foliage）需要更多高斯，当前模型容量不足。

### 4.2 其他配置差异（次要）

| 模块 | v15 A100 E1b | v14 monitor |
|------|--------------|-------------|
| LPIPS 启动 | 24000, final 0.006 | 16000, final 0.010 |
| color align 启动 | 25000, final 0.001 | 26000, final 0.0015 |
| stage3 LR | geo×0.40, color×0.55 | geo×0.75, color×0.80 |
| PSNR loss | 22000 起，final 0.008 | 未启用 |

这些晚期差异可能存在，但**阶段轨迹显示主要分叉发生在 densify 结束之后**，因此 densify 策略是首要怀疑对象。

### 4.3 本次实验自身的正常表现

尽管不如历史最佳，本次训练曲线本身健康：

- PSNR 峰值在 28500
- 末段 PSNR/SSIM 微落 + LPIPS 微好（与 v14/v15 3080 观察一致）
- chronic views（00008/11/16/18/21）相对 7000→30000 整体仍大幅改善
- 但相对 v14，`28500→30000` 有 **1/5 chronic views 出现 triple regression**

---

## 5. 与历史实验的逐视角对比

### 5.1 相对 v14 monitor `ours_30000`

| 指标 | 改善视角数 | 退化视角数 |
|------|-----------|-----------|
| PSNR | 2 / 25 | 23 / 25 |
| SSIM | 0 / 25 | 25 / 25 |
| LPIPS | 0 / 25 | 25 / 25 |

### 5.2 相对 v4_restore / v8

| 对比基准 | 平均 dPSNR | PSNR 改善视角 |
|----------|-----------|--------------|
| vs v4_restore | -0.154 dB | 3 / 25 |
| vs v8 soft | -0.172 dB | 4 / 25 |
| vs v14 monitor 28500 | -0.208 dB | 1 / 25 |

### 5.3 回退最严重的视角

| 视角 | dPSNR vs v14 | 主要误差区域 |
|------|-------------|-------------|
| **00002.png** | **-0.480** | bg_highfreq、background |
| 00003.png | -0.383 | background、subject |
| 00015.png | -0.353 | wheel、frame |
| 00005.png | -0.349 | wheel_front、subject |
| 00020.png | -0.347 | wheel_rear、bg_highfreq |
| 00018.png | -0.339 | bg_highfreq（-0.84 dB）、background |
| 00004.png | -0.319 | background、frame_mid |
| 00001.png | -0.169 | frame_mid、wheel_front、subject |

### 5.4 相对更好的视角（例外）

| 视角 | dPSNR vs v14 | 说明 |
|------|-------------|------|
| 00022.png | +0.165 | 少数受益视角 |
| 00011.png | +0.051 | 历史 chronic view，本次略好 |
| 00008.png | -0.015 | 基本持平 |

---

## 6. 渲染图像空间差异（v15 相对 v14 更差的位置）

基于 GT 区域裁剪 PSNR 与 ref|v14|v15|GT 对比分析（分辨率均为 1600×1063）。

### 6.1 全视角平均区域回退

| 区域 | 平均 dPSNR (v15−v14) | 视觉表现 |
|------|---------------------|----------|
| wheel_rear | **-0.268 dB** | 后轮辐条、轮胎边缘更糊 |
| wheel_front | **-0.250 dB** | 前轮细节、金属高光被抹平 |
| subject | **-0.250 dB** | 车体、人物主体整体偏软 |
| frame_mid | **-0.182 dB** | 车架中段、handlebar junction 几何/颜色漂移 |
| background | **-0.168 dB** | 背景树丛、天空过渡区颗粒感加重 |
| bg_highfreq | -0.056 dB | 远景高频 foliage 在 00018 等视角损失大 |

**结论：不是单点 bug，而是全局表达能力不足**，所有预定义区域一致回退。

### 6.2 典型视角的视觉差异

#### 00002 / 00004（低 PSNR 侧视，历史 hard-regression 视角）

- 草地、背景树叶更糊，颗粒感更重
- 车辐条、细管边界更软
- GT 差分热力图在背景区误差显著增大

#### 00001 / 00005 / 00000（近景）

- 前后轮、车架局部 PSNR 低 0.4–0.6 dB
- 金属/漆面高光被抹平，像“少了一层高斯”

#### 00018 / 00020（高 PSNR 远景）

- 远景树丛、建筑边缘损失更大
- 00018 的 bg_highfreq 区域 dPSNR 达 **-0.84 dB**

#### 00011 / 00022（例外）

- 00011 chronic view 略好（+0.05 dB）
- 00022 明显更好（+0.17 dB）
- 无法改变全局 23/25 视角 PSNR 回退的事实

---

## 7. 阶段选择与 gate 观察（本次实验内部）

来源：`bicycle_v15_a100_e1b_densify7k_stage_selection_summary.md`

| 项目 | 值 |
|------|-----|
| PSNR 峰值 | `ours_28500` (25.340) |
| SSIM 峰值 | `ours_28500` (0.760) |
| LPIPS 最优 | `ours_30000` (0.216) |
| 质量主指标 | `ours_28500` |
| 28500→30000 chronic triple regression | 1 / 5 |

末段质量/感知 split 依旧存在：选 28500 保 PSNR/SSIM，选 30000 略优 LPIPS 但 PSNR/SSIM 微落。

---

## 8. 综合判断

### 8.1 本次实验的价值

| 方面 | 评价 |
|------|------|
| 训练稳定性 | 正常，阶段曲线符合 v15 预期 |
| 模型紧凑性 | 优（~1.1 GB vs ~2.1 GB） |
| 全分辨率质量 | **未达历史最佳** |
| 3080→A100 迁移 | E1b densify7k 策略需针对全分辨率重新验证 |

### 8.2 历史实验排名（A100 全分辨率）

1. **v14 monitor `ours_28500`** — 全局 + gate 双优，推荐作为 A100 论文/对比基准
2. v14 monitor `ours_30000` — 交付 checkpoint，LPIPS 略优
3. v8 color_geo_soft — 全局高但 40% 视角 PSNR 回退
4. v4_restore — 稳定生产基线
5. **本次 v15 A100 E1b** — 低于上述全部

---

## 9. 建议

### 9.1 论文与对比表

- A100 全分辨率表继续使用 **v14 monitor** 或 **v4_restore** 作为 baseline / best
- **不要把 3080 的 26.69 dB 写进 A100 全分辨率表**
- 本次 v15 可作为「E1b 迁 A100 失败/待优化案例」记录

### 9.2 后续实验方向

1. **densify_until 消融**（优先）：7200 vs 10k vs 12k vs 18k，观察 A100 全分辨率 PSNR 能否回到 25.5+
2. 若坚持 E1b 思路：可尝试 **延长 densify 但保留 late_stabilize**，而非简单沿用 3080 的 7200
3. 固定 v14 配置为 A100 对照组，每次只改一个变量

### 9.3 本次交付

- **质量主指标 checkpoint：`ours_28500`**
- 感知主指标备选：`ours_30000`（LPIPS 略优，PSNR 在 0.10 dB budget 内）

---

## 10. 数据来源

| 文件 | 用途 |
|------|------|
| `bicycle_v15_a100_e1b_densify7k_checkpoint_summary.md` | 本次 checkpoint 指标 |
| `bicycle_v15_a100_e1b_densify7k_stage_selection_summary.md` | 阶段选择与 gate |
| `bicycle_v15_a100_e1b_densify7k/results.json` | 全局 metrics |
| `bicycle_v15_a100_e1b_densify7k/per_view.json` | 逐视角 metrics |
| `TAGSs/output/bicycle_integrated_v4_restore/` | v4 基线 |
| `TAGSs/output/bicycle_v8_color_geo_soft/` | v8 最优 |
| `TAGSs/output/bicycle_v14_v8_monitor_28500_select/` | v14 A100 最佳 |
| `reference/3080_e1b/` | 3080 配置参考（不可比绝对值） |
| 区域分析脚本 | `scripts/bicycle_region_boxes.json` + GT 裁剪 PSNR |

---

*本文档整合自 2026-08-26 实验分析会话，供 v15 A100 迁移决策与论文对比参考。*
