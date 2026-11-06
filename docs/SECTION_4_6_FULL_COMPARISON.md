# v3 §4.6 全对比综合报告 (Three-Group + 1/2 物理像素验证)

> **生成时间**: 2026-10-15 (v3 计划首版), 2026-10-17 重大修订 (分辨率策略)
> **数据源**: 三组对比 + 1/2 物理像素验证
> **分辨率策略 (2026-10-17 修订)**: 主实验统一采用 **1/4 物理像素** (--images images_2 + --resolution 2), 配 schedule warmup 1/16→1/8→1/4 物理像素. 1/2 物理像素验证 (C.3 / std) 验证可扩展性. 详见 §1.0
> **场景**: 4 个 Mip-NeRF 360 场景 (bicycle/garden/bonsai/room)
> **wandb 集成**: 默认关闭 (`$env:WANDB_ENABLED = "1"` 启用), 每次 run 自动 log (psnr/ssim/lpips + scene/group/seed), 见 EXP4_RUN_BOOK.md §10
> **当前阶段**: Stage 1 (Preprocess 自检), 2026-10-15 启动 — 详见 `docs/EXP4_RUN_BOOK.md`

---

## 1.0) 2026-10-17 分辨率策略重大修订

### 1.0.1 旧版 (v3 首版, 2026-10-15) 的问题

| 现象 | 根因 |
|---|---|
| 主实验 14k iter 跑了 6h, 远长于 v2 1/4 训练 | schedule 写 `1200:4,4000:2,total:1` 但 `--images images_2 + --resolution -1` 把所有分辨率压到 1600px (因为 orig_w>1600 触发 `global_down = orig_w/1600`), schedule 的 1/4→1/2→1.0 实际只切了 400/800/1600 px 训练 (不是 1/16→1/8→1/4 物理像素) |
| C 组分辨率敏感性失效 | `--images images_2/4/` 切换源图, 但 `--resolution -1` 又把训练像素拉回 1600, C.1/C.2/C.3 实际差异 < 100 像素, 等于白做 |
| 用户 "切 1/4" 的决策没省到算力 | 因为 `images_2` 已经是 1/2 物理源图, 切到 `images_4` (1/4 物理源图) 在 `--resolution -1` 模式下训练像素从 1600 → 1237, 几乎无变化 |

### 1.0.2 新版 (2026-10-17 修订) 的方案

**关键认识**:
- `Scene.loadCam` 公式: `resolution = orig_w / (global_down * resolution_scale)`
  - `global_down = orig_w / args.resolution` (--resolution -1 时为 1600)
  - `resolution_scale` 来自 `ResolutionSchedule.get_scale(iteration)` (即 1.0/2.0/4.0 三档 ratio)
- 因此**两个独立的旋钮**:
  - `--images <dir>` 决定源图目录 (orig_w)
  - `--resolution <int>` 决定主阶段训练像素 = orig_w / resolution
  - `resolution_schedule` 决定 warmup 比例 (1.0→2.0→4.0)

**新方案: 统一 --images, 区分 --resolution**

| 组 | --images | --resolution | 主阶段训练像素 (bicycle) | schedule warmup | 物理像素位置 |
|---|:---:|:---:|---|---|---|
| **A.1-A.4 主实验** | images_2 | **2** | 1236 (1/4 物理) | 1/16 → 1/8 → 1/4 物理 | **1/4 物理 (主)** |
| **B.0-B.4 消融** | images_2 | **2** | 1236 (1/4 物理) | 1/16 → 1/8 → 1/4 物理 | **1/4 物理 (主)** |
| **C.1 极小** | images_2 | **8** | 309 (1/16 物理) | (无 schedule) | 1/16 物理 |
| **C.2 中等** (=A/B) | images_2 | **2** | 1236 (1/4 物理) | 1/16 → 1/8 → 1/4 物理 | 1/4 物理 |
| **C.3 大** | images_2 | **1** | 2473 (1/2 物理) | (无 schedule) | 1/2 物理 |
| **std 验证** | images_2 | **1** | 2473 (1/2 物理) | (无 schedule) | 1/2 物理 (= C.3) |

**核心收益**:
- A/B 主实验主阶段训练像素 = 1236 (1/4 物理), 比旧版 1600 节省算力 **~40%** (像素与算力大致成正比)
- 1/4 物理对应 SCANeRF/Mip-NeRF 360 论文的 baseline 设置, 主观上几乎无质量损失
- C 组三档分辨率真正区分 (1/16 vs 1/4 vs 1/2 物理, 像素 0.32M / 1.0M / 4.0M, 4× 间隔)
- A/B 与 C.2 完全相同配置 → 跨组可比性最大化

### 1.0.3 对照表 (旧版 vs 新版)

| 维度 | 旧版 (2026-10-15) | 新版 (2026-10-17) |
|---|---|---|
| 主实验主阶段像素 | 1600 px (= 1/3 物理) | **1236 px (= 1/4 物理)** |
| 主实验 warmup | 1/4→1/2→1.0 = 400/800/1600 px | **1/16→1/8→1/4 = 154/309/618 px (warmup) + 1236 px (主)** |
| C.1 物理像素 | 1237 px (= 1/4 物理) | **309 px (= 1/16 物理)** |
| C.2 物理像素 | 1600 px (= 1/3 物理) | **1236 px (= 1/4 物理)** |
| C.3 物理像素 | 1600 px (= 1/3 物理) | **2473 px (= 1/2 物理)** |
| C 组分辨率倍数 | 1.0× / 1.0× / 1.0× (失效) | **8× / 2× / 1× = 8× 间隔 (有效)** |
| 单 run 时间 (A.4 bicycle) | 14h (跑到 14k) | **预计 6-7h (跑到 30k)** |
| 48 runs 总时间 | ~28d (估算 14h/run) | **~13d (估算 6.5h/run)** |
| wandb tag | `1/2_res` | `1/4_res` (主) / `1/16_res` (C.1) / `1/2_res` (C.3, std) |

---

## -1) 当前进度快照 (2026-10-15)

| 场景 | depth_any_out | normals_from_pretrain | mono_depth | depth_params | Stage 1 状态 |
|---|:---:|:---:|:---:|:---:|---|
| bicycle | ✅ 194 | ✅ 388 | ✅ 195 | ✅ | **DONE** |
| garden | ❌ | ❌ | ❌ | ❌ | **PENDING** |
| bonsai | ❌ | ❌ | ❌ | ❌ | **PENDING** |
| room | ❌ | ❌ | ❌ | ❌ | **PENDING** |

**Stage 1 立即可执行指令** (见 `docs/EXP4_RUN_BOOK.md` §7):

```powershell
# 一键跑完 3 场景的 4 个子步骤
.\scripts\run_prepare_extra_scenes.ps1
```

**Stage 1 完成后 → Stage 2 (QA baseline) 立即可执行**:

```powershell
.\scripts\run_qa_baseline.ps1
```

---

## 0) v3 与 v2 计划的关键差异

| 维度 | v2 (高可靠性版) | v3 (分辨率优化版) | 理由 |
|---|---|---|---|
| **主实验分辨率** | 1/4 (images_4) | **1/4 物理像素 (--images images_2 + --resolution 2)** | 旧版 1/2 物理像素跑 1600px 训练, 不如直接 1/4 物理像素 1236px 效率; 1/2 物理像素改为 C.3 / std 验证用 |
| **三组对比结构** | 4 个独立 §4.2-§4.5 扫参 | **三组 + 1/2 物理像素验证** | 用户要求重组: Group A 主要方法对比, Group B 模块消融, Group C 分辨率敏感性 |
| **§4.2-§4.5 扫参** | 主体 | **保留, 复用同一脚本** | 原始扫参仍是模块内归因的核心证据, 不删除 |
| **1/2 物理像素验证** | 仅作为后续扩展 | **v3 显式补做** | 验证 1/4 物理像素结论的可扩展性, 2-3 场景 |

---

## 1) 三组对比 + 标准分辨率验证 (总览)

### 1.1 Group A: 主要方法对比 (1/4 物理像素, 4 方法 × 4 场景 × 3 seed = 48 runs, v3 用户决策 3 seed)

| 组 | 名称 | 描述 | 与 Group B 的关系 |
|---|---|---|---|
| A.1 | Raw3DGS | 原始 3DGS, 无任何 TAGS 模块 (对照组) | = B.0 |
| A.2 | TAGS_Geo | 3DGS + 几何先验 (depth + normal, 后期介入) | = B.1 |
| A.3 | TAGS_SPA | 3DGS + GaussianSpa 稀疏化 (无几何) | (Group B 未单独跑, 互补) |
| A.4 | Full_TAGS | 3DGS + Geo + SPA + LPIPS staged + 多分辨率 | = B.4 |

**配置**: `--images images_2 --resolution 2` (= 1/4 物理像素主阶段, 配 schedule 1/16→1/8→1/4 warmup, 详见 §1.0)

**指标** (论文主表):
- 质量: PSNR / SSIM / LPIPS
- 效率: 训练时间 (s) / 峰值显存 (GB) / 渲染 FPS
- 资源: 最终高斯数量 (N) / 模型文件大小 (M)

### 1.1.1 算力预算（高斯数）统一控制（2026-10-16 新增）

**核心约定**: Group A/B/C 所有方法组在统一算力预算下比较, 即**所有组共享 `--max_gaussians 2,500,000` 硬上限**. 这是为了在"同等算力"下比较"质量谁更高", 而不是"谁的模型更稠密/更稀疏" — 后者会让审稿人质疑"TAGS 点数少当然快".

| 维度 | 统一设置 | 备注 |
|---|---|---|
| 硬上限 | `--max_gaussians 2,500,000` | `densify_and_prune` 后超出会 raise, 4 个 Group A 脚本 (A/B/C/std) 全部统一 |
| 默认剪枝阈值 | `--prune_opacity_threshold 0.02` (Inria 原版 0.005 偏保守) | 高于 Inria 原版, 给 SPA 留更多候选 |
| Inria 原版剪枝 | `--prune_min_visibility 0`, `--prune_max_screen_size -1` | 走 Inria 默认的 opacity 剪枝路径 |
| 增殖控制 | `--densify_until_fraction 0.50`, `--prune_only_from_fraction 0.50` | 30k iter 中前 15k 自由增殖, 后 15k 只剪不增 |
| A.1 (baseline) 的额外约束 | `--sparse_mode none`, `--prune_ratio2 0.0` | **彻底关掉 SPA**, 让 A.1 严格等价于 Inria 原版 |
| A.3 (TAGS_SPA) 的额外约束 | `--sparse_mode spa`, `--prune_ratio2 0.04` | SPA 主动剪枝, 同样 2.5M 上限下点数会 < A.1 |

**预期最终点数** (供实验前 sanity check):

| 组 | bicycle | garden | bonsai | room | 解释 |
|---|---:|---:|---:|---:|---|
| A.1 Raw3DGS | ~1.8–2.3M | ~1.5–2.0M | ~1.2–1.7M | ~0.8–1.3M | 受 `prune_opacity_threshold=0.02` 控, 自然剪枝 |
| A.2 TAGS_Geo | ~1.5–2.0M | ~1.3–1.8M | ~1.0–1.5M | ~0.7–1.1M | 几何先验收敛更快, 主动剪枝更多 |
| A.3 TAGS_SPA | ~1.0–1.5M | ~0.9–1.4M | ~0.7–1.1M | ~0.5–0.8M | SPA 主动删低贡献点 |
| A.4 FullTAGS | ~0.8–1.3M | ~0.7–1.1M | ~0.6–1.0M | ~0.4–0.7M | Geo + SPA 双重稀疏 |

> 论文中描述: *"All methods are evaluated under a fixed 2.5M-Gaussian computational budget enforced by `--max_gaussians`. This isolates quality differences from sparsity artifacts: any PSNR/SSIM/LPIPS gain over Raw3DGS reflects module contributions, not reduced model capacity."*

**注意事项**:
- 这 2.5M 是**所有 1/4 物理像素主实验的"公平预算"**; 1/16 物理 (C.1) 不强制, 1/2 物理 (C.3 / std) 也保持同样上限, 但显存吃紧时会自动 raise
- 后期替换 baseline 为标准 3DGS (`train.py`) 时, 仍设 `--max_gaussians 2,500,000` (但 `train.py` 当前不支持该参数, 需要 fork 一份; **见 exp4_runner.py 是否需要 backport**)
- 跑完后 `num_points_mean` 是核心 sanity check: A.1 应最稠密, A.4 应最稀疏, 否则说明模块没生效

**输出文件**:
- `output/exp4_stats/exp4v3_A_method_comparison.csv` (跨场景主表)
- `output/exp4_stats/exp4v3_A_per_scene.csv` (per-scene 详细)
- `docs/exp4v3_A_method_comparison.md` (论文表 4-X 主要方法对比)

---

### 1.2 Group B: 模块消融实验 (1/4 物理像素, 5 组 × 4 场景 × 3 seed = 60 runs, v3 用户决策 3 seed)

逐项启用 TAGS 的 3 个核心模块, 验证每个模块的独立贡献. 与 §4.5 多模块消融的区别: 只关心 3 个核心模块 (Geo, SPA, Staged), 5 组线性"逐项启用"消融, 避免 2^k 析因的组合爆炸.

**配置**: 与 Group A 完全一致 (`--images images_2 --resolution 2`), 保证跨组可比性.

| 组 | 名称 | Geo | SPA | Staged | 与 §4.5 关系 |
|---|---|:---:|:---:|:---:|---|
| B.0 | Raw3DGS | off | off | off | = §4.5.A |
| B.1 | Geo only | on | off | off | (新) |
| B.2 | Geo + SPA | on | on | off | (新) |
| B.3 | Geo + Staged | on | off | on | (新) |
| B.4 | FullTAGS | on | on | on | = §4.5.F |

**关键点**:
- "除被消融的模块外, 其他设置保持一致" — 这是 v3 用户明确要求
- 通过 B.1 vs B.2: 验证 SPA 独立贡献
- 通过 B.1 vs B.3: 验证 Staged 独立贡献
- 通过 B.2 vs B.4: 验证 Staged 独立贡献 (在 SPA 基础上)
- 通过 B.3 vs B.4: 验证 SPA 独立贡献 (在 Staged 基础上)
- 通过 B.2 vs B.3: 互相对照, 哪个模块贡献更大

**输出文件**:
- `output/exp4_stats/exp4v3_B_module_ablation.csv`
- `docs/exp4v3_B_module_ablation.md`

> **算力预算统一**: 与 Group A 完全一致, 所有 B.X 组共享 `--max_gaussians 2,500,000` 硬上限 (见 §1.1.1). 5 组 (B.0–B.4) 在同等算力下做"逐项启用"消融, 任何 PSNR 提升 = 模块独立贡献.

---

### 1.3 Group C: 分辨率敏感性实验 (1/16 vs 1/4 vs 1/2 物理像素, 4 场景 × 3 seed = 36 runs)

完整 TAGSs 在 3 种物理像素分辨率下分别训练与评估. 观察分辨率变化对:
- 重建质量 (PSNR/SSIM/LPIPS)
- 资源消耗 (训练时间, 峰值显存)
- 稀疏化效果 (最终高斯数, 模型大小)

| 组 | 名称 | --images | --resolution | 主阶段训练像素 (bicycle) | 描述 |
|---|---|:---:|:---:|---|---|
| C.1 | Res_1_16 | images_2 | 8 | 309 × 206 | 1/16 物理像素 + 完整 TAGS (无 schedule 切换, 已是最小) |
| C.2 | Res_1_4  | images_2 | 2 | 1236 × 822 | 1/4 物理像素 + 完整 TAGS (**= A/B 主实验**) |
| C.3 | Res_1_2  | images_2 | 1 | 2473 × 1643 | 1/2 物理像素 + 完整 TAGS (= std 验证) |

**注意**:
- **1/4 物理**是主实验分辨率 (Group A/B + C.2 用), 详见 §1.0
- **1/16 物理**是 C 组低分辨率端点, 验证在极低像素下完整 TAGS 的鲁棒性
- **1/2 物理**是 v3 补充验证, 验证 1/4 物理的结论可扩展性
- 所有 3 组都用相同 seed (0,1,2), 便于跨分辨率配对 t 检验

**输出文件**:
- `output/exp4_stats/exp4v3_C_resolution_sensitivity.csv`
- `docs/exp4v3_C_resolution_sensitivity.md`

---

### 1.4 1/2 物理像素验证 (3 场景 × 完整 TAGS × 3 seed = 9 runs)

v3 用户要求: "挑选 2-3 个具有代表性的 Mip-NeRF 360 场景做高分辨率验证"

代表性场景选取:
- **bicycle** (户外复杂, 论文主表 1)
- **bonsai**  (室内复杂, 论文主表 2)
- **garden**  (户外中等, 论文主表 3)

完整 TAGSs (与 Group C.C.3 完全一致) 在 1/2 物理像素下训练与评估, 验证 1/4 物理像素 (Group A/B + C.2) 得出的结论是否能扩展到 1/2 物理像素.

**配置**: `--images images_2 --resolution 1` (= C.3 模式, 跨验证可读)

**输出文件**:
- `output/exp4_stats/exp4v3_std_resolution_validation.csv`
- `docs/exp4v3_std_resolution_validation.md`

---

## 2) 实验矩阵总览

> **注 (2026-10-15 用户决策)**: 主实验统一采用 **3 seed (0,1,2)** 而非原计划的 5 seed.
> 所有 PS1 包装默认通过 `$env:E4_SEEDS = "0,1,2"` 设置. 节省 ~37% runs (~635 → ~399).
> 配对 t-test 仍可用 (df=2), 箱线图能识别离群点.

| 实验 | 场景数 | 分辨率 | 组数 | Seed | 总 runs | 主要输出 |
|---|---|---|---|---|---|---|
| **Group A** 主要方法对比 | 4 | 1/2 | 4 | **3** | **48** | `exp4v3_A_method_comparison.csv` |
| **Group B** 模块消融 | 4 | 1/2 | 5 | **3** | **60** | `exp4v3_B_module_ablation.csv` |
| **Group C** 分辨率敏感性 | 4 | 1/4/1/2/标准 | 3 | 3 | 36 | `exp4v3_C_resolution_sensitivity.csv` |
| **std 验证** | 3 | 标准 | 1 | 3 | 9 | `exp4v3_std_resolution_validation.csv` |
| **§4.2 稀疏化扫参** | 3 | 1/2 | 5 | **3** | **45** | `exp4_2_paired_t.csv` |
| **§4.3 LPIPS×SPA 协同** | 3 | 1/2 | 5 | **3** | **45** | `exp4_3_paired_t.csv` |
| **§4.4 几何时机/强度** | 4 | 1/2 | 6 | **3** | **72** | `exp4_4_anova.csv` |
| **§4.5 多模块消融** | 4 | 1/2 | 7 | **3** | **84** | `exp4_5_anova.csv` |
| **合计** | | | | | **~399 runs** (-37%) | |

### 2.1) Stage 2 QA baseline (3 seed)

| 场景 | Seed | 总 runs | 阈值 (≥) |
|---|---|---|---|
| bicycle | 0,1,2 | 3 | 23.4 dB (1/2) |
| garden | 0,1,2 | 3 | 24.4 dB (1/2) |
| bonsai | 0,1,2 | 3 | 26.3 dB (1/2) |
| room | 0,1,2 | 3 | 27.2 dB (1/2) |
| **合计** | | **12 runs** | |

### 2.2) wandb 同步 (默认关闭, 2026-10-15 集成)

| 项 | 状态 | 说明 |
|---|---|---|
| **登录** | ✅ 已登录 | entity=`jsweni-neu` (jsweninlife), 验证 ping: `https://wandb.ai/jsweni-neu/tags-ablation/runs/hvmz7b62` |
| **默认开关** | ❌ OFF (不打扰) | 设 `$env:WANDB_ENABLED = "1"` 启用 |
| **项目名** | `tags-ablation` | PS1 自动填默认值, 可被 `$env:WANDB_PROJECT` 覆盖 |
| **Run group** | 按 PS1 填 | A=`exp4v3_A_method_comparison`, B=`exp4v3_B_module_ablation`, C=`exp4v3_C_resolution_sensitivity`, std=`exp4v3_std_resolution_validation`, qa_baseline=`exp4v3_qa_baseline` |
| **Tags** | 自动按 `E4_RESOLUTION` 映射 | `resolution=1`→`1/2_res`, `resolution=2`→`1/4_res` (主), `resolution=8`→`1/16_res` (C.1) |
| **Log 粒度** | 每 run 1 个 wandb run | `run_full()` 末尾自动调 `_log_to_ledger_and_wandb`, 写 (psnr/ssim/lpips/num_points + scene/group/seed) |
| **本地 cache** | `wandb/` 目录 | 离线时暂存, 联网自动 sync |
| **重置** | `Remove-Item env:WANDB_*` | 清 env 即关 |

启用示例:
```powershell
$env:WANDB_ENABLED = "1"
$env:E4_IMAGES_DIR = "images_2"
$env:E4_SEEDS = "0,1,2"
$env:E4_SCENES = "bicycle"
.\scripts\run_exp4v3_A_method_comparison.ps1
# → https://wandb.ai/jsweni-neu/tags-ablation
```

### 2.3) 极小分辨率 (C.1, 用户决策快备选, 2026-10-17 修订)

| 项 | 1/4 物理 (主实验默认, A/B/C.2) | 1/16 物理 (C.1 极小) |
|---|---|---|
| `E4_RESOLUTION` | `2` | `8` |
| `E4_IMAGES_DIR` | `images_2` | `images_2` (统一) |
| 主阶段训练像素 (bicycle) | 1236 | 309 |
| 显存 (RTX 3080) | 2-3 GB | 0.5-1 GB |
| 30k iter 时间 (bicycle 1 scene 1 seed) | ~6.5h | ~1.5h |
| PSNR 损失 | baseline (参考) | -3 ~ -5 dB (预期, 1/16 物理太粗) |
| wandb tag | `1/4_res` | `1/16_res` |
| 数据准备 | 4 场景 194/185/292/311 帧 | 4 场景 194/185/292/311 帧 (同预处理) |

**适用场景**:
- 烟囱测试: 1/16 物理 + 1 seed 验证整个 pipeline 通不通 (~1.5h)
- GPU 紧张 (8G 以下): 1/16 物理唯一选择
- 赶时间: 1/16 物理 + 3 seed 总时间 ~4.5h vs 1/4 物理 + 3 seed ~19.5h, 代价 -3 ~ -5 dB (主结论只能用 1/4 物理出)

**1/2 物理** (C.3 / std 验证) 仅作上限验证, 用 `run_exp4v3_std_resolution_validation.ps1` (默认 `--images-dir images_2 --resolution 1`)。

---

## 3) 实施顺序与依赖

1. **预处理阶段 A (Preprocess 自检, 1 周)**:
   - 改造 `export_bicycle_mono_depth.py` 为 `scripts/prepare_extra_scenes.py`
   - 对 garden/bonsai/room 跑 Depth Anything V2 推理 + `depth_params.json` 标定 + 法线估计
2. **预处理阶段 B (QA gate, 2 周)**:
   - `python scripts/qa_baseline_check.py --images-dir images_2` (1/2 分辨率基线)
   - 4 场景各跑 no-pretrain baseline 30k 训练 (3 seed), 验证 PSNR ≥ 23-27 dB (1/2 阈值)
3. **预处理阶段 C (reliability, 1 周)**:
   - `python scripts/qa_preprocess_reliability.py` (mono_depth 标定可靠性)
4. **Group A 主要方法对比 (2-3 周)**:
   - `python scripts/run_exp4v3_A_method_comparison.py --scenes bicycle,garden,bonsai,room --seeds 0,1,2`
5. **§4.2 稀疏化扫参 (3 周)**:
   - `python scripts/run_exp4_2_sweep.py --images-dir images_2 --seeds 0,1,2`
6. **§4.3 LPIPS×SPA 协同 (3 周)**:
   - `python scripts/run_exp4_3_collaboration.py --images-dir images_2 --seeds 0,1,2`
7. **Group B 模块消融 (2-3 周)**:
   - `python scripts/run_exp4v3_B_module_ablation.py --scenes bicycle,garden,bonsai,room --seeds 0,1,2`
8. **§4.4 几何时机/强度 (3-4 周)**:
   - `python scripts/run_exp4_4_geometry.py --images-dir images_2 --seeds 0,1,2`
9. **§4.5 多模块消融 (3-4 周)**:
   - `python scripts/run_exp4_5_ablation.py --images-dir images_2 --seeds 0,1,2`
10. **Group C 分辨率敏感性 (1-2 周)**:
    - `python scripts/run_exp4v3_C_resolution_sensitivity.py --scenes bicycle,garden,bonsai,room --seeds 0,1,2`
11. **标准分辨率验证 (1 周, 最后做)**:
    - `python scripts/run_exp4v3_std_resolution_validation.py --scenes bicycle,bonsai,garden --seeds 0,1,2`
12. **文档撰写 (1-2 周)**:
    - `docs/exp4v3_A_method_comparison.md`
    - `docs/exp4v3_B_module_ablation.md`
    - `docs/exp4v3_C_resolution_sensitivity.md`
    - `docs/exp4v3_std_resolution_validation.md`

总时间估算: **20-25 周 (5-6 个月)**, 假设单 A100 全程跑满

---

## 4) 复现指令

### 4.0 跑 Stage 1 (Preprocess, 4 子步骤) — 当前阶段第一步

```powershell
# 一键跑完 garden/bonsai/room 的 4 个子步骤 (depth -> normal -> params -> export)
.\scripts\run_prepare_extra_scenes.ps1

# 单场景烟囱测试 (推荐先用这个验证 GPU 推理可用)
$env:E4_SCENES = "garden"
$env:E4_STAGE = "depth"
.\scripts\run_prepare_extra_scenes.ps1

# 完成后状态自检
python scripts/stage1_status_check.py
# 期望: 4 场景全部 STAGE_1_DONE
```

### 4.0.1 跑 Stage 2 (QA baseline, 1/4 物理像素, 3 seed)

```powershell
# v3 用户决策: 全部用 3 seed (节省 37% runs)
$env:E4_SEEDS = "0,1,2"

# 1/4 物理像素 4 场景 × 3 seed × 30k iter (12 runs)
# 注意: --images 统一 images_2, --resolution 2 决定主阶段 1/4 物理像素
$env:E4_RESOLUTION = "2"
.\scripts\run_qa_baseline.ps1

# 1/16 物理像素备选 (若 1/4 OOM 或赶时间)
$env:E4_RESOLUTION = "8"
.\scripts\run_qa_baseline.ps1
```

### 4.1 跑 Group A (主要方法对比, 80 runs)

```powershell
# 全部 80 runs
.\scripts\run_exp4v3_A_method_comparison.ps1

# 单场景单组
$env:E4_SCENES = "bicycle"
$env:E4_GROUPS = "A4"
$env:E4_SEEDS = "0,1,2"   # v3 用户决策 3 seed
.\scripts\run_exp4v3_A_method_comparison.ps1

# Smoke
$env:E4_SMOKE = "1"
.\scripts\run_exp4v3_A_method_comparison.ps1
```

### 4.2 跑 Group B (模块消融, 100 runs)

```powershell
.\scripts\run_exp4v3_B_module_ablation.ps1
$env:E4_GROUPS = "B4"
$env:E4_SMOKE = "1"
.\scripts\run_exp4v3_B_module_ablation.ps1
```

### 4.3 跑 Group C (分辨率敏感性, 36 runs)

```powershell
.\scripts\run_exp4v3_C_resolution_sensitivity.ps1
# 只跑标准分辨率 (C.3)
$env:E4_GROUPS = "C.3"
.\scripts\run_exp4v3_C_resolution_sensitivity.ps1
```

### 4.4 跑标准分辨率验证 (9 runs)

```powershell
.\scripts\run_exp4v3_std_resolution_validation.ps1
$env:E4_SCENES = "bicycle"
$env:E4_SEEDS = "0"
.\scripts\run_exp4v3_std_resolution_validation.ps1
```

### 4.5 跑 §4.2-§4.5 扫参 (保留 v2 设计, 改 1/2 分辨率)

```powershell
# §4.2
$env:E4_IMAGES_DIR = "images_2"
.\scripts\run_exp4_2_sweep.ps1

# §4.3
$env:E4_IMAGES_DIR = "images_2"
.\scripts\run_exp4_3_collaboration.ps1

# §4.4
$env:E4_IMAGES_DIR = "images_2"
.\scripts\run_exp4_4_geometry.ps1

# §4.5
$env:E4_IMAGES_DIR = "images_2"
.\scripts\run_exp4_5_ablation.ps1
```

---

## 5) 关键文件清单 (v3 新增/修改)

### 5.1 新增脚本

| 路径 | 描述 |
|---|---|
| `scripts/run_exp4v3_A_method_comparison.py` | Group A: 4 方法对比 (1/4 物理像素, --images images_2 --resolution 2) |
| `scripts/run_exp4v3_A_method_comparison.ps1` | 同上, PowerShell 包装 |
| `scripts/run_exp4v3_B_module_ablation.py` | Group B: 5 组逐项启用消融 |
| `scripts/run_exp4v3_B_module_ablation.ps1` | 同上, PowerShell 包装 |
| `scripts/run_exp4v3_C_resolution_sensitivity.py` | Group C: 3 分辨率 × 完整 TAGS |
| `scripts/run_exp4v3_C_resolution_sensitivity.ps1` | 同上, PowerShell 包装 |
| `scripts/run_exp4v3_std_resolution_validation.py` | 标准分辨率验证 (2-3 场景) |
| `scripts/run_exp4v3_std_resolution_validation.ps1` | 同上, PowerShell 包装 |
| `scripts/run_prepare_extra_scenes.ps1` | Stage 1 预处理 (4 子步骤一键跑) |
| `scripts/run_qa_baseline.ps1` | Stage 2 QA baseline (4 场景 × 3 seed 一键跑) |
| `scripts/stage1_status_check.py` | 零依赖状态盘点 (验证 Stage 1 完成度) |

### 5.2 修改的现有脚本 (1/4 物理像素为默认, --images images_2 + --resolution 2)

| 路径 | 改动 |
|---|---|
| `scripts/exp4_runner.py` | `run_full()` 增加 `resolution` 参数 (默认 `2`), train_cmd 注入 `--resolution {resolution}` |
| `scripts/qa_baseline_check.py` | 支持 `--resolution` 参数, 阈值按分辨率调整 (1/4 物理像素阈值 0.95x, 1/16 物理像素阈值 0.85x) |
| `scripts/run_exp4_2_sweep.py` | 增加 `--resolution` 参数 (默认 `2`) |
| `scripts/run_exp4_3_collaboration.py` | 增加 `--resolution` 参数 (默认 `2`) |
| `scripts/run_exp4_4_geometry.py` | 增加 `--resolution` 参数 (默认 `2`) |
| `scripts/run_exp4_5_ablation.py` | 增加 `--resolution` 参数 (默认 `2`) |
| `scripts/run_exp4_2_sweep.ps1` | 默认 `--resolution 2`, 可通过 `E4_RESOLUTION` 环境变量覆盖 |
| `scripts/run_exp4_3_collaboration.ps1` | 同上 |
| `scripts/run_exp4_4_geometry.ps1` | 同上 |
| `scripts/run_exp4_5_ablation.ps1` | 同上 |

### 5.3 新增文档

| 路径 | 描述 |
|---|---|
| `docs/SECTION_4_6_FULL_COMPARISON.md` (本文件) | v3 总览: 三组 + 标准分辨率验证 |
| `docs/exp4v3_A_method_comparison.md` | Group A 详细报告 (待跑完后回填) |
| `docs/exp4v3_B_module_ablation.md` | Group B 详细报告 (待跑完后回填) |
| `docs/exp4v3_C_resolution_sensitivity.md` | Group C 详细报告 (待跑完后回填) |
| `docs/exp4v3_std_resolution_validation.md` | 标准分辨率验证报告 (待跑完后回填) |

### 5.4 新增输出目录

| 路径 | 描述 |
|---|---|
| `output/exp4v3_A_{scene}/A.X_{method}/train/seed{N}/` | Group A 训练产物 |
| `output/exp4v3_B_{scene}/B.X_{ablation}/train/seed{N}/` | Group B 训练产物 |
| `output/exp4v3_C_{scene}/C.X_Res_*/train/seed{N}/` | Group C 训练产物 |
| `output/exp4v3_std_{scene}/Std_FullTAGS/train/seed{N}/` | 标准分辨率验证训练产物 |
| `output/exp4_stats/exp4v3_*.csv` | 全部 v3 汇总 CSV |
| `output/exp4_stats/qa_baseline_results.csv` | QA gate 1/4 物理像素基线 |

---

## 6) 风险与回退 (v3 关键风险)

- **1/4 物理像素 OOM**: 监控前 5 个 run 的 nvidia-smi, 若 OOM 立即降级到 1/16:
  - `python scripts/run_exp4v3_A_method_comparison.py --resolution 8`
  - `python scripts/run_exp4v3_B_module_ablation.py --resolution 8`
  - 同步改 1/16 物理的 QA threshold (V3_THRESHOLDS_SIXTEENTH, 0.85x of 1/4)
- **1/4 物理像素时间超预算**: 若单 run > 8 小时, 降级到 1/16
- **1/2 物理像素 (C.3 / std) 显存/时间最重**: 仅 9 runs, 跑不起来可降低 seed 数到 1 或完全省略
- **Group C.C.3 跑不动**: 单独跳过, 不影响 1/4 物理主结论
- **存储**: ~399 runs × 30k iter × 1/4 物理像素预计 250-350 GB, 提前规划 `output/` 磁盘配额
- **数据预处理**: garden/bonsai/room 需先跑 `prepare_extra_scenes.py` 生成 mono_depth + normals
- **wandb 同步失败**: 默认 OFF 不影响主流程; 启用后若断网, runs 暂存 `wandb/` 目录, 联网自动 sync; 同步报错可 `wandb sync wandb/run-XXX` 手动补
- **1/16 物理像素误用**: 默认 `E4_RESOLUTION=2` (1/4 物理), 不设即主实验; 1/16 物理 (C.1) 仅在显式设 `E4_RESOLUTION=8` 时启用, PSNR 损失 -3 ~ -5 dB (预期, 1/16 太粗)

---

**生成时间**: 2026-10-15 (首版), 2026-10-17 (分辨率策略重大修订: --images + --resolution 联合控制, 1/4 物理像素主实验)
**版本**: v3.1 (分辨率优化版, 3 seed, wandb 集成, 1/4 物理主 + 1/16/1/2 物理验证)
**作者**: TAGS 实验自动规划
**配套文档**: `EXP4_RUN_BOOK.md` (操作手册), `EXP4_RESUME_SKILL.md` (断点续跑)
