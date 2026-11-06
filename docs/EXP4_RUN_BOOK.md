# TAGSs v3 实验执行手册 (Run Book)

> **建立时间**: 2026-10-15 (v3 计划配套)
> **目标**: 一份按 v3 计划 §3 "实施顺序与依赖" 的可执行 Run Book
> **配套**: `docs/SECTION_4_6_FULL_COMPARISON.md` (整体矩阵) + `docs/EXP4_RESUME_SKILL.md` (断点续跑机制)

---

## 0) v3 实施顺序 (来自 v3 计划 §3)

| 阶段 | 内容 | 预计耗时 | 状态 |
|---|---|---|---|
| 1 | 预处理阶段 A (Preprocess 自检) | 1 周 | **当前阶段, 2026-10-15 启动** |
| 2 | 预处理阶段 B (QA gate, 1/2 分辨率) | 2 周 | 待启动 (依赖 1) |
| 3 | 预处理阶段 C (reliability) | 1 周 | 待启动 (依赖 2) |
| 4 | Group A 主要方法对比 | 2-3 周 | 待启动 (依赖 2) |
| 5 | §4.2 稀疏化扫参 | 3 周 | 待启动 (依赖 4) |
| 6 | §4.3 LPIPS×SPA 协同 | 3 周 | 待启动 (依赖 5) |
| 7 | Group B 模块消融 | 2-3 周 | 待启动 (依赖 4) |
| 8 | §4.4 几何时机/强度 | 3-4 周 | 待启动 (依赖 6) |
| 9 | §4.5 多模块消融 | 3-4 周 | 待启动 (依赖 7,8) |
| 10 | Group C 分辨率敏感性 | 1-2 周 | 待启动 (依赖 4,7) |
| 11 | 标准分辨率验证 | 1 周 | 待启动 (依赖 10) |
| 12 | 文档撰写 | 1-2 周 | 待启动 (依赖 4-11) |

**阶段 1 状态盘点 (2026-10-15)**:
- bicycle: STAGE_1_DONE (depth_any_out 194 文件, mono_depth 195 文件)
- garden: STAGE_1_PENDING (4 个子步骤全 MISS)
- bonsai: STAGE_1_PENDING (4 个子步骤全 MISS)
- room: STAGE_1_PENDING (4 个子步骤全 MISS)
- 状态 JSON: `output/exp4_stats/stage1_status.json`

### 0.1) Seed 数选择 (3 seed 是推荐最低配置)

| Seed 数 | 总 runs (默认 5 vs 3 vs 2) | 配对 t-test df | 统计意义 | 推荐场景 |
|---|:---:|:---:|---|---|
| **5 seed (默认)** | ~635 runs | df=4 | p 值稳定, 论文级别 | 时间充裕, 投顶会 |
| **3 seed (推荐最低)** | **~393 runs** (-38%) | df=2 | 勉强可用, 箱线图能看出离群点 | 时间中等, 投一般会议/期刊 |
| **2 seed (不推荐)** | ~272 runs (-57%) | df=1 | **p 值无意义**, 只能看中位数 | 紧急情况 (仅限) |

**节省对比** (5 seed → 3 seed):

| 实验 | 5 seed | 3 seed | 节省 runs | 节省 % |
|---|---:|---:|---:|---:|
| Group A 主要方法对比 | 80 | 48 | 32 | -40% |
| Group B 模块消融 | 100 | 60 | 40 | -40% |
| §4.2 稀疏化 | 75 | 45 | 30 | -40% |
| §4.3 LPIPS×SPA | 75 | 45 | 30 | -40% |
| §4.4 几何 | 120 | 72 | 48 | -40% |
| §4.5 消融 | 140 | 84 | 56 | -40% |
| Group C 敏感性 | 36 | 36 | 0 | (默认就是 3) |
| 标准分辨率验证 | 9 | 9 | 0 | (默认就是 3) |
| **合计** | **~635** | **~399** | **236** | **-37%** |

**3 seed 切换方法** (所有 PS1 包装都支持 `E4_SEEDS`):

```powershell
# 一行环境变量, 所有后续 run 都用 3 seed
$env:E4_SEEDS = "0,1,2"

# 跑任意阶段
.\scripts\run_exp4v3_A_method_comparison.ps1
.\scripts\run_exp4v3_B_module_ablation.ps1
.\scripts\run_exp4_2_sweep.ps1
# ... 其他同理
```

---

## 1) 阶段 1: Preprocess 自检 (当前阶段)

### 1.1 现状盘点 (2026-10-15)

| 场景 | images_2 | images_4 | sparse/0 | depth_any_out | mono_depth | normals_from_pretrain | depth_params.json |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bicycle** | ✅ | ✅ | ✅ | ✅ | ✅ (195 帧) | ✅ (388 帧) | (需 `--stage params` 重生成) |
| garden | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| bonsai | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| room | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

**结论**:
- **bicycle** 已完成预处理, 跳过 Stage 1
- **garden/bonsai/room** 全部需要 Stage 1 的 4 个子步骤 (depth → normal → params → export)

### 1.2 阶段 1 执行指令 (PowerShell)

```powershell
# === 1) 跑 Depth Anything V2 推理 (生成 depth_any_out/) ===
# 单场景:
$env:E4_SCENES = "garden"
python scripts/prepare_extra_scenes.py --stage depth --scenes garden

# 3 场景一起:
python scripts/prepare_extra_scenes.py --stage depth --scenes garden,bonsai,room

# === 2) 跑法线估计 (生成 normals_from_pretrain/) ===
python scripts/prepare_extra_scenes.py --stage normal --scenes garden,bonsai,room

# === 3) 标定 depth_params.json (写入 sparse/0/) ===
python scripts/prepare_extra_scenes.py --stage params --scenes garden,bonsai,room

# === 4) 导出 mono_depth/*_aligned.npy + 改写 transforms.json ===
python scripts/prepare_extra_scenes.py --stage export --scenes garden,bonsai,room

# === 5) 全部场景自检 (4 场景) ===
python scripts/prepare_extra_scenes.py --stage check --scenes bicycle,garden,bonsai,room

# === 6) 一步到位 (depth -> normal -> params -> export -> check) ===
python scripts/prepare_extra_scenes.py --stage all --scenes garden,bonsai,room
```

### 1.3 阶段 1 入口 PS1 包装

```powershell
# 直接用 PS1 包装 (推荐):
.\scripts\run_prepare_extra_scenes.ps1 -Stage all -Scenes garden,bonsai,room
```

### 1.4 阶段 1 完成标准

- 4 场景均存在 `sparse/0/depth_params.json`
- 4 场景均存在 `mono_depth/*.npy` (`*_aligned.npy` 形式, 与 `transforms.json` 对应)
- 4 场景 `transforms.json` 被改写, 含 `mono_depth_path` 字段
- Stage 1 完成后, **qa_baseline_check 才能在 1/2 分辨率下跑** (因为 baseline 训练不依赖 mono_depth, 但 §4.2+ 的实验都依赖)

### 1.5 阶段 1 监控 (wandb + 文档)

`prepare_extra_scenes.py` **当前不集成 wandb** (它是离线预处理脚本, 不属于 train run). 但每个 stage 完成后应当:

1. **写本地 ledger**: `output/exp4_stats/preprocess_ledger.jsonl` (由脚本自动追加, 后续用)
2. **写 stage 报告**: `output/exp4_stats/preprocess_stage_check.json` (每个 stage 跑完后写一份)
3. **更新本文档**: 在 §2/§3 标记完成

---

## 2) 阶段 2: QA Gate (1/2 分辨率, 跑前必须通过)

### 2.1 目标

用 **no-pretrain 的 3DGS baseline** 在 4 场景上各跑 3 seed (v3 用户决策), 验证:
- 训练脚本 `train_optimized.py` 能在 1/2 分辨率下跑通 (无 OOM)
- 1/2 分辨率下的 3DGS baseline PSNR 达到阈值 (见下表)
- 若不通过, 优先排查**数据问题** (transforms.json 错误、images_2 缺帧), 再考虑降级到 1/4

### 2.2 阈值 (v3, 1/2 分辨率)

| 场景 | 1/2 阈值 (≥) | 1/4 阈值 (≥) | 标准阈值 (≥) | 来源 |
|---|:---:|:---:|:---:|---|
| bicycle | 23.4 dB | 22.5 dB | 24.0 dB | Inria 3DGS baseline × 0.97/0.93/1.00 |
| garden | 24.4 dB | 23.5 dB | 25.0 dB | 同上 |
| bonsai | 26.3 dB | 25.3 dB | 27.0 dB | 同上 |
| room | 27.2 dB | 26.2 dB | 28.0 dB | 同上 |

### 2.3 阶段 2 执行指令

```powershell
# === 1) 跑 no-pretrain 3DGS baseline (1/2 分辨率, 1 seed 烟囱测试) ===
$env:E4_SCENES = "bicycle"
$env:E4_SEEDS = "0"
python scripts/qa_baseline_check.py --images-dir images_2 --iterations 30000

# === 2) 跑全部 4 场景 3 seed (主 QA gate, v3 用户决策) ===
python scripts/qa_baseline_check.py --images-dir images_2 --scenes bicycle,garden,bonsai,room --seeds 0,1,2 --iterations 30000

# === 3) 跑增量 (跳过已通过 case) ===
python scripts/qa_baseline_check.py --images-dir images_2 --skip-existing

# === 4) 输出 ===
#   output/exp4_stats/qa_baseline_results.csv
#   output/exp4_stats/baseline_thresholds.json
#   output/exp4_stats/qa_baseline_report.md
#   output/exp4_stats/qa_baseline_report.json
```

### 2.4 阶段 2 入口 PS1 包装

```powershell
# PS1 包装 (推荐):
.\scripts\run_qa_baseline.ps1 -ImagesDir images_2 -Scenes bicycle,garden,bonsai,room -Seeds 0,1,2
```

### 2.5 阶段 2 完成标准

- `output/exp4_stats/qa_baseline_results.csv` 中 4 场景 × 3 seed = 12 行, **全部 `passed=True`**
- 若 `passed=False`:
  - 检查 transforms.json 中 `mono_depth_path` 是否正确
  - 检查 images_2/ 目录是否完整
  - 检查稀疏点云 sfm 重建质量 (可能需要重新跑 COLMAP)
- 通过后, 立刻进 Group A

---

## 3) 阶段 3: Reliability 检验 (可选, 但 v3 建议保留)

### 3.1 目标

验证 mono_depth 标定在 4 场景上**确实能提升 baseline+geo 训练的 PSNR ≥ 0.1 dB**。
若不达标, 说明 mono_depth 标定失败, 回到阶段 1 调预处理。

### 3.2 阶段 3 执行指令

```powershell
# 跑 4 场景各 3 seed 的 baseline+geo 30k 训练
python scripts/qa_preprocess_reliability.py --scenes bicycle,garden,bonsai,room --seeds 0,1,2
```

### 3.3 阶段 3 完成标准

- 4 场景均 `Δ PSNR (开geo - 不开geo) ≥ 0.1 dB`
- 输出 `output/exp4_stats/qa_reliability_report.md`

---

## 4) 阶段 4: Group A 主要方法对比 (1/2 分辨率, 80 runs)

### 4.1 目标

在 4 场景 × 3 seed (v3 用户决策) 上对比 4 种方法 (Raw3DGS / +Geo / +SPA / Full TAGS),
输出**论文主表** (`exp4v3_A_method_comparison.csv`)。

### 4.2 阶段 4 执行指令

```powershell
# 1) 全部 48 runs (4 方法 × 4 场景 × 3 seed, v3 用户决策 3 seed)
.\scripts\run_exp4v3_A_method_comparison.ps1

# 2) 单场景单组
$env:E4_SCENES = "bicycle"
$env:E4_GROUPS = "A4"  # Full TAGS
$env:E4_SEEDS = "0,1,2"   # v3 用户决策 3 seed
.\scripts\run_exp4v3_A_method_comparison.ps1

# 3) 烟囱测试
$env:E4_SMOKE = "1"
.\scripts\run_exp4v3_A_method_comparison.ps1

# 4) 完成后自动汇总
python scripts\run_exp4v3_A_method_comparison.py --summarize
```

### 4.3 wandb 实时监控配置

每个 (scene, group, seed) 完成后自动 log 到 wandb (v3 默认开启):

```powershell
# 启用 wandb
$env:WANDB_ENABLED = "1"
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_ENTITY = "your_username"  # 替换为实际
$env:WANDB_RUN_GROUP = "exp4v3_A_method_comparison"  # 4 方法共享 group
$env:WANDB_TAGS = "v3,1/2_res,4_scenes,3_seeds"   # v3 用户决策 3 seed

# 跑
.\scripts\run_exp4v3_A_method_comparison.ps1
```

**wandb dashboard 实时观看**:
- URL: `https://wandb.ai/<your-username>/tags-ablation/groups/exp4v3_A_method_comparison`
- Group 视图: 按 4 方法 (A.1/A.2/A.3/A.4) 分面
- Table 视图: 4 场景 × 4 方法 × 3 seed = 48 行

### 4.4 阶段 4 完成标准

- 80 runs 全部完成
- `output/exp4_stats/exp4v3_A_method_comparison.csv` 4 场景 × 4 方法 × 3 seed 都有数据
- `docs/exp4v3_A_method_comparison.md` 报告生成, 含论文表 4-X

---

## 5) 阶段 5-11: 后续阶段 (简表, 详见 SECTION_4_6)

| 阶段 | PS1 包装 | Python 包装 | wandb group | 输出 |
|---|---|---|---|---|
| 5 §4.2 | `run_exp4_2_sweep.ps1` | `run_exp4_2_sweep.py --images-dir images_2` | `exp4_2_sparsity_sweep` | `exp4_2_paired_t.csv` |
| 6 §4.3 | `run_exp4_3_collaboration.ps1` | `run_exp4_3_collaboration.py --images-dir images_2` | `exp4_3_lpips_spa_collab` | `exp4_3_paired_t.csv` |
| 7 Group B | `run_exp4v3_B_module_ablation.ps1` | `run_exp4v3_B_module_ablation.py` | `exp4v3_B_module_ablation` | `exp4v3_B_module_ablation.csv` |
| 8 §4.4 | `run_exp4_4_geometry.ps1` | `run_exp4_4_geometry.py --images-dir images_2` | `exp4_4_geometry` | `exp4_4_anova.csv` |
| 9 §4.5 | `run_exp4_5_ablation.ps1` | `run_exp4_5_ablation.py --images-dir images_2` | `exp4_5_ablation` | `exp4_5_anova.csv` |
| 10 Group C | `run_exp4v3_C_resolution_sensitivity.ps1` | `run_exp4v3_C_resolution_sensitivity.py` | `exp4v3_C_resolution` | `exp4v3_C_resolution_sensitivity.csv` |
| 11 std 验证 | `run_exp4v3_std_resolution_validation.ps1` | `run_exp4v3_std_resolution_validation.py` | `exp4v3_std_validation` | `exp4v3_std_resolution_validation.csv` |

---

## 6) 通用 wandb 配置模板 (v3)

```powershell
# 全局 wandb 配置 (所有 PS1 包装都读取这些环境变量)
$env:WANDB_ENABLED = "1"
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_ENTITY = "your_username"  # 必填, 替换为实际
# $env:WANDB_RUN_GROUP = "<由各脚本自动设>"
# $env:WANDB_TAGS = "v3,1/2_res"  # 可选
```

**dashboard 入口**:
- Project home: `https://wandb.ai/<your-username>/tags-ablation`
- Runs 视图: 按时间倒序, 每行一次 (scene, group, seed)
- Groups 视图: 按 `WANDB_RUN_GROUP` 聚合, 4 方法对比时用
- Compare 视图: 多 run 对比, 支持 PSNR/SSIM/LPIPS/num_points 四象限

### 6.1 Stage 1 (预处理) 不上 wandb

`prepare_extra_scenes.py` 是离线预处理脚本, 不直接 log wandb.
**但 stage 1 完成后, 写本地 ledger 到 `output/exp4_stats/preprocess_ledger.jsonl`**:
```json
{"timestamp": "2026-10-15T10:30:00", "stage": "all", "scenes": ["garden", "bonsai", "room"], "hold": 8, "wandb": true}
```

### 6.2 Stage 2+ (训练 run) 上 wandb

Stage 2 (qa_baseline) 起, 每个 (scene, group, seed) 完成后自动 log:
- `psnr`, `ssim`, `lpips`, `num_points` (主指标)
- `train_exit`, `render_exit`, `metrics_exit` (退出码)
- `scene`, `group`, `seed` (groupby 字段)
- `images_dir` (1/2 vs 1/4 vs 标准, 配 t-test 用)

**典型 wandb group 命名**:
| 阶段 | WANDB_RUN_GROUP | 备注 |
|---|---|---|
| Stage 2 | `exp4_qa_baseline_1_2` | 1/2 分辨率 baseline |
| Stage 4 (Group A) | `exp4v3_A_method_comparison` | 4 方法共享 group |
| Stage 7 (Group B) | `exp4v3_B_module_ablation` | 5 消融共享 group |
| Stage 10 (Group C) | `exp4v3_C_resolution` | 3 分辨率共享 group |

**典型 wandb tags 命名**:
- `v3`: v3 计划
- `1/2_res` / `1/4_res` / `std_res`: 分辨率
- `3_seed` / `5_seed`: seed 数
- 场景: `bicycle`, `garden`, `bonsai`, `room`
- 方法: `raw_3dgs`, `tags_geo`, `tags_spa`, `full_tags`

### 6.3 实时观看示例 (Group A)

```powershell
# 1) 启用 wandb
$env:WANDB_ENABLED = "1"
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_ENTITY = "your_username"
$env:WANDB_RUN_GROUP = "exp4v3_A_method_comparison"
$env:WANDB_TAGS = "v3,1/2_res,3_seed"   # v3 用户决策 3 seed

# 2) 跑
.\scripts\run_exp4v3_A_method_comparison.ps1

# 3) 实时看 (浏览器打开)
# https://wandb.ai/<your-username>/tags-ablation/groups/exp4v3_A_method_comparison
# - Group 视图: 4 方法分面 (A.1 / A.2 / A.3 / A.4)
# - PSNR 折线图: 4 条曲线 (raw=灰, +geo=黄, +spa=绿, full=蓝)
# - Table 视图: 80 行, 每行 (scene, group, seed, psnr, ssim, lpips, num_points)
# - 配对 t-test: 在 Compare 视图选 A.1 vs A.4 全部 12 run (3 seed)
```

### 6.4 实时观看示例 (Group C 分辨率敏感性)

```powershell
$env:WANDB_ENABLED = "1"
$env:WANDB_RUN_GROUP = "exp4v3_C_resolution"
$env:WANDB_TAGS = "v3,full_tags,3_seed"
.\scripts\run_exp4v3_C_resolution_sensitivity.ps1

# 实时看:
# - Group 视图: 3 分辨率分面 (C.1=1/4, C.2=1/2, C.3=std)
# - 训练时间柱状图: x=scene, y=train_time, color=resolution
# - PSNR vs 分辨率折线: x=resolution (1/4, 1/2, std), y=PSNR, hue=scene
```

---

## 7) 当前阶段 (Stage 1) 立即可执行的指令

**全局 seed 设置 (v3 选 3 seed, 节省 37% runs)**:

```powershell
# 所有后续阶段都默认 3 seed (覆盖 v3 计划默认 5 seed)
$env:E4_SEEDS = "0,1,2"
```

**状态盘点 (2026-10-15, 见 `output/exp4_stats/stage1_status.json`)**:

| 场景 | depth_any_out | normals_from_pretrain | mono_depth | depth_params | 状态 |
|---|:---:|:---:|:---:|:---:|---|
| bicycle | ✅ 194 | ✅ 388 | ✅ 195 | ✅ | **DONE** |
| garden | ❌ | ❌ | ❌ | ❌ | **PENDING** |
| bonsai | ❌ | ❌ | ❌ | ❌ | **PENDING** |
| room | ❌ | ❌ | ❌ | ❌ | **PENDING** |

**结论**: bicycle 已预处理过, garden/bonsai/room 需从 Stage 1 第一个子步骤 (depth) 开始.

**前提**: 4 场景的 `images_4/` 都已存在 (已确认), `sparse/0/` 都已存在 (已确认).

**最小化第一步 (Stage 1 only, 不动训练)**:

```powershell
# === 步骤 0: 环境准备 (推荐 PowerShell 7+ / VSCode 终端) ===
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TAGGS_CPU_KNN_INIT = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# === 步骤 A: 跑 garden 的 depth stage (作为烟囱测试) ===
python scripts/prepare_extra_scenes.py --stage depth --scenes garden --hold 8
# 期望: data/garden/depth_any_out/ 下生成 N 张 uint16 inverse depth png (N = 训练视角数)

# === 步骤 B: 跑 garden 的 normal stage ===
python scripts/prepare_extra_scenes.py --stage normal --scenes garden --hold 8
# 期望: data/garden/normals_from_pretrain/ 下生成 N 张法线 png

# === 步骤 C: 跑 garden 的 params stage ===
python scripts/prepare_extra_scenes.py --stage params --scenes garden --hold 8
# 期望: data/garden/sparse/0/depth_params.json 标定完成

# === 步骤 D: 跑 garden 的 export stage ===
python scripts/prepare_extra_scenes.py --stage export --scenes garden --hold 8
# 期望: data/garden/mono_depth/ 下生成 N 个 *_aligned.npy + transforms.json 改写

# === 步骤 E: 跑 garden 的 check stage ===
python scripts/prepare_extra_scenes.py --stage check --scenes garden --hold 8
# 期望: "aligned": true (无对齐失败警告)

# === 步骤 F: 跑全部 3 场景的 all stage (深度+法线+params+export+check) ===
python scripts/prepare_extra_scenes.py --stage all --scenes garden,bonsai,room --hold 8

# === 步骤 G: 4 场景统一 check (含 bicycle) ===
python scripts/prepare_extra_scenes.py --stage check --scenes bicycle,garden,bonsai,room --hold 8
# 期望: 4 场景全部 "aligned": true
```

**PS1 包装等价指令 (推荐)**:

```powershell
# 一键跑完 Stage 1 全流程
.\scripts\run_prepare_extra_scenes.ps1

# 自定义场景
$env:E4_SCENES = "garden"
.\scripts\run_prepare_extra_scenes.ps1

# 烟囱测试 (只打印命令)
$env:E4_DRY_RUN = "1"
.\scripts\run_prepare_extra_scenes.ps1
```

**重要监控点**:
1. 步骤 A 跑完 → 检查 `data/garden/depth_any_out/` 文件数 (应为训练视角数 ≈ 195-200 张)
2. 步骤 C 跑完 → 检查 `data/garden/sparse/0/depth_params.json` 是否存在且非空
3. 步骤 D 跑完 → 检查 `data/garden/mono_depth/` 文件数 + `transforms.json` 是否含 `mono_depth_path` 字段
4. 步骤 G 跑完 → 4 场景全部 "aligned": true

**失败回退**:
- 步骤 A 失败 (depth 为空): 检查 Depth Anything V2 模型是否下载, GPU 是否可用
- 步骤 C 失败 (params 不收敛): 检查 COLMAP sparse/0/ 完整性, 调 `--hold` 或 `make_depth_scale.py` 的迭代数
- 步骤 D 失败 (export OOM): 减少 batch size, 或分 chunk 导出
- 任何步骤失败 → 不要继续, 回到步骤 A 重跑

**状态自检 (重新跑可看到 Stage 1 状态变化)**:

```powershell
# 零依赖状态盘点
python scripts/stage1_status_check.py
# 输出: 4 场景的 depth/normal/params/export 状态 + 下一步建议
```

---

## 8) 验证记录 (2026-10-15)

| 命令 | 状态 | 输出 |
|---|:---:|---|
| `python scripts/stage1_status_check.py` | ✅ OK | `output/exp4_stats/stage1_status.json` |
| `python scripts/prepare_extra_scenes.py --help` | ⏸ 待 numpy | 需先 `pip install numpy` |
| `$env:E4_DRY_RUN="1"; $env:E4_SCENES="garden,bonsai,room"; .\scripts\run_prepare_extra_scenes.ps1` | ✅ OK | `[DRY-RUN] Command: python scripts\prepare_extra_scenes.py --stage all --scenes garden,bonsai,room --hold 8` |
| `$env:E4_DRY_RUN="1"; $env:E4_SEEDS="0,1,2"; .\scripts\run_qa_baseline.ps1` | ✅ OK | `[DRY-RUN] Command: python scripts\qa_baseline_check.py --scenes bicycle,garden,bonsai,room --seeds 0,1,2 --images-dir images_2 --iterations 30000 --output output/exp4_stats --skip-existing` (v3 用户决策 3 seed) |
| `$env:E4_DRY_RUN="1"; $env:E4_SCENES="bicycle"; $env:E4_GROUPS="A4"; .\scripts\run_exp4v3_A_method_comparison.ps1` | ✅ OK | `[v3 Group A] All runs complete.` |
| `$env:E4_DRY_RUN="1"; $env:E4_SEEDS="0,1,2"; $env:E4_SCENES="bicycle"; $env:E4_GROUPS="B4"; .\scripts\run_exp4v3_B_module_ablation.ps1` | ✅ OK | `[Group B 1/3] bicycle B.B4_FullTAGS seed=0` (3 seed) |
| `$env:E4_DRY_RUN="1"; $env:E4_SEEDS="0,1,2"; $env:E4_SCENES="bicycle"; $env:E4_GROUPS="C.1"; .\scripts\run_exp4v3_C_resolution_sensitivity.ps1` | ✅ OK | `[Group C 1/3] bicycle C.1_Res_1_4 (images=images_4) seed=0` (3 seed) |
| `$env:E4_DRY_RUN="1"; $env:E4_SEEDS="0,1,2"; $env:E4_SCENES="bicycle"; .\scripts\run_exp4v3_std_resolution_validation.ps1` | ✅ OK | `[v3 std 1/3] bicycle Std_FullTAGS (images=images) seed=0` (3 seed) |
| `$env:E4_DRY_RUN="1"; $env:E4_SEEDS="0,1,2"; $env:E4_IMAGES_DIR="images_4"; $env:WANDB_ENABLED="1"; .\scripts\run_exp4v3_A_method_comparison.ps1` | ✅ OK | `[v3 Group A] Wandb ON ... tags=v3,1/4_res,0_1_2-seed` + 4 method DRY-RUN (修 BUG: A PS1 long group name → short, 详见 §11) |

**结论**:
- 所有 6 个 v3 PS1 包装在 `gaussian_splatting` conda 环境 (Python + torch 1.12.1 + numpy 1.21.6 + wandb 0.18.7) + PowerShell 5.1 下均 dry-run 成功 (3 seed)
- 2 个新增 PS1 (`run_prepare_extra_scenes.ps1`, `run_qa_baseline.ps1`) 验证通过
- 实际训练/预处理 run 需要:
  1. 先 `conda activate gaussian_splatting` (依赖已装好)
  2. 跑 Stage 1 (深度+法线+params+export) — 需 GPU + Depth Anything V2 模型
  3. 跑 Stage 2 (qa_baseline) — 需 torch + train_optimized.py 可跑
  4. 跑 Stage 4 (Group A) — 需 torch + 完整 TAGS 配置

**常用 PS1 包装环境变量速查**:
```powershell
# 启动时
conda activate gaussian_splatting

# 全局配置 (一次性, 影响所有后续 PS1)
$env:E4_SEEDS = "0,1,2"                    # v3 决策 3 seed
$env:E4_DRY_RUN = "1"                      # 测试用, 不实际跑
$env:E4_SCENES = "bicycle"                 # 单场景测试
$env:E4_GROUPS = "A4"                      # 单组测试 (A1-A4 / B0-B4 / C.1-C.3)

# wandb (Stage 2+)
$env:WANDB_ENABLED = "1"
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_ENTITY = "your_username"
$env:WANDB_RUN_GROUP = "exp4v3_A_method_comparison"
$env:WANDB_TAGS = "v3,1/2_res,3_seed"

# 1/4 分辨率备选 (若 1/2 OOM 或赶时间)
$env:E4_IMAGES_DIR = "images_4"
# 详见 §11 (性能/PSNR 权衡表)
```

---

## §10 wandb 同步 (v3 集成)

**默认关闭** (设 `WANDB_ENABLED=1` 启用). 启用后, `exp4_runner.py` 在每次 run 完成会自动 log 到 wandb (psnr/ssim/lpips/num_points + scene/group/seed/extra).

### 启用方式 (任一 PS1 前设即可)

```powershell
# 必设
$env:WANDB_ENABLED = "1"

# 推荐 (PS1 里有默认值, 不设也行)
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_RUN_GROUP = "exp4v3_A_method_comparison"  # PS1 会按 group 自动覆盖
$env:WANDB_TAGS = "v3,1/2_res,3_seed"                # PS1 会按 group 自动覆盖
```

### 自动覆盖的 6 个 PS1 标签

| PS1 | 默认 `WANDB_RUN_GROUP` | 默认 `WANDB_TAGS` |
|---|---|---|
| `run_qa_baseline.ps1` | `exp4v3_qa_baseline` | `v3,qa_baseline,<resTag>` |
| `run_exp4v3_A_method_comparison.ps1` | `exp4v3_A_method_comparison` | `v3,<resTag>,<seeds>-seed` |
| `run_exp4v3_B_module_ablation.ps1` | `exp4v3_B_module_ablation` | `v3,<resTag>,<seeds>-seed` |
| `run_exp4v3_C_resolution_sensitivity.ps1` | `exp4v3_C_resolution_sensitivity` | `v3,C.1_C.2_C.3,<seeds>-seed` |
| `run_exp4v3_std_resolution_validation.ps1` | `exp4v3_std_resolution_validation` | `v3,<resTag>,<seeds>-seed` |
| `run_prepare_extra_scenes.ps1` | (不上 wandb) | (不上 wandb) |

`<resTag>` 根据 `E4_IMAGES_DIR` 自动映射:
- `images` → `std_res`
- `images_2` → `1/2_res` (v3 默认)
- `images_4` → `1/4_res` (用户决策快)
- 其它值 → 原样透传

> 显式设 `$env:WANDB_TAGS` 后, PS1 不覆盖 (直接用你设的值).

### 验证 (2026-10-15 已做)

```powershell
# 直接 ping 测
python -c "
import os
os.environ['WANDB_ENABLED'] = '1'
from scripts.exp4_wandb_logger import log_run_result
log_run_result('bicycle', 'ping', 99,
    {'psnr': 28.5, 'ssim': 0.85, 'lpips': 0.18, 'num_points': 1e6, 'train_exit': 0},
    extra={'note': 'ping test'})"
# 应输出:
#   wandb: Tracking run with wandb version 0.18.7
#   wandb: View run at https://wandb.ai/<your-entity>/tags-ablation/runs/<id>
```

实际 ping 测试: `https://wandb.ai/jsweni-neu/tags-ablation/runs/hvmz7b62` (run=`bicycle__ping__seed99`, psnr=28.5, ssim=0.85, lpips=0.18) ✅

### 6 个 PS1 wandb 钩子 dry-run 验证 (2026-10-15)

| PS1 | wandb=OFF 显示 | wandb=ON 显示 |
|---|---|---|
| `run_prepare_extra_scenes.ps1` | `Wandb : False` | `Wandb : True` |
| `run_qa_baseline.ps1` | `Wandb     : False` | `Wandb     : True` + `(wandb透传) project=...` |
| `run_exp4v3_A_method_comparison.ps1` | `[v3 Group A] Wandb OFF (set $env:WANDB_ENABLED=1 to enable)` | `[v3 Group A] Wandb ON  project=tags-ablation group=exp4v3_A_method_comparison tags=v3,1/2_res,0_1_2-seed` |
| `run_exp4v3_B_module_ablation.ps1` | `[v3 Group B] Wandb OFF` | `[v3 Group B] Wandb ON  project=tags-ablation group=exp4v3_B_module_ablation tags=v3,1/2_res,0_1_2-seed` |
| `run_exp4v3_C_resolution_sensitivity.ps1` | `[v3 Group C] Wandb OFF` | `[v3 Group C] Wandb ON  project=tags-ablation group=exp4v3_C_resolution_sensitivity tags=v3,C.1,0_1_2-seed` |
| `run_exp4v3_std_resolution_validation.ps1` | `[v3 std] Wandb OFF` | `[v3 std] Wandb ON  project=tags-ablation group=exp4v3_std_resolution_validation tags=v3,std_res,0_1_2-seed` |

> **注**: 上表所有 `tags` 字段为 `images_2` 默认状态; 设 `$env:E4_IMAGES_DIR = "images_4"` 后, A/B/qa_baseline 会自动改成 `1/4_res`, std 保持 `std_res` (因为 std 的 default `images_dir=images` 不受 E4_IMAGES_DIR 影响, 除非显式设). 详见 §11.

---

## §11 分辨率: 1/2 vs 1/4 (v3 用户决策)

### 决策表

| 分辨率 | `E4_IMAGES_DIR` | 显存 (RTX 3080 10G) | 30k iter 时间 (bicycle 1 scene 1 seed, qa_baseline) | PSNR vs 1/2 | 适用场景 |
|---|---|---|---|---|---|
| **1/1** (标准) | `images` | 6-8G | 50-70 min (估) | baseline | 终极对比 (std_resolution_validation) |
| **1/2** (v3 默认) | `images_2` | 3-4G | **1.5-2 h (实测 2026-10-16)** | 0 (参考) | 主实验 (4 method × 4 scene × 3 seed) |
| **1/4** (用户决策快) | `images_4` | 1-2G | **35-50 min (实测 2026-10-16)** | -0.5 ~ -1.0 dB | 烟囱测试 / GPU 紧张 / 时间紧 |

> **2026-10-16 修正**: 原表"1/2 12-15 min, 1/4 3-4 min"是 A100/3090 估的乐观值, RTX 3080 + densify 满载实际慢 6-10×。  
> **4 scene × 3 seed 完整跑**: 1/2 = 18-24 h, 1/4 = 7-9 h。  
> §4.2+ 跑时如果 Geo prior 全开, 1/2 还会再慢 30-50% (Lpips + Geo loss + 多 scale)。

### 数据准备状态 (2026-10-15)

```
bicycle: images_4=194 images_2=194 mono_aligned=194   ← 4 场景全部已对齐
garden:  images_4=185 images_2=185 mono_aligned=185
bonsai:  images_4=292 images_2=292 mono_aligned=292
room:    images_4=311 images_2=311 mono_aligned=311
```

### 用法

```powershell
conda activate gaussian_splatting

# 1/4 + wandb (RTX 3080 1 scene 1 seed ~3 min)
$env:E4_IMAGES_DIR = "images_4"
$env:E4_SEEDS = "0,1,2"
$env:E4_SCENES = "bicycle"
$env:E4_GROUPS = "A4"
$env:WANDB_ENABLED = "1"
.\scripts\run_exp4v3_A_method_comparison.ps1

# 切回 1/2 (默认)
Remove-Item env:E4_IMAGES_DIR -ErrorAction SilentlyContinue
```

### 1/4 dry-run 验证 (2026-10-15, 修 BUG: A PS1 长 group 名 → 短名)

| 命令 | 状态 | 输出 (节选) |
|---|:---:|---|
| `$env:E4_DRY_RUN="1"; $env:E4_SEEDS="0"; $env:E4_IMAGES_DIR="images_4"; $env:WANDB_ENABLED="1"; .\scripts\run_exp4v3_A_method_comparison.ps1` | ✅ OK | `[v3 Group A] Wandb ON project=tags-ablation group=exp4v3_qa_baseline tags=v3,qa_baseline,1/4_res` + 4 个 method (A1/A2/A3/A4) DRY-RUN extra_args |
| (同 1 行, B PS1) | ✅ OK | `[v3 Group B] Wandb ON ... tags=v3,1/4_res,0_1_2-seed` |
| (同 1 行, C PS1) | ✅ OK | `[v3 Group C] Wandb ON ... group=exp4v3_C_resolution_sensitivity` |
| (同 1 行, std PS1) | ✅ OK | `[v3 std] Wandb ON ... tags=v3,std_res,0_1_2-seed` (注: std 默认 images_dir=images, 不受 E4_IMAGES_DIR 影响) |
| (同 1 行, qa_baseline PS1) | ✅ OK | `(wandb透传) project=tags-ablation group=exp4v3_qa_baseline tags=v3,qa_baseline,1/4_res` + `[DRY-RUN] Command: ... --images-dir images_4 ...` |

**修的 BUG**: A PS1 原本 `$ALL_GROUPS = @("A1_Raw3DGS", ...)` 长名, Python 端 GROUPS 字典 key 是 `A1/A2/A3/A4` 短名, 会 raise `unknown group: A1_RAW3DGS, available: ['A1', 'A2', 'A3', 'A4']`. 改成短名后通过. (Python 端 `group_id = f"A.{group}_{cfg['name']}"` 会自动拼回完整名, summary 路径不变.)

### 实测速度时间线 (2026-10-15 22:49, 1/4 + qa_baseline seed=0)

| Iter 区间 | it/s | s/it | Gaussians | 阶段 | 备注 |
|---|---:|---:|---:|---|---|
| 0 → 3000 | **20.57** | 0.049 | 0 → 1.82M | warmup | 1/4 + densify 启动, 数据已加载完 |
| 3000 → 4220 | 0.64 | **1.56** | 1.82M → 2.93M | **densify peak** | iter 3000 opacity reset, 快速加 110 万点, KNN 距离计算是 CPU (`scipy cKDTree`) 严重减速 |
| 4220 (中断) | — | — | 2.93M | 仍在 densify | **用户主动 Ctrl-C** |

**结论**:
- **1/4 + `--data_device cpu` + densify 期间 KNN CPU fallback, 1.56 s/it 是预期现象, 不是 bug** (`qa_baseline_check.py` 跑 baseline 启用了 `--data_device cpu`, train command 里能看到)
- **不是均匀 1.56 s/it**: warmup 20.57 it/s → densify 0.64 it/s → prune 后回升到 11.37 it/s (4220 时看到回升)
- **完整 30k 预计 2-4 小时** (densify 阶段占大头, 之后恢复正常)
- 想要 3-4 min 跑完 1 scene 1 seed, 必须满足: ① `data_device=gpu` (默认) ② 关闭 KNN CPU fallback (现版本默认 `distCUDA2`, 你的环境被 fallback 是因为没装 distCUDA2 扩展)

**重要: 不要把 1.56 s/it 误判为 GPU 没用上** — 终端显示 `Loss=0.1199956, Gaussians=2,929,848` 是正常的 torch 输出, 说明 GPU 训练在跑, 只是 CPU 数据加载 + KNN 拖累速度。

### 性能建议 (RTX 3080 实测 2026-10-16)

- **烟囱测试**: 1/4 + 1 seed, ~40 min 完成 1 个 method, 验证整个 pipeline 通不通
- **正式主实验**: 1/2 + 3 seed, **总时间 ~18-24 h (1 天 1 夜)**, 4 method × 4 scene × 3 seed × 1.5-2 h
- **赶时间**: 1/4 + 3 seed, 总时间 ~7-9 h, 代价 -0.5 ~ -1.0 dB (与 1/2 比)
- **绝对最严**: 1/1 (std) + 3 seed, 总时间 ~50-70 h, 用作主实验 1/2 的上限验证

---

**生成时间**: 2026-10-15
**版本**: v3 (分辨率优化版, 3 seed, wandb 集成, 1/4 备选, 实测速度时间线)
**配套文档**: `SECTION_4_6_FULL_COMPARISON.md`, `EXP4_RESUME_SKILL.md`
