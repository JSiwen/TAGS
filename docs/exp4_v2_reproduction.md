# TAGS 第 4 章 消融实验 — 复现命令 (v2 高可靠性版)

> 完整论文: *第四章 分阶段几何—感知协同优化策略设计与实验分析* §4.2-§4.5
> 数据集: Mip-NeRF 360 的 4 个代表性场景 — bicycle (户外复杂) / garden (户外中等) / bonsai (室内复杂) / room (室内中等)
> 总 runs: **450** (5+5+6+7=23 组 × 4 场景 × 5 seed = 460, 部分用 3 场景)
> Seed: 0, 1, 2, 3, 4 (5 seed, 报告 95% CI)
> 估计时间: A100 全程跑满 5 个月

## 前置: 验证数据

```bash
# 验证 4 场景 images_4 都已就绪
ls data/bicycle/images_4
ls data/garden/images_4
ls data/bonsai/images_4
ls data/room/images_4
```

---

## Step 0: 预处理 (一次性, 4 场景)

### 0a) bicycle 已就绪, 只需 check
```bash
python scripts/prepare_extra_scenes.py --stage check --scenes bicycle
```

### 0b) garden/bonsai/room 需跑 Depth Anything V2 + 法线 + depth_params
```bash
# 全部 3 阶段一起跑 (depth + normal + params + export + check)
# 注意: 需要 Depth-Anything-V2 / Depth-Anything-3 环境, 命令可能需要按本地路径调整
python scripts/prepare_extra_scenes.py --stage all --scenes garden,bonsai,room
```

### 0c) 自检 4 场景都通过
```bash
python scripts/prepare_extra_scenes.py --stage check --scenes bicycle,garden,bonsai,room
```

---

## 实时记录机制 (v2 关键功能, 默认开启)

**每次 `(scene, group, seed)` run 完成, 自动:**
1. 追加一行到 `output/exp4_stats/exp4_ledger.jsonl` (append-only, 带文件锁)
2. 刷新 `output/exp4_stats/exp4_ledger.csv` 视图
3. 刷新对应 phase 的 `docs/exp4_<phase>_results.md` (含 mean ± SD + 95% CI)
4. **如果** `WANDB_ENABLED=1`, 调用 wandb.log() 实时 log 到云端 dashboard

**启用 wandb 实时预览 (可选):**
```powershell
# 一次性安装
pip install wandb
wandb login

# 启用方式 (在每次跑 §4.2/§4.3/§4.4/§4.5 之前 export)
$env:WANDB_ENABLED = "1"
$env:WANDB_PROJECT = "tags-ablation"
$env:WANDB_ENTITY = "your_username"   # 可选
$env:WANDB_RUN_GROUP = "exp4_5_Full"   # 可选, 用于 wandb dashboard 分组
$env:WANDB_TAGS = "bicycle,full,5seed" # 可选, 逗号分隔

# 然后正常跑训练脚本即可, 每个 seed 自动 log
python scripts/run_exp4_5_ablation.py --scenes bicycle --groups F --seeds 0
```

**手动补 log (run 完成但 wandb 没开):**
```powershell
# 用 ledger 已有数据重 log
python scripts/exp4_wandb_logger.py --scene bicycle --group 4_5_F_Full_TAGS --seed 0
```

**手动重生成所有 docs (例如 docs 文件被误删):**
```powershell
python scripts/exp4_doc_writer.py --refresh-all
```

**查看 ledger 当前状态:**
```powershell
python scripts/exp4_ledger.py --list
```

---

## Step 1: QA gate A — 跑 no-pretrain 3DGS baseline 验证 PSNR 阈值

```bash
# 4 场景 × 5 seed = 20 个 baseline 训练 (30000 iter)
python scripts/qa_baseline_check.py

# 输出:
#   output/exp4_stats/qa_baseline_results.csv
#   output/exp4_stats/qa_baseline_report.md
# 阈值: bicycle >= 24.0, garden >= 25.0, bonsai >= 27.0, room >= 28.0 (dB)
# 任一场景不通过 -> 回到 Step 0 重做预处理
```

## Step 2: QA gate B — 跑 baseline+geo 验证 mono_depth 标定可靠性

```bash
# 4 场景 × 3 seed = 12 个 baseline+geo 训练
python scripts/qa_preprocess_reliability.py

# 输出:
#   output/exp4_stats/qa_reliability_results.csv
#   output/exp4_stats/qa_reliability_report.md
# 阈值: baseline+geo 比 baseline 高 >= 0.10 dB
# 任一场景不通过 -> mono_depth 标定失败, 回到 Step 0
```

---

## Step 3: §4.2 — 高斯稀疏化强度选择 (3 场景 × 5 组 × 5 seed = 75 runs)

```bash
# 跑全 75 runs (30000 iter each)
python scripts/run_exp4_2_sweep.py

# 汇总 + 配对 t 检验
python scripts/summarize_exp4_234.py --sections 4_2

# 选定 prune_ratio2 后, 修改 scripts/run_exp4_3_collaboration.py 顶部:
#   PRUNE_RATIO2_DEFAULT = <选定值>
```

---

## Step 4: §4.3 — 感知损失 × 稀疏化协同 (3 场景 × 5 组 × 5 seed = 75 runs)

```bash
# 跑全 75 runs
python scripts/run_exp4_3_collaboration.py --prune-ratio2 0.04

# 汇总 + 配对 t 检验
python scripts/summarize_exp4_234.py --sections 4_3
```

---

## Step 5: §4.4 — 几何先验时机与强度 (配对设计, 4 场景 × 6 独立组 × 5 seed = 120 runs)

```bash
# 跑 120 runs
python scripts/run_exp4_4_geometry.py

# 汇总 + ANOVA + 配对 t
python scripts/summarize_exp4_234.py --sections 4_4
```

---

## Step 6: §4.5 — 多模块协同消融 (2^k + negative control, 4 场景 × 7 组 × 5 seed = 140 runs)

```bash
# 跑 140 runs
python scripts/run_exp4_5_ablation.py

# 汇总 + 论文表 4-X 复现
python scripts/summarize_exp4_5.py

# 跨场景汇总
python scripts/summarize_exp4_5_cross_scene.py
```

---

## Step 7: 可视化

```bash
# 误差棒散点 + 配对差异图 + 户外/室内箱线图
python scripts/plot_exp4.py
# 输出:
#   output/exp4_stats/exp4_5_ablation_scatter.png
#   output/exp4_stats/exp4_5_indoor_outdoor_boxplot.png
#   output/exp4_stats/exp4_4_paired_diff.png
```

---

## 增量 / 重跑 (任一阶段都可)

```bash
# 跳过已完成 (summary.json 已存在且 psnr 非空)
python scripts/run_exp4_5_ablation.py --skip-existing

# 单场景单组
python scripts/run_exp4_5_ablation.py --scenes bicycle --groups F --seeds 0
```

---

## 关键产物清单

| 路径 | 内容 |
|---|---|
| `output/exp4_{scene}/4_2_*/summary.json` | §4.2 每 (scene, group, seed) 的 PSNR/SSIM/LPIPS |
| `output/exp4_{scene}/4_3_*/summary.json` | §4.3 同上 |
| `output/exp4_{scene}/4_4_*/summary.json` | §4.4 同上 |
| `output/exp4_{scene}/4_5_*/summary.json` | §4.5 同上 |
| `output/exp4_5/ablation_matrix.csv` | 论文表 4-X 复现 (4 场景 × 7 组) |
| `output/exp4_5/indoor_vs_outdoor.csv` | 户外/室内对比 + 配对 t |
| `output/exp4_stats/exp4_*_paired_t.csv` | 各阶段配对 t 检验 |
| `output/exp4_stats/exp4_4_anova.csv` | §4.4 双因素 ANOVA |
| `output/exp4_stats/exp4_5_negative_control.csv` | 4.5.G vs 4.5.C 配对检验 (消融方向验证) |
| `output/exp4_stats/*.png` | 可视化 |

---

## 故障排查

- **LOSS 报 `unknown schedule staged`**: 检查 `experiments/loss_builder.py::_lpips_weight` 是否有 "staged" 分支
- **mono_depth 缺失**: 回到 Step 0 跑 `prepare_extra_scenes.py --stage all`
- **TRAIN NaN**: 大概率 depth_params 标定失败, 跑 `utils/make_depth_scale.py --source data/{scene}/sparse/0 --depths data/{scene}/depth_any_out --images data/{scene}/images_4 --out data/{scene}/sparse/0/depth_params.json --hold 8` 重新标定
- **render/metrics 报文件找不到**: 检查 `output/exp4_{scene}/4_X_*/train/seed{N}/test/ours_30000/` 下是否生成
- **scipy 缺失**: `pip install scipy matplotlib` (汇总和可视化脚本需要)
- **summarize 输出 "WARNING: scipy not available"**: 配对 t 检验将被跳过, 需先 `pip install scipy`
