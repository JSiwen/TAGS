# TAGSs v3 实施计划 (2026-10-17 修订版)

> **生成时间**: 2026-10-17
> **配套**: `docs/SECTION_4_6_FULL_COMPARISON.md` (§1.0 详细解释分辨率策略修订理由)
> **目标**: 让 4 组实验 (A/B/C/std) 在 2026-10-17 修订的分辨率策略下正确跑出可发表结果

---

## 0) 当前现状 (2026-10-17 13:00 之前)

| 项 | 状态 |
|---|---|
| **A.4 bicycle seed=0 已跑** | 14212 iter / 6h, 14k 检查点写入, 但 14k 之后进程死了 (OOM / 死机 / 主动停? 不确定) |
| **A.4 cfg 实际跑过的配置** | `--images images_2 --resolution -1 (默认)` → 训练像素 1600px (不是 1/2 物理 2473px!) |
| **A.4 schedule 行为** | `resolution_schedule = "2397:4,7989:2,30000:1"` 实际生效为 400/800/1600 px 三段 (1/4→1/2→1/0 of 1600, 不是 1/16→1/8→1/4 物理像素) |
| **B/C/std 未开始** | 全新 |
| **文档** | v3 §4.6 写"1/2 分辨率"但实际跑的是 1/3 物理像素, 文档与代码不一致 |

## 1) 2026-10-17 修订后的"标准配置"

所有实验**统一**用 `--images images_2`, **区分**用 `--resolution`:

| 维度 | 数值 |
|---|---|
| `--images` | `images_2` (1/2 物理源图, 2473×1643 for bicycle) |
| `--resolution` | `2` (A/B/C.2) / `8` (C.1) / `1` (C.3, std) |
| 主阶段训练像素 (bicycle) | `1236` (A/B/C.2) / `309` (C.1) / `2473` (C.3, std) |
| 主阶段物理像素位置 | 1/4 (A/B/C.2) / 1/16 (C.1) / 1/2 (C.3, std) |
| warmup schedule | `1200:4,4000:2,30000:1` (1/16→1/8→1/4 物理像素) |
| 高斯预算 | `--max_gaussians 2,500,000` (A/B/C/std 共享) |

---

## 2) 实施时间线 (2026-10-17 ~ 2026-11-15, 4 周)

### Phase 0: 衔接 (2026-10-17 当天, 0.5 天)

- [x] 改 `experiments/resolution_aware_schedule.py` 的 `_scale_resolution_schedule` 注释 (4/2/1 写法是正确语义)
- [x] 改 `scripts/exp4_runner.py` 加 `resolution` 参数, train_cmd 注入 `--resolution`
- [x] 改 `scripts/run_exp4v3_A_method_comparison.py` 加 `--resolution` 默认 2, `_common_lr_args` 加 `--images images_2 --resolution 2`
- [x] 改 `scripts/run_exp4v3_B_module_ablation.py` 同上
- [x] 改 `scripts/run_exp4v3_C_resolution_sensitivity.py`: GROUPS 改成 3 档 resolution (8/2/1), 统一 `--images images_2`
- [x] 改 `scripts/run_exp4v3_std_resolution_validation.py`: 加 `--images images_2 --resolution 1` (1/2 物理像素, = C.3 模式)
- [x] 改 `docs/SECTION_4_6_FULL_COMPARISON.md` 加 §1.0 解释 + 全文同步"1/2 分辨率"→"1/4 物理像素"术语
- [ ] **烟囱测试**: 1 个完整 A.4 bicycle seed=0 run 验证新配置真的切到 1/4 物理像素
  - 预期输出: `cfg_args.resolution=2`, `cfg_args.images=images_2`, `cfg_args.resolution_schedule=1200:4,4000:2,30000:1`
  - 前 1200 iter 训练像素 = 309×206 (1/16 物理)
  - 1200-4000 iter = 618×411 (1/8 物理)
  - 4000+ iter = 1236×822 (1/4 物理)

### Phase 1: Stage 1 + 2 衔接 (2026-10-17 ~ 10-20, 3 天)

- [ ] Stage 1: 跑 `prepare_extra_scenes.py` 准备 garden/bonsai/room 的 depth + normal + params
  - 1 场景 ~30 min, 3 场景 ~1.5h
- [ ] Stage 2: 跑 `qa_baseline_check.py` 在 1/4 物理像素下做 QA gate
  - 1 场景 1 seed ~6h, 4 场景 3 seed = 72h
  - **必须**用新 `--resolution 2` 配置 (qa_baseline_check.py 也要同步改)
  - 阈值需下调 (1/4 物理比 1/2 物理 PSNR 低 ~1 dB)

### Phase 2: Group A 主方法对比 (2026-10-21 ~ 11-04, 2 周)

- [ ] 启动 A.4 bicycle seed=0 (作为新 baseline, 旧 14k 的扔掉)
  - 预计 ~6.5h/run, 1 场景 1 seed
- [ ] 跑 A.4 全 12 runs (4 场景 × 3 seed): ~3 天
- [ ] 跑 A.1/A.2/A.3 全 36 runs: ~9 天
- [ ] 汇总 `exp4v3_A_method_comparison.csv` + `exp4v3_A_per_scene.csv`
- [ ] 写 `docs/exp4v3_A_method_comparison.md` (论文表 4-X)

### Phase 3: Group B 模块消融 (2026-11-04 ~ 11-11, 1 周)

- [ ] 跑 B.0 ~ B.4 全 60 runs: ~15 天? (调后估算)
- [ ] 实际: B 5 组共享大部分配置, 预计 ~13h/run × 60 = 32 天, **需要并行 / 跑更长**
  - **建议**: 先跑 1 场景 1 seed (5 runs ~1.5 天) 做可行性, 再决定是否并行
- [ ] 汇总 `exp4v3_B_module_ablation.csv`
- [ ] 写 `docs/exp4v3_B_module_ablation.md`

### Phase 4: Group C 分辨率敏感性 (2026-11-11 ~ 11-18, 1 周)

- [ ] 跑 C.1 12 runs (4 场景 × 3 seed): ~1.5 天 (1/16 物理像素极快)
- [ ] 跑 C.2 12 runs: ~3 天 (与 A/B 同)
- [ ] 跑 C.3 12 runs: ~6 天 (1/2 物理像素, 4× 算力, 预计 ~24h/run)
- [ ] 汇总 `exp4v3_C_resolution_sensitivity.csv`
- [ ] 写 `docs/exp4v3_C_resolution_sensitivity.md`

### Phase 5: 1/2 物理像素验证 (2026-11-18 ~ 11-25, 1 周)

- [ ] 跑 std 9 runs (3 场景 × 3 seed, 1/2 物理): ~9 天
  - 实际: 1/2 物理像素 1 run ~24h, 9 run = 216h = 9 天
  - 串行跑时间最长, 考虑 **3 场景并行 + 每场景 1 seed** (3 并行 × 3 seed = 9 单 run = 9 天)
- [ ] 汇总 `exp4v3_std_resolution_validation.csv`
- [ ] 写 `docs/exp4v3_std_resolution_validation.md`

### Phase 6: 文档与论文撰写 (2026-11-25 ~ 12-02, 1 周)

- [ ] 汇总 §4.6 总报告 (`SECTION_4_6_FULL_COMPARISON.md` 回填数据)
- [ ] 论文 §4.5/§4.6 图表生成
- [ ] 论文 §4.5/§4.6 文字撰写
- [ ] 所有 wandb run 公开链接, 准备 supplementary

**总时间**: ~6 周 (2026-10-17 ~ 2026-12-02)

---

## 3) 风险与回退

| 风险 | 概率 | 回退方案 |
|---|---|---|
| 1/4 物理像素 (1236 px) 训练 PSNR 比预期低 (>-1 dB 损失) | 中 | 改用 1/2 物理像素 (C.3) 为主, 1/4 物理 (A/B) 改为 C.2 端点 |
| 1/2 物理像素 (2473 px) 显存 OOM | 中 | 降 `--max_gaussians` 到 1.5M, 或只跑 1 seed |
| 烟囱测试发现新配置 bug | 低 | 退回 `--images images_4 + --resolution 1` 旧方案, 1/4 物理源图直接训练 (1237 px, 1/4 物理源图 = 1/4 物理像素) |
| 4 周时间不够 | 中 | 砍 B (1 周) 或 std (1 周), C 组只跑 2 场景 (36 → 18 runs) |

---

## 4) 关键决策记录 (2026-10-17)

| 决策 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| 主实验主阶段训练像素 (bicycle) | 1600 px (1/3 物理) | **1236 px (1/4 物理)** | 1/4 物理是 Mip-NeRF 360 论文 baseline, 算力更省, 质量几乎无损失 |
| A/B/C 源图目录 | 混用 images_2/4/ | **统一 images_2** | 避免 baseline dataset 不一致 |
| C 组分辨率控制 | `--images` 切换 | **`--resolution` 切换** | 旧方案被 `--resolution -1` 默认值压回 1600, 失去敏感性 |
| A.4 旧 14k 跑点 | 14212 iter (1600 px 训练) | **重跑** (1236 px 训练) | 旧数据与新配置不可比, 重跑是正确选择 |
| 1/2 物理像素实验 | 作为主实验 | **作为 C.3 / std 验证** | 1/2 物理 4 场景跑 30k iter 总时间 7-8 天, 改为 9 runs 验证 |
| 1/4 物理源图备选 | 旧 C.1 用 `images_4` | **改用 C.1 `--resolution 8` (= 1/16 物理)** | 旧方案等效 C.2 (1/4 物理), 没区分度; 1/16 物理才有区分度 |

---

## 5) 烟囱测试验证清单 (Phase 0 完成前必过)

```powershell
# 1. 语法检查
python -c "import ast; ast.parse(open('scripts/exp4_runner.py').read())"
python -c "import ast; ast.parse(open('scripts/run_exp4v3_A_method_comparison.py').read())"
python -c "import ast; ast.parse(open('scripts/run_exp4v3_B_module_ablation.py').read())"
python -c "import ast; ast.parse(open('scripts/run_exp4v3_C_resolution_sensitivity.py').read())"
python -c "import ast; ast.parse(open('scripts/run_exp4v3_std_resolution_validation.py').read())"

# 2. dry-run A.4 (1 scene, 1 seed, 1000 iter 看 warmup 像素)
$env:E4_SCENES = "bicycle"
$env:E4_GROUPS = "A4"
$env:E4_SEEDS = "0"
$env:E4_ITERATIONS = "1000"
python scripts/run_exp4v3_A_method_comparison.py --dry-run

# 3. 真实跑 1 个完整 run 验证 (预计 ~6.5h, 不要放在 phase 0 做)
# 建议先跑 3000 iter 验证 warmup 切到 1/4 物理像素 (1/2 物理)
$env:E4_ITERATIONS = "3000"
python scripts/run_exp4v3_A_method_comparison.py

# 4. 检查 cfg_args
Get-Content output\exp4v3_A_bicycle\A.A4_Full_TAGS\train\seed0\cfg_args | Select-String "resolution"
# 预期: resolution=2, images=images_2, resolution_schedule=1200:4,4000:2,30000:1
```

---

## 6) 长期 (2026-12+) 可选扩展

- [ ] **多 GPU 训练**: 单 30k iter 6.5h, 4 张 GPU 并行可压缩到 1.5 天
- [ ] **更细的 resolution 网格**: C 组改成 5 档 (1/32, 1/16, 1/8, 1/4, 1/2 物理) 完整曲线
- [ ] **其他数据集**: DTU, Tanks&Temples, ScanNet++ 验证泛化性
- [ ] **Ablation on resolution schedule**: 不切 / 1/4→1/2 / 1/16→1/8→1/4 三种切法对比

---

**生成时间**: 2026-10-17
**作者**: TAGS 实验自动规划 (基于 yori 决策)
**配套**: `docs/SECTION_4_6_FULL_COMPARISON.md` (§1.0 详细), `docs/EXP4_RUN_BOOK.md` (操作手册)
