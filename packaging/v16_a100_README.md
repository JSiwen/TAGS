# TAGSs v16 → A100 实验包

本压缩包是当前 **3080 最优** 的 v16 resolution-aware 代码快照，用于迁到 A100 做 **bicycle 全分辨率** 实验。

| 项 | 3080 本包参考（`reference/3080_v16/`） | A100 本包默认 |
|---|---|---|
| Case | `bicycle_v16_3080_resolution_aware` | `bicycle_v16_a100_resolution_aware` |
| 图像 | `images_4` | `images` |
| resolution | `800` | `-1`（全分辨率） |
| 调度 | RAS planner + adaptive controller | **同一套代码**，按有效像素自动拉长 densify / SPA / late loss |
| 迭代 | 30k，`--eval --llffhold 8` | 相同 |
| 推荐 ckpt | `ours_28500` | 同样先看 28500，并核对 `schedule_plan.json` |
| 3080 指标 | PSNR **26.9117** / SSIM **0.837794** / LPIPS **0.135433** | **绝对值不可与 3080 对比** |

相对上一版 3080 最佳（v15 E1b densify7k `ours_28500`：26.6930 / 0.828843 / 0.149942）：

- PSNR **+0.219 dB**
- SSIM **+0.0090**
- LPIPS **−0.0145**

v16 与 v15 E1b 的关键差异：**不再写死 `densify_until=7200`**。训练开始后由 Resolution-Aware Schedule Planner 按输入分辨率生成阶段边界；运行时 Adaptive Controller 可在容量不足时把 densify 再往后延（上限 18000）。

A100 全分辨率下 `resolution_scale` 会接近上限 2.5，planner 预计把 densify 拉到 **约 18000**（不再是 7200）。这正是上次 E1b 全分辨率容量不足要验证的点。

---

## 1. 解压

```bash
# 在 A100 上
mkdir -p /root/highspeedstorage/TAGS-A100/TAGSs
cd /root/highspeedstorage/TAGS-A100/TAGSs
tar -xzf TAGSs_v16_a100.tar.gz
cd TAGSs_v16_a100
```

解压后所有命令默认在**包根目录**执行。

---

## 2. 环境（若机器上已有 `gaussian_splatting` 可跳过）

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

子模块未编译成功时：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install opencv-python joblib numpy Pillow scipy tensorboard
```

自检：

```bash
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print('OK', torch.cuda.is_available())"
python -m unittest tests.test_resolution_aware_schedule
```

建议：`export TAGGS_CPU_KNN_INIT=1`（避免 GPU KNN 初始化 OOM，需 `scipy`）。

---

## 3. 准备 bicycle 数据

**本包不含任何图像**（原图、深度 PNG 均未打包）。请直接使用 A100 上已有数据：

```
/root/highspeedstorage/TAGS-A100/data/bicycle/
  images/
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
  sparse/0/depth_params.json
  depth_any_out/
```

用 `SOURCE` / `DEPTHS` 指向该目录即可，不必拷进本包。

---

## 4. 启动 v16（推荐：planner + adaptive）

```bash
conda activate gaussian_splatting

# 若数据不在默认路径，先覆盖：
export SOURCE=/root/highspeedstorage/TAGS-A100/data/bicycle
export DEPTHS=/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out
export OUTPUT_ROOT=/root/highspeedstorage/TAGS-A100/TAGSs/output/v16_resolution_aware_a100
export GPU_ID=0

source scripts/configs/bicycle_v16_resolution_aware.env
bash scripts/run_bicycle_v16_resolution_aware.sh
```

等价的一条命令：

```bash
SOURCE=/root/highspeedstorage/TAGS-A100/data/bicycle \
DEPTHS=/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out \
OUTPUT_ROOT=/root/highspeedstorage/TAGS-A100/TAGSs/output/v16_resolution_aware_a100 \
bash scripts/run_bicycle_v16_resolution_aware.sh
```

只评测、不训练：

```bash
RUN_TRAIN=0 RUN_RENDER_METRICS=1 bash scripts/run_bicycle_v16_resolution_aware.sh
```

强制重训：

```bash
SKIP_COMPLETED=0 bash scripts/run_bicycle_v16_resolution_aware.sh
```

Planner-only 消融（关掉运行时 densify 延长）：

```bash
VARIANT=planner_only bash scripts/run_bicycle_v16_resolution_aware.sh
```

---

## 5. 关键超参

损失权重与 v15 E1b / late_stabilize **对齐**；阶段边界由 RAS 生成，不再手写 7200/13k/24k。

| 模块 | 3080 实测（scale≈1.0） | A100 全分辨率预期（scale 接近 2.5） |
|---|---|---|
| densify_until（planner） | 7200 | ≈ **18000**（clamp 上限） |
| adaptive 延长 | 7200 → **9200**（capacity_extension） | 仅当点数未到 `ADAPTIVE_POINT_BUDGET=8M` 才会再延，上限仍 18000 |
| SPA | planner 9200–17200；runtime 改到 11200–19200 | densify 结束后 +2000 起，窗口约 8k |
| LPIPS | start 22500, final 0.006, max 0.010 | 约 0.75 × 30k，且不早于 SPA 末段 |
| color align | start 17200, final 0.001 | ≈ SPA end |
| PSNR loss | start 21000, final 0.008 | 约 0.70 × 30k |
| stage3 LR | geo×0.40, color×0.55 | 相同 |
| 点增长门控 | `TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD=50000` | 相同 |
| 分辨率课程 | `1200:4,4000:2,30000:1` | 边界按 scale 拉伸 |

训练一开始就会写 `schedule_plan.json`。先确认 `resolution_scale`、`densify_until_iter`、`spa_*` 再离开现场。

完整 CLI 见 `scripts/run_bicycle_v16_resolution_aware.sh`。3080 当时的 `cfg_args` / `schedule_plan.json` 在 `reference/3080_v16/`。

---

## 6. 产出与怎么看结果

默认输出：

```
$OUTPUT_ROOT/bicycle_v16_a100_resolution_aware/
  cfg_args
  schedule_plan.json
  results.json
  per_view.json
  point_cloud/iteration_28500/
  point_cloud/iteration_30000/
$OUTPUT_ROOT/bicycle_v16_a100_resolution_aware_stage_selection_summary.md
```

- **质量主指标**：先看 `ours_28500`（3080 上 PSNR/SSIM/LPIPS 三点同时最优）
- **对照**：A100 全分辨率的 v14 monitor / v4_restore，**不要**和 3080 的 26.91 混排
- 成功标准（相对上次失败的 v15 A100 E1b 25.340 / 0.760 / 0.217）：
  - 至少追上 v14 monitor `ours_28500`（25.548 / 0.775 / 0.187）
  - `schedule_plan.json` 里 densify 明显长于 7200
  - 28500 高斯数不应再只有 v14 的一半（v15 A100 当时约 4.39M vs v14 约 8.53M）

---

## 7. 本包包含 / 不包含

**包含**

- 训练入口：`train_optimized.py`、`train.py`、`render.py`、`metrics.py`
- 核心代码：`arguments/`、`experiments/`（含 RAS planner + adaptive controller）、`gaussian_renderer/`、`scene/`、`utils/`、`lpipsPyTorch/`
- v16 启动脚本、summarize 工具、RAS 单测
- CUDA 扩展**源码**（需在 A100 上重新编译）
- 文档：`README.md`、`docs/`、`EXPERIMENT_SUMMARY.md`、`VERSION_HISTORY.md`、`DEPENDENCIES_README.md`
- 3080 v16 参考指标（`reference/3080_v16/`）
- 上一版 3080 最佳 v15 E1b 指标（`reference/3080_e1b/`），便于对照

**不包含**

- 任何图像（原图 / 深度 PNG / 子模块示例图）
- `data/bicycle/` 场景数据与 COLMAP
- `output/` checkpoint / ply
- `Depth-Anything-*`、`third_party/`（训练不依赖）
- 本地 `build/`、`__pycache__`

---

## 8. 常见问题

1. **脚本 `bad interpreter` / `$'\r'`**  
   `sed -i 's/\r$//' scripts/*.sh scripts/configs/*.env`

2. **显存不足**  
   保持 `TAGGS_CPU_KNN_INIT=1`；仍 OOM 时可临时 `IMAGES=images_4 RESOLUTION=1600` 做冒烟，正式表仍应用 `images` + `-1`。

3. **缺少 `depth_params.json`**  
   `python utils/make_depth_scale.py --base_dir data/bicycle --depths_dir data/bicycle/depth_any_out`

4. **LPIPS 权重下载失败**  
   首次运行需要联网拉取 VGG；离线机器需预先缓存 torchvision / torch hub 权重。

5. **densify 仍停在 7200**  
   检查 `schedule_plan.json` 的 `resolution_scale` 是否接近 1.0。若 `IMAGES` 误指到 `images_4` 或 `RESOLUTION=800`，planner 会按 3080 协议处理。全分辨率必须是 `images` + `-1`。
