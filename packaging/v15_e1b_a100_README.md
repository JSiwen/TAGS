# TAGSs v15 E1b → A100 实验包

本压缩包是当前效果最好的 **v15 E1b densify7k** 代码快照，用于迁到 A100 做**全分辨率** bicycle 实验。

| 项 | 3080 参考（本包 `reference/3080_e1b/`） | A100 本包默认 |
|---|---|---|
| Case | `bicycle_v15_3080_e1b_densify7k` | `bicycle_v15_a100_e1b_densify7k` |
| 图像 | `images_4` | `images` |
| resolution | `800` | `-1`（全分辨率） |
| densify | 500 → **7200** | 相同 |
| 迭代 | 30k，`--eval --llffhold 8` | 相同 |
| 推荐 ckpt | `ours_28500` | 同样先看 28500 |
| 3080 指标 | PSNR **26.6930** / SSIM **0.828843** / LPIPS **0.149942** | **绝对值不可与 3080 对比** |

E1b 核心差异：把 densify 从默认 18k / E1 的 12k 提前停在 **7200**，配合 `late_stabilize` 晚期损失。

---

## 1. 解压

```bash
# 在 A100 上
mkdir -p /root/highspeedstorage/TAGS-A100/TAGSs
cd /root/highspeedstorage/TAGS-A100/TAGSs
tar -xzf TAGSs_v15_e1b_a100.tar.gz
cd TAGSs_v15_e1b_a100
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

## 4. 启动 E1b（推荐）

```bash
conda activate gaussian_splatting

# 若数据不在默认路径，先覆盖：
export SOURCE=/root/highspeedstorage/TAGS-A100/data/bicycle
export DEPTHS=/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out
export OUTPUT_ROOT=/root/highspeedstorage/TAGS-A100/TAGSs/output/v15_e1b_a100
export GPU_ID=0

source scripts/configs/bicycle_v15_e1b_a100.env
bash scripts/run_bicycle_v15_e1b_a100.sh
```

等价的一条命令：

```bash
SOURCE=/root/highspeedstorage/TAGS-A100/data/bicycle \
DEPTHS=/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out \
OUTPUT_ROOT=/root/highspeedstorage/TAGS-A100/TAGSs/output/v15_e1b_a100 \
bash scripts/run_bicycle_v15_e1b_a100.sh
```

只评测、不训练：

```bash
RUN_TRAIN=0 RUN_RENDER_METRICS=1 bash scripts/run_bicycle_v15_e1b_a100.sh
```

强制重训：

```bash
SKIP_COMPLETED=0 bash scripts/run_bicycle_v15_e1b_a100.sh
```

---

## 5. E1b 关键超参（与 3080 胜出配置对齐）

| 模块 | 值 |
|---|---|
| densify | from 500, until **7200** |
| geo prior | depth final 0.05, normal 0.02, warmup 5k, decay 12k–25k |
| SPA | 13k–25k, prune_ratio2=0.04, hybrid |
| 分辨率调度 | `1200:4,4000:2,30000:1` |
| LPIPS | start 24k, final 0.006, max 0.010, interval 10 |
| color align | start 25k, final 0.001, max 0.002 |
| PSNR loss | start 22k, final 0.008, max 0.012 |
| stage3 LR | geo×0.40, color×0.55 |
| 点增长门控 | `TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD=50000` |

完整 CLI 见 `scripts/run_bicycle_v15_30k_densify_fix.sh`；3080 当时的 `cfg_args` 在 `reference/3080_e1b/cfg_args`。

---

## 6. 产出与怎么看结果

默认输出：

```
$OUTPUT_ROOT/bicycle_v15_a100_e1b_densify7k/
  cfg_args
  results.json
  per_view.json
  point_cloud/iteration_28500/
  point_cloud/iteration_30000/
$OUTPUT_ROOT/bicycle_v15_a100_e1b_densify7k_stage_selection_summary.md
```

- **质量主指标**：`ours_28500`（3080 上 PSNR/SSIM 峰值）
- **感知主指标**：`ours_30000`（LPIPS 通常略好，PSNR 可能回落约 0.00x dB）
- 论文对比请对 **A100 全分辨率** 的 origin / v4_restore / v14，不要把 26.69 写进全分辨率表

---

## 7. 本包包含 / 不包含

**包含**

- 训练入口：`train_optimized.py`、`train.py`、`render.py`、`metrics.py`
- 核心代码：`arguments/`、`experiments/`、`gaussian_renderer/`、`scene/`、`utils/`、`lpipsPyTorch/`
- E1b / A100 启动脚本与 summarize 工具
- CUDA 扩展**源码**（需在 A100 上重新编译）
- 文档：`README.md`、`docs/`、`EXPERIMENT_SUMMARY.md`、`VERSION_HISTORY.md`、`DEPENDENCIES_README.md`
- 3080 E1b 参考指标（`reference/3080_e1b/`）

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
