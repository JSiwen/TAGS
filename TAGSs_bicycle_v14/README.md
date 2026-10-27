# TAGSs bicycle v14 迁移包

本压缩包是 **bicycle v14 实验（`bicycle_v14_v8_monitor_28500_select`）** 的可迁移快照，
包含复现它所需的**脚本、完整训练代码、文档，以及 bicycle 对应的深度图与原图**。

| 项 | 值 |
|---|---|
| v14 最优 case | `bicycle_v14_v8_monitor_28500_select`（5 个候选中唯一 **gate pass**） |
| 指标（A100 原跑） | SSIM **0.774495** / PSNR **25.517925** / LPIPS **0.186443** |
| Gate reference | `bicycle_v8_color_geo_soft`（25.512505 / 0.774602 / 0.186343） |
| 协议 | `--iterations 30000`，`--eval --llffhold 8`，seed 0 |
| 输入分辨率 | `images_4`（1237×822） |
| 阶段 checkpoint | 7000 / 18000 / 24000 / 25500 / 28500 / 30000 |

v14 的核心结论是：`28500` 与 `30000` 必须**并列**作为候选，`28500` 才是 SSIM/PSNR 最优点。
详见 `output/bicycle_v14_experiment_architecture.md`（架构）与 `output/bicycle_v14_polish_checkpoint_summary.md`（结果）。

---

## 1. 解压

```bash
unzip TAGSs_bicycle_v14.zip -d /path/to/dest
cd /path/to/dest/TAGSs_bicycle_v14
```

Windows（PowerShell）：

```powershell
Expand-Archive -Path TAGSs_bicycle_v14.zip -DestinationPath D:\path\to\dest
```

---

## 2. 环境

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

CUDA 扩展需在目标机重新编译（本包只含源码）：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install opencv-python joblib numpy Pillow scipy tensorboard plyfile
```

自检：

```bash
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print('OK', torch.cuda.is_available())"
python -m unittest discover tests
```

建议导出 `export TAGGS_CPU_KNN_INIT=1`（避免 GPU KNN 初始化 OOM，需要 `scipy`）。

> 若目标机是 Windows 且已有编译好的 `.pyd`，本包**不含**二进制，请自行安装或复用现有环境。

---

## 3. 数据（已随包携带，无需重跑 Depth-Anything）

```
data/bicycle/
  images_4/          194 张原图，1237x822        <- 1237 协议输入
  depth_any_out/    194 张 uint8 逆深度 PNG     <- --depths 指向这里
  sparse/0/         COLMAP + depth_params.json  <- 不要重新生成 depth_params
  transforms.json / transforms.fullres.json
  poses_bounds.npy / sparse_pc.ply
  README.md         数据说明
```

`depth_params.json` 里的 `med_scale = 0.4677118023188449` 是原始标定值，
换机器后重新跑 `utils/make_depth_scale.py` 会改变结果，**必须原样保留**。

---

## 4. 运行 v14

### 4.1 Linux / A100：多卡 5 case 全量（原始 v14 论文脚本）

```bash
export SOURCE=/path/to/TAGSs_bicycle_v14/data/bicycle
export DEPTHS=$SOURCE/depth_any_out
export OUTPUT_ROOT=/path/to/output
export GPU_IDS_CSV=0,1,2,3

bash scripts/run_bicycle_v14_polish_checkpoint.sh
```

该脚本会跑满 v14 的 5 个候选 case（guard_recheck / polish_soft075 / polish_soft050 /
refine_end30000 / monitor_28500_select），并在结束后调用
`scripts/summarize_v4restore_ablation.py` 生成 `bicycle_v14_polish_checkpoint_summary.{json,md}`。

常用开关：

| 变量 | 作用 |
|---|---|
| `ONLY_CASE=bicycle_v14_v8_monitor_28500_select` | 只跑最优那个 case |
| `RUN_TRAIN=0 RUN_RENDER_METRICS=1` | 只评测已有 checkpoint |
| `SKIP_COMPLETED_CASES=0` | 强制重训 |
| `GPU_IDS_CSV=0` | 单卡串行 |
| `GATE_REFERENCE_CASE=...` | 换 gate 基准 |

### 4.2 Windows / 3080：只跑 monitor_28500_select（当前 1237 复现协议）

```powershell
$env:RERUN_CASE = "v14"
powershell -ExecutionPolicy Bypass -File scripts\run_bicycle_rerun_1237.ps1
```

或在脚本里改 `$IMAGES = "images_4"`、`$RESOLUTION = 1`、`$ITERATIONS = 30000`、`$SEED = 0`。
该驱动会自动：断点续训（读最新 `chkpnt*.pth`）→ 逐 stage 渲染 → `metrics.py` 出 `results.json` / `per_view.json`。

> 注意：`depth_params.json` 已随包，目标机只读不写。

---

## 5. 汇总与出图

```bash
# 5 case 排名 + gate
python scripts/summarize_v4restore_ablation.py \
  --output_root $OUTPUT_ROOT \
  --baseline_case bicycle_integrated_v4_restore \
  --gate_reference_case bicycle_v8_color_geo_soft \
  --cases bicycle_v14_v8_monitor_28500_select \
  --method ours_30000 \
  --stage_methods ours_7000 ours_18000 ours_24000 ours_25500 ours_28500 ours_30000 \
  --out_md v14_summary.md --out_json v14_summary.json

# stage TensorBoard 导出
python scripts/export_stage_metrics_to_tensorboard.py \
  --case_dir $OUTPUT_ROOT/bicycle_v14_v8_monitor_28500_select \
  --gate_reference_dir $OUTPUT_ROOT/bicycle_v8_color_geo_soft \
  --final_method ours_30000 \
  --stage_methods ours_7000 ours_18000 ours_24000 ours_25500 ours_28500 ours_30000

# 训练进度 / live log 汇总
python scripts/read_train_progress.py
python scripts/summarize_live_logs.py
```

---

## 6. 本包包含 / 不包含

**包含**

- 训练入口：`train.py`、`train_optimized.py`、`render.py`、`metrics.py`、`convert.py`、`full_eval.py`、`export_ply.py`
- 核心代码：`arguments/`、`experiments/`（geo prior / resolution schedule / sparse / balanced late controller 等）、
  `gaussian_renderer/`、`scene/`、`utils/`、`lpipsPyTorch/`、`tests/`
- v14 脚本：`run_bicycle_v14_polish_checkpoint.sh`、`run_bicycle_rerun_1237.ps1`、
  `_launch_bicycle_rerun_detached.ps1`、`summarize_v4restore_ablation.py`、
  `summarize_bicycle_rerun_1237.py`、`export_stage_metrics_to_tensorboard.py` 等
- CUDA 扩展**源码**（含 glm headers），需在目标机编译
- 数据：`data/bicycle/{images_4, depth_any_out, sparse/0}` + `depth_params.json` + transforms
- 文档：`output/bicycle_v14_experiment_architecture.md`、
  `output/bicycle_v14_experiment_evolution_architecture_analysis.md`、
  `output/bicycle_v14_polish_checkpoint_summary.{md,json}`、`EXPERIMENT_SUMMARY.md`、
  `VERSION_HISTORY.md`、`DEPENDENCIES_README.md`、`docs/THESIS_SYSTEM_DESIGN.md` 等
- 参考产物：`reference/local_1237_run/`（cfg_args、cameras.json、live_logs、训练日志）

**不包含**

- 任何 checkpoint / 高斯点云（`output/**/*.pth`、`*.ply` 模型、`input.ply`）
- `normals_from_pretrain/`（2.4 GB）、`mono_depth/`（752 MB）
  —— v14 的法线先验由深度现推（`experiments/geo_prior.py` 的 `depth_to_normals`），不需要外挂法线图
- `images/`、`images_2/`、`images_8/` 全分辨率图（只带 1237 协议用的 `images_4`）
- `Depth-Anything-2/3`、`third_party/`（预处理工具链，训练不依赖）
- `wandb/`、`__pycache__/`、`build/`、`.git/`、编译产物（`*.so` / `*.pyd`）

---

## 7. 常见问题

1. **`.sh` 脚本 `bad interpreter` / `$'\r'`**
   `sed -i 's/\r$//' scripts/*.sh`（本包已统一转成 LF，一般不会发生）

2. **CUDA 扩展编译失败**
   确认 `submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp` 存在；
   需要匹配目标机 torch / CUDA 版本，`pip install --no-build-isolation submodules/...`

3. **`depth_params.json` 缺失或被改**
   从本包 `data/bicycle/sparse/0/depth_params.json` 恢复，不要重跑 `make_depth_scale.py`

4. **显存不足（1237×822）**
   `export TAGGS_CPU_KNN_INIT=1`；仍 OOM 可临时 `--resolution 2` 冒烟，
   但正式表必须 `images_4` + `--resolution -1`（即原生 1237×822）

5. **LPIPS 权重下载失败**
   首次运行需联网拉 VGG；离线机请预缓存 torchvision / torch hub 权重

6. **想直接看 v14 结果而不重训**
   `reference/local_1237_run/` 里有 cfg_args 与 live_logs，
   指标见 `output/bicycle_v14_polish_checkpoint_summary.md`

---

## 8. 相关文档索引

| 文件 | 内容 |
|---|---|
| `output/bicycle_v14_experiment_architecture.md` | v14 实验架构（候选矩阵、阶段、决策规则） |
| `output/bicycle_v14_experiment_evolution_architecture_analysis.md` | v8→v14 演进分析 |
| `output/bicycle_v14_polish_checkpoint_summary.md` | 5 case 排名 + gate 明细 |
| `EXPERIMENT_SUMMARY.md` | 全项目实验总览 |
| `VERSION_HISTORY.md` | 版本谱系（v4 / v8 / v13 / v14 / v15 / v16） |
| `data/bicycle/README.md` | 数据目录说明 |
| `MANIFEST.txt` | 本包完整文件清单 |