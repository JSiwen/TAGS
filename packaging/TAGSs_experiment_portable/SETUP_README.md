# TAGSs 实验环境移植指南

本压缩包包含在新机器上复现 TAGSs 训练 / 渲染 / 评估所需的**全部代码与启动脚本**（不含场景数据与训练输出）。

解压后目录结构应类似：

```
TAGSs_core/
├── SETUP_README.md          ← 本文件（移植说明）
├── DEPENDENCIES_README.md   ← 依赖细节
├── environment.yml
├── train_optimized.py       ← 主训练入口
├── train.py / render.py / metrics.py / ...
├── arguments/  experiments/  gaussian_renderer/  scene/  utils/
├── lpipsPyTorch/
├── scripts/                 ← 实验启动与辅助脚本
└── submodules/              ← CUDA 扩展源码（需本地编译）
```

---

## 1. 硬件与系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04（推荐）或 Windows 10/11 |
| GPU | NVIDIA，支持 CUDA 11.6 |
| 显存 | 全分辨率 bicycle 建议 ≥ 24 GB；3080 等可用降分辨率脚本 |
| Python | 3.7.13（由 conda 环境固定） |
| CUDA Toolkit | 11.6（与 PyTorch 1.12.1 匹配） |
| 编译器 | Linux: `nvcc`；Windows: Visual Studio Build Tools + CUDA |

---

## 2. 解压与目录放置

1. 将压缩包解压到目标机器任意路径，例如：
   - Linux: `/home/user/TAGSs_core`
   - Windows: `D:\TAGSs\TAGSs_core`
2. **不要**把 `output/` 或大型 `data/` 放进本代码包；数据需单独准备（见第 4 节）。
3. 之后所有命令默认在**项目根目录**执行。

---

## 3. 创建 Conda 环境

### 3.1 一键创建（推荐）

```bash
cd /path/to/TAGSs_core
conda env create --file environment.yml
conda activate gaussian_splatting
```

若子模块未自动编译成功，在项目根目录手动安装：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install opencv-python joblib numpy Pillow scipy tensorboard
```

### 3.2 Windows 额外步骤

1. 安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（含 C++ 桌面开发）与 CUDA 11.6。
2. 可用项目内脚本编译子模块：

```powershell
conda activate gaussian_splatting
.\scripts\install_cuda_submodules.bat
```

3. 若 Pillow DLL 异常：

```powershell
.\scripts\fix_gaussian_splatting_pillow.ps1
```

4. 环境自检：

```powershell
.\scripts\test_train_environment.ps1
```

### 3.3 验证安装

```bash
conda activate gaussian_splatting
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print('OK', torch.cuda.is_available())"
```

可选：

```bash
python -c "import fused_ssim; print('fused-ssim OK')"
```

---

## 4. 准备实验数据

### 4.1 本包已包含的深度文件

解压后即可使用，**无需再跑深度估计**：

```
data/bicycle/
├── depth_any_out/                 # 194 张单目逆深度 PNG（已打包）
└── sparse/0/depth_params.json     # 深度尺度标定（已打包）
```

### 4.2 仍需自行准备的内容

标准 bicycle 实验还需要原图与 COLMAP（本包为控制体积未打包）：

```
data/bicycle/
├── images/                        # 原图（全分辨率）
├── images_4/ 或 images_8/         # 可选，3080 低分辨率脚本使用
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    ├── points3D.ply               # 可选
    └── depth_params.json          # 已随包提供，勿覆盖丢失
```

将 [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) bicycle 的 `images/` 与 `sparse/0/` 拷贝进 `data/bicycle/`，与本包深度文件合并即可。

> **注意**：启用 `--depths` / geo prior 时必须保留 `depth_params.json`。若误删，可在本机有完整数据时重跑：
> `python utils/make_depth_scale.py --base_dir data/bicycle --depths_dir data/bicycle/depth_any_out`

---

## 5. 运行实验

### 5.1 推荐基线：perceptual-front / geo-late（v4_restore）

**Linux**（务必覆盖脚本里的默认路径）：

```bash
conda activate gaussian_splatting
export SOURCE=/path/to/data/bicycle
export DEPTHS=/path/to/data/bicycle/depth_any_out
export MODEL=/path/to/output/bicycle_integrated_v4_restore
export TAGGS_CPU_KNN_INIT=1
bash scripts/run_bicycle_v4_restore.sh
```

**Windows**：

```powershell
conda activate gaussian_splatting
$env:SOURCE = "D:\TAGSs\TAGSs_core\data\bicycle"
$env:DEPTHS = "D:\TAGSs\TAGSs_core\data\bicycle\depth_any_out"
$env:TAGGS_MODEL = "D:\TAGSs\TAGSs_core\output\bicycle_integrated_perceptual_front_geo_late"
.\scripts\run_bicycle_v4_restore.ps1
```

跳过训练后的 render/metrics：

```powershell
$env:RUN_RENDER_METRICS = "0"
.\scripts\run_bicycle_v4_restore.ps1
```

### 5.2 v15 等后续配置

| 脚本 | 场景 |
|------|------|
| `scripts/run_bicycle_v15_e1b_a100.sh` | Linux A100，v15 E1b 主配方 |
| `scripts/run_bicycle_v15_30k_densify_fix.sh` / `.ps1` | 3080 / Linux densify-fix（E1 / E1b） |
| `scripts/run_bicycle_v16_resolution_aware.sh` | v16 resolution-aware 主线 |
| `scripts/configs/bicycle_v15_*.env*` / `bicycle_v16_*.env` | 环境变量配置 |

Linux 示例：

```bash
source scripts/configs/bicycle_v15_e1b_a100.env   # 若存在
export SOURCE=... DEPTHS=... OUTPUT_ROOT=...
bash scripts/run_bicycle_v15_e1b_a100.sh
```

### 5.3 手动调用主入口

```bash
python train_optimized.py \
  -s data/bicycle \
  -m output/my_run \
  --depths data/bicycle/depth_any_out \
  --eval --llffhold 8 \
  --iterations 30000 \
  # ... 其余参数见 scripts/run_bicycle_*.sh|ps1
```

训练完成后评估：

```bash
python render.py -m output/my_run --skip_train --quiet
python metrics.py -m output/my_run --split test
```

---

## 6. 常用环境变量

| 变量 | 说明 |
|------|------|
| `SOURCE` | 场景根目录（含 `images/`、`sparse/`） |
| `DEPTHS` | 深度 PNG 目录 |
| `MODEL` / `TAGGS_MODEL` | 输出目录（Windows 优先用 `TAGGS_MODEL`） |
| `OUTPUT_ROOT` | v15 多 case 输出根目录 |
| `ITERATIONS` | 训练迭代数，默认多为 30000 |
| `TAGGS_CPU_KNN_INIT=1` | CPU scipy KNN 初始化，避免 GPU KNN OOM（需 scipy） |
| `RUN_RENDER_METRICS=0` | 跳过训练后自动 render/metrics |
| `CUDA_VISIBLE_DEVICES` / `GPU_ID` | 指定 GPU |
| `CONDA_ENV` | conda 环境名，默认 `gaussian_splatting` |

> Linux 启动脚本内常硬编码 `/root/highspeedstorage/...`，换机后**必须**用环境变量覆盖。

---

## 7. 本压缩包包含 / 不包含

### 包含

- 训练与评估入口：`train_optimized.py`、`train.py`、`render.py`、`metrics.py` 等
- 核心库：`arguments/`、`experiments/`、`gaussian_renderer/`、`scene/`、`utils/`、`lpipsPyTorch/`
- 启动与分析脚本：`scripts/`（含 `configs/`）
- CUDA 扩展**源码**：`submodules/`（不含本地 `build/`、`.egg-info`）
- 环境与文档：`environment.yml`、`DEPENDENCIES_README.md`、实验说明文档
- bicycle 深度先验：`data/bicycle/depth_any_out/` + `sparse/0/depth_params.json`

### 不包含（需另行准备）

| 内容 | 原因 |
|------|------|
| `data/bicycle/images*` 与 COLMAP bin | 体积大，需按第 4 节自行准备；**深度图已打包** |
| `output/` | 训练结果 / checkpoint，体积极大 |
| `third_party/` | 仅参考实现，训练不依赖 |
| `Depth-Anything-*` | 仅用于生成深度先验 |
| `__pycache__/`、编译产物 | 换机后需重新编译 |

---

## 8. 常见问题

1. **子模块编译失败**  
   检查 `nvcc`、`CUDA_HOME` 是否指向 CUDA 11.6，且与 `pytorch==1.12.1+cu116` 一致。

2. **首次运行下载权重**  
   LPIPS / ImageNet 预训练权重会通过网络下载；离线机器需事先缓存到对应 torch hub / torchvision 目录。

3. **显存不足**  
   设置 `TAGGS_CPU_KNN_INIT=1`；或使用 `images_4` / 降低 `--resolution` 的 3080 脚本。

4. **缺少 depth_params.json**  
   运行 `utils/make_depth_scale.py` 生成后再开 geo prior。

5. **Windows Smart App Control**  
   可能拦截 `fbgemm.dll`；按系统策略放行或换用兼容的 PyTorch 构建。

---

## 9. 快速检查清单

- [ ] 解压代码包到目标路径  
- [ ] `conda env create -f environment.yml` 并激活  
- [ ] CUDA 子模块 `import` 成功  
- [ ] 深度目录已随包解压（`depth_any_out` + `depth_params.json`）
- [ ] 自行放入 `images/` 与 COLMAP `sparse/0/*.bin`  
- [ ] 用环境变量覆盖 `SOURCE` / `DEPTHS` / `MODEL`（或 `TAGGS_MODEL`）  
- [ ] 运行 `scripts/run_bicycle_v4_restore.sh` 或 `.ps1`  

更多依赖细节见同目录 `DEPENDENCIES_README.md`；版本与实验设计见 `EXPERIMENT_SUMMARY.md`、`VERSION_HISTORY.md`。

