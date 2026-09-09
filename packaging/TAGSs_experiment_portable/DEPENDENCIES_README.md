# TAGSs 依赖安装说明

本文档列出运行 TAGSs 核心训练与评估流程所需的全部依赖及推荐安装方式。

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04（推荐）或 Windows 10 |
| GPU | NVIDIA GPU，支持 CUDA 11.6 |
| 显存 | 建议 ≥ 24 GB（全分辨率训练 bicycle 场景） |
| Python | 3.7.13 |
| CUDA Toolkit | 11.6 |

## 方式一：Conda 一键安装（推荐）

项目根目录提供 `environment.yml`，包含 Conda 与 pip 依赖：

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

`environment.yml` 中已声明的依赖：

### Conda 包

| 包名 | 版本 | 用途 |
|------|------|------|
| `python` | 3.7.13 | 运行时 |
| `pytorch` | 1.12.1 | 深度学习框架 |
| `torchvision` | 0.13.1 | 图像变换与数据加载 |
| `torchaudio` | 0.12.1 | PyTorch 生态配套 |
| `cudatoolkit` | 11.6 | CUDA 运行时 |
| `plyfile` | — | PLY 点云读写 |
| `tqdm` | — | 训练进度条 |
| `pip` | 22.3.1 | pip 包管理 |

### pip 包（本地编译扩展）

| 包名 | 安装路径 | 用途 |
|------|----------|------|
| `diff-gaussian-rasterization` | `submodules/diff-gaussian-rasterization` | 可微高斯光栅化（核心渲染器） |
| `simple-knn` | `submodules/simple-knn` | KNN 距离计算，用于高斯初始化 |
| `fused-ssim` | `submodules/fused-ssim` | 加速 SSIM 损失（可选，无则回退 PyTorch 实现） |
| `opencv-python` | PyPI | 图像读写、深度图处理 |
| `joblib` | PyPI | 并行工具（评估脚本） |

若 Conda 创建环境后 pip 子模块未自动编译，请手动安装：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install opencv-python joblib
```

## 方式二：手动 pip 安装

```bash
# 1. 安装 PyTorch（需匹配 CUDA 11.6）
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# 2. 安装 Python 依赖
pip install plyfile tqdm opencv-python joblib numpy Pillow

# 3. 编译安装 CUDA 子模块（必须在项目根目录执行）
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

## 隐式 / 代码中使用的额外依赖

以下包未写入 `environment.yml`，但运行时代码会用到：

| 包名 | 是否必需 | 用途 | 安装命令 |
|------|----------|------|----------|
| `numpy` | **必需** | 数值计算（scene、utils、scripts） | `pip install numpy` |
| `Pillow` | **必需** | 图像加载（dataset_readers、metrics） | `pip install Pillow` |
| `scipy` | 条件必需 | `TAGGS_CPU_KNN_INIT=1` 时用于 CPU KNN 初始化 | `pip install scipy` |
| `tensorboard` | 可选 | 训练日志（`train_optimized.py`、`train.py`） | `pip install tensorboard` |

> **说明**：`lpipsPyTorch` 已内置于项目 `lpipsPyTorch/` 目录，无需额外安装 LPIPS 包。

## Git 子模块

以下 CUDA 扩展以 Git 子模块形式维护（已包含在 `submodules/` 中）：

| 子模块 | 来源 |
|--------|------|
| `submodules/diff-gaussian-rasterization` | [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-splatting)（分支 `dr_aa`） |
| `submodules/simple-knn` | [bkerbl/simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) |
| `submodules/fused-ssim` | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) |

若从 Git 克隆（非本压缩包），需使用：

```bash
git clone --recursive <repo_url>
```

## 数据准备额外工具（非训练运行时依赖）

生成深度先验时需单独安装 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)，该工具不在 TAGSs 核心依赖中。

深度尺度标定使用项目内置脚本：

```bash
python utils/make_depth_scale.py
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `TAGGS_CPU_KNN_INIT=1` | 使用 CPU scipy KNN 初始化，避免 GPU KNN OOM（需安装 scipy） |

## 验证安装

```bash
conda activate gaussian_splatting
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print('OK')"
```

可选验证 fused-ssim：

```bash
python -c "import fused_ssim; print('fused-ssim OK')"
```

## 主要入口脚本

| 脚本 | 功能 |
|------|------|
| `train_optimized.py` | TAGSs 集成优化训练（主入口） |
| `train.py` | 原始 3DGS 训练 |
| `render.py` | 渲染测试/训练视角 |
| `metrics.py` | 计算 PSNR / SSIM / LPIPS |
| `convert.py` | COLMAP 数据预处理 |
| `export_ply.py` | 导出高斯 PLY |
| `scripts/run_bicycle_v4_restore.sh` | 当前最佳配置训练脚本 |

## 硬件与编译注意事项

1. **CUDA 版本必须匹配**：PyTorch 1.12.1 + cudatoolkit 11.6，子模块编译时自动检测本机 CUDA。
2. **编译失败**：确认已安装 `nvcc` 及对应 CUDA toolkit，且 `CUDA_HOME` 指向正确路径。
3. **显存不足**：设置 `TAGGS_CPU_KNN_INIT=1`，或在脚本中使用 `--data_device cpu`。
