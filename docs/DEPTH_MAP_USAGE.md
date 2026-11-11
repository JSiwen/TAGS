# TAGS 深度图使用说明

本文结合 TAGS 项目聊天记录、现有训练代码和预处理脚本，说明深度图的生成、对齐、加载与训练用途。聊天中曾有一段深度排查未完成；以下把已能从当前源码确认的行为与尚未验证的部分分开说明。

## 先区分三种深度数据

| 数据 | 常见路径 | 含义 | 在 TAGS 中的用途 |
|---|---|---|---|
| Depth Anything V2 原始预测 | `data/<scene>/depth_any_out/*.png` | 每张图单独归一化、量化为 8-bit 的相对深度/逆深度先验，不是米制真值 | COLMAP 场景的 `--depths` 输入；训练时转换并与渲染器深度比较 |
| 标定参数 | `data/<scene>/sparse/0/depth_params.json` | 每张图的 `scale`、`offset`，由单目预测和 COLMAP 稀疏点拟合 | 加载相对深度前做仿射对齐 |
| 对齐后的米制深度 | `data/<scene>/mono_depth/*_aligned.npy` | exporter 根据 `depth_params.json` 得到的 z-depth（米） | 主要用于 DN-Splatter/Coolermap 数据接口；TAGS 的 `--depths` 加载器不直接读取 `.npy` |

不要把彩色可视化拼图、原始预测、对齐后的米制深度和传感器深度互换使用。特别是 `mono_depth` 的 `.npy` 不能直接传给当前 TAGS `--depths` 参数。

## TAGS 训练中深度先验如何起作用

1. `-s/--source_path` 指向场景目录；`--depths` 是该目录下面的深度子目录名。COLMAP 相机图像名（不含扩展名）用于查找同名 PNG 和 `depth_params.json` 条目。
2. `scene/dataset_readers.py` 默认读取 `sparse/0/depth_params.json`，计算有效 `scale` 的中位数，并把逐视角参数附在相机信息上；也可用 `--depth_params_path` 指向独立校准文件，避免覆盖原数据。
3. `utils/camera_utils.py` 按 `--depth_input_encoding` 解码深度 PNG；默认 `auto` 会将 uint8 视为 `/255` 的单目逆深度，将 uint16 保留为兼容性自动判断。`scene/cameras.py` resize 到训练分辨率，应用 `depth = depth * scale + offset`，并用离群 `scale` 过滤不可靠视角。
4. 训练渲染包中的 `render_pkg["depth"]` 与相机先验比较。基础路径在 `train.py` 使用带指数权重日程的 L1 深度损失；实验路径在 `experiments/geo_prior.py` 使用可选的 edge-aware log-L1、Pearson 项、可靠性 mask 和独立权重日程。
5. 深度图是训练正则项，不是用于初始化 COLMAP 相机或替代 RGB 监督的稠密真值。法线先验可由深度图梯度计算，因此深度/法线先验通常共享同一张输入图。

基础 L1 默认权重从 `depth_l1_weight_init=1.0` 衰减到 `depth_l1_weight_final=0.01`。实验几何先验默认关闭；常见 E4 配置通过 `--enable_geo_prior --geo_prior_depth` 显式启用，常见深度末端权重为 `0.04` 或 `0.05`，并按实验配置设置 warmup/decay。

## 生成和准备单目深度

在仓库根目录运行，模型权重默认从 `Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth` 读取。`run.py` 每张输入图单独做 min-max 归一化为 8-bit，因此其数值不代表米；训练前必须经过对齐，且要保证输入/输出文件名与 COLMAP 图像名一致。

```powershell
cd D:\TAGSs\TAGSs_core\Depth-Anything-V2
python run.py `
  --img-path ..\data\bicycle\images_4 `
  --outdir ..\data\bicycle\depth_any_out `
  --encoder vitl --input-size 518 --pred-only --grayscale
```

`--pred-only --grayscale` 很重要：上游脚本默认保存 RGB 与彩色深度并排的可视化拼图，不能直接当作先验图输入。灰度模式会把灰度值复制到 3 个 PNG 通道。推理使用 GPU（若 CUDA 不可用会回退到 CPU），此处未执行推理验证。

## 标定、导出与训练命令

理论上的标准预处理顺序是：先有 undistorted COLMAP `sparse/0`，再生成与图像逐视角对应的单目 PNG，然后拟合 scale/offset，最后检查所有图像名与尺度参数。

```powershell
cd D:\TAGSs\TAGSs_core
python utils\make_depth_scale.py `
  --base_dir data\bicycle `
  --depths_dir data\bicycle\depth_any_out `
  --model_type bin `
  --output output\depth_audit\bicycle_depth_params_corrected.json
```

`--output` 可将校准结果写到独立文件；训练时用 `--depth_params_path` 读取，不必改写原始 `sparse/0/depth_params.json`。拟合时会把 COLMAP 采样点的 x/y 坐标整理成二维 map，再逐点取回所有逆深度样本。

`scripts/prepare_extra_scenes.py` 把多场景预处理拆为 depth、normal、params、export、check 阶段；`docs/EXP4_RUN_BOOK.md` 有 garden/bonsai/room 的场景化命令。`scripts/export_bicycle_mono_depth.py` 可将 bicycle 的预测导出为 `mono_depth/*_aligned.npy`，并可改写 `transforms.json`；这条导出主要是 DN-Splatter/Coolermap 适配，不等同于 TAGS 训练加载步骤。

此前排查发现的两个风险已经在当前代码中修正，但还要用目标环境复核：

- 旧 `camera_utils.py` 曾把任何最大像素值大于 50 的图都按毫米传感器深度解释。Bicycle 的 uint8 Depth Anything 图最大值为 255，约一半像素会因此落入毫米分支。当前 `auto` 按 dtype 将 uint8 解码为 `raw/255`；复现实验可显式传 `--depth_input_encoding monocular_inverse`。`legacy_threshold` 只用于复现旧问题，不用于正式训练。
- 旧 `make_depth_scale.py` 对 `cv2.remap` 结果使用 `[..., 0]`，可能只取一个采样值。当前版本保留所有映射采样点，并检查零/非有限离差；使用 `--output` 将重算标定写到新文件，再通过 `--depth_params_path` 加载。

这只证明代码路径已针对问题修正，不证明旧实验使用的远端数据、标定和代码版本完全一致，也不证明新的几何监督已经有效。重算出来的标定应先审计 scale/offset、目标深度值域和与 COLMAP 逆深度的对应关系；传感器深度与单目相对深度必须明确区分。

当输入格式和拟合代码已修复/验证后，基础训练接口形式如下（`--depths` 相对 `-s` 的目录；启用几何实验路径时加相应开关）：

```powershell
python train.py `
  -s data\bicycle `
  -m output\bicycle_geo_depth `
  --images images_4 --resolution 1 `
  --depths depth_any_out `
  --depth_input_encoding monocular_inverse `
  --enable_geo_prior --geo_prior_depth `
  --depth_prior_loss_type edge_aware_log_l1 `
  --depth_prior_weight_init 0.0 `
  --depth_prior_weight_final 0.04 `
  --iterations 30000
```

上面是参数形态示例，不是已通过训练验证的推荐实验配置。应复制对应实验 runner 的完整参数和数据划分，例如 `scripts/run_exp4_5_ablation.ps1`，不要只复用这段简化命令进行论文对比。

## 训练前检查清单

- `depth_any_out` 中每个训练视角有一张同名 PNG；内容是灰度预测图（可为 3 个通道重复灰度值），不是彩色拼图。
- 图像分辨率与深度图对应。若改用 `images_4` 等缩放图像，训练图、深度图和相机内参须保持同一图像坐标系。
- `depth_params.json` 的键与 COLMAP 图像 stem 完全一致；scale、offset 全部有限，正 scale 的分布没有明显离群值。
- 用少量视角抽查加载后深度形状、数值范围、mask 和训练日志里的 depth loss；确认单位和权重后再跑完整实验。
- 记录预测权重/编码、预处理分辨率、scale/offset 版本、训练分辨率、深度损失类型与 schedule。不同数据格式或标定版本的结果不能直接比较。
- Bicycle 的 `depth_any_out` 若为 uint8 Depth Anything 输出，训练命令显式指定 `--depth_input_encoding monocular_inverse`；不要使用仅供诊断的 `legacy_threshold`。

几何结果评价可使用 `scripts/evaluate_depth_per_view.py` 或 `scripts/geometry_debug.py`。前者是深度一致性比较；后者可做渲染深度导出、反投影和 SfM 稀疏参考评价。二者都不能把相对单目深度自动变成独立真值。

## 相关文件

- 生成：`Depth-Anything-V2/run.py`
- 单目与 COLMAP 标定：`utils/make_depth_scale.py`
- 数据集路径与相机加载：`scene/dataset_readers.py`、`utils/camera_utils.py`、`scene/cameras.py`
- 基础深度损失：`train.py`
- 实验深度/法线先验：`experiments/geo_prior.py`、`experiments/training_engine.py`
- 多场景准备：`scripts/prepare_extra_scenes.py`、`docs/EXP4_RUN_BOOK.md`
- bicycle 的米制深度导出：`scripts/export_bicycle_mono_depth.py`
- DN 对照实验的数据格式与单位注意事项：`docs/BICYCLE_E4_VS_DN.md`
