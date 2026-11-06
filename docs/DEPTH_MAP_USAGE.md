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
2. `scene/dataset_readers.py` 读取 `sparse/0/depth_params.json`，计算有效 `scale` 的中位数，并把逐视角参数附在相机信息上。
3. `utils/camera_utils.py` 读取深度 PNG 并转换成逆深度；`scene/cameras.py` resize 到训练分辨率，应用 `depth = depth * scale + offset`，并用离群 `scale` 过滤不可靠视角。
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
  --model_type bin
```

`scripts/prepare_extra_scenes.py` 把多场景预处理拆为 depth、normal、params、export、check 阶段；`docs/EXP4_RUN_BOOK.md` 有 garden/bonsai/room 的场景化命令。`scripts/export_bicycle_mono_depth.py` 可将 bicycle 的预测导出为 `mono_depth/*_aligned.npy`，并可改写 `transforms.json`；这条导出主要是 DN-Splatter/Coolermap 适配，不等同于 TAGS 训练加载步骤。

**当前源码检查发现两个会影响“直接照抄命令开训”的问题，需先处理并重新验证：**

- `utils/camera_utils.py` 把任何最大像素值大于 50 的输入当作毫米传感器深度。Depth Anything V2 输出为 8-bit，通常也大于 50；但 `utils/make_depth_scale.py` 按 `/255` 归一化并据此拟合参数。训练加载和标定因此不在同一数值域，可能把单目预测错误地当毫米值。
- `utils/make_depth_scale.py` 对 `cv2.remap` 的采样结果再执行 `[..., 0]`。在当前传入一维采样坐标的写法下，OpenCV 通常返回一维向量，这个索引会只取首个采样值，使 MAD/scale 拟合失效或产生非有限参数。

以上是静态源码检查的结论，不代表已重跑脚本或证明既有实验输出全部无效。**在确认这两处处理方式前，不要把当前 8-bit `depth_any_out` + 新生成的 `depth_params.json` 作为可靠几何监督启动新训练。**传感器深度要与单目相对深度分开配置和标定；不能仅凭“像素值大于 50”猜测单位。

当输入格式和拟合代码已修复/验证后，基础训练接口形式如下（`--depths` 相对 `-s` 的目录；启用几何实验路径时加相应开关）：

```powershell
python train.py `
  -s data\bicycle `
  -m output\bicycle_geo_depth `
  --images images_4 --resolution 1 `
  --depths depth_any_out `
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
