# bicycle：E4 胜出配方 vs DN-Splatter

把室内胜出配方 `lpips_ms_w005` 原样迁到 Mip-NeRF 360 `bicycle`，并在同一套 COLMAP 上重训 DN-Splatter。DN 论文没有室外 Mip-NeRF 360 数字，vr_room 的 26.4 / 0.86 / 0.134 不能拿来填这张表。

这不是 bicycle 上的 v15/v16 再调参。v15/v16 只作内部诊断，不进 vs DN 主表。E4 的 2.5M cap 和 LPIPS 0.05 比本地 bicycle 配方更强，PSNR/SSIM 可能低于 v16。

## 锁定协议

| 项 | TAGS | DN-Splatter |
|----|------|-------------|
| 数据 | `data/bicycle` | 同一目录 |
| 图像 | `images_4` | `images_4`（改写 `transforms.json`） |
| 分辨率 | `--resolution 1`，1237×822 | `--downscale-factor 1`，1237×822 |
| 划分 | `--eval --llffhold 8`，按文件名排序后每 8 张取 1 张测试 | `eval_mode=interval`，`eval_interval=8`，natsort 后同样取余 |
| 迭代 | 30,000，先 seed=0 | 30,000 |
| 单目深度 | `depth_any_out` + `sparse/0/depth_params.json` | 由前者导出的 `mono_depth/*_aligned.npy`，不跑 Zoe |
| 法线 | 由深度梯度（TAGS geo prior） | 已有 `normals_from_pretrain/`，datamanager 会缩放到图像尺寸 |
| 评分 | 仓库 [metrics.py](../metrics.py) 的 VGG LPIPS | 渲染后再用同一个 `metrics.py` |

`images_4` 实测为 1237×822。两边都在这一档上训练。TAGS 的 `--resolution 1` 走整数倍率分支，宽高除以 1，得到原图像素。DN `--downscale-factor 1` 同样是 `images_4` 原尺寸。若要回到本地 3080 的宽边 800，设 `$env:BICYCLE_RESOLUTION = "800"`。

## 冻结的 TAGS 配方

与 [scripts/run_vr_room_e4_perceptual_ablation.ps1](../scripts/run_vr_room_e4_perceptual_ablation.ps1) 的 `lpips_ms_w005` 相同，只换数据：

| 参数 | 值 |
|------|-----|
| 感知损失 | 多尺度 LPIPS 0.05（尺度 1 / 0.5 / 0.25，权重 0.5 / 0.3 / 0.2），21k step |
| `max_gaussians` | 2,500,000 |
| `prune_opacity_threshold` | 0.02 |
| `densify_until_fraction` / `prune_only_from_fraction` | 0.50 |
| geo | depth+normal，`edge_aware_log_l1`，warmup 4k，decay 10k–20k |
| 关闭 | resolution-aware、SPA、late_stabilize、color/psnr/detail 损失 |

输出：`output/bicycle_e4_lpips_ms_w005/`。

## DN 设定

官方单目推荐，不改成 TAGS 的 log-L1：

- `--pipeline.model.depth-loss-type PearsonDepth`
- `--pipeline.model.use-depth-loss True`
- `--pipeline.model.depth-lambda 0.2`（这个仓库默认是 0，打开 depth loss 时会断言必须大于 0；0.2 是 DN README 的推荐权重）
- `--pipeline.model.use-normal-loss True`
- `--pipeline.model.use-normal-tv-loss True`
- `--pipeline.model.normal-supervision mono`
- `--pipeline.datamanager.dataparser.load-normals True`（否则 mono 分支没有 `batch["normal"]`）

`data/bicycle/transforms.json` 原来指向全分辨率 `images/`（4946×3286）。训练前脚本会备份为 `transforms.fullres.json`，再把路径改到 `images_4` 并按宽高比缩放焦距和主点。`mono_depth/*_aligned.npy` 现有文件是全分辨率；`--overwrite` 会换成 `images_4` 尺寸的度量深度。

深度导出与 `utils/make_depth_scale.py` 一致：uint8 逆深度除以 255，乘 `depth_params` 的 scale/offset，再取倒数得到 z。DN 的 datamanager 若分辨率仍不一致，会再 resize 一次。

## 已知对不齐

- 光栅器：Inria CUDA rasterizer 对 gsplat。
- 法线：TAGS 用法线来自深度梯度；DN 用 `normals_from_pretrain`（外部网络，加载时缩放到图像大小）。
- 位姿：DN 经 nerfstudio `transforms.json`，`orientation_method=none`、`auto_scale_poses=false`，仍不是 COLMAP 的逐字节拷贝。
- TAGS 运行时 `utils/camera_utils.py` 在深度图最大值 > 50 时把像素当成毫米传感器深度。bicycle 的 Depth Anything 是 uint8 逆深度，scale/offset 是按 `/255` 拟合的。导出给 DN 的图用 `/255` 这条路径。TAGS 训练若走进毫米分支，geo 监督和导出图不是同一个量。

## 测试流程

两个 conda 环境共用 `D:\TAGSs\TAGSs_core\data\bicycle` 和 `output\`。评分回到 `gaussian_splatting`，因为 LPIPS 用仓库 [metrics.py](../metrics.py)。DN 的训练和渲染在已安装 nerfstudio + `dn_splatter` 的环境里跑 [scripts/run_bicycle_dn_other_env.ps1](../scripts/run_bicycle_dn_other_env.ps1)。那个脚本可以留在本仓库，也可以单文件拷走；拷走时带上 `-Data` 和 `-OutputRoot`。

### 1. gaussian_splatting：TAGS 训练，并准备 DN 数据

```powershell
conda activate gaussian_splatting
cd D:\TAGSs\TAGSs_core

# 可选烟雾：TAGS 1k iter，LPIPS 从 700 启动
$env:E4_SMOKE = "1"
.\scripts\run_bicycle_e4_winner.ps1
Remove-Item Env:E4_SMOKE -ErrorAction SilentlyContinue

# 正式 30k
.\scripts\run_bicycle_e4_winner.ps1

# 导出 images_4 度量深度，并把 transforms.json 从 images/ 改到 images_4
.\scripts\run_bicycle_dn_splatter.ps1 -Stage export
.\scripts\run_bicycle_dn_splatter.ps1 -CheckOnly
```

`-CheckOnly` 确认：`mono_depth/*_aligned.npy` 与 `images_4` 一一对应，且 hold-8 测试张数与按文件名排序的 TAGS 划分相同。导出完成前不要进 DN 环境开训。

### 2. dn-splatter 环境：训练和渲染

```powershell
conda activate dn-splatter
cd D:\TAGSs\TAGSs_core

.\scripts\run_bicycle_dn_other_env.ps1 -CheckOnly

# 可选烟雾：1000 step，写到 bicycle_e4_vs_dn_smoke，不占用正式 seed0
.\scripts\run_bicycle_dn_other_env.ps1 -Smoke

# 正式 30k，然后渲染 eval，并导出 ply
.\scripts\run_bicycle_dn_other_env.ps1
```

脚本被拷到别处时：

```powershell
.\run_bicycle_dn_other_env.ps1 -CheckOnly `
    -Data D:\TAGSs\TAGSs_core\data\bicycle `
    -OutputRoot D:\TAGSs\TAGSs_core\output
.\run_bicycle_dn_other_env.ps1 `
    -Data D:\TAGSs\TAGSs_core\data\bicycle `
    -OutputRoot D:\TAGSs\TAGSs_core\output
```

`ns-train` 和 `python` 用当前环境 PATH 上的命令。需要指定解释器时设 `DN_PYTHON`、`DN_NS_TRAIN`、`DN_NS_EXPORT`。已有 `config.yml`、渲染图或 `splat.ply` 时默认跳过对应步骤；重跑加 `-Force`。

### 3. 回到 gaussian_splatting：同一套 metrics.py

```powershell
conda activate gaussian_splatting
cd D:\TAGSs\TAGSs_core
python scripts\summarize_bicycle_e4_vs_dn.py
```

DN 渲染要落在 `output/bicycle_dn_renders/eval/{pred,gt}/rgb/`。汇总脚本把它整理成 `output/bicycle_dn_splatter/test/dn-splatter/{renders,gt}/` 再打分。交接路径写在 `output/bicycle_dn_other_env_done.txt`。

## 输出

- `output/bicycle_e4_lpips_ms_w005/`
- `output/bicycle_dn_train/bicycle_e4_vs_dn/seed0/`
- `output/bicycle_dn_renders/eval/{pred,gt}/rgb/`
- `output/bicycle_dn_splatter/test/dn-splatter/{renders,gt}/`（给 `metrics.py`）
- `output/bicycle_e4_vs_dn_summary.json`
- `output/bicycle_e4_vs_dn_summary.csv`
