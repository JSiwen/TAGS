# TAGS Bicycle v16：AutoDL A800 复现、消融与复核方案

**版本：** 1.2　 **日期：** 2026-11-06　 **主环境：** AutoDL Linux / NVIDIA A800　 **数据：** Bicycle `images_4`

本文把历史配方复现、可比较基线、法线来源消融和独立验证分开。命令均使用 Bash，假设已进入 AutoDL 上的 TAGSs 仓库 checkout，并使用项目现有 `gaussian_splatting` Conda 环境。数据路径按实际挂载位置设置。为了与原始 3DGS Bicycle 的 `-r 4` 协议一致，所有 Bicycle 组都读取同一 Mip-NeRF 360 `images_4` 目录，并设置 TAGS `--resolution 1`，即不对已下采样的 `images_4` 再缩放；目标渲染尺寸应与原始 3DGS 的 `images_4` GT 逐张一致（该数据通常约为 1237×822，运行时以实际图像尺寸为准）。固定 30k、`--eval --llffhold 8`。**不采用 v16 A100 脚本默认的 `images` / `resolution=-1`，也不再用 `resolution=800`**，因为两者都不是本次选定的 `images_4` 原生尺寸协议。

所有组从头训练，使用新输出目录，不覆盖旧结果，也不从未经验证的 checkpoint 恢复。A800 是本方案内的统一对比硬件；RTX 3080 历史结果只作历史参考，不能与 A800 数字混成同硬件对比。

## 1. 实验问题与证据边界

1. v16 的 `images_4 @ resolution=1` 原生尺寸配置能否迁移到 AutoDL A800 并完整运行？
2. 其余设置相同时，深度先验、当前深度导出法线和外部法线分别带来什么图像质量与几何变化？
3. 冻结候选后，这些变化能否在未参与筛选的数据/场景上复现？

目前 Bicycle R0/G2 三种子汇总报告的是小幅、同方向差异：配对均值 ΔPSNR `+0.014766 dB`、ΔSSIM `+0.0001139`、ΔLPIPS `−0.00003718`。其 PSNR 均值仍低于 `25.50 dB` 门槛，且官方 test split 已参与候选筛选。因此仅作探索性参考，不能作为独立测试结论。详见 [`BICYCLE_R0_G2_THREE_SEED_SUMMARY.md`](BICYCLE_R0_G2_THREE_SEED_SUMMARY.md)。

`evaluate_depth_per_view.py` 测量的是渲染深度与输入单目深度先验的一致性，不是绝对几何精度。若论文主张几何准确度，必须有独立真值（例如扫描深度/网格）及相应误差指标。不同 GPU、图像尺寸、训练迭代数、划分、指标实现或预算协议的历史结果不得合并排名。

## 2. 阶段 0：AutoDL 环境、代码与数据审计（不训练）

在 AutoDL 登录节点或训练容器中进入**实际 checkout 根目录**。下面路径沿用历史 AutoDL 常见目录；若实例挂载路径不同，只修改变量，不要改协议参数：

```bash
cd /root/autodl-tmp/TAGS_github_complete_20261105
export ROOT="$(git rev-parse --show-toplevel)"
export DATA_ROOT="/root/autodl-tmp/TAGS/data"
export SOURCE="$DATA_ROOT/bicycle"
export DEPTH_DIR="$SOURCE/depth_any_out"
export OUTROOT="/root/autodl-tmp/TAGS/output/v16_a800_images4_native_20261106"
export AUDIT="$OUTROOT/audit"
mkdir -p "$AUDIT" "$OUTROOT/logs"
git rev-parse HEAD | tee "$AUDIT/git_commit.txt"
git status --short | tee "$AUDIT/git_status.txt"
nvidia-smi | tee "$AUDIT/nvidia_smi.txt"
python - <<'PY' | tee "$AUDIT/torch_cuda.txt"
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
python train_optimized.py --help > "$AUDIT/train_help.txt" 2>&1
python render.py --help > "$AUDIT/render_help.txt" 2>&1
python metrics.py --help > "$AUDIT/metrics_help.txt" 2>&1
```

确认关键文件存在并记录数据概况：

```bash
test -f "$SOURCE/sparse/0/cameras.bin" && echo cameras.bin:OK
test -f "$SOURCE/sparse/0/images.bin" && echo images.bin:OK
test -f "$SOURCE/sparse/0/depth_params.json" && echo depth_params.json:OK
test -d "$SOURCE/images_4" && echo images_4:OK
test -d "$DEPTH_DIR" && echo depth_any_out:OK
find "$SOURCE/images_4" -maxdepth 1 -type f | wc -l | tee "$AUDIT/image_count.txt"
find "$DEPTH_DIR" -maxdepth 1 -type f -name '*.png' | wc -l | tee "$AUDIT/depth_count.txt"
```

检查 PNG 通道、尺寸和值域，以及标定覆盖/有限性：

```bash
ROOT="$SOURCE" python - <<'PY' | tee "$AUDIT/depth_input_audit.txt"
import json, math, os
from pathlib import Path
import cv2
import numpy as np
root = Path(os.environ['ROOT'])
images, depths = root/'images_4', root/'depth_any_out'
with (root/'sparse/0/depth_params.json').open(encoding='utf-8') as f:
    params = json.load(f)
image_stems = {p.stem for p in images.iterdir() if p.is_file()}
depth_files = {p.stem:p for p in depths.glob('*.png')}
print('images_4:',len(image_stems),'depth PNG:',len(depth_files),'calibration entries:',len(params))
print('missing depth:',sorted(image_stems-depth_files.keys())[:30])
print('missing calibration:',sorted(image_stems-params.keys())[:30])
print('extra depth:',sorted(depth_files.keys()-image_stems)[:30])
for stem in sorted(depth_files)[:12]:
    a=cv2.imread(str(depth_files[stem]),cv2.IMREAD_UNCHANGED)
    print(stem, None if a is None else (a.shape,str(a.dtype),float(np.nanmin(a)),float(np.nanmax(a)),bool(np.isfinite(a).all())))
bad=[]
for k,v in params.items():
    try:
        if not math.isfinite(float(v['scale'])) or not math.isfinite(float(v['offset'])): bad.append(k)
    except Exception: bad.append(k)
print('invalid calibration:',bad[:30],'count:',len(bad))
PY
```

**继续门槛：**图像、同名深度图和标定条目对应；图像与深度尺寸/变换一致；标定值有限；并通过静态源码审阅确认实际深度编码走对了 `utils/camera_utils.py` 的解码分支。当前加载器会把最大像素值大于 50 的深度图视作毫米传感器深度。若 `depth_any_out` 是 8-bit Depth Anything 相对深度，必须先修复或明确转换输入与加载逻辑，再核验训练实际读入的张量值域、mask 和尺度。审计未通过时，不运行任何启用深度/法线先验的完整训练。不要为了通过检查而覆盖原始深度文件。

A800 的 GPU 可用、CUDA/扩展导入和数据访问均需确认。若 AutoDL checkout 或数据不在上述历史路径，以 `pwd`、`git rev-parse --show-toplevel` 和实际挂载目录为准；将正确路径写入 manifest。

## 3. 公共训练配置（Bash）

确认阶段 0 通过后，在同一个 shell session 定义公共参数。Linux 下 `--depths` 可传该数据目录的绝对路径；数据读取器会以场景路径和深度目录组成输入路径。每组固定相同参数、A800、环境和依赖版本：

```bash
export ROOT="$(git rev-parse --show-toplevel)"
export SOURCE="/root/autodl-tmp/TAGS/data/bicycle"
export DEPTH_DIR="$SOURCE/depth_any_out"
export OUTROOT="/root/autodl-tmp/TAGS/output/v16_a800_images4_native_20261106"
export LOGROOT="$OUTROOT/logs"
mkdir -p "$OUTROOT" "$LOGROOT"
cd "$ROOT"

COMMON=(
  -s "$SOURCE" --images images_4 --depths "$DEPTH_DIR"
  --resolution 1 --data_device cpu --eval --llffhold 8
  --iterations 30000 --densify_from_iter 500
  --save_iterations 28500 30000 --no_save_schedule_point_clouds
  --densify_grad_threshold 0.00016 --disable_viewer
  --lpips_loss_weight_final 0.006 --lpips_loss_max_weight 0.010
  --lpips_loss_interval 10 --psnr_loss_weight_final 0.008
  --psnr_loss_max_weight 0.012 --color_align_weight_final 0.0010
  --color_align_max_weight 0.0020 --color_align_std_weight 0.25
  --enable_resolution_adaptive --enable_resolution_aware_schedule
  --enable_adaptive_stage_controller --enable_group_lr_schedule
  --lr_geo_mult_stage1 1.10 --lr_geo_mult_stage2 1.00
  --lr_geo_mult_stage3 0.40 --lr_color_mult_stage1 0.75
  --lr_color_mult_stage2 1.05 --lr_color_mult_stage3 0.55
  --enable_geometry_controller --enable_live_loss_logging
  --live_loss_log_interval 1 --enable_sparse_optimization
  --sparse_mode spa --spa_prune_ratio 0.05 --prune_ratio2 0.04
  --spa_metric hybrid --ras_ref_pixels 0 --ras_scale_min 1.0
  --ras_scale_max 2.5 --ras_densify_base 7200 --ras_densify_min 10000
  --ras_densify_max 18000 --ras_spa_delay 2000
  --adaptive_densify_extend_step 2000 --adaptive_densify_max_iter 18000
  --adaptive_point_budget 8000000
)

run_tags() {
  local name="$1" seed="$2"; shift 2
  local model="$OUTROOT/$name" log="$LOGROOT/$name.log"
  if [[ -e "$model" ]]; then echo "Refusing existing output: $model" >&2; return 2; fi
  local free_kb min_free_kb
  free_kb="$(df -Pk "$OUTROOT" | awk 'NR == 2 {print $4}')"
  min_free_kb="$((6 * 1024 * 1024))"
  if [[ -n "$free_kb" && "$free_kb" -lt "$min_free_kb" ]]; then
    echo "Need at least 6 GiB free for the 28500 and 30000 PLYs; have $((free_kb / 1024)) MiB." >&2
    return 3
  fi
  mkdir -p "$model"
  CUDA_VISIBLE_DEVICES=0 TAGGS_CPU_KNN_INIT=1 \
    python train_optimized.py "${COMMON[@]}" -m "$model" --seed "$seed" "$@" \
    2>&1 | tee "$log"
}
```

使用全新模型名；不要向 `run_tags` 传与公共参数重复的选项。若 shell 的 `pipefail` 未开启，训练管道退出码可能由 `tee` 覆盖，建议在定义函数前执行 `set -o pipefail`。`TAGGS_CPU_KNN_INIT=1` 与 A800 包装脚本的建议一致；记录是否启用。

## 4. 阶段 1：v16 历史配方迁移复现

这一步保留用户给出的 v16 损失、控制器和 SPA 参数，在 A800 上 seed0 从头训练；只将输入明确固定为原始 3DGS 同一份 `images_4`，且 `--resolution 1` 保持该目录原生尺寸。它回答“该配方在统一图像输入后能否迁移运行”，不单独证明几何先验有效，也不与旧 RTX 3080 `resolution=800` 结果合并为同协议对比。

```bash
set -o pipefail
run_tags v16_repro_seed0 0 \
  --enable_geo_prior --geo_prior_depth --geo_prior_normal \
  --depth_prior_weight_final 0.05 --normal_prior_weight_final 0.02 \
  --normal_prior_tv_weight 0.003 --geo_prior_start_iter 0
```

检查训练启动后立即生成的 `schedule_plan.json`：记录 `cur_pixels`、`resolution_scale`、densify、SPA 和阶段边界。因为 `images_4` 原生像素数高于之前使用的 800 像素宽度设置，resolution-aware planner 可能据此改变 densify、SPA 和阶段时序；将其视为本次协议的一部分并留存计划文件。所有 A800 对照组必须得到一致的输入尺寸和计划逻辑。训练结束后：

```bash
MODEL="$OUTROOT/v16_repro_seed0"
python render.py -m "$MODEL" --iteration 30000 --skip_train --quiet
python metrics.py -m "$MODEL" --split test
python scripts/evaluate_depth_per_view.py -m "$MODEL" --iteration 30000 --split test
```

留存 `cfg_args`、`schedule_plan.json`、训练曲线（若产出）、日志、最终 PLY、`results.json`、`per_view.json` 和 `depth_per_view.json`。逐项核对 test camera ID、GT/render 尺寸、iteration、PLY 点数和训练完成状态。

## 5. 阶段 2：深度/法线损失贡献对照

保持阶段 1 其它设置完全不变，仅关闭深度与法线监督。`--enable_geometry_controller` 是单独机制，公共参数中继续开启；因此该组比较的是深度/法线损失的增量，不代表关闭所有几何相关机制。

```bash
run_tags v16_nogeo_seed0 0
```

seed0 后若进入正式候选比较，基线/候选都追加 seed1、seed2，采用配对种子：

```bash
run_tags v16_repro_seed1 1 --enable_geo_prior --geo_prior_depth --geo_prior_normal --depth_prior_weight_final 0.05 --normal_prior_weight_final 0.02 --normal_prior_tv_weight 0.003 --geo_prior_start_iter 0
run_tags v16_nogeo_seed1 1
run_tags v16_repro_seed2 2 --enable_geo_prior --geo_prior_depth --geo_prior_normal --depth_prior_weight_final 0.05 --normal_prior_weight_final 0.02 --normal_prior_tv_weight 0.003 --geo_prior_start_iter 0
run_tags v16_nogeo_seed2 2
```

每个 run 分别执行阶段 1 的 render、metrics 和 depth consistency 评估命令。报告最终高斯数与训练耗时。当前 v16 没有把最终点数锁为完全相等；如主张等预算，必须先实现并对所有组一致启用相同的预算规则，不能只在训练后裁剪候选。

## 6. 阶段 3：法线来源消融

仅在深度输入链路通过审计后进行。所有组固定同一深度监督、seed、图像分辨率、训练预算、其余 v16 参数。

**A：只有深度先验，无正常损失**

```bash
run_tags v16_depthonly_seed0 0 \
  --enable_geo_prior --geo_prior_depth --depth_prior_weight_final 0.05 \
  --geo_prior_start_iter 0
```

**B：当前深度导出法线**

```bash
run_tags v16_depth_currentnormal_seed0 0 \
  --enable_geo_prior --geo_prior_depth --geo_prior_normal \
  --depth_prior_weight_final 0.05 --normal_prior_weight_final 0.02 \
  --normal_prior_tv_weight 0.003 --geo_prior_start_iter 0
```

当前 `depth_to_normals` 对逆深度作有限差分，是图像平面近似且不使用相机内参反投影。因此 B 应称为“depth-derived image-space normal”，不能称严格三维表面法线误差。

**C：外部模型法线（代码接口实现后才可执行）**

现有 CLI 没有独立法线目录参数，以下命令是接口实现后的约定，不是当前可运行命令。候选模型固定为 DN-Splatter 支持的 DSINE；实现时需保证目标法线与可微渲染法线在坐标系、方向、编码、resize、有效 mask 上一致。只替换目标图、却让预测端继续使用不同定义的深度导出法线，不构成公平对照。

```bash
NORMAL_DIR="$SOURCE/normals_dsine"
run_tags v16_depth_externalnormal_seed0 0 \
  --enable_geo_prior --geo_prior_depth --geo_prior_normal \
  --depth_prior_weight_final 0.05 --normal_prior_weight_final 0.02 \
  --normal_prior_tv_weight 0.003 --geo_prior_start_iter 0 \
  --normal_prior_source external --normal_maps_dir "$NORMAL_DIR"
```

记录 DSINE 仓库/权重版本、预处理、输出编码、法线文件清单与哈希、训练 mask 和反向传播计算图。若实现需要更换可微预测法线方式，也应视为 C 组方法定义的一部分并记录，不能与 B 组混称同算法。参考 [DN-Splatter 论文](https://arxiv.org/abs/2403.17822) 与[官方实现](https://github.com/maturk/dn-splatter)。

先用 seed0 选择一个候选并冻结所有参数，再对基线和候选各完成 seed1、seed2。test split 不用于二次选参；最新 R0/G2 的历史三种子结果由于已参与筛选，继续标注为探索性。

## 7. 阶段 4：冻结后的独立验证

仓库虽有 `bonsai`、`garden`、`room` 等目录，但要先确认 AutoDL 上实际数据、图像目录、COLMAP、深度输入和标定都完整。建议以 `garden` 为首选验证场景；若审计失败，则选择第一个完整通过审计的其它场景，并在 manifest 记录选择依据。验证场景不用于调参。

```bash
export SOURCE=/root/autodl-tmp/TAGS/data/garden  # 仅当目录结构审计通过
export DEPTH_DIR="$SOURCE/depth_any_out"
# 按冻结组的 COMMON 条件，将 images_4 改为该场景实际且固定的图像目录；
# 同步固定该场景实际训练分辨率，并按第 3 节重新定义 COMMON；
# 更新后基线和候选必须完全相同。
run_tags garden_baseline_seed0 0
# 将获选组 manifest 中的参数原样填入数组，然后运行：
CANDIDATE_ARGS=(--enable_geo_prior --geo_prior_depth --geo_prior_normal)  # 示例，替换成冻结组完整参数
run_tags garden_candidate_seed0 0 "${CANDIDATE_ARGS[@]}"
```

候选参数数组必须逐字复制获选组 manifest 的完整损失/监督开关，而不是只保留上面示例的三个开关。由于新场景的图像目录和尺寸可能不同，不将 Bicycle 的 `images_4` 尺寸生搬过去；先冻结场景内的分辨率和划分，再保证基线/候选一致。若无独立几何真值，只报告新场景图像质量和先验一致性，不将后者称作绝对几何泛化。

## 8. 每次运行后的检查和记录

```bash
MODEL="$OUTROOT/v16_repro_seed0"  # 替换成该次模型目录
find "$MODEL" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort
cat "$MODEL/cfg_args"
if [[ -f "$MODEL/schedule_plan.json" ]]; then cat "$MODEL/schedule_plan.json"; fi
python render.py -m "$MODEL" --iteration 30000 --skip_train --quiet
python metrics.py -m "$MODEL" --split test
python scripts/evaluate_depth_per_view.py -m "$MODEL" --iteration 30000 --split test
```

若训练流程已自动生成对应渲染/指标，重复执行前确认不会覆盖唯一原始产物。每个 run 保存 manifest（JSON 或 Markdown），字段至少包括：

| 字段 | 记录内容 |
|---|---|
| 实验身份 | 唯一 ID、对照组/候选组、seed、开始结束时间、状态 |
| 代码和环境 | git commit/status、Python/PyTorch/CUDA、GPU 型号、扩展版本 |
| 命令与计划 | 完整命令、`cfg_args`、`schedule_plan.json`、有效像素和阶段边界 |
| 数据 | scene、图像子目录、每张原图/GT/render 宽高、LLFF hold、训练/测试 camera ID 清单；确认与原始 3DGS `-r 4` 输出尺寸一致 |
| 先验 | 深度/法线模型与权重、编码/单位、预处理、校准、文件摘要/hash、mask |
| 输出 | 日志、曲线、checkpoint（如有）、最终 PLY、渲染、metrics、逐视角结果 |
| 结果 | PSNR、SSIM、LPIPS、几何真值指标、先验一致性、N-final、训练耗时 |
| 偏差 | 与冻结配置的变化、失败日志证据、排除理由、未验证项 |

纳入主表的条件：完整训练；目标 camera ID、分辨率和 iteration 正确；深度/法线输入通过校验；指标、PLY、点数、manifest 完整；配对组仅在声明变量上不同。未满足的 run 保留并标记 `excluded_from_primary_comparison`，不删除失败数据。三种子报告均值、样本标准差和配对差值；逐视角胜率仅作补充。小幅变化不称统计显著；单场景不推导普遍有效；test 参与筛选的结果标为探索性。

## 9. 限制和待办

- AutoDL A800 是迁移复现目标，不等同 RTX 3080 原环境；质量数值按硬件/协议分组记录。Bicycle 统一使用与原始 3DGS `-r 4` 相同的 `images_4` 输入，并以 TAGS `--resolution 1` 保持其原生尺寸。旧 3080 `resolution=800` 结果不可与本次数字直接合并。
- 深度编码/单位判断是完整训练前阻断项；本文不声称当前深度链路已验证正确。
- 外部法线支持尚未实现；C 组须先开发 CLI/数据加载/可微预测端并检查坐标与梯度，再执行训练。
- 旧 checkpoint 的学习率状态及 SPA 配置曾出现恢复遗漏。本方案从头训练；以后若用 resume，先单独验证连续/恢复一致性。
- 深度先验一致性不是独立真值几何精度。论文几何结论需要有效的扫描/网格真值或其它独立几何评估。
- 远端路径以 AutoDL 实际 checkout 和挂载为准。每次运行都保存该次 commit 与干净/脏工作区状态；不能把无关本地改动、数据集、权重、凭据或生成训练产物加入代码提交。
