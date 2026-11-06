# TAGSs_core 几何评测脚本：远程服务器使用说明

此包包含几何评测脚本、配置、测试与说明，不含模型、数据集、完整 TAGSs_core 仓库或已编译 CUDA 扩展。可以在包目录中运行脚本，通过 `--repo` 指向服务器已有的 TAGSs_core，不要求覆盖远程训练代码。

## 1. 服务器准备

需要已有可运行的 TAGSs_core 及其训练环境。参考本机验证环境为 Python 3.7.1、torch 1.12.1+cu116；远程可使用其已有兼容环境，无需照搬 Windows 环境或这些精确版本。CPU 评价依赖 numpy/scipy；GPU 导出还需要 NVIDIA GPU、兼容驱动、torch、项目自身依赖，以及**支持第三项加权逆深度输出**的 `diff_gaussian_rasterization` 扩展。脚本启动时用已知 Z=2 的单高斯检测实际深度语义。仅能导出 RGB 的原始 rasterizer 不满足要求，请使用对应 TAGSs_core 源码编译的扩展。

模型与 SfM 必须来自同一坐标系，不假定米制，不进行尺度拟合。仅需以下数据，不需要稠密点云、RGB 或单目深度先验：

```text
/path/to/TAGSs_core/
  scene/colmap_loader.py
  scene/gaussian_model.py
  scene/cameras.py
  gaussian_renderer/...
  utils/...                      # 项目自身模块与其依赖
  data/bicycle/sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  output/.../point_cloud/iteration_30000/point_cloud.ply
```

相机要求去畸变 PINHOLE/SIMPLE_PINHOLE，暂不支持 COLMAP 文本模型和其他畸变模型。当前 holdout 协议按图像名排序每8个选一个，不读取自定义 test.txt。

Linux 上解压并激活你已经用于 TAGSs_core 训练的环境：

```bash
unzip TAGSs_geometry_eval_portable.zip -d /path/to/tools
conda activate YOUR_TAGSS_ENV

# 请将这三个路径替换成服务器上的真实绝对路径。
PKG=/path/to/tools/geometry_eval_portable
REPO=/path/to/TAGSs_core
OUT=/path/to/evaluation_results/bicycle_v16

# 仅在该环境缺少 CPU 依赖时执行，不会安装 torch 或 CUDA 扩展。
python -m pip install -r "$PKG/requirements-cpu.txt"

# 验证 CPU 几何运算与每次运行的记录行为。
python -m unittest discover -s "$PKG/tests" -p test_geometry_debug.py -v
```

ZIP 中的 `SHA256SUMS.txt` 可以在包目录内用 `sha256sum -c SHA256SUMS.txt` 核验。

## 2. 预检

以下命令适用于数据和模型布局与本机一致的情况。配置内路径相对 `--repo` 解析。`--config`、`--input-run` 的相对路径按当前工作目录解析；推荐全部传绝对路径。

```bash
python "$PKG/scripts/geometry_debug.py" \
  --repo "$REPO" \
  --config "$PKG/configs/bicycle_v16_25views.json" \
  --output "$OUT" \
  --stage preflight
```

预检只检查模型文件、相机、SfM 和配置，不加载 CUDA 扩展、不产生几何指标。实际 GPU 能否渲染需下一步验证。缺失模型会返回非零退出码并记录失败。

## 3. 先做 3 视角小样本测量

```bash
python "$PKG/scripts/geometry_debug.py" \
  --repo "$REPO" \
  --config "$PKG/configs/bicycle_v16_25views.json" \
  --output "$OUT" \
  --stage run \
  --max-views 3 \
  --max-side 320
```

该命令完成深度导出、反投影、多视角一致性、SfM 稀疏参考测量以及全部指标文件的保存。它只是小样本检查，不能将其数字直接与 25 视角/640 像素的结果比较。

## 4. 正式执行本次 25 视角协议

```bash
python "$PKG/scripts/geometry_debug.py" \
  --repo "$REPO" \
  --config "$PKG/configs/bicycle_v16_25views.json" \
  --output "$OUT" \
  --stage run
```

默认 iteration_30000、25个holdout视角、最长边640像素、stride=4、alpha≥0.5；多视角一致要求相对深度差≤2%且循环像素误差≤1。共享过滤SfM轨迹不足50的视角对跳过。SfM要求轨迹长度≥3、重投影误差≤2像素。bicycle 有194个注册视角，holdout=8时共25个测试视角。

若模型或数据路径不同，用以下命令覆盖，无需编辑脚本：

```bash
python "$PKG/scripts/geometry_debug.py" \
  --repo "$REPO" \
  --config "$PKG/configs/bicycle_v16_25views.json" \
  --source-path /path/to/datasets/bicycle \
  --checkpoint /path/to/trained_model/point_cloud/iteration_30000/point_cloud.ply \
  --method my_method_iteration30000 \
  --output "$OUT" \
  --stage run
```

其他数据集需检查测试视角数，必要时用 `--max-views 3` 先调试；不足配置要求数量时会记录失败。比较不同方法须固定同一数据、坐标系、视角、分辨率、阈值、alpha和采样密度，只更换 checkpoint/method；每次运行仍生成新的记录目录。

## 5. 每次运行会记录什么

`--output` 现在表示**输出根目录**。没有指定时，使用配置的 output；配置缺失或无法解析时，回退到 `--repo/output/geometry_records`。相对的 output 路径相对 `--repo` 解析。

每次启动会立即创建类似目录：

```text
$OUT/20261027T130813_123456Z_run_a1b2c3d4/
  record.json                # 从 running 更新到 ok/failed/interrupted
  config_snapshot.json       # 解析后并应用命令行覆盖的配置
  config_original.json       # 可读取时保存原始配置文件
  console.log                # Python 标准输出、标准错误和异常堆栈
  run_status.json            # 或 preflight_status/evaluate_status
  summary.json               # 状态及汇总指标；失败/预检不伪造指标
  artifacts.json             # 完成时生成，列出文件大小与SHA256
  manifest.json              # 预检成功后保存输入哈希、相机、环境等
  metrics.json               # 测量成功后保存完整逐项指标
  per_view.csv               # 成功测量的逐视角指标
  per_pair.csv               # 成功测量的逐有向视角对指标及跳过状态
  REPORT.md                  # 成功测量的简要指标报告
  <image_id>.npz             # GPU run产生：Z、alpha、加权逆深度、mask、K、w2c
  surface_samples.npy/.ply   # 测量阶段产生：未融合的反投影采样点
```

命令一开始会打印 `RUN_DIR=...`，结束会打印包含绝对记录路径的 JSON。时间戳为UTC，附微秒和随机标识；重复执行同一命令也不会覆盖前次结果。记录包含完整参数列表、Linux可复现命令、执行目录、机器信息、Python解释器、脚本哈希、配置哈希、起止时间、耗时和退出码。Windows复制复现时请使用README的PowerShell命令而不是JSON中的POSIX转义命令。

正常结束退出码0；异常1；可捕获的Ctrl+C中断130。无效JSON、配置缺失、模型缺失、导入失败和Python运行异常也有对应记录。命令行参数错误会在当前目录 `geometry_records` 下记录 `cli_error`；`--help`仅显示帮助，不创建实验。

前提是输出目录可写且有可用磁盘空间。断电、kill -9、进程崩溃等不能保证写出结束状态，但事先写入的 `record.json` 会保留 running，用于辨认未完成任务。日志只捕获Python层输出，不保证捕获原生CUDA库直接写入的内容。`artifacts.json`不包含自身的哈希。

## 6. 复用已导出的深度，仅运行CPU测量

把 GPU run 开头打印的完整目录填入 `INPUT_RUN`，不要只填写输出根目录。支持本包导出的记录，也支持旧脚本生成且包含manifest及NPZ的目录。

```bash
INPUT_RUN=/path/to/evaluation_results/bicycle_v16/ACTUAL_TIMESTAMP_run_ID

python "$PKG/scripts/geometry_debug.py" \
  --repo "$REPO" \
  --config "$PKG/configs/bicycle_v16_25views.json" \
  --output "$OUT" \
  --input-run "$INPUT_RUN" \
  --stage evaluate
```

CPU复测创建新的 evaluate 记录目录，不修改输入run。不需要GPU或模型PLY，仍需要原SfM文件和仓库的纯numpy COLMAP loader。它核对SfM哈希、视角和测量配置，并记录输入深度文件哈希。若输入run由3视角/320像素命令生成，复测也必须加 `--max-views 3 --max-side 320`。更换服务器路径时可覆盖source-path；不要更改测量协议。

## 7. Windows本地执行

```powershell
$py = 'C:/Users/yori/miniconda3/envs/gaussian_splatting/python.exe'
$pkg = 'C:/Users/yori/Documents/ChatGPT/TAGS/geometry_eval_portable'
& $py "$pkg/scripts/geometry_debug.py" `
  --repo 'D:/TAGSs/TAGSs_core' `
  --config "$pkg/configs/bicycle_v16_25views.json" `
  --output 'D:/TAGSs/TAGSs_core/output/geometry_records/bicycle_v16' `
  --stage run
```

## 8. 指标解释和限制

depth不是原始渲染器的第三项直接取倒数。脚本导出 `Z=alpha/sum(T_i*alpha_i/Z_i)`，是混合高斯表面代理。AbsRel按有效SfM观测加权；多视角一致率按通过可见性判别的样本加权；空集合写null。有效观测/像素覆盖率、跳过数、遮挡占比也随结果提供。

SfM与训练初始化相关，并非独立稠密真值；预测遮挡判断可能排除模型错误。因此不能用本包直接宣称稠密Chamfer、完整性或F-score，也不能在未评估其他方法时宣称方法优劣。完整几何定义见 `docs/GEOMETRY_PROTOCOL.md`；该文件保留旧版背景，执行指令和记录行为以本README为准。

包内 `VALIDATION.md` 记录本地实际验证范围。远程Linux环境尚未在本次会话中执行，不声称已验证远程CUDA兼容性。
