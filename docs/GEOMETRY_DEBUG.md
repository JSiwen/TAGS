> 运行记录版本已升级：每次运行生成独立子目录，evaluate 必须提供 --input-run。执行命令请以 docs/GEOMETRY_PORTABLE_USAGE.md 为准；下文旧版命令仅保留为历史背景。

# 3DGS 统一几何评测：TAGSs_core 最小 Debug 版

本版实现模型深度导出、反投影、多视角重投影与 SfM 稀疏参考评价，不需要稠密真值点云。它验证几何评测管线是否能运行，不能替代独立真值上的几何精度实验。

## 来源与接入边界

2026-10-27，已读取聊天“比较几何指标可行性”（6ad86506-a93c-83ea-bf0e-da7a6f3a7bad）。接口返回用户需求和正文引用占位符，未提供旧方案正文或附件。因此这里按本次明确需求重建最小实现，不能声称逐字恢复或完整同步了旧方案。原始 3DGS 工作环境本次未修改。

目标：`D:/TAGSs/TAGSs_core`。只新增独立脚本、配置、测试和文档，不改现有训练器、CUDA 源码、深度先验数据或用户原有改动。运行加载现成 PLY，不重新训练。

## 实现与语义

- `scripts/geometry_debug.py`：`preflight` 检查 COLMAP 二进制输入、固定小样本、模型路径并记录哈希；`run` 导出深度并评价；`evaluate` 从导出重新运行 CPU 评价，并核对配置、相机和 SfM 文件哈希。
- `configs/geometry_bicycle_debug.json`：bicycle、迭代 1000 的 TAGSs e4 模型、3 个 holdout 视角、最长边 320、stride=4、alpha≥0.5。该模型只供 Debug，不代表最终训练结果。
- `tests/test_geometry_debug.py`：合成平面、非平凡相机姿态往返、平移相机、遮挡、前方不一致、空掩码、非有限深度、正确与缩放错误的 SfM 对照。

仓库 `gaussian_renderer/__init__.py` 将第三项输出命名为 `depth`；但 `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:431-452` 实际累积 `S=sum(T_i*alpha_i/Z_i)`。它不是 Z，也不是可直接取倒数的归一化逆深度。

脚本以黑背景、白色高斯预计算颜色渲染一次，同时得到 `A=sum(T_i*alpha_i)` 和 S，导出 `Z=A/S`，即条件平均逆深度的倒数。它仍是高斯混合得到的表面代理，不能称为真实表面交点或中值深度。无效位置 Z=0，并以单独 mask 排除。每次 GPU 运行先以已知 Z=2 的单个高斯验证已安装扩展的深度语义，失败即停止。

直接使用现有 `GaussianModel.load_ply`、底层 `GaussianRasterizer` 和 `MiniCam`，避免加载全部 RGB 和深度先验。绕过现有 render 封装的 `override_color` 参数问题：封装传递空 SH 张量，同时传预计算颜色，安装版扩展将其视为冲突；脚本向底层明确传 `shs=None`。未修改训练渲染路径。

COLMAP 外参 `X_cam=R*X_world+t`，正 Z 在相机前方；反投影为 `X_world=R.T*(Z*K^-1*[u,v,1]-t)`。支持去畸变 PINHOLE/SIMPLE_PINHOLE，其他模型明确拒绝。缩放后 K 使用实际宽高比例，COLMAP 主点乘比例再减 0.5 转为整数数组坐标；投影矩阵显式保留主点，符合 rasterizer 的 NDC→像素公式。深度单位为 COLMAP 场景单位，不假定米制，不进行拟合对齐。

## 视角、掩码与评价规则

1. 按完整图像名排序，每 8 个抽一个 holdout 候选；固定第一个为锚点，按过滤后共享 SfM 轨迹数选出另外两个，重叠相同时按名称排序。这个集合只由参考数据决定，与模型结果无关。要求同配置跨方法运行，核对 manifest 中视角和输入哈希；有自定义 test.txt 的实验应另行固定对应划分，当前配置仅用 LLFF holdout=8。
2. SfM 点要求重投影误差≤2 像素、轨迹长度≥3；每个视角只使用其实际观测过的过滤点。输出投影位置最近像素的深度 AbsRel、RMSE、有效数量及有效覆盖率。无效深度不计入误差，同时必须报告有效覆盖率，防止空预测取得虚假低误差。
3. 对每一有向视角对要求共享过滤轨迹≥50。源有效深度按 stride=4 采样，反投影后投到目标；排除出界、相机后方和目标无效深度。目标用最近像素采样。
4. `z_projected > z_target*(1+0.02)` 记为遮挡候选，不纳入可见误差。可见点同时满足目标相对深度差≤2% 和目标深度反投影回源的循环像素误差≤1，才判为一致。输出源样本数、目标有效数、遮挡数、评分数、一致率、循环误差和深度误差；同时给出未剔除遮挡的全部目标有效点 AbsRel。遮挡判别依赖预测本身，可能把预测错误排除，须联合所有计数及双向结果解释。
5. 参考点到本视角采样表面的单向最近邻距离，除以固定场景尺度 R；R 为全部注册相机中心到其坐标中位数的距离之 90% 分位数。报告阈值 0.01R/0.02R/0.05R 内的稀疏覆盖率。该采样结果依赖分辨率与 stride，不能与不同采样配置直接比较。

不输出稠密 Chamfer、表面精度、完整性或 F-score：SfM 缺失区域没有可靠的负例和完整表面。SfM 还是训练初始化相关的弱参考；多视角一致也可能来自共同错误。论文中的独立几何精度比较仍需要如 Tanks and Temples 的独立真值、官方裁剪/配准与评价流程。本次没有实施那些流程或形成多方法排名。

## 在仓库中复现

在仓库根目录的 PowerShell 执行；当前确认可用的是 `gaussian_splatting` 环境。CPU 评价仅需 numpy/scipy，GPU 导出需要项目 CUDA 扩展和兼容 torch。

```powershell
$py = 'C:/Users/yori/miniconda3/envs/gaussian_splatting/python.exe'
& $py scripts/geometry_debug.py --config configs/geometry_bicycle_debug.json --stage preflight
& $py -m unittest discover -s tests -p test_geometry_debug.py -v
& $py scripts/geometry_debug.py --config configs/geometry_bicycle_debug.json --stage run
& $py scripts/geometry_debug.py --config configs/geometry_bicycle_debug.json --stage evaluate
```

只测试本次新增文件，避免触发整个仓库已有测试。跨方法时复制配置，只更换 method、checkpoint、output；保持其余参数与坐标系一致，使用各自独立输出目录。显式选择迭代数，不自动搜索最新模型。不要对来自不同 SfM 坐标系的模型直接评价。

默认输出 `output/geometry_bicycle_debug/`：

- `manifest.json`：配置、相机内外参、观测点 ID、输入与模型 SHA256、运行环境和单高斯校验结果。
- `<image_id>.npz`：float Z、alpha、weighted_invdepth、bool mask、K、w2c。
- `surface_samples.npy/.ply`：反投影采样点，非融合、非去重点云。
- `metrics.json`：每视角 SfM 指标和双向多视角指标。
- `preflight_status.json`、`run_status.json`、`evaluate_status.json`：阶段成功/失败、实际错误与时间。失败时请以对应阶段 status 为准，目录中可能保留上一次成功输出。

`--output` 可指定其他目录。`evaluate` 无需再次渲染，但必须保留当初的输入和 manifest；模型输出哈希防篡改与长期不可变实验登记未在该最小版中实现。

实际测试与同步核验见 `docs/GEOMETRY_DEBUG_RESULTS.md`。
