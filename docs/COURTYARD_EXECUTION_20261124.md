# Courtyard 完整执行手册（2026-11-24）

本轮目的：在未用于权重选择的 ETH3D Courtyard 上，确认 Playground 的扫描几何收益是否复现。四组固定为 3DGS、TAGS noGeo、depth-only、depth+DSINE w1；配对 seeds 0/1/2，30k，`images_4 --resolution 1 --eval --llffhold 8`。主指标固定扫描 F1@2cm；同时报告六阈值 Accuracy/Completeness/F1、PSNR/SSIM/LPIPS、Gaussian 数、时间和显存。DSINE 角度仅代表先验一致性。

已通过 SSH 核对：Courtyard 三个数据包已下载、已解压；38 张注册相机、四个 PINHOLE 标定；A800 80GB，检查时空闲；`/root/autodl-tmp` 可用约 50GB，`/autodl-fs` 挂载整体可用约 282GB（不等于个人配额）。`dsine` 为 Python 3.10，但尚缺 torch。TAGS 的 Python 3.7 环境保持原样。本次已完成 CPU 场景转换，38张输入和33/5划分已核验；深度/法线生成与训练尚未执行。两条GPU rasterizer的各10个合成案例和各5个真实Camera加载检查均已通过；38个相机均携带完整K。当前不存在 Courtyard 训练结果。本手册提供后续执行命令，不代表这些阶段已运行。

执行器已随本次任务提供：`scripts/courtyard_pipeline.py`、`scripts/courtyard_env.sh`。远端部署目录为 `/autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124`。小型包包含本机 DSINE v02 源码及许可证，不含数据或权重。两份权重使用本机已有文件并校验 SHA256，避免下载不同模型。首次执行按 0—13 顺序；每个阶段失败就停止，保留日志。不要跳过检查或删除旧模型来强行继续。

## 0. 本机 PowerShell：上传已有权重

当前缺口是服务器没有已核验的两份权重。下面上传约 1.63GB；不是训练。脚本包在本次任务中上传，若需要重新部署，先执行小包上传和解压指令。新开 PowerShell 后整段复制即可：

```powershell
$sshArgs = @('-p','38636','-i','C:\Users\yori\.ssh\id_ed25519_tags_autodl_20261112','-o','IdentitiesOnly=yes','-o','UserKnownHostsFile=D:\TAGSs\TAGSs_core\output\tags_architecture_ablation_20261111\autodl_38636_known_hosts','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes')
$scpArgs = @('-P','38636','-i','C:\Users\yori\.ssh\id_ed25519_tags_autodl_20261112','-o','IdentitiesOnly=yes','-o','UserKnownHostsFile=D:\TAGSs\TAGSs_core\output\tags_architecture_ablation_20261111\autodl_38636_known_hosts','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes')
ssh @sshArgs root@connect.nma1.seetacloud.com 'mkdir -p /autodl-fs/data/eth3d_courtyard/tools/weights'
if ($LASTEXITCODE -ne 0) { throw 'Remote directory preparation failed' }
scp @scpArgs 'D:\TAGSs\TAGSs_core\output\courtyard_full_runbook_20261124\courtyard_commands_20261124.tar.gz' root@connect.nma1.seetacloud.com:/autodl-fs/data/eth3d_courtyard/
if ($LASTEXITCODE -ne 0) { throw 'Command bundle upload failed' }
ssh @sshArgs root@connect.nma1.seetacloud.com 'set -e; tar -xzf /autodl-fs/data/eth3d_courtyard/courtyard_commands_20261124.tar.gz -C /autodl-fs/data/eth3d_courtyard/tools; cd /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124; sha256sum -c SHA256SUMS'
if ($LASTEXITCODE -ne 0) { throw 'Command bundle validation failed' }
scp @scpArgs 'D:\TAGSs\TAGSs_core\checkpoints\depth_anything_v2_vitl.pth' root@connect.nma1.seetacloud.com:/autodl-fs/data/eth3d_courtyard/tools/weights/
if ($LASTEXITCODE -ne 0) { throw 'DA-V2 weight upload failed' }
scp @scpArgs 'C:\Users\yori\DSINE\projects\dsine\checkpoints\exp001_cvpr2024\dsine.pt' root@connect.nma1.seetacloud.com:/autodl-fs/data/eth3d_courtyard/tools/weights/
if ($LASTEXITCODE -ne 0) { throw 'DSINE weight upload failed' }
ssh @sshArgs root@connect.nma1.seetacloud.com
```

以下均在登录后的 **Linux 终端**执行。断线重登后重新 `source` 环境脚本。不要把 PowerShell 的变量语法放进 Linux。

## 1. 固定路径、核验资源和权重

目标：明确当前实验目录、GPU 和实际磁盘；两份权重完全一致。停止条件：哈希不符、GPU 有其它任务或可用空间不足。建议本轮输出预留至少 100GiB：这是容量规划余量，不是已测得的 Courtyard 占用；实际配额还要看平台。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
hostname
nvidia-smi
df -h /root/autodl-tmp /autodl-fs
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
cd "$DATA/tools/weights"
printf '%s\n' \
  'a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345  depth_anything_v2_vitl.pth' \
  '629472c66ea9ec83bab6ba7c957ad7d983bb9c91063a54ad74ff5b8229cbf22f  dsine.pt' | sha256sum -c -
"$PY" -c 'import shutil,os; p=os.environ["NEW_OUT"]; free=shutil.disk_usage(p).free; print("free GiB",free/2**30); assert free >= 100*2**30, "Need output storage review before full matrix"'
```

## 2. 补齐独立先验环境

使用已存在的 `dsine` Python 3.10 环境，不在 TAGS Python 3.7 内升级 PyTorch。PyTorch 2.5.1 / torchvision 0.20.1 / CUDA 12.1 组合来自[官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。模型分别来自 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 和 [DSINE](https://github.com/baegwangbin/DSINE)。这里固定版本用于复现，未声称服务器已安装成功。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PRIOR_PY" -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$PRIOR_PY" -m pip install --no-cache-dir numpy==1.26.4 pillow==10.4.0 opencv-python-headless==4.10.0.84 geffnet==1.0.2
"$PRIOR_PY" -c 'import torch,torchvision,numpy,cv2,geffnet; print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); print(torch.ones(1,device="cuda")*2)'
"$PRIOR_PY" -m pip freeze > "$NEW_OUT/audit/prior_environment.txt"
"$PY" -m pip freeze > "$NEW_OUT/audit/tags_environment.txt"
```

任何下载、导入或 CUDA 检查失败即停止。不要去掉版本限制后盲目继续。

## 3. 转换 Courtyard 输入

目标：生成 38 张 `images_4` 和二进制 COLMAP；保留原相机姿态、原生分辨率 K、三维点及观测坐标，图像名去掉子目录。实际训练分辨率约 1550×1034，33 训练视角、5 测试视角。源 JPEG 经 LANCZOS 四分之一尺寸、quality95 写入；这是显式记录的本轮输入定义。尺寸或相机数异常即停止。完成后重复调用只校验并复用；部分生成目录会拒绝覆盖，需先检查原因。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" "$PIPE" prepare 2>&1 | tee "$NEW_OUT/audit/prepare.log"
cat "$NEW_SCENE/preparation.json"
```

## 4. 生成 DA-V2 深度并用 SfM 标定

目标：同一 vitl 权重、输入尺寸 518；保存原始 float32 和单通道 uint16 相对逆深度，加载分母明确为 65535；标定只用 COLMAP SfM，不读取扫描真值。所有四组共享此数据定义。标定采用 median + mean absolute deviation，稀疏点误差≤2、track≥3，采样坐标 `x*sx−0.5,y*sy−0.5`；这是本轮显式冻结的准备实现，不能声称与未保存源码的旧准备器逐位相同。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PRIOR_PY" "$PIPE" depth 2>&1 | tee "$NEW_OUT/audit/depth_generation.log"
"$PY" "$PIPE" calibrate 2>&1 | tee "$NEW_OUT/audit/depth_calibration.log"
```

停止条件：预测退化／非有限值、有效 SfM 对应不足 10 个、无有效标定。不得用扫描 F1 调整 scale/offset。先验生成可以在冻结前重跑；冻结后脚本会阻止重生成。

## 5. 生成 DSINE 法线

目标：H×W×3 float32、单位向量、相机坐标 xyz；输入使用真实缩放 K 和 padding 主点平移。DSINE 的像素中心约定为 x+0.5，不能直接套用渲染器整数像素 K。先 pad 黑色 RGB，再归一化，使用完整权重严格加载；无需另外下载已被完整 checkpoint 覆盖的 EfficientNet 权重。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PRIOR_PY" "$PIPE" normals 2>&1 | tee "$NEW_OUT/audit/normal_generation.log"
```

停止条件：权重键不匹配、CUDA OOM、形状错误、NaN/Inf。不要通过翻转法线方向来提升测试角度。

## 6. 完整输入与投影验收

目标：38/38 输入完备，法线单位长度误差<0.01；每张标定 scale 位于中位数的 0.2—5 倍内，避免 loader 静默禁用；正逆深度覆盖率>50%（技术失败门槛，不是论文质量结论）。两条 rasterizer 和真实 Camera 流程均运行原投影测试。这里会运行少量 GPU 测试，不是训练。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" "$PIPE" audit 2>&1 | tee "$NEW_OUT/audit/input_check.log"
"$PY" "$PIPE" projection 2>&1 | tee "$NEW_OUT/audit/projection_check.log"
cat "$NEW_OUT/audit/courtyard_projection.json"
```

停止条件：任意检查失败；代码 commit/diff 不再匹配 Playground 投影验收记录。保留原日志并修复技术问题，不修改验收 JSON 伪造通过。

## 7. 冻结全部 12 条训练命令

当前无 Courtyard 分数参与选择。直接读取 Playground 四组 seed0 的完成 manifest，除 source、model、seed、depth/calibration/normal 路径外，要求其余命令完全一致。内部历史 batch 名仍是 `garden`，实际 `-s` 必须是 Courtyard。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" "$PIPE" freeze 2>&1 | tee "$NEW_OUT/audit/all_commands_dry_run.txt"
cat "$NEW_OUT/audit/frozen_protocol.json"
```

目标产物：完整 12 条 argv、输入哈希、代码身份、评价器和先验源码身份。停止条件：参考 manifest 缺失、非路径参数变化、输入/投影验收未通过。第一次冻结后，不再根据 Courtyard 指标改变权重、分辨率、alpha、stride、种子或组别。

## 8. 四组 seed0 完整训练

**从这一步开始才是正式训练。** 本次规划任务没有执行。四组顺序运行，各保存 28,500 和 30,000 PLY；保留现有协议的计算预算、损失与初始化。Playground四组每seed训练合计约2.92小时，三seed约8.76小时，尚未包含先验生成和评测；这仅是参考预算，Courtyard实际耗时未知。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" -u "$PIPE" train --seeds 0 2>&1 | tee "$NEW_OUT/seed0_train.log"
```

另开 Linux 终端看当前任务：

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
nvidia-smi
ps -eo pid,etime,args | grep -E '[t]rain_optimized.py|[3]dgs_seeded/train.py|[c]ourtyard_pipeline.py train'
find "$NEW_OUT/garden" -name experiment_manifest.json -print -exec cat {} \;
tail -n 40 "$NEW_OUT/garden/tags_depth_dsine_w1_seed0/train.log"
```

最后一条在 DSINE 组尚未创建时会提示不存在，不能据此判断前面的组失败。训练日志在每个模型的 `train.log`，外层只显示组别开始/结束。前台训练终端保持连接；Ctrl+C 会打断当前训练，wrapper 不承诺自动恢复。`trained` 且 command 一致可跳过；unfinished/different run 会拒绝重跑，先检查日志和 checkpoint，不能删除完成标记伪装全新运行。

## 9. seed0 统一 RGB／深度／扫描评测

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" "$PIPE" evaluate --seeds 0 2>&1 | tee "$NEW_OUT/seed0_rgb_depth.log"
"$PY" "$PIPE" geometry --seeds 0 2>&1 | tee "$NEW_OUT/seed0_geometry.log"
```

RGB 与深度统一使用30k和同5个测试视角。扫描导出使用全38个相机、max_side1600、alpha0.5、stride2、holdout1；读取 `raw/courtyard/dslr_scan_eval/scan_alignment.mlp`，阈值1/2/5/10/20/50cm、voxel_size1cm，继承 Playground 表面导出规则。模型目录下 `followup_eval_30000/complete.json`、`geometry/<run>/complete.json` 必须存在。

停止条件：模型缺失、渲染失败、测试视角数不符、非有限 PLY、扫描日志没有六档结果。完整 seed0 的低分也要保留；只有技术错误影响验收，不按分数删组或调参。

## 10. 补齐 seeds1、2 并统一评测

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" -u "$PIPE" train --seeds 1 2 2>&1 | tee "$NEW_OUT/seeds12_train.log"
"$PY" "$PIPE" evaluate --seeds 1 2 2>&1 | tee "$NEW_OUT/seeds12_rgb_depth.log"
"$PY" "$PIPE" geometry --seeds 1 2 2>&1 | tee "$NEW_OUT/seeds12_geometry.log"
```

目标12/12完成。不要用“seed0 不好看”取消特定方法；若预算不足，报告不完整矩阵，而不是将可用种子混成三种子结论。

## 11. Courtyard 与 Playground 法线补评测

每次把12个模型一起送入同一评价器，使用共同有效像素交集、signed 角度、等视角权重。必须报告覆盖率；低覆盖率的低角差不能单独视作改进。诊断图是评价器默认的部分视角，不宣称包含全部测试图。

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" "$PIPE" normal-eval 2>&1 | tee "$NEW_OUT/courtyard_normal_eval.log"
"$PY" "$PIPE" playground-normals 2>&1 | tee "$NEW_OUT/playground_normal_eval.log"
```

Playground 补评测可在步骤8前单独执行，不要求Courtyard训练完成，但要求旧12个30k模型和输入仍存在。已有输出时脚本拒绝覆盖；保留第一次结果。这里的 DSINE 参考参与过训练，不能支持“独立真实法线精度提升”。若论文必须主张真实法线精度，下一轮需要另设激光扫描表面法线、可见性/边界掩码及坐标一致性验证，当前脚本没有实现该独立指标。

## 12. 导出结果与证据包

```bash
source /autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/courtyard_env.sh
"$PY" "$PIPE" summarize 2>&1 | tee "$NEW_OUT/audit/summary.log"
"$PY" "$PIPE" pack 2>&1 | tee "$NEW_OUT/audit/pack_result.txt"
sha256sum "$ROOT/audit/courtyard_projection_v2_confirm_20261124_evaluations.tar.gz"
```

summarize生成run_table.csv、summary.json，给出三种子均值/样本标准差及三个预设比较的同seed差值；F1差值单位为百分点。包内保存 JSON/log/csv/patch 等小型证据和 SHA256；不打包大 PLY、原始扫描、权重或凭据。30k模型和表面文件留在原目录，禁止据此删除。`pack` 会检查12份训练状态、RGB和扫描完成标记，以及Courtyard法线结果；后续回传包仍须核对指标条目、视角和覆盖率，不把“成功打包”等同论文通过。

## 13. 本机 PowerShell：下载结果

```powershell
scp -P 38636 -i 'C:\Users\yori\.ssh\id_ed25519_tags_autodl_20261112' -o IdentitiesOnly=yes -o 'UserKnownHostsFile=D:\TAGSs\TAGSs_core\output\tags_architecture_ablation_20261111\autodl_38636_known_hosts' -o StrictHostKeyChecking=yes -o BatchMode=yes root@connect.nma1.seetacloud.com:/autodl-fs/data/tags_architecture_ablation_20261112/audit/courtyard_projection_v2_confirm_20261124_evaluations.tar.gz 'C:\Users\yori\Desktop\test\'
if ($LASTEXITCODE -ne 0) { throw 'Evidence download failed' }
Get-FileHash 'C:\Users\yori\Desktop\test\courtyard_projection_v2_confirm_20261124_evaluations.tar.gz' -Algorithm SHA256
```

## 结果判读与停止／继续规则

先给12条逐run表，再算三种子均值±样本标准差、同seed配对差；不将38个表面视角或5个测试图当作独立训练重复。预先关注：depth-only−noGeo、DSINE−depth-only、DSINE−3DGS。主指标为F1@2cm，不能按结果临时换到更好看的阈值。图像指标、时间/显存/Gaussian数一同展示，低扫描绝对分数和失败区域必须如实写明。

Courtyard 如果复现，论文可扩大为两个室外扫描场景上的证据；仍不能证明普适提升或室内有效。若方向反转，应先报告场景依赖，保留负结果；任何基于Courtyard调参后的运行改为探索性。后续独立室内场景优先考虑 ETH3D Delivery Area，下载和检查后另行冻结协议。不要同时铺开更多场景或权重搜索，使当前缺失证据再次被搁置。

当前四组可以检验新增损失的增量作用，但noGeo保留的其它机制、缺失normal-only、固定Gaussian/时间预算和外部几何方法对照，会限制模块因果性与效率/领先性主张。是否补这些实验取决于论文实际结论；不能仅凭这12次运行写成全面SOTA。

## 本次交付的验证边界

本地通过4项CPU检查（uint16编码/标定、COLMAP转换及篡改检测、六阈值日志解析、同seed配对统计），以及Python3.7语法和Bash语法检查。服务器实际完成CPU场景转换与TAGS/3DGS两条GPU投影/Camera测试，日志存于新输出的audit目录。两份大权重尚未由本任务上传；依赖安装、DA-V2/DSINE推理、输入全验收、正式freeze、12次训练及新场景评测尚未运行，不能把本手册当作端到端已通过记录。
