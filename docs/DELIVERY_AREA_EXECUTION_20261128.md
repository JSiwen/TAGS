# Delivery Area 执行手册（2026-11-28）

本手册补充 `output/courtyard_review_20261127/Courtyard实验分析与下一步计划.html`。当前已知结论保持：两个室外扫描场景支持几何增量收益；法线一致性不是独立真值法线精度；Courtyard 的完整组相对 depth-only 平均 PSNR 下降0.070dB，训练成本增加。

**本次连接状态：**使用既有私钥、公钥认证模式和严格known_hosts检查尝试连接，TCP建立后两次在SSH banner exchange超时，未进入公钥认证。没有成功登录，不能确认服务器当前GPU、剩余配额、数据或代码状态；以下入口已完成本地测试，尚未部署/执行。不要把历史目录当成本次在线核验。

## 一次部署，以后无需重新设置路径

下面第一段只在本机Windows PowerShell执行。密钥不上传；若SSH仍在握手阶段超时，先在AutoDL控制台检查实例开机状态、当前SSH端口和地址是否仍相同。不要关闭主机密钥检查。

```powershell
$daKey = 'C:\Users\yori\.ssh\id_ed25519_tags_autodl_20261112'
$daKnown = 'UserKnownHostsFile=D:\TAGSs\TAGSs_core\output\tags_architecture_ablation_20261111\autodl_38636_known_hosts'
ssh -o ConnectTimeout=15 -p 38636 -i $daKey -o IdentitiesOnly=yes -o $daKnown -o StrictHostKeyChecking=yes -o BatchMode=yes root@connect.nma1.seetacloud.com 'mkdir -p /autodl-fs/data/eth3d_delivery_area/tools/commands_20261128'
if ($LASTEXITCODE -ne 0) { throw 'SSH未连接，停止部署' }
scp -P 38636 -i $daKey -o IdentitiesOnly=yes -o $daKnown -o StrictHostKeyChecking=yes 'D:\TAGSs\TAGSs_core\scripts\delivery_area\pipeline.py' 'D:\TAGSs\TAGSs_core\scripts\delivery_area\run.sh' 'D:\TAGSs\TAGSs_core\scripts\delivery_area\install.sh' root@connect.nma1.seetacloud.com:/autodl-fs/data/eth3d_delivery_area/tools/commands_20261128/
if ($LASTEXITCODE -ne 0) { throw '上传失败' }
ssh -p 38636 -i $daKey -o IdentitiesOnly=yes -o $daKnown -o StrictHostKeyChecking=yes -o BatchMode=yes root@connect.nma1.seetacloud.com 'bash /autodl-fs/data/eth3d_delivery_area/tools/commands_20261128/install.sh'
if ($LASTEXITCODE -ne 0) { throw '安装失败' }
ssh -p 38636 -i $daKey -o IdentitiesOnly=yes -o $daKnown -o StrictHostKeyChecking=yes root@connect.nma1.seetacloud.com
```

安装创建 `/usr/local/bin/tagsda`，每次调用自动设置ROOT、FOLLOW、CUDA、OMP及解释器路径；不修改`.bashrc`，不需要手动source、export、cd或conda activate。数据和脚本保存在`/autodl-fs/data`；如果重建容器导致`/usr/local/bin`丢失，只需重新执行上述install.sh。非交互环境PATH未含`/usr/local/bin`时可使用完整命令`/usr/local/bin/tagsda`。

固定路径：

| 用途 | 路径 |
|---|---|
| 训练代码 | `/autodl-fs/data/tags_architecture_ablation_20261112/code` |
| 3DGS代码 | `/autodl-fs/data/tags_architecture_ablation_20261112/3dgs_seeded` |
| 原始下载与解包 | `/autodl-fs/data/eth3d_delivery_area/downloads`、`raw/delivery_area` |
| 训练场景 | `/autodl-fs/data/eth3d_delivery_area/tags_delivery_area` |
| 深度 | `/autodl-fs/data/eth3d_delivery_area/delivery_area_mono_prior` |
| 新实验输出 | `/autodl-fs/data/tags_architecture_ablation_20261112/audit/delivery_area_projection_v2_confirm_20261128` |
| 复用权重 | `/autodl-fs/data/eth3d_courtyard/tools/weights` |
| 复用DSINE源码 | `/autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124/dsine_source` |

## 1. 环境、数据和先验准备（Linux）

每个步骤成功后再执行下一步。以下不启动正式训练；depth/normals/projection会运行GPU推理或短测试。无需重复安装已经正常工作的Courtyard环境。

```bash
tagsda preflight
tagsda download
tagsda prepare
tagsda depth
tagsda calibrate
tagsda normals
tagsda audit
tagsda projection
tagsda freeze
```

`preflight`会打印GPU和存储状态，并检查TAGS/DSINE解释器和两个共享入口哈希；不通过时先修复环境或核对代码变化，不跳过检查。缺少7z时可在服务器安装`p7zip-full`后重跑download。下载固定为官方去畸变图像、scan_eval、occlusion三个包，下载后7z完整性检查并保存SHA256；已有解包文件不会覆盖。官方说明Delivery Area有44张室内图像：[ETH3D数据列表](https://www.eth3d.net/datasets)。

输入验收：44份图像/标定/法线对应，38训练/6测试；完整K，逆深度uint16编码，有限值与有效标定尺度，单位法线、哈希匹配。先验只用SfM标定，不用扫描GT。投影检查覆盖TAGS和3DGS两路径。冻结继承Playground完整已完成命令，只改场景、路径和seed，不重新套用旧Bicycle默认配方。

冻结结果与12条实际训练命令可直接查看：

```bash
cat /autodl-fs/data/tags_architecture_ablation_20261112/audit/delivery_area_projection_v2_confirm_20261128/audit/frozen_protocol.json
```

四组固定3DGS/noGeo/depth-only/depth+DSINE，seed=0/1/2；30k主终点，保存28500/30000。深度final=0.05，法线final=0.02、init=min=0，TV=0.003；w1只是标签。训练分辨率日程与自适应规则继承，不声称实际增密触发完全相同。扫描导出max_views=44，RGB/法线6测试视角。

## 2. 正式训练指令

建议使用tmux防止网络断开导致训练终止；若服务器没有tmux，先安装或使用已有终端管理器。进入会话后逐条执行：

```bash
tmux new -s tags_delivery_area
```

```bash
tagsda train --seeds 0
tagsda evaluate --seeds 0
tagsda geometry --seeds 0
```

seed0仅检查技术失败，不按分数取消较差组的后续种子。确认没有输入/训练/评价错误后：

```bash
tagsda train --seeds 1 2
tagsda evaluate --seeds 1 2
tagsda geometry --seeds 1 2
tagsda normal-eval
tagsda summarize
tagsda pack
```

Ctrl+B再按D脱离tmux；新SSH终端用`tmux attach -t tags_delivery_area`返回。入口会拒绝GPU已有计算任务、并发矩阵、未完成旧run或不同命令的旧输出；训练前检查80GiB挂载余量，同时需要人工核对个人配额。失败后先看日志，不删除manifest强行复用目录；完整同配置run可以跳过。脚本不承诺断点续训。

每次调用自动写入输出目录`audit/阶段_UTC时间_PID.log`，各组完整训练日志在`garden/组名_seedN/train.log`。使用历史garden目录名只是兼容入口，不是场景身份。评价前会检查已有point_cloud链接；坏链接须核实目标后修复，不能绕过保护。

查看进度，无需路径变量：

```bash
nvidia-smi
tail -n 30 /autodl-fs/data/tags_architecture_ablation_20261112/audit/delivery_area_projection_v2_confirm_20261128/garden/tags_depth_dsine_w1_seed0/train.log
find /autodl-fs/data/tags_architecture_ablation_20261112/audit/delivery_area_projection_v2_confirm_20261128/garden -name experiment_manifest.json -exec cat {} \;
```

预计排程先预留约30—45小时训练缓冲，另计先验和评价；这是依据Courtyard30.24小时的规划值，待新场景seed0后更新，不是性能保证。低分不是技术失败；保持完整矩阵。主终点固定F1@2cm，同时保留六阈值、RGB、法线一致性和成本。建议未来预登记实用门槛：法线分支F1增量≥1个百分点、三seed同方向；平均PSNR下降≤0.1dB、LPIPS相对恶化≤2%。这不是显著性检验。

## 3. 导出与下载

`tagsda pack`生成以下文件，并打印SHA256；保留模型在服务器，不覆盖已有包。新pack包含cfg_args，仍需P0补齐评价器源码与环境等证据。

```bash
sha256sum /autodl-fs/data/tags_architecture_ablation_20261112/audit/delivery_area_projection_v2_confirm_20261128_evaluations.tar.gz
```

本机PowerShell下载，无需前面终端的变量：

```powershell
scp -P 38636 -i 'C:\Users\yori\.ssh\id_ed25519_tags_autodl_20261112' -o IdentitiesOnly=yes -o 'UserKnownHostsFile=D:\TAGSs\TAGSs_core\output\tags_architecture_ablation_20261111\autodl_38636_known_hosts' -o StrictHostKeyChecking=yes root@connect.nma1.seetacloud.com:/autodl-fs/data/tags_architecture_ablation_20261112/audit/delivery_area_projection_v2_confirm_20261128_evaluations.tar.gz 'C:\Users\yori\Desktop\test\'
Get-FileHash 'C:\Users\yori\Desktop\test\delivery_area_projection_v2_confirm_20261128_evaluations.tar.gz' -Algorithm SHA256
```

## P0/P1及后续主张的处理

进入新场景前先补Courtyard原训练/评价cfg_args、代码版本、PLY/软链接身份和全视角诊断；旧模型继续复用，不因本次新增入口重训。旧计划中的Linux补档命令仍适用。Playground法线若未生成，可在新入口安装且identity检查通过后执行`tagsda playground-normals`，它使用旧12个模型，不训练，结果归档在新输出的audit/playground_normals。已有结果优先导出，不重复计算。

当前法线评估仍是DSINE一致性。扫描真值法线、全视角诊断、normal-only/LPIPS-off/TV-off和DN-Splatter需要各自的数据接口与冻结协议。它们不是上述12次确认矩阵的必需依赖；本手册没有把尚未实现、未核对的外部方法接口写成可直接运行的训练命令。优先完成现有模型补评、Delivery Area确认和匹配外部对照，再按论文主张选择消融。

## 验证边界

本地4项CPU检查通过：44相机场景转换与篡改检测、uint16编码/标定、六阈值扫描日志解析、三seed配对汇总；Python3.7语法及两份Bash脚本语法通过。服务器SSH握手超时，因此下载/实际目录结构、GPU投影、先验推理、冻结、12次训练、评价和全链路打包均尚未验证。远端共享脚本漂移时preflight/identity应停止，不能静默忽略。
