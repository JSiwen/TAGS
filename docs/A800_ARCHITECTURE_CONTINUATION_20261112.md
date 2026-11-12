# A800 架构消融接续记录（2026-11-12）

本次接续已批准的 4 组 × seeds 0–2 A800 Bicycle 实验。训练源码仍冻结于
`94cd6b1903ca8e102eb34d70cf58cb27aac3f558`，远端工作区干净。本次输入校验修复未部署到正在运行的训练 checkout，未重启或覆盖任何训练。

## 核验时的状态

截至北京时间约 18:06，原 `tags-a800-20261112` screen 队列仍在运行：

| 组别 | seed 0 状态 | PSNR | SSIM | LPIPS | 最终高斯数 |
|---|---|---:|---:|---:|---:|
| 3DGS | 30k 完成；原渲染失败，已有手动恢复指标 | 25.227497 | 0.765495 | 0.208958 | 4,895,375 |
| TAGS noGeo | 30k、RGB 与深度先验评估完成 | 25.534264 | 0.786424 | 0.172165 | 7,112,558 |
| no-group-LR | 30k、RGB 与深度先验评估完成 | 25.476252 | 0.784717 | 0.171640 | 7,111,822 |
| depth-only | 正在训练，核验时超过 18k | — | — | — | 尚非最终点数 |

seeds 1、2 仍由原队列按顺序执行。本表只是状态快照，不能当作三种子结论。
no-group-LR seed 0 相对 noGeo：PSNR −0.058012 dB、SSIM −0.001707、LPIPS −0.000524；没有复现本地 SSIM 改善方向。
3DGS 与 TAGS 点数不同，不能把视觉差异称作同模型容量优势。训练环境也不同：3DGS 为 PyTorch 2.1.0/cu118，TAGS 为 1.12.1/cu116；统一离线评估使用 TAGS 环境。

远端任务根目录：`/autodl-fs/data/tags_architecture_ablation_20261112`。
运行目录：`output/a800_bicycle_images4_native_20261112`。
所有 Bicycle 结果仍是探索性结果，先验一致性不代表独立几何精度。

## 3DGS 评估恢复

原队列已加载的 Python 代码不会因编辑磁盘脚本而更新，因此不修改正在运行的队列。
新增 `scripts/repair_architecture_evaluations.py` 等待原队列 PID 6710 结束，且要求存在队列完成记录。
随后对训练成功而原评估失败的 3DGS run 补做显式 `--llffhold 8` 的渲染、RGB 指标和先验一致性评估。

补评估使用独立 `evaluation_recovery_v1` 目录，保存原 manifest 副本、完整命令、退出码、日志和最终 PLY 哈希；通过软链接读取原点云。
原 manifest 中的失败状态、首次日志及 seed 0 的手动恢复产物均保留。汇总程序应优先读取成功的 recovery manifest，不能继续只看旧 manifest。
补评估会占用 GPU，因此仅在训练队列退出后执行。

已启动的 screen 名称为 `tags-eval-repair-20261112`；当前记录为 `audit/evaluation_repair_status.json`，启动后核验状态是 `waiting_for_training_queue`。这不表示补评估已完成。

## Garden 先验修复与审计

首次远端深度审计失败，原因是旧 `data_stage4/garden/depth_any_out/*.png` 为原图、50 像素间隔和彩色深度的拼接预览。185 个文件均不是二维标量深度。旧加载器取第一通道并 resize，导致文件数量、Camera shape 和有限值检查可以通过，但输入含义错误；旧标定和由其导出的 mono_depth 也不用于本轮。

保留所有旧文件；在本地 RTX 3080、PyTorch 2.4.0+cu118 下用现有 Depth Anything V2 ViT-L 权重和 input-size 518 重新生成 185 张单通道 uint8 PNG。输出目录为 `output/tags_architecture_ablation_20261111/garden_depth_repaired/scalar`。

权重 SHA-256：`a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345`。
生成 manifest 记录模型源码、权重、185 张输入图像和185张输出深度的哈希。修复后的标定工具输出新 JSON，不覆盖数据目录的旧标定。

远端后续训练必须使用：

```bash
ROOT=/autodl-fs/data/tags_architecture_ablation_20261112
SOURCE="$ROOT/data_stage4/garden"
DEPTHS="$ROOT/data_stage4/garden_depth_repaired/scalar"
CALIB="$ROOT/data_stage4/garden_depth_repaired/depth_params_corrected.json"
# TAGS 参数：
# -s "$SOURCE" --images images_4 --resolution 1 --eval --llffhold 8
# --depths "$DEPTHS" --depth_input_encoding monocular_inverse
# --depth_params_path "$CALIB"
```

修复标定 SHA-256：`e169bb97aae63af7c481bae5d204ccb811b0798de70a1bcbe6a45f8676b760c9`。
远端 `audit/stage4_garden_corrected_input_audit.json` 已通过：185 图像/深度/标定，161 train、24 test，1297×840，深度有限且对齐，无尺度保护拒绝视角。
`audit/garden_transfer_verification.json` 对 185 图像和185深度逐文件哈希核验，mismatch=0。
这证明输入传输和加载链路通过审计，不是几何真值验证。

## 本次代码检查与后续门槛

- 修正 `prepare_extra_scenes.py` 默认深度命令，加 `--pred-only --grayscale`。
- 深度解码器拒绝彩色通道；重复灰度三通道仍兼容。
- `generate_scalar_depth.py` 提供无需 matplotlib 的单通道导出和来源清单，拒绝覆盖非空输出目录。
- 6 项深度解码/标定采样测试通过；新增脚本语法检查通过；185 张真实图像推理、标定、上传和远端 Camera loader 审计完成。
- 3DGS 补评估脚本仅验证了 CLI、语法和远端等待状态；实际 GPU 补评估尚待原队列完成。

阶段四尚未启动。必须先收齐并审计 Bicycle 12 个 run（包括恢复评估），保存完整训练证据及点云哈希，给出配对均值/样本标准差和 Pareto 取舍，再记录晋级候选及完整冻结配置。Garden 输入通过审计不替代这一门槛。
本次同步回本地的配置/日志快照不包含远端点云、渲染或完整训练事件文件，因此不称为最终完整归档。
