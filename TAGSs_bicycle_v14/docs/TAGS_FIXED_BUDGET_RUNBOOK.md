# 固定预算实验运行说明（2026-10-29）

本次实现：PSNR smoothstep 25k–27k 倍率、geometry 最终统一缩放、SPA 固定目标选择（保护集合超过预算则报错）、最终预算报告、500 步 CSV、相机 ID/尺寸清单、v2 checkpoint（模型/优化器、曝光/曝光优化器、随机状态、相机队列、SPA、难视角 EMA）。默认倍率为 1，目标数为 0，保持旧行为。SPA 目标只对 S 组设置；R0/P/G 不改变已有裁剪规则。

支持范围：单 GPU、固定时间表；不支持动态阶段控制器的公平恢复。旧 tuple checkpoint 仍可用于普通训练，但固定预算恢复会拒绝它。完整恢复只是工程接入，必须通过 GPU 的不中断/恢复 R0 实验才能证明可复现。

## 训练命令

在 gaussian_splatting 环境、项目根目录前台执行。BASE 必须是你确认的同分辨率 30k 正式基准目录，包含 cfg_args 和 point_cloud/iteration_30000/point_cloud.ply。驱动器从 PLY 读取 N0，从 cfg_args 继承图像源、分辨率、全部已声明训练参数；不会猜测基准。当前搜索未定位到文档 25.2944317 对应的真实基准或 25k checkpoint，不能替你填入其他实验。

```powershell
conda activate gaussian_splatting
cd D:\TAGSs\TAGSs_core
$BASE = "你的30k正式基准绝对目录"
$DATA = "D:\TAGSs\TAGSs_core\data\bicycle"
$DEPTH = "$DATA\depth_any_out"
# 不加 --execute 只打印完整命令与 N0，先核对图像源/分辨率/权重
python scripts/run_fixed_budget_sweep.py --baseline $BASE --source $DATA --depths $DEPTH --case R0 --output output/tags_budget_sweep/reference_seed0
# 确认后，从头重训原配置，生成新版25k/30k checkpoint（继承基准裁剪，不强制凑数）
python scripts/run_fixed_budget_sweep.py --baseline $BASE --source $DATA --depths $DEPTH --case R0 --output output/tags_budget_sweep/reference_seed0 --execute
# 先做恢复对照，与上面不中断30k比较
$CKPT = "output/tags_budget_sweep/reference_seed0/chkpnt25000.pth"
python scripts/run_fixed_budget_sweep.py --baseline $BASE --source $DATA --depths $DEPTH --case R0 --checkpoint $CKPT --output output/tags_budget_sweep/R0_seed0 --execute
# R0复现通过后，逐个前台执行；case可替换P2/G1/G2
python scripts/run_fixed_budget_sweep.py --baseline $BASE --source $DATA --depths $DEPTH --case P1 --checkpoint $CKPT --output output/tags_budget_sweep/P1_seed0 --execute
# 第二轮从0开始。S0/S1/S2分别保护0.05/0.08/0.10，同一N0选择保留对象
python scripts/run_fixed_budget_sweep.py --baseline $BASE --source $DATA --depths $DEPTH --case S1 --seed 0 --output output/tags_budget_sweep/S1_seed0 --execute
# 正式渲染/指标（换成实际候选目录）
python render.py -m output/tags_budget_sweep/P1_seed0 --iteration 30000 --skip_train
python metrics.py -m output/tags_budget_sweep/P1_seed0 --split test
```

Linux/AutoDL 使用相同 Python 驱动器，替换绝对路径；无需 PowerShell。输出目录必须不存在。训练失败后请保留目录，重跑选择新目录。没有自动启动训练、render或metrics。

## 评价与未完成验证边界

本驱动继承官方 llffhold=8 并在30k评价官方test，manifest明确标记 official_test_used=true；目前没有新增独立开发集。若用这些命令筛参，必须如实报告官方测试集参与筛选，不能宣称独立开发验证结果。PG1应在单变量选择后手工传参，未提供自动选择。

检查 camera_manifest.json、budget_report.json、training_curves.csv、experiment_manifest.json 和 runtime.json。预算同时要求 N_final<=N0 且差值<=0.5%；不足预算不会补点。质量目标使用未截断指标。逐视角指标仍由原 metrics.py 输出 per_view.json；本次未改指标实现、未自动生成方案的四种汇总CSV，也未完成跨run尺寸/视角断言、数据版本哈希和多seed汇总。不能把训练成功当作完整实验验收。

验证：24项后期调度断言、4组驱动器模拟dry-run、Python compileall通过；尚未运行GPU短训、checkpoint恢复数值对照、完整训练或质量验收。模拟数据在 output/fixed_budget_driver_fixture，仅用于驱动器验证，不能用作基准。


## AutoDL：unrecognized arguments 的修复

如果报错不认识 psnr_final_multiplier / geo_scale / final_target_gaussians，说明只拿到了新版驱动器，训练模块仍旧。应更新整套代码。远端代码在 codex/fixed-budget-experiments-20261029 分支，main 尚未合并。

为保留服务器旧工作区，可创建独立工作树（新目录必须不存在）：

```bash
cd /root/autodl-tmp/TAGS
git fetch https://github.com/JSiwen/TAGS.git codex/fixed-budget-experiments-20261029
git worktree add --detach /root/autodl-tmp/TAGS_fixed_budget_20261029 FETCH_HEAD
cd /root/autodl-tmp/TAGS_fixed_budget_20261029/TAGSs_bicycle_v14
export DATA=/root/autodl-tmp/TAGS/TAGSs_bicycle_v14/data/bicycle
export DEPTH="$DATA/depth_any_out"
# BASE 必须指向实际TAGS的完整30k基准，不是原始gaussian-splatting/output。
export BASE=/替换为实际TAGS基准目录
python scripts/run_fixed_budget_sweep.py --baseline "$BASE" --source "$DATA" --depths "$DEPTH" --case R0 --output output/tags_budget_sweep/reference_seed0_v2
# dry-run确认后，执行同一命令并加 --execute。
```

新版驱动器会在创建输出目录前检查八个配套模块的更新标记和基准训练配置；不完整就报错。--execute还会先加载训练CLI的--help，发现导入故障时不创建输出目录。此检查不等价于GPU短训。旧失败目录无需删除，新目录用reference_seed0_v2。不要删除新参数来绕过版本错误。

特别注意：原始3DGS的cfg_args一般只有模型/渲染参数，无法恢复TAGS几何、SPA、感知损失及学习率设置。即使PLY可读并给出N0，也不构成TAGS配置冻结。原有只检查模型参数的模拟dry-run未覆盖此错误；现新增3项preflight回归检查，均通过。
