# 固定预算实验运行说明（2026-10-29）

本次实现：PSNR smoothstep 25k–27k 倍率、geometry 最终统一缩放、SPA 固定目标选择（保护集合超过预算则报错）、最终预算报告、500 步 CSV、相机 ID/尺寸清单、v2 checkpoint（模型/优化器、曝光/曝光优化器、随机状态、相机队列、SPA、难视角 EMA）。默认倍率为 1，目标数为 0，保持旧行为。SPA 目标只对 S 组设置；R0/P/G 不改变已有裁剪规则。

支持范围：单 GPU；新增动态阶段控制器状态保存/恢复（需要含 adaptive_stage 字段的新版 checkpoint）。旧 tuple checkpoint 仍可用于普通训练，但固定预算恢复会拒绝它。完整恢复只是工程接入，必须通过 GPU 的不中断/恢复 R0 实验才能证明可复现。

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

本驱动继承官方 llffhold=8 并在30k评价官方test，manifest明确标记 official_test_used=true；目前没有新增独立开发集。若用这些命令筛参，必须如实报告官方测试集参与筛选，不能宣称独立开发验证结果。PG1在单变量筛选后固定选择P2+G2（PSNR倍率1.5、geo_scale=0.5），驱动器不自动筛选候选。

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


## 动态阶段基准支持更新

如果 cfg_args 中 enable_adaptive_stage_controller=True，保留此开关与 resolution_schedule，不能为绕过报错改成False。新版驱动器已移除该拦截。checkpoint新增 adaptive_stage 字段，记录动态 densify_until_iter、SPA start/end 和上次扩展迭代；同时恢复有效densify边界、fullres边界和已有质量EMA。旧checkpoint如缺少该字段，动态恢复仍会拒绝；请从0运行reference生成新checkpoint。

驱动器保留基准记录的test_iterations（若历史配置缺失，默认7000/30000，需人工核对），防止评价时序改变动态质量决策。resolution_schedule='1853:4,6177:2,30000:1'会沿用；如同时启用resolution-aware planner，仍由原planner按其配置计算，dry-run仅展示输入参数。

新增3项CPU控制器恢复检查：恢复后下一次扩展与连续运行一致、相同边界不重复执行、开关不匹配拒绝。加上3项preflight共6项通过。尚未完成真实GPU的不中断/恢复R0复现验证。动态延长可能使SPA跨越25k；若新reference的25k时SPA尚未结束，不能把25k后段候选描述为“已完成SPA”的固定拓扑实验，必须先核对实际阶段与点数日志。


## 无 GitHub 网络时离线更新

使用 tags_fixed_budget_offline.zip，可在本地用 scripts/build_fixed_budget_patch.py 从发布分支构建。上传ZIP到AutoDL任意位置，不需要解压或连接GitHub。进入原项目根目录执行：

```bash
cd /root/autodl-tmp/TAGS/TAGSs_bicycle_v14
python /root/autodl-tmp/tags_fixed_budget_offline.zip --target "$PWD"
python -m unittest discover -s tests -p 'test_fixed_budget*.py'
python scripts/run_fixed_budget_sweep.py --baseline "$BASE" --source "$DATA" --depths "$DEPTH" --case R0 --output output/tags_budget_sweep/reference_seed0_offline
# 核对输出后同一命令加 --execute 前台启动。
```

安装器先校验包内SHA256，然后备份所有将覆盖的已有文件为项目根目录下fixed_budget_backup_*.zip，再安装并逐个核验。补丁包含完整experiments Python模块、参数定义、训练入口、驱动器、测试及文档，不包含数据或模型。根目录同名驱动器替换为兼容入口，转发至scripts/中的正式版本。安装不会启动训练。若目录此前已存在，应选择新实验输出名。


## 7000步保存失败：磁盘空间与PLY保存频率

2026-10-30日志：7000步训练和test/train评估成功，在保存PLY时剩余1110 MiB，低于估算1658 MiB而停止。不得通过降低TAGGS_SAVE_PLY_MARGIN_BYTES绕过容量不足。本次修复：固定预算开启时，resolution-aware planner保留显式save_iterations；评估/阶段日志仍按原时间表执行，25k/30k checkpoint不变。普通训练旧行为保持。

服务器先检查：
```bash
df -h . /root/autodl-tmp
du -h --max-depth=2 /root/autodl-tmp 2>/dev/null | sort -h | tail -n 25
find output/tags_budget_sweep/reference_seed0_offline -maxdepth 2 -type f -name 'chkpnt*.pth' -ls
```

建议为本轮预留至少20 GiB，并视checkpoint实际大小增加；这不是容量保证。扩容、转移已备份结果或换用空间足够的挂载盘均可；不要删除基准、数据或未完成checkpoint。如果只有PLY而无完整checkpoint，不符合公平Resume要求，需从0重跑并选择新输出目录。仅出现Saving Gaussians不代表完整checkpoint已保存成功。修复包须重新安装；安装本身也需要备份旧文件的空间。


## 旧PyTorch恢复时报weights_only错误

如果读取25k checkpoint时报TypeError: weights_only is an invalid keyword argument for Unpickler，这是旧torch.load把未知参数转发给pickle造成的版本兼容问题。入口改用experiments/checkpoint_io.py：检测torch.load签名，只在明确支持weights_only时传False，旧版本不传此参数。checkpoint内容和训练超参数不变，不需要重训reference。加载失败的输出目录已创建，重试请选R0_resume_seed0_v2等新目录，保留原失败目录。

3项加载器回归检查覆盖旧签名、新签名完整状态、损坏文件异常不被重试掩盖。尚未在服务器旧PyTorch/GPU上实际恢复验证。


## 2026-11-02：PG1组合候选

根据用户AutoDL输出，R0_resume_seed0_v2为PSNR25.3045673/SSIM0.7859414/LPIPS0.1723897；G2_seed0_retry1为25.3299465/0.7860729/0.1723238，N_final=7206331，N0=7234487，两个budget标志均为true。以上为用户提供的单种子日志，未在本地重新核验原始结果文件、相机尺寸、每视角退化或几何伪影，不构成多种子稳定性结论。

PG1现在支持--case PG1，固定P2+G2=(1.5,0.5)。保持相同reference的25k checkpoint、seed0、原BASE和数据路径；从25k恢复，不能从G2的30k结果继续。优先与G2和R0比较。若组合无额外收益，保留较简单的G2，不把微小差异解释成稳定提升。

```bash
export TORCH_HOME=/root/autodl-fs/torch-cache
export REF="$PWD/output/tags_budget_sweep/reference_seed0_diskfix"
python scripts/run_fixed_budget_sweep.py --baseline "$BASE" --source "$DATA" --depths "$DEPTH" --case PG1 --checkpoint "$REF/chkpnt25000.pth" --output /root/autodl-fs/output/tags_budget_sweep/PG1_seed0
# 核对打印参数后，同一命令添加--execute。运行前检查df -h /root/autodl-fs。
```

PG1尚未进行GPU训练。质量目标仍未达到；后续需要每视角/尺寸/预算核验和预先确定的多个种子验证。官方test参与了本轮筛选，应明确披露。


## 2026-11-03：S0+PG1后期LPIPS减弱单变量

用户日志显示PG1相对G2的25个视角中22个PSNR提高，无视角回退超过0.2dB，但均值增益仅0.0044766dB。实际PSNR倍率1.5、geo_scale0.5均已生效，法线损失为0，hard-view重加权关闭；不能通过修改其max参数构成有效消融。LPIPS为linear，22500起，0到0.006，每10步计算一次。

新增--case PG1L1：保留PG1的PSNR1.5/geo0.5，只将lpips_final_multiplier设置0.5，使用与PSNR相同的25k-27k smoothstep区间。默认1.0不改变旧组。不能把0.006*0.5*1/10简单当作等价全步权重：当前间隔计算语义保持不变。

若继承上述linear配置，则27k原LPIPS权重0.0036，有效0.0018；28.5k原0.0048，有效0.0024；30k原0.006，有效0.003（balanced controller/hard-view multiplier都为1时）。25k倍率仍为1，避免恢复第一步突然减半。CSV增加原始LPIPS权重、倍率、最终有效权重、raw/weighted loss；采样未计算的步不伪造loss值。

仍用原BASE读取N0，从S0_seed0/chkpnt25000.pth恢复，输出建议S0_PG1_L1_seed0。对照是S0_PG1_seed0，不是从它的30k继续训练。只测试此单变量；未有真实GPU结果，不能保证提升或达标。严格约束PSNR>=25.50、SSIM>=0.785、LPIPS<=0.175及预算/同视角尺寸；单次通过后需独立复验。本轮官方测试集已参与筛选，应明确披露。


## 2026-11-03：PG1D1训练GT梯度细节消融

用户所提供GT/render图像均为1237x822。对附件解码后计算PSNR=22.53033408dB，与此前00004渲染的22.53033447吻合（身份仍以原文件为准）。render-GT全图RGB均值差约(-0.03297,-0.03160,-0.01889)，全局通道均值偏差占总平方误差14.58%；以GT灰度梯度80%分位分出的约20%像素占误差41.26%。这些仅为单帧诊断，不修改正式指标，也不证明全场景几何正确、曝光错误或边缘损失必然有效。附件并非19.2dB最差视角的证据。

新增--case PG1D1（输出建议S0_PG1_D1_seed0）：保持PG1的PSNR1.5、geo0.5、LPIPS倍率1。仅启用已有_detail_loss训练GT梯度损失，init=0，start=27000，final=0.01，27k到30k按原实现线性渐入；edge quantile沿用基准。此项是梯度匹配，不是新实现的边缘加权RGB，二者不可混称。基准detail loss必须原本关闭，max_weight须>=0.01，否则拒绝。

仍从S0的同一25k checkpoint恢复，不从30k继续训练，不叠加PG1L1。只有训练相机GT参与loss/mask；测试图仅用于诊断。CSV补记细节raw loss、最终weight、weighted loss和mask fraction。14项既有回归检查通过；新候选另做参数dry-run。尚无GPU或质量验证。如出现更锐利但PSNR/SSIM降低，保留为失败消融，不继续无界增加detail权重。


## 2026-11-03：恢复学习率重复缩放修复

代码审计发现GroupLRScheduler._base_lrs只在内存按param_group对象id缓存，旧v2 checkpoint未保存。恢复优化器后首次apply把已经乘过stage倍率的LR当基准，导致scaling/rotation及f_dc/f_rest/opacity重复缩放；xyz因每步从scheduler计算不受同样问题影响。原配置geo_stage3=.4、color_stage3=.55，意味着受影响组恢复后又额外乘.4/.55。此前R0均值差小于工程阈值不能证明恢复状态完整，已存在确定的状态遗漏。

新checkpoint按optimizer组名保存group_lr_base_lrs，恢复时映射至新组对象。旧v2 checkpoint无此字段时，用gaussians.training_setup(opt)后、optimizer.restore之前捕获的新鲜初始组LR恢复（须保持原优化器配置与原训练一致）。不从已经缩放的checkpoint LR反推。旧实验保留为legacy-resume协议结果；不得与修复后的候选直接混作同一恢复对照。

下一步优先从S0_seed0/chkpnt25000.pth跑R0，输出S0_R0_lrfix_seed0，与原不中断S0对照。先验证恢复；之后PG1如需比较必须以新R0为对照重跑。D1用户指标与PG1几乎一致，不继续增加detail权重。修复经CPU组LR恢复检查，未做GPU连续/恢复一致性验证。


## 2026-11-03：统一post-SPA数量下限协议（可选，默认关闭）

seed1用户轨迹：25k=7202214，28.5k=7198609，29k=7198113，30k=7197180。严格最低数量7198315在29k被跌破，符合每250步几何清理继续删点的代码路径。cfg_args的spa_end_iter可能不含运行时动态覆盖，不可独自作为最终时序证据。

新--budget-policy post-spa-floor显式启用post_spa_budget_floor。保留0.5%门槛，精确整数下限Nmin=N0-N0//200。SPA完成后，几何清理只允许删到Nmin；原保护mask与候选条件保留，候选过多时优先删除其中opacity最低者。既不复制补点，也不末次回填。已低于下限的模型报错，不虚构严格通过。默认legacy不改变旧结果；此规则不约束其他剪枝器，最终预算报告仍必须检查。

几何统计进入training_curves.csv，包含floor、候选数、因floor阻止的数量与实际裁剪数。新报告和experiment_manifest记录budget_policy。规则须对全部种子与对照/候选统一应用；旧报告不得改标签冒充新协议。seed0旧轨迹从未触及下限时预计不受影响，仍需记录/核验，而非直接宣称与新协议一致。

先从seed1原S0的25k完整checkpoint运行R0 --seed 1 --budget-policy post-spa-floor，输出S0_R0_floor_seed1；不是从已低于下限的30k继续。原BASE及N0不变。该比较引入新数量策略，不是原协议的无改动恢复验证。20项CPU回归检查通过；尚未GPU验证策略、质量或泛化。


## 2026-11-03：恢复时必须保留源checkpoint的SPA结构配置

seed1 floor恢复报错时约27k数量7170103，已明显低于下限。失败命令确实未传spa_target_gaussians，因此恢复入口用BASE的默认0覆盖了S0训练的7234487。旧v2只保存SPA张量和final_pruned，未保存结构配置；若seed1动态SPA尚未在25k完成，后续final prune就可能按prune_ratio2及30000上限额外裁剪。上一版仅约束SPA完成后的geometry，未覆盖此配置遗漏。不能去掉下限断言绕过错误，也不能假定seed1与seed0一样25k时已完成SPA。

驱动器现在读取checkpoint同目录cfg_args，只继承SPA/稀疏结构配置，不覆盖新候选的geo/PSNR/LPIPS等损失选择；缺少源cfg_args拒绝恢复。新checkpoint嵌入sparse_config，恢复时先应用它，再恢复adaptive_stage运行时边界，避免把静态cfg_args当作最终SPA时序。打印[Resume SPA]的target/protect/end/final_pruned/points，manifest保存源cfg路径、哈希和继承内容。原BASE仍用于N0和其他基准设置。

应从原S0_seed1/chkpnt25000.pth重跑R0 --seed 1 --budget-policy post-spa-floor，输出S0_R0_floor_seed1_v2。启动命令中必须包含--spa_target_gaussians 7234487，不能从此次失败中间点数继续。此次源配置继承属于恢复修复，下限策略仍是独立的新协议；所有后续对照/种子须统一采用。原始从0训练的S0不受恢复配置遗漏影响；此前恢复候选若SPA已经final_pruned则此SPA终末裁剪遗漏不再触发，但未检查的seed不可默认等价。

4项新增CPU检查覆盖源结构继承、损失权重不被覆盖、缺失配置拒绝、checkpoint结构配置保存恢复及非法覆盖拒绝；共24项通过。GPU/真实seed1轨迹尚待验证。
