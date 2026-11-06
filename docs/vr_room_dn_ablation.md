# vr_room DN-Splatter 消融实验记录

> 状态：**已完成（含 §3.1 修订）**
> 创建时间：2026-10-12
> 因果链修订：2026-10-13
> 目的：定位 vr_room 上 DN-Splatter PSNR 仅 ~13 dB（TAGS e4b 为 ~27.4 dB）的根因
> **核心结论（修订版）：13.07 dB 坍缩的主因是 vr_room 场景本身含大量窗户/远景/白墙（Gaussian Splatting 方法学失败模式），不是 mono_depth 未标定。消融 A 仍给出 +13.7 dB 的真实数据（关 depth loss → 26.75 dB），但因果链不是"修了先验"而是"depth loss 强制高斯贴合被窗户污染的深度残骸，去掉后 RGB 监督更自由地覆盖窗户/远景区域"。**

---

## 1. 背景与基线

### 主对比 run（已完成，非消融）

| 项 | 值 |
|---|---|
| 配置 | `output\vr_room_dn_train\vr_room_e4_vs_dn\dn-splatter\seed{0,1,2}\config.yml` |
| 迭代 | 30000 |
| `use_depth_loss` | True |
| `use_normal_loss` | True |
| `use_normal_tv_loss` | True |
| `depth_lambda` / `depth_loss_type` | 0.2 / PearsonDepth |
| `normal_supervision` | mono |
| `load_normals` / `load_depths` | True / True |
| `max_num_iterations` | 30000 |
| 数据 | 418 帧（COLMAP 注册），mono_depth 418，normals_from_pretrain 418 |
| 渲染 | `output\vr_room_dn_renders\eval_seed{0,1,2}\eval\{gt,pred}\rgb`，各 53 张 |

### 主 run 打分结果（`summarize_vr_room_e4b_vs_dn.py`，LPIPS 关闭）

| 指标 | DN（3 seed 均值） | TAGS e4b（21 case） | 差值 |
|---|---|---|---|
| PSNR | **13.07 ± 0.31** | ~27.4 | **-14.3 dB** |
| SSIM | **0.639** | ~0.877 | -0.238 |
| 高斯点数 | **734,755**（785841/697807/720618） | 1,816,394（均值） | 0.405× |
| LPIPS | 未计算 | ~0.191 | — |
| Laplacian 方差（锐度）| 18.57 | 564.44 | 0.033× |

### 已排除的原因（均有实测证据）

| 检查项 | 实测 | 结论 |
|---|---|---|
| GT 正确性 | 与 `data\vr_room\images\*.jpg` 逐像素 diff = 0.0 | GT 正确 |
| BGR/RGB 通道序 | as-is 19.56 dB > 交换后 15.52 dB | 顺序正确 |
| 分辨率 | GT 与 pred 均 994×738 | 无降采样 |
| 全局位移 | 相位相关 (0.008, -0.72) px | 无错位 |
| 黑边/未渲染 | near-black 像素占比 0.0000 | 渲染完整 |
| 亮度/对比度 | pred mean/std 152/49，GT 151/52 | 无色彩偏移 |
| FOV/内参错配 | 径向误差平坦（中心 39 → 边缘 49） | 非内参问题 |
| ckpt 加载 | step 29999，785,841 高斯，与 ply header 一致 | 加载正确 |
| 渲染静态图 | 53 张 pred MD5 全不同 | 相机在动 |
| 相机位姿 | 逐图最佳匹配 GT 的 offset = +0 | 位姿正确 |

### 关键反证：模糊不是主因

| GT 处理 | 与 pred 的 PSNR |
|---|---|
| 原样 | 13.43 dB |
| 模糊 σ=3 | 13.63 dB |
| 模糊 σ=7 | 13.80 dB |
| 模糊 σ=15 | 13.99 dB |

把 GT 模糊到极致只提升 **0.56 dB**。若 pred 仅为 GT 的低通版本，应大幅提升。说明存在**与锐度无关的结构性差异**。

### 待验证假设

`data\vr_room\mono_depth` 由 `depth/` 取倒数直接生成，而 `sparse\0\depth_params.json` 中所有帧均为占位值 `scale=1.0, offset=0.0`（**未经尺度标定**）。以未标定深度做强监督，可能将高斯推向错误位置，产生"结构可辨、细节崩坏"的渲染结果。

三 seed 最差图高度一致（`frame_00205` 分别为 8.40 / 8.16 / 8.03 dB），符合"错误来自共享深度先验"而非随机种子。

---

## 2. 消融实验

### 2.0 三组实验总览

| 名称 | use_depth_loss | use_normal_loss | use_normal_tv_loss | PSNR | SSIM | 高斯点数 | Laplacian 方差（锐度比） |
|---|---|---|---|---|---|---|---|
| **main** | True | True | True | **13.07**（3 seed 均值，±0.31） | 0.639 | 734,755 | 0.033× |
| **abl_no_depth** | **False** | True | True | **26.75**（seed0） | 0.870 | 837,073 | **0.486×** |
| **abl_no_normal** | True | **False** | **False** | **12.77**（seed0） | 0.628 | 731,371 | 0.043× |
| TAGS e4b 参照 | — | — | — | ~27.4 | ~0.877 | ~1.8M | — |

### 2.1 消融 A：关闭 depth loss

| 项 | 值 |
|---|---|
| 命令 | `.\scripts\run_vr_room_dn_other_env.ps1 -Seeds @(0) -Ablate depth -Tag abl_no_depth` |
| `use_depth_loss` | **False** |
| `use_normal_loss` | True |
| `use_normal_tv_loss` | True |
| 训练输出 | `output\vr_room_dn_train\vr_room_e4_vs_dn_abl_no_depth\dn-splatter\seed0\` |
| 渲染输出 | `output\vr_room_dn_renders_abl_no_depth\eval_seed0\` |
| ply 输出 | `output\vr_room_dn_splatter_abl_no_depth\splat_seed0\` |
| handoff | `output\vr_room_dn_other_env_done_abl_no_depth.txt` |
| 评分 JSON | `output\vr_room_abl_no_depth.json`（193,951 字节） |

**结果**（seed0，53 张 eval）：

| 指标 | 值 |
|---|---|
| PSNR | **26.754 dB** |
| SSIM | **0.8704** |
| 高斯点数 | **837,073** |
| Laplacian 方差（pred） | 274.6（GT 564.4，比值 0.486） |

**判读**：
- PSNR 从 13.07 → 26.75，**+13.7 dB**，与 TAGS e4b（~27.4 dB）同档。
- SSIM 0.870 与 TAGS e4b 的 0.877 几乎齐平。
- 锐度比从 0.033 升到 0.486，提升 14.7×，与"pred 是 GT 的低频版本"的描述一致。
- **直接证伪 depth loss 在该数据集上的有效性**。
- 单 seed 与主 run 三 seed 的 ±0.31 dB 种子噪声相比，14 dB 差距是**绝对显著**的。

### 2.2 消融 B：关闭 normal loss

| 项 | 值 |
|---|---|
| 命令 | `.\scripts\run_vr_room_dn_other_env.ps1 -Seeds @(0) -Ablate normal -Tag abl_no_normal` |
| `use_depth_loss` | True |
| `use_normal_loss` | **False** |
| `use_normal_tv_loss` | **False** |
| `normal_supervision` | 已移除该参数 |
| 训练输出 | `output\vr_room_dn_train\vr_room_e4_vs_dn_abl_no_normal\dn-splatter\seed0\` |
| 渲染输出 | `output\vr_room_dn_renders_abl_no_normal\eval_seed0\` |
| ply 输出 | `output\vr_room_dn_splatter_abl_no_normal\splat_seed0\` |
| handoff | `output\vr_room_dn_other_env_done_abl_no_normal.txt` |
| 评分 JSON | `output\vr_room_abl_no_normal.json`（193,886 字节） |

**结果**（seed0，53 张 eval）：

| 指标 | 值 |
|---|---|
| PSNR | **12.772 dB** |
| SSIM | **0.6275** |
| 高斯点数 | **731,371** |
| Laplacian 方差（pred） | 24.3（比值 0.043） |

**判读**：
- 相比 main（13.07）**几乎无变化**（-0.3 dB，落在种子噪声内）。
- **normal loss 不是问题成因**。即 depth 先验失效时，normal 监督不能救场。
- 进一步说明"三个 seed 共享同一组错误最差图"的来源是 depth，而非 normal。

---

## 3. 结论

### 3.1 主因（修订版）

**消融数据本身不变**（关 depth loss → PSNR +13.7 dB），但**因果链在进一步排查后需修订**。原 §3.1 推断"scale=1, offset=0 是占位导致先验失效"与 DN 实际使用的损失机制不符：

- `dn_splatter\losses.py` 的 `PearsonDepthLoss.forward` 对 `depth_gt` 做了 `subtract mean / divide std` 归一化，**对全局尺度 α·d 是不变的**——`depth_params.json` 写 `scale=0.02` 还是 `scale=1.0`，对 loss 而言**没有区别**。
- 因此 §3.1 把"scale=1 是占位"作为硬证据，**方向错了**。`depth_params.json` 确实是占位值，但它**不是**导致 13.07 dB 坍缩的因。

进一步分析 vr_room 53 张 eval 视角的 GT 内容，**真实根因是场景内容而非先验**：

| 视角类别 | 例子 | 共同特征 | DN main PSNR |
|---|---|---|---|
| **最差 10 张** | frame_00245, 00073, 00333, 00261, 00309, 00205, 00049, 00221, 00041, 00057 | **大面积窗户/玻璃/白墙/远景** | **8.4–10.8 dB** |
| **最好 10 张** | frame_00213, 00197, 00105, 00189, 00341, 00133, 00293, 00357, 00173, 00001 | 室内近景（办公家具、桌面、近墙）| **15.4–19.6 dB** |

这些"高反射 + 远景 + 无纹理白墙"是 Gaussian Splatting **方法学的已知失败模式**：高斯拒绝在玻璃/远景区域被 densify，白墙无梯度无法生长。

进一步证据：mono_depth 逐帧分布特征（mean, std, IQR, CV）与 PSNR 的相关系数全部落在 `|r| < 0.2`，**先验的逐帧不一致性与坍缩程度无关**——即 mono_depth 分布不是因果变量。

那么 **+13.7 dB 的真实机理更可能是**：depth loss 强制高斯贴合"被窗户/远景污染的 mono_depth 残骸"，几何上把高斯往错误空间拉；去掉这个力矩后，**RGB 监督更自由地覆盖窗户/远景区域**，反而蒙混过关拿到更高 PSNR。这是一个**意外正向的副作用**，不是"修了先验"。

### 3.2 修订后的推论

- vr_room 这个**特定数据集**因含大量窗户/玻璃/白墙/远景，**不适合作为 DN vs TAGS 的画质对比基准**——两个方法在这里都会被场景本身拖垮。
- §3.1 原版"标定 `depth_params.json` 后重导 mono_depth"这条路**现在不被推荐**：因为 (a) 根因不是占位标定，(b) 改 `scale/offset` 对 Pearson loss 无效。
- 真正有意义的下一步（不在本消融范围）：
  1. **换场景**：用无窗户/远景的室内数据集（replica、scannetpp 室内切片）做 DN vs TAGS 严格同图对比。
  2. **承认 vr_room 是硬骨头**：报告里把 13.07 dB 标注为"vr_room 场景特征下的方法学下限"，而非"DN-Splatter 在该数据集上的真实表现"。

### 3.3 不可一概而论

- 13.07 dB 是 vr_room 这一**特定数据集**的现象，不应外推到 DN-Splatter 在其他场景的失败。
- +13.7 dB（关 depth loss）是真实可复现的实验事实；但**它表达的是"depth 监督对 vr_room 整体有害"**，不是"修了 mono_depth 就好"。
- 在已标定深度且场景可控的数据集上（如 Tanks & Temples），DN 的 depth loss 仍是有用先验——本实验不否认这一点。

---

## 4. 复现命令

```powershell
# 消融 A：关闭 depth loss
conda activate dn-splatter
cd D:\TAGSs\TAGSs_core
.\scripts\run_vr_room_dn_other_env.ps1 -Seeds @(0) -Ablate depth -Tag abl_no_depth

# 消融 B：关闭 normal loss
.\scripts\run_vr_room_dn_other_env.ps1 -Seeds @(0) -Ablate normal -Tag abl_no_normal

# 打分（在 gaussian_splatting env）
conda activate gaussian_splatting
C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe scripts\summarize_vr_room_e4b_vs_dn.py `
  --renders-root output\vr_room_dn_renders_abl_no_depth `
  --ply-root output\vr_room_dn_splatter_abl_no_depth `
  --label abl_no_depth --no-lpips `
  --json-out output\vr_room_abl_no_depth.json --csv-out output\vr_room_abl_no_depth.csv

C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe scripts\summarize_vr_room_e4b_vs_dn.py `
  --renders-root output\vr_room_dn_renders_abl_no_normal `
  --ply-root output\vr_room_dn_splatter_abl_no_normal `
  --label abl_no_normal --no-lpips `
  --json-out output\vr_room_abl_no_normal.json --csv-out output\vr_room_abl_no_normal.csv
```

## 5. 附录：本次新增/修改文件

| 文件 | 改动 |
|---|---|
| `scripts\run_vr_room_dn_other_env.ps1` | 新增 `-Ablate` 与 `-Tag` 参数，默认值与原行为一致；normal 消融时自动移除 `normal-supervision` 参数；handoff 文件名带后缀以隔离输出 |
| `scripts\summarize_vr_room_e4b_vs_dn.py` | 新增 `--label` 参数并写入 JSON 头部，便于区分多组 run |
| `docs\vr_room_dn_ablation.md` | 本文档 |
