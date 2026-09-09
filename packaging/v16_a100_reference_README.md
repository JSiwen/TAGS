# 3080 v16 参考结果

本目录来自本机已跑完的**当前最优**实验，供 A100 对照配置与调度，**不是**全分辨率可比数字。

| 项 | 值 |
|---|---|
| Case | `bicycle_v16_3080_resolution_aware` |
| 协议 | `images_4`，`resolution=800`，30k，`--eval --llffhold 8` |
| 代码 | RAS planner + adaptive stage controller |
| 推荐 checkpoint | `ours_28500`（PSNR / SSIM / LPIPS 三点同时最优） |
| PSNR / SSIM / LPIPS | **26.9117 / 0.837794 / 0.135433** |
| Final `ours_30000` | 26.9058 / 0.837601 / 0.135439 |
| planner densify_until | 7200（scale≈1.0） |
| runtime override | iter 7200 `capacity_extension` → densify 9200，SPA 11200–19200 |
| 训练 | RTX 3080，2026-08-31 |

相对上一版 3080 最佳 v15 E1b densify7k `ours_28500`（26.6930 / 0.828843 / 0.149942）：

| 指标 | 差值 |
|---|---|
| PSNR | **+0.2187 dB** |
| SSIM | **+0.00895** |
| LPIPS | **−0.01451** |

A100 全分辨率结果请单独记录，不要与上表混排。那边的对照应是 v14 monitor（25.548 / 0.775 / 0.187）和失败的 v15 E1b A100（25.340 / 0.760 / 0.217）。
