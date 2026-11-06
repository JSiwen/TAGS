# perceptual-front / geo-late 实验架构

TAGS 深度图的生成、标定、训练输入与当前单位注意事项见 [`docs/DEPTH_MAP_USAGE.md`](docs/DEPTH_MAP_USAGE.md)。

Bicycle v16 的复现、深度/法线消融、独立验证与运行记录规范见 [`docs/TAGS_EXPERIMENT_OPTIMIZATION_PROTOCOL.md`](docs/TAGS_EXPERIMENT_OPTIMIZATION_PROTOCOL.md)。

## 设计要点

| 阶段 | 迭代区间（默认 30k） | 内容 |
|------|----------------------|------|
| 早期感知 | 1 → ~8000 | LPIPS + PSNR 监督，几何先验关闭 |
| 中期困难视角 | 8000 → 25000 | 仅损失重加权（不改采样），LPIPS 逐步衰减 |
| SPA 稀疏化 | 13000 → 25000 | GaussianSpa（全分辨率 + delay 后启动） |
| 晚期几何 | 26000 → 30000 | depth + normal 先验 warmup（法线仅 cosine，无 TV） |

## 安装

将本包内文件按目录结构覆盖到 TAGSs 仓库根目录（保留相对路径）：

```
TAGSs_core/
  arguments/__init__.py
  experiments/loss_builder.py
  experiments/geo_prior.py
  experiments/training_engine.py
  scripts/run_bicycle_v4_restore.ps1
```

## 训练

```powershell
cd D:\TAGSs\TAGSs_core
.\scripts\run_bicycle_v4_restore.ps1
```

可选环境变量：

```powershell
$env:TAGGS_MODEL = "D:\TAGSs\TAGSs_core\output\bicycle_integrated_perceptual_front_geo_late"
$env:RUN_RENDER_METRICS = "0"   # 跳过训练后 render/metrics
.\scripts\run_bicycle_v4_restore.ps1
```

## 默认输出

`output/bicycle_integrated_perceptual_front_geo_late`

## 关键参数摘要

- LPIPS: init=0.004, start iter=1, interval=8, 逐步衰减至 0
- PSNR loss: init=0.020, final=0.010, start iter=1
- Hard-view: `--enable_hard_view_loss_reweight`, start=8000, max=1.20
- Geo prior: start=26000, warmup_until=28500（SPA 结束后）
- Normal: cosine only, `normal_prior_tv_weight=0.0`
