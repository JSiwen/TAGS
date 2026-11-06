#!/usr/bin/env python3
"""v2 wandb 集成 shim (可选, 延迟导入).

设计原则:
  - 默认禁用, 仅当环境变量 WANDB_ENABLED=1 启用, 不污染现有流程
  - 延迟 import: 没装 wandb 也不会报错, 仅打印 warning
  - 与 exp4_runner.py 集成: 每个 (scene, group, seed) run 完成后自动 log

启用方法:
  set WANDB_ENABLED=1
  set WANDB_PROJECT=tags-ablation
  set WANDB_ENTITY=your_username
  set WANDB_RUN_GROUP=exp4_5_Full  (可选, 用于 wandb dashboard 分组)
  set WANDB_TAGS=bicycle,full_tags,5_seed  (可选, 逗号分隔)

用法 (在 exp4_runner.py 末尾自动调用, 不需要用户手动调用):
  from scripts.exp4_wandb_logger import log_run_result
  log_run_result(scene, group, seed, metrics_dict, config_dict)
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

# 全局缓存 wandb 模块
_WANDB = None
_WANDB_INITIALIZED = False
_WANDB_RUN = None


def _is_enabled() -> bool:
    return os.environ.get("WANDB_ENABLED", "0").strip() in ("1", "true", "True", "yes")


def _get_wandb():
    """延迟导入 wandb, 避免被 cwd 下的 wandb/ 日志目录 shadow 成空 namespace package.

    repo 根目录 D:\\TAGSs\\TAGSs_core\\wandb 存放 wandb 落地的 run-XXX/ 目录,
    Python 3.3+ implicit namespace package 会让 `import wandb` 拿到这个空目录,
    而不是真正安装的 wandb 包 (hasattr(wandb,'init') == False).
    修复: 先验证 init 存在, 不存在就 pop 出来重 import, 再不行就给清晰的错误.
    """
    global _WANDB
    if _WANDB is not None:
        return _WANDB
    if not _is_enabled():
        return None
    try:
        import wandb  # type: ignore
        # Sanity check: 真正的 wandb 包必须有 .init
        if not hasattr(wandb, "init"):
            # cwd 下的 wandb/ 目录 (无 __init__.py) 把真包 shadow 了
            cwd_wandb = (Path.cwd() / "wandb").resolve()
            if cwd_wandb.is_dir() and not (cwd_wandb / "__init__.py").is_file():
                warnings.warn(
                    f"[wandb] shadowed by namespace pkg at {cwd_wandb}. "
                    "Popping from sys.modules and retrying import."
                )
                sys.modules.pop("wandb", None)
                # 也把 sys.path 上指向该目录的 entry 摘掉
                sys.path[:] = [
                    p for p in sys.path
                    if Path(p).resolve() != cwd_wandb
                ]
                import wandb as wandb_retry  # type: ignore
                wandb = wandb_retry
                if not hasattr(wandb, "init"):
                    raise ImportError(
                        "wandb still has no .init after removing shadow. "
                        "Install with: pip install wandb"
                    )
            else:
                raise ImportError(
                    f"wandb module has no .init (looks like a namespace stub). "
                    "Install with: pip install wandb"
                )
        _WANDB = wandb
        return wandb
    except ImportError as e:
        warnings.warn(
            f"[wandb] WANDB_ENABLED=1 but wandb unavailable: {e}. "
            "Install with: pip install wandb"
        )
        return None


def _ensure_run(
    config: Dict,
    run_name: str,
    *,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[list] = None,
):
    """确保 wandb run 已初始化, 幂等.

    Args:
        project: wandb project (默认 $WANDB_PROJECT or "tags-ablation")
        entity: wandb entity (默认 $WANDB_ENTITY or None)
        group: wandb run group (默认 $WANDB_RUN_GROUP or None)
        tags: list of tags (默认 $WANDB_TAGS)
    """
    global _WANDB_INITIALIZED, _WANDB_RUN
    wandb = _get_wandb()
    if wandb is None:
        return None
    # 每次都重 init: 0.18.7 finish 后 wandb.run 会变 None, 用它判断更稳.
    if _WANDB_INITIALIZED and getattr(wandb, "run", None) is not None:
        return _WANDB_RUN

    # 优先用函数参数, 再 fallback env, 最后默认值
    proj = project or os.environ.get("WANDB_PROJECT", "tags-ablation")
    ent = entity if entity is not None else (os.environ.get("WANDB_ENTITY", "") or None)
    grp = group if group is not None else (os.environ.get("WANDB_RUN_GROUP", "") or None)
    if tags is None:
        tags_env = os.environ.get("WANDB_TAGS", "")
        tags = [t.strip() for t in tags_env.split(",") if t.strip()] or None

    _WANDB_RUN = wandb.init(
        project=proj,
        entity=ent,
        name=run_name,
        group=grp,
        tags=tags,
        config=config,
        reinit=True,  # 允许多次调用 init (每个 seed 一个 run)
    )
    _WANDB_INITIALIZED = True
    return _WANDB_RUN


def log_run_result(
    scene: str,
    group: str,
    seed: int,
    metrics: Dict,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> bool:
    """把一次 run 的结果 log 到 wandb.

    Args:
        scene: 场景名 (bicycle/garden/bonsai/room)
        group: 实验组 ID (4_2_C_Reference / 4_5_F_Full_TAGS 等)
        seed: 随机种子
        metrics: {psnr, ssim, lpips, num_points, train_exit, render_exit, metrics_exit}
        config: 实验配置 dict (会被 log 到 wandb.config)
        extra: 其它要 log 的标量 (例如 geo_eval/pearson)

    Returns:
        True if logged, False if wandb 不可用
    """
    wandb = _get_wandb()
    if wandb is None:
        return False

    run_name = f"{scene}__{group}__seed{seed}"
    cfg = dict(config or {})
    cfg.update({
        "scene": scene,
        "group": group,
        "seed": seed,
    })
    run = _ensure_run(cfg, run_name)
    if run is None:
        return False

    log_dict = {
        "seed": seed,
        "scene": scene,
        "group": group,
    }
    if "psnr" in metrics:
        log_dict["psnr"] = float(metrics["psnr"])
    if "ssim" in metrics:
        log_dict["ssim"] = float(metrics["ssim"])
    if "lpips" in metrics:
        log_dict["lpips"] = float(metrics["lpips"])
    if "num_points" in metrics and metrics["num_points"] is not None:
        log_dict["num_points"] = int(metrics["num_points"])
    if "train_exit" in metrics:
        log_dict["train_exit"] = int(metrics["train_exit"])
    if extra:
        for k, v in extra.items():
            log_dict[k] = v

    wandb.log(log_dict)
    # 每个 seed 完成后 finish, 避免长 run 卡住
    wandb.finish()
    # 清全局状态, 防止下次 _ensure_run 因 reinit 撞墙
    global _WANDB_INITIALIZED, _WANDB_RUN
    _WANDB_INITIALIZED = False
    _WANDB_RUN = None
    return True


# ============================================================
#  v2 实时曲线 (v3 design, 2026-10-16)
#  exp4_live_to_wandb.py 读 train_iter_metrics.jsonl,
#  把 stats.* 里的指标以 iteration 为 step 推送到 wandb.
#  设计目标: 不侵入 train_optimized.py, 仅观察 live_logs/ 即可.
# ============================================================

def init_live_run(scene: str, group: str, seed: int,
                  config: Optional[Dict] = None,
                  run_name: Optional[str] = None) -> Optional[object]:
    """为单次 (scene, group, seed) 启动一个 wandb run, 准备接收实时 metrics.

    Args:
        scene / group / seed: 标识这次 run
        config: dict, 会被写入 wandb.config (可写 cfg_args / 调度参数)
        run_name: 自定义 run name, 默认 "{scene}__{group}__seed{seed}_live"
    """
    wandb = _get_wandb()
    if wandb is None:
        return None
    if run_name is None:
        run_name = f"{scene}__{group}__seed{seed}_live"
    cfg = dict(config or {})
    cfg.update({"scene": scene, "group": group, "seed": seed, "live": True})
    return _ensure_run(cfg, run_name)


def log_live_iter(payload: Dict, step: int) -> bool:
    """把 train_iter_metrics.jsonl 的一行 stats 推到 wandb (以 step=iteration).

    Args:
        payload: 一行 jsonl (必须有 iteration 和 stats)
        step: wandb 的 x 轴, 用 iteration
    """
    wandb = _get_wandb()
    if wandb is None:
        return False
    if getattr(wandb, "run", None) is None:
        return False
    stats = payload.get("stats") or {}
    log_dict = {
        "train/loss": float(payload.get("total_loss", 0.0)),
        "train/l1": float(stats.get("train/l1", 0.0)),
        "train/ssim": float(stats.get("train/ssim", 0.0)),
        "train/psnr": float(stats.get("train/psnr", 0.0)),
        "train/lpips_proxy": float(stats.get("train/lpips_proxy", 0.0)),
        "scene/total_points": float(stats.get("scene/total_points", 0.0)),
        "scene/budget_utilization": float(stats.get("scene/budget_utilization", 0.0)),
        "train/resolution_scale": float(stats.get("train/resolution_scale", 0.0)),
        "train/fullres_ready": float(stats.get("train/fullres_ready", 0.0)),
    }
    # GeoPrior (后期介入 26000+, 早期 weight=0)
    if stats.get("geo/depth_weight"):
        log_dict["geo/depth_weight"] = float(stats["geo/depth_weight"])
        log_dict["geo/depth_loss"] = float(stats.get("geo/depth_loss", 0.0))
    if stats.get("geo/normal_weight"):
        log_dict["geo/normal_weight"] = float(stats["geo/normal_weight"])
        log_dict["geo/normal_loss"] = float(stats.get("geo/normal_loss", 0.0))
    # LPIPS
    if "train/lpips_loss" in stats:
        log_dict["train/lpips_loss"] = float(stats["train/lpips_loss"])
    wandb.log(log_dict, step=step)
    return True


def finish_live_run() -> bool:
    """结束实时 run (主 run 结束后调用)."""
    wandb = _get_wandb()
    if wandb is None:
        return False
    if getattr(wandb, "run", None) is None:
        return False
    wandb.finish()
    # 清全局状态, 防止下次 reinit 撞墙
    global _WANDB_INITIALIZED, _WANDB_RUN
    _WANDB_INITIALIZED = False
    _WANDB_RUN = None
    return True


def main():
    """CLI 入口: 手动 log 一条结果 (主要用于 retry / 补 log)."""
    import argparse
    parser = argparse.ArgumentParser(description="手动 log 一条结果到 wandb")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--psnr", type=float)
    parser.add_argument("--ssim", type=float)
    parser.add_argument("--lpips", type=float)
    parser.add_argument("--num-points", type=int)
    parser.add_argument("--ledger", default="output/exp4_stats/exp4_ledger.jsonl",
                        help="也可直接读 ledger 一行 jsonl 来 log")
    args = parser.parse_args()

    if args.ledger and Path(args.ledger).is_file():
        # 读 ledger 找匹配条目
        with Path(args.ledger).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if (entry.get("scene") == args.scene
                        and entry.get("group") == args.group
                        and entry.get("seed") == args.seed):
                    ok = log_run_result(
                        scene=args.scene, group=args.group, seed=args.seed,
                        metrics=entry.get("metrics", {}),
                        config=entry.get("config", {}),
                        extra=entry.get("extra"),
                    )
                    print(f"logged: ok={ok}")
                    return
        print(f"NOT FOUND: {args.scene} {args.group} seed={args.seed}")
        return

    metrics = {}
    if args.psnr is not None:
        metrics["psnr"] = args.psnr
    if args.ssim is not None:
        metrics["ssim"] = args.ssim
    if args.lpips is not None:
        metrics["lpips"] = args.lpips
    if args.num_points is not None:
        metrics["num_points"] = args.num_points
    ok = log_run_result(args.scene, args.group, args.seed, metrics)
    print(f"logged: ok={ok}")


if __name__ == "__main__":
    main()
