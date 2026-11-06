#!/usr/bin/env python3
"""v3 live-metrics → wandb watcher (2026-10-16).

设计目的:
  train_optimized.py 不调 wandb.log() (避免侵入式改源码).
  它已经在 live_logs/train_iter_metrics.jsonl 里每行写一条 metric.
  这个 watcher 监控这个文件, 把新行增量推到一个新的 wandb run
  (run name = "<scene>__<group>__seed<seed>_live"), step=iteration.

用法:
  # 1) 确保环境变量已设 (与主 run 共享)
  set WANDB_ENABLED=1
  set WANDB_PROJECT=tags-ablation
  set WANDB_ENTITY=jsweni-neu
  set WANDB_RUN_GROUP=exp4v3_A_method_comparison

  # 2) 启动 watcher (后台, 跟主训练同一个 venv)
  python scripts/exp4_live_to_wandb.py \
      --jsonl output/exp4v3_A_bicycle/bicycle/A.A4_Full_TAGS/train/seed0/live_logs/train_iter_metrics.jsonl \
      --scene bicycle --group A.A4_Full_TAGS --seed 0

特性:
  - 启动时会把 jsonl 已有的行全部推 (从 iteration 0 开始), 不漏曲线.
  - 每 N 秒 poll 一次 (默认 30s, 与训练 iter 间隔匹配).
  - 主动检测: 文件大小没变 + 文件被 truncated (变小) 就重读.
  - 主 run 结束 (jsonl 停止增长 N 次 poll 后) 自动 finish wandb.
  - crash-safe: --cursor 记录行号, 重启能续传.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# repo 根 -> import scripts.exp4_wandb_logger
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.exp4_wandb_logger import init_live_run, log_live_iter, finish_live_run  # type: ignore  # noqa: E402


def read_payloads(path: Path) -> List[Dict]:
    """读 jsonl, 返回所有 payload. 损坏行跳过."""
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run_watcher(
    jsonl: Path,
    scene: str,
    group: str,
    seed: int,
    config: Optional[Dict] = None,
    poll_interval: float = 30.0,
    stall_polls_before_finish: int = 3,
    cursor_file: Optional[Path] = None,
) -> None:
    """主循环: 读 jsonl -> log_live_iter."""
    if cursor_file is None:
        cursor_file = jsonl.with_suffix(".wandb_cursor")

    # 1) 启动 wandb run
    run = init_live_run(scene, group, seed, config=config)
    if run is None:
        print(f"[watcher] wandb init failed (WANDB_ENABLED={os.environ.get('WANDB_ENABLED')!r})")
        return
    print(f"[watcher] wandb run started: {run.name}  url={run.url}")

    # 2) 读 cursor (上次处理到的行号)
    last_count = 0
    if cursor_file.is_file():
        try:
            last_count = int(cursor_file.read_text(encoding="utf-8").strip() or "0")
            print(f"[watcher] resuming from line {last_count}")
        except (ValueError, OSError):
            last_count = 0

    # 3) poll loop
    last_size = -1
    stall_count = 0
    while True:
        # 检测文件被 truncated (变小) -> 重置
        try:
            cur_size = jsonl.stat().st_size
        except FileNotFoundError:
            cur_size = 0
        if cur_size < last_size:
            print(f"[watcher] file truncated ({last_size} -> {cur_size}), reset")
            last_count = 0
        last_size = cur_size

        # 读并推
        payloads = read_payloads(jsonl)
        if len(payloads) > last_count:
            new_payloads = payloads[last_count:]
            ok = 0
            for p in new_payloads:
                step = int(p.get("iteration", 0))
                if log_live_iter(p, step=step):
                    ok += 1
            cursor_file.write_text(str(len(payloads)), encoding="utf-8")
            last_count = len(payloads)
            print(f"[watcher] {jsonl.name}: pushed {ok} new iters (total {last_count}, last_iter={payloads[-1].get('iteration')})")
            stall_count = 0
        else:
            # 文件大小没变, 但仍可能 append (训练停在 30k 时)
            if cur_size > 0:
                stall_count += 1
                if stall_count >= stall_polls_before_finish:
                    print(f"[watcher] no growth for {stall_count} polls, finish wandb run")
                    finish_live_run()
                    return

        time.sleep(poll_interval)


def main():
    p = argparse.ArgumentParser(description="Watch train_iter_metrics.jsonl and push to wandb.")
    p.add_argument("--jsonl", required=True, type=Path,
                   help="live_logs/train_iter_metrics.jsonl 路径")
    p.add_argument("--scene", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cfg-args", type=Path, default=None,
                   help="可选, 读 cfg_args 文件并写入 wandb.config")
    p.add_argument("--poll-interval", type=float, default=30.0,
                   help="poll 间隔秒数 (默认 30)")
    p.add_argument("--stall-polls", type=int, default=3,
                   help="连续 N 次没增长就 finish (默认 3)")
    args = p.parse_args()

    cfg: Optional[Dict] = None
    if args.cfg_args and args.cfg_args.is_file():
        try:
            # cfg_args 实际是 Namespace repr. 简化: 读整个文件成字符串作为 description.
            raw = args.cfg_args.read_text(encoding="utf-8")
            cfg = {"cfg_args_text": raw[:1000]}  # truncate 防超 1MB
        except OSError as e:
            print(f"[watcher] failed to read cfg_args: {e}")

    run_watcher(
        jsonl=args.jsonl,
        scene=args.scene,
        group=args.group,
        seed=args.seed,
        config=cfg,
        poll_interval=args.poll_interval,
        stall_polls_before_finish=args.stall_polls,
    )


if __name__ == "__main__":
    main()
