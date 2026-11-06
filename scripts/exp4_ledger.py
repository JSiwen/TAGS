#!/usr/bin/env python3
"""v2 实验总账 (Ledger) - 每次 (scene, group, seed) 完成后追加一行 JSONL.

设计原则:
  - append-only JSONL, 避免并发写丢失
  - 文件锁 (fcntl) 在 Linux/Mac 上工作, Windows 上退化为 best-effort
  - 集中记录所有实验结果, 是文档自动生成 + wandb 重放 的唯一数据源
  - 同时写一个 compact 的 ledger.csv 方便 Excel/查看

输出:
  output/exp4_stats/exp4_ledger.jsonl     (append-only, 1 行 / run)
  output/exp4_stats/exp4_ledger.csv        (扁平化视图, 每次刷新)

调用方式:
  from scripts.exp4_ledger import record_run
  record_run(scene, group, seed, metrics, config, extra=None)
"""

import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# fcntl 仅在 Linux/Mac 可用, Windows 退化为 best-effort (见文件级注释)
try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_DIR = REPO_ROOT / "output" / "exp4_stats"
DEFAULT_LEDGER_JSONL = DEFAULT_LEDGER_DIR / "exp4_ledger.jsonl"
DEFAULT_LEDGER_CSV = DEFAULT_LEDGER_DIR / "exp4_ledger.csv"

# 进程内写锁 (避免同一进程多次 append 写 CSV 冲突)
# 用 RLock (可重入), 让 record_run 内部调 refresh_csv 不死锁
_LOCK = threading.RLock()
_CSV_FIELDS = [
    "timestamp", "scene", "group", "seed", "phase",
    "psnr", "ssim", "lpips", "num_points",
    "train_exit", "render_exit", "metrics_exit",
    "config_json", "extra_json",
]


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _flatten_config(config: Optional[Dict]) -> Dict:
    """把 config dict 扁平成单层 dict, 复杂值用 repr 存."""
    out = {}
    for k, v in (config or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def record_run(
    scene: str,
    group: str,
    seed: int,
    metrics: Dict,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
    phase: Optional[str] = None,
    ledger_path: Path = DEFAULT_LEDGER_JSONL,
) -> Dict:
    """追加一条 run 结果到 ledger.

    Returns: 写入的 dict (含 timestamp 等自动字段)
    """
    _ensure_dir(ledger_path)
    if phase is None:
        # 从 group_id 推断: 4_2_* / 4_3_* / 4_4_* / 4_5_*
        phase = group.split("_")[0] + "_" + group.split("_")[1] if "_" in group else "unknown"
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "scene": scene,
        "group": group,
        "seed": int(seed),
        "phase": phase,
        "metrics": {
            "psnr": float(metrics["psnr"]) if metrics.get("psnr") is not None else None,
            "ssim": float(metrics["ssim"]) if metrics.get("ssim") is not None else None,
            "lpips": float(metrics["lpips"]) if metrics.get("lpips") is not None else None,
            "num_points": int(metrics["num_points"]) if metrics.get("num_points") is not None else None,
            "train_exit": int(metrics.get("train_exit", 0) or 0),
            "render_exit": int(metrics.get("render_exit", 0) or 0),
            "metrics_exit": int(metrics.get("metrics_exit", 0) or 0),
        },
        "config": _flatten_config(config),
        "extra": extra or {},
    }
    with _LOCK:
        # 写 jsonl
        with open(ledger_path, "a", encoding="utf-8") as f:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (AttributeError, OSError):
                    pass
            try:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (AttributeError, OSError):
                        pass
        # 刷新 CSV 视图
        refresh_csv(ledger_path)
    return entry


def iter_entries(ledger_path: Path = DEFAULT_LEDGER_JSONL) -> Iterable[Dict]:
    """读 ledger 全部条目, 按时间排序."""
    if not ledger_path.is_file():
        return iter([])
    out = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out.sort(key=lambda x: (x.get("timestamp", ""), x.get("scene", ""), x.get("group", ""), x.get("seed", 0)))
    return iter(out)


def refresh_csv(
    ledger_path: Path = DEFAULT_LEDGER_JSONL,
    csv_path: Path = DEFAULT_LEDGER_CSV,
):
    """从 JSONL 重新生成 CSV 视图."""
    _ensure_dir(csv_path)
    with _LOCK:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for entry in iter_entries(ledger_path):
                row = {
                    "timestamp": entry.get("timestamp"),
                    "scene": entry.get("scene"),
                    "group": entry.get("group"),
                    "seed": entry.get("seed"),
                    "phase": entry.get("phase"),
                    "psnr": entry.get("metrics", {}).get("psnr"),
                    "ssim": entry.get("metrics", {}).get("ssim"),
                    "lpips": entry.get("metrics", {}).get("lpips"),
                    "num_points": entry.get("metrics", {}).get("num_points"),
                    "train_exit": entry.get("metrics", {}).get("train_exit"),
                    "render_exit": entry.get("metrics", {}).get("render_exit"),
                    "metrics_exit": entry.get("metrics", {}).get("metrics_exit"),
                    "config_json": json.dumps(entry.get("config", {}), ensure_ascii=True) if entry.get("config") else "",
                    "extra_json": json.dumps(entry.get("extra", {}), ensure_ascii=True) if entry.get("extra") else "",
                }
                writer.writerow(row)


def list_scenes_groups_seeds(ledger_path: Path = DEFAULT_LEDGER_JSONL) -> Dict:
    """汇总 ledger 包含的 (scene, group, seed) 三元组, 用于 doc_writer."""
    entries = list(iter_entries(ledger_path))
    out = {}
    for e in entries:
        s = e.get("scene", "")
        g = e.get("group", "")
        sd = e.get("seed", 0)
        out.setdefault(s, {}).setdefault(g, set()).add(sd)
    return out


def main():
    """CLI: 刷新 CSV 视图 / 列出当前 ledger 状态."""
    import argparse
    parser = argparse.ArgumentParser(description="v2 experiment ledger")
    parser.add_argument("--refresh-csv", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_JSONL))
    args = parser.parse_args()

    ledger_path = Path(args.ledger)

    if args.refresh_csv:
        refresh_csv(ledger_path)
        print(f"refreshed: {DEFAULT_LEDGER_CSV}")
    if args.list:
        if not ledger_path.is_file():
            print("ledger is empty")
            return
        entries = list(iter_entries(ledger_path))
        print(f"total entries: {len(entries)}")
        scenes_groups = list_scenes_groups_seeds(ledger_path)
        for scene in sorted(scenes_groups):
            for group in sorted(scenes_groups[scene]):
                seeds = sorted(scenes_groups[scene][group])
                print(f"  {scene:10s}  {group:30s}  seeds={seeds}")


if __name__ == "__main__":
    main()
