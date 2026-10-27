#!/usr/bin/env python3
"""Read latest training progress from model output directory."""

import argparse
import json
import re
import sys
from pathlib import Path


def latest_saved_iteration(model_dir: Path):
    best = 0
    pc_root = model_dir / "point_cloud"
    if pc_root.is_dir():
        for child in pc_root.iterdir():
            m = re.fullmatch(r"iteration_(\d+)", child.name)
            if m:
                best = max(best, int(m.group(1)))
    for path in model_dir.glob("chkpnt*.pth"):
        m = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def latest_tensorboard_step(model_dir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None, None

    event_files = sorted(model_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None, None

    ea = EventAccumulator(str(model_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    step = None
    loss = None
    points = None

    if "train/total_loss" in tags:
        scalars = ea.Scalars("train/total_loss")
        if scalars:
            step = scalars[-1].step
            loss = scalars[-1].value
    if "scene/total_points" in tags:
        pts = ea.Scalars("scene/total_points")
        if pts:
            points = pts[-1].value
    return step, loss, points


def latest_live_loss(model_dir: Path):
    log_path = model_dir / "live_logs" / "train_iter_metrics.jsonl"
    if not log_path.is_file():
        return None
    last_row = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last_row = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last_row


def stage_snapshot_count(model_dir: Path):
    path = model_dir / "live_logs" / "stage_snapshots.jsonl"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Read TAGSs training progress.")
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--target_iter", type=int, default=6000)
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.is_dir():
        print(f"model_dir missing: {model_dir}", file=sys.stderr)
        raise SystemExit(1)

    saved = latest_saved_iteration(model_dir)
    step, loss, points = latest_tensorboard_step(model_dir)
    live = latest_live_loss(model_dir)
    stage_count = stage_snapshot_count(model_dir)

    current = step if step is not None else saved
    pct = (100.0 * current / args.target_iter) if args.target_iter > 0 else 0.0

    print(f"current_iter={current}")
    print(f"target_iter={args.target_iter}")
    print(f"progress_pct={pct:.1f}")
    print(f"saved_checkpoint_iter={saved}")
    if step is not None:
        print(f"tensorboard_step={step}")
    if loss is not None:
        print(f"train_total_loss={loss:.6f}")
    if points is not None:
        print(f"scene_total_points={int(points)}")
    if live is not None:
        print(f"live_log_iter={int(live.get('iteration', 0))}")
        print(f"live_log_total_loss={float(live.get('total_loss', 0.0)):.6f}")
        print(f"live_log_stage={live.get('late_stage', 'n/a')}")
    print(f"live_stage_snapshots={stage_count}")


if __name__ == "__main__":
    main()
