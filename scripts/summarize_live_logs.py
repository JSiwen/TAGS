#!/usr/bin/env python3
"""Summarize live training logs for stage/loss analysis."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_case_dirs(model_dir, output_root, cases):
    if model_dir:
        return [Path(model_dir)]
    root = Path(output_root)
    if cases:
        return [root / name for name in cases]
    discovered = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        iter_log = child / "live_logs" / "train_iter_metrics.jsonl"
        if iter_log.is_file():
            discovered.append(child)
    return discovered


def parse_focus_iters(spec):
    if not spec:
        return set()
    result = set()
    for token in str(spec).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.add(int(token))
        except ValueError:
            continue
    return result


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tail_average(rows, key, window):
    if not rows:
        return None
    tail = rows[-window:] if window > 0 else rows
    vals = [_safe_float(row.get(key, 0.0), 0.0) for row in tail if key in row]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _mean_dict(rows, key, window):
    if not rows:
        return {}
    tail = rows[-window:] if window > 0 else rows
    sums = {}
    counts = {}
    for row in tail:
        payload = row.get(key, {}) or {}
        if not isinstance(payload, dict):
            continue
        for k, v in payload.items():
            fv = _safe_float(v, None)
            if fv is None:
                continue
            sums[k] = sums.get(k, 0.0) + fv
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts.get(k, 0) > 0}


def summarize_case(model_dir: Path, tail_window: int, focus_iters):
    iter_rows = read_jsonl(model_dir / "live_logs" / "train_iter_metrics.jsonl")
    stage_rows = read_jsonl(model_dir / "live_logs" / "stage_snapshots.jsonl")

    summary = {
        "case": model_dir.name,
        "model_dir": str(model_dir),
        "iter_log_exists": bool(iter_rows),
        "stage_log_exists": bool(stage_rows),
        "iter_rows": len(iter_rows),
        "stage_rows": len(stage_rows),
    }
    if not iter_rows:
        summary["error"] = "missing_iter_log"
        return summary

    latest = iter_rows[-1]
    summary["iter_first"] = int(iter_rows[0].get("iteration", 0))
    summary["iter_last"] = int(latest.get("iteration", 0))
    summary["total_loss_latest"] = _safe_float(latest.get("total_loss", 0.0))
    summary["total_loss_tail_avg"] = _tail_average(iter_rows, "total_loss", tail_window)

    best_row = min(iter_rows, key=lambda row: _safe_float(row.get("total_loss", 1e18), 1e18))
    summary["total_loss_best"] = _safe_float(best_row.get("total_loss", 0.0))
    summary["total_loss_best_iter"] = int(best_row.get("iteration", 0))

    stage_counts = {}
    for row in iter_rows:
        stage = str(row.get("late_stage", "unknown"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    summary["late_stage_counts"] = stage_counts

    summary["operation_losses_tail_avg"] = _mean_dict(iter_rows, "operation_losses", tail_window)
    summary["operation_weights_tail_avg"] = _mean_dict(iter_rows, "operation_weights", tail_window)
    summary["operation_losses_latest"] = latest.get("operation_losses", {}) or {}
    summary["operation_weights_latest"] = latest.get("operation_weights", {}) or {}

    stage_by_iter = {}
    for row in stage_rows:
        it = int(row.get("iteration", 0))
        stage_by_iter[it] = row
    stage_iters = sorted(stage_by_iter.keys())
    if focus_iters:
        stage_iters = [it for it in stage_iters if it in focus_iters]

    stage_snapshots = []
    for it in stage_iters:
        row = stage_by_iter[it]
        eval_payload = row.get("eval", {}) or {}
        test_eval = eval_payload.get("test", {}) or {}
        train_eval = eval_payload.get("train", {}) or {}
        train_snapshot = row.get("train_snapshot", {}) or {}
        stage_snapshots.append(
            {
                "iteration": it,
                "num_points": int(row.get("num_points", 0)),
                "test_psnr": _safe_float(test_eval.get("psnr", None), None),
                "test_ssim": _safe_float(test_eval.get("ssim", None), None),
                "test_lpips": _safe_float(test_eval.get("lpips", None), None),
                "train_psnr": _safe_float(train_eval.get("psnr", None), None),
                "train_ssim": _safe_float(train_eval.get("ssim", None), None),
                "train_lpips": _safe_float(train_eval.get("lpips", None), None),
                "train_total_loss": _safe_float(train_snapshot.get("train/total_loss", None), None),
                "train_photo_loss": _safe_float(train_snapshot.get("train/photo_loss", None), None),
                "train_lpips_loss": _safe_float(train_snapshot.get("train/lpips_loss", None), None),
                "train_psnr_loss": _safe_float(train_snapshot.get("train/psnr_loss", None), None),
            }
        )
    summary["stage_snapshots"] = stage_snapshots

    final_stage = stage_snapshots[-1] if stage_snapshots else None
    summary["final_stage_eval"] = final_stage
    summary["rank_test_psnr"] = final_stage["test_psnr"] if final_stage and final_stage["test_psnr"] is not None else None
    summary["rank_test_lpips"] = final_stage["test_lpips"] if final_stage and final_stage["test_lpips"] is not None else None
    return summary


def _fmt_num(value, digits=6):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _top_items(data, top_n):
    items = sorted(data.items(), key=lambda kv: abs(_safe_float(kv[1], 0.0)), reverse=True)
    return items[:top_n]


def build_markdown(summaries, tail_window):
    lines = [
        "# Live Log Summary",
        "",
        f"- Cases: `{len(summaries)}`",
        f"- Tail window: `{tail_window}`",
        "",
    ]

    ranked = []
    for s in summaries:
        if s.get("rank_test_psnr") is None:
            continue
        ranked.append(s)
    ranked.sort(key=lambda x: x["rank_test_psnr"], reverse=True)

    if ranked:
        lines.extend(
            [
                "## Ranking by final stage test PSNR",
                "",
                "| Rank | Case | Final stage iter | Test PSNR | Test SSIM | Test LPIPS |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, s in enumerate(ranked, start=1):
            final_stage = s.get("final_stage_eval") or {}
            lines.append(
                f"| {idx} | `{s['case']}` | {final_stage.get('iteration', 'n/a')} | "
                f"{_fmt_num(final_stage.get('test_psnr', None), 4)} | "
                f"{_fmt_num(final_stage.get('test_ssim', None), 6)} | "
                f"{_fmt_num(final_stage.get('test_lpips', None), 6)} |"
            )
        lines.append("")

    for s in summaries:
        lines.extend(
            [
                f"## Case `{s['case']}`",
                "",
                f"- Model dir: `{s['model_dir']}`",
                f"- Iter log rows: `{s.get('iter_rows', 0)}` | Stage rows: `{s.get('stage_rows', 0)}`",
            ]
        )
        if s.get("error"):
            lines.append(f"- Error: `{s['error']}`")
            lines.append("")
            continue

        lines.extend(
            [
                f"- Iteration coverage: `{s['iter_first']} -> {s['iter_last']}`",
                f"- Total loss latest: `{_fmt_num(s.get('total_loss_latest', None), 6)}`",
                f"- Total loss tail avg: `{_fmt_num(s.get('total_loss_tail_avg', None), 6)}`",
                f"- Total loss best: `{_fmt_num(s.get('total_loss_best', None), 6)}` @ `{s.get('total_loss_best_iter', 'n/a')}`",
                "",
                "### Late-stage sample counts",
                "",
            ]
        )
        for stage, count in sorted((s.get("late_stage_counts") or {}).items()):
            lines.append(f"- `{stage}`: `{count}`")

        lines.extend(["", "### Operation loss/weight (tail avg)", ""])
        loss_items = _top_items(s.get("operation_losses_tail_avg", {}), top_n=12)
        weight_items = _top_items(s.get("operation_weights_tail_avg", {}), top_n=12)
        if loss_items:
            lines.append("| Loss term | Tail avg |")
            lines.append("| --- | ---: |")
            for key, value in loss_items:
                lines.append(f"| `{key}` | {_fmt_num(value, 6)} |")
            lines.append("")
        else:
            lines.append("- No operation loss terms found.")
            lines.append("")
        if weight_items:
            lines.append("| Weight term | Tail avg |")
            lines.append("| --- | ---: |")
            for key, value in weight_items:
                lines.append(f"| `{key}` | {_fmt_num(value, 6)} |")
            lines.append("")
        else:
            lines.append("- No operation weight terms found.")
            lines.append("")

        lines.extend(["### Stage snapshots", ""])
        snapshots = s.get("stage_snapshots", [])
        if not snapshots:
            lines.append("- No stage snapshots found.")
            lines.append("")
        else:
            lines.append(
                "| Iter | Points | test PSNR | test SSIM | test LPIPS | train total | photo | lpips loss | psnr loss |"
            )
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in snapshots:
                lines.append(
                    f"| {row['iteration']} | {row['num_points']} | "
                    f"{_fmt_num(row.get('test_psnr', None), 4)} | "
                    f"{_fmt_num(row.get('test_ssim', None), 6)} | "
                    f"{_fmt_num(row.get('test_lpips', None), 6)} | "
                    f"{_fmt_num(row.get('train_total_loss', None), 6)} | "
                    f"{_fmt_num(row.get('train_photo_loss', None), 6)} | "
                    f"{_fmt_num(row.get('train_lpips_loss', None), 6)} | "
                    f"{_fmt_num(row.get('train_psnr_loss', None), 6)} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize live logs from v15 experiments.")
    parser.add_argument("--model_dir", type=Path, default=None, help="Single model directory to summarize.")
    parser.add_argument("--output_root", type=Path, default=None, help="Output root containing case directories.")
    parser.add_argument("--cases", nargs="+", default=None, help="Case folder names under --output_root.")
    parser.add_argument(
        "--focus_stage_iters",
        type=str,
        default="",
        help="Optional comma-separated iterations to keep in stage table.",
    )
    parser.add_argument("--tail_window", type=int, default=200, help="Window for tail average statistics.")
    parser.add_argument("--out_md", type=Path, required=True, help="Path to write markdown summary.")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional path to write raw JSON summary.")
    args = parser.parse_args()

    if args.model_dir is None and args.output_root is None:
        raise SystemExit("Provide either --model_dir or --output_root.")

    case_dirs = parse_case_dirs(args.model_dir, args.output_root, args.cases)
    if not case_dirs:
        raise SystemExit("No case directories found.")
    focus_iters = parse_focus_iters(args.focus_stage_iters)

    summaries = [summarize_case(case_dir, max(1, args.tail_window), focus_iters) for case_dir in case_dirs]

    md = build_markdown(summaries, max(1, args.tail_window))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "cases": summaries,
                    "tail_window": max(1, args.tail_window),
                    "focus_stage_iters": sorted(focus_iters),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(args.out_md)


if __name__ == "__main__":
    main()
