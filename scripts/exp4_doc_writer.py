#!/usr/bin/env python3
"""v2 文档自动生成器 - 每次 run 完成后调用, 立刻把结果写入 docs/.

设计目标 (用户需求 "看一眼当前所有实验结果都能够立刻记录在文档中"):
  - 单次 run 完成 -> 立刻更新 docs/exp4_<phase>_results.md 中该 (scene, group, seed) 行
  - 跨场景汇总表自动刷新
  - 失败 run 标记为 "FAILED" 而不是删除, 保留失败证据
  - 不覆盖用户已手写的 docs/exp4_*.md 解释段落: 只更新固定 "## Latest Results" 章节

实现:
  - 用 sentinel 行标记 "## Latest Results" 章节的起止
  - 每次写入: 读原 docs 文件 -> 找到 sentinel 区域 -> 替换该区域 -> 写回
  - 章节不存在则自动追加 (首次)

输出 (每个 phase 一个文件):
  docs/exp4_2_results.md
  docs/exp4_3_results.md
  docs/exp4_4_results.md
  docs/exp4_5_results.md
  docs/exp4_qa_results.md  (QA gate 结果)

用法:
  # 在 exp4_runner.run_full 末尾自动调用 (无需用户手动)
  from scripts.exp4_doc_writer import write_run_result_to_doc
  write_run_result_to_doc(scene, group, seed, metrics, phase)

  # CLI: 一次性重新生成所有 docs
  python scripts/exp4_doc_writer.py --refresh-all
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_ledger import (  # type: ignore  # noqa: E402
    DEFAULT_LEDGER_JSONL, iter_entries, list_scenes_groups_seeds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

SENTINEL_BEGIN = "<!-- AUTO-GENERATED Latest Results BEGIN (do not edit) -->"
SENTINEL_END = "<!-- AUTO-GENERATED Latest Results END -->"

PHASE_TO_DOC = {
    "4_2": "exp4_2_results.md",
    "4_3": "exp4_3_results.md",
    "4_4": "exp4_4_results.md",
    "4_5": "exp4_5_results.md",
    "QA":  "exp4_qa_results.md",
}


def _phase_from_group(group: str) -> str:
    """从 group id 推断 phase: 4_2_C_Reference -> 4_2."""
    if group.startswith("4_2_"):
        return "4_2"
    if group.startswith("4_3_"):
        return "4_3"
    if group.startswith("4_4_"):
        return "4_4"
    if group.startswith("4_5_"):
        return "4_5"
    if group.startswith("qa_") or group.startswith("QA"):
        return "QA"
    return "4_5"  # default


def _format_metrics_row(scene: str, group: str, seed: int, entry: Dict) -> str:
    """格式化单行 markdown."""
    metrics = entry.get("metrics", {}) if entry else {}
    psnr = metrics.get("psnr")
    ssim = metrics.get("ssim")
    lpips = metrics.get("lpips")
    n = metrics.get("num_points")
    train_exit = metrics.get("train_exit", 0)
    render_exit = metrics.get("render_exit", 0)
    metrics_exit = metrics.get("metrics_exit", 0)

    if psnr is None:
        status = f"FAILED (train={train_exit} render={render_exit} metrics={metrics_exit})"
        psnr_s = ssim_s = lpips_s = n_s = "—"
    else:
        status = "OK"
        psnr_s = f"{psnr:.4f}"
        ssim_s = f"{ssim:.4f}" if ssim is not None else "—"
        lpips_s = f"{lpips:.4f}" if lpips is not None else "—"
        n_s = f"{int(n):,}" if n is not None else "—"
    return f"| {scene} | {group} | {seed} | {psnr_s} | {ssim_s} | {lpips_s} | {n_s} | {status} |"


def _format_summary_table(entries: List[Dict], title: str = "Per-Run Results") -> str:
    """格式化整个 phase 的 markdown 汇总表."""
    if not entries:
        return f"_{title}: (no entries yet)_\n"
    # 按 (scene, group, seed) 排序
    entries.sort(key=lambda e: (e.get("scene", ""), e.get("group", ""), e.get("seed", 0)))
    lines = [
        f"### {title}",
        "",
        f"_Last refreshed: {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "| scene | group | seed | PSNR | SSIM | LPIPS | num_points | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(_format_metrics_row(
            e.get("scene", ""), e.get("group", ""), e.get("seed", 0), e
        ))
    return "\n".join(lines) + "\n"


def _per_scene_group_aggregate(entries: List[Dict]) -> str:
    """按 (scene, group) 聚合, 输出 mean ± std + 95% CI."""
    import math
    try:
        from scipy import stats as sc_stats
        has_scipy = True
    except ImportError:
        has_scipy = False
    by_sg: Dict[Tuple[str, str], List[float]] = {}
    for e in entries:
        psnr = e.get("metrics", {}).get("psnr")
        if psnr is None:
            continue
        key = (e.get("scene", ""), e.get("group", ""))
        by_sg.setdefault(key, []).append(float(psnr))
    if not by_sg:
        return ""
    lines = [
        "",
        "### Per-(Scene, Group) Mean ± SD",
        "",
        "| scene | group | n_seeds | PSNR mean | PSNR std | 95% CI low | 95% CI high |",
        "|---|---|---|---|---|---|---|",
    ]
    for (scene, group), psnrs in sorted(by_sg.items()):
        n = len(psnrs)
        m = sum(psnrs) / n
        if n >= 2:
            sd = math.sqrt(sum((p - m) ** 2 for p in psnrs) / (n - 1))
        else:
            sd = 0.0
        if has_scipy and n >= 2 and sd > 0:
            sem = sd / math.sqrt(n)
            t_crit = sc_stats.t.ppf(0.975, df=n - 1)
            ci_low = m - t_crit * sem
            ci_high = m + t_crit * sem
            ci_low_s = f"{ci_low:.4f}"
            ci_high_s = f"{ci_high:.4f}"
        else:
            ci_low_s = ci_high_s = "n/a"
        lines.append(
            f"| {scene} | {group} | {n} | {m:.4f} | {sd:.4f} | {ci_low_s} | {ci_high_s} |"
        )
    return "\n".join(lines) + "\n"


def _write_to_doc_with_sentinel(doc_path: Path, body: str, phase: str):
    """读 / 写 docs/exp4_<phase>_results.md, 保留 sentinel 之外的章节."""
    if not doc_path.is_file():
        # 新建文件, 写一个完整模板
        title = f"# §{phase.replace('_', '.')} — Auto-Recorded Results"
        body_full = (
            f"{title}\n\n"
            f"本文件由 `scripts/exp4_doc_writer.py` 自动生成, 每次 run 完成会刷新.\n"
            f"可手写其他章节, 但不要修改 `{SENTINEL_BEGIN}` 与 `{SENTINEL_END}` 之间的内容.\n\n"
            f"{SENTINEL_BEGIN}\n\n"
            f"{body}\n\n"
            f"{SENTINEL_END}\n"
        )
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(body_full, encoding="utf-8")
        return
    text = doc_path.read_text(encoding="utf-8")
    if SENTINEL_BEGIN in text and SENTINEL_END in text:
        # 替换 sentinel 区域
        pattern = re.compile(
            re.escape(SENTINEL_BEGIN) + r".*?" + re.escape(SENTINEL_END),
            re.DOTALL,
        )
        new_block = f"{SENTINEL_BEGIN}\n\n{body}\n\n{SENTINEL_END}"
        text = pattern.sub(new_block, text)
    else:
        # 追加 sentinel 区域
        text = text.rstrip() + f"\n\n## Latest Results (auto-updated)\n\n{SENTINEL_BEGIN}\n\n{body}\n\n{SENTINEL_END}\n"
    doc_path.write_text(text, encoding="utf-8")


def write_run_result_to_doc(
    scene: str,
    group: str,
    seed: int,
    metrics: Dict,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
):
    """单条 run 结果: 写 ledger + 刷新对应 docs 文件.

    这是 exp4_runner.run_full 末尾应调用的入口.
    """
    from scripts.exp4_ledger import record_run
    entry = record_run(
        scene=scene, group=group, seed=seed, metrics=metrics,
        config=config, extra=extra,
        phase=_phase_from_group(group),
    )
    # 刷新对应 phase 的 doc
    refresh_phase_doc(_phase_from_group(group))
    return entry


def refresh_phase_doc(phase: str, ledger_path: Path = DEFAULT_LEDGER_JSONL):
    """刷新整个 phase 的 doc, 包含 phase 下所有 (scene, group, seed)."""
    if phase not in PHASE_TO_DOC:
        return
    # 过滤 ledger
    entries = [e for e in iter_entries(ledger_path) if _phase_from_group(e.get("group", "")) == phase]
    if not entries:
        body = "_(no runs recorded yet)_\n"
    else:
        body = _format_summary_table(entries, title=f"§{phase.replace('_', '.')} Per-Run Results")
        body += _per_scene_group_aggregate(entries)
    doc_path = DOCS_DIR / PHASE_TO_DOC[phase]
    _write_to_doc_with_sentinel(doc_path, body, phase)


def refresh_all_docs(ledger_path: Path = DEFAULT_LEDGER_JSONL):
    """CLI 入口: 一次性刷新所有 phase 的 docs."""
    for phase in PHASE_TO_DOC:
        refresh_phase_doc(phase, ledger_path)
        print(f"refreshed: docs/{PHASE_TO_DOC[phase]}")


def main():
    parser = argparse.ArgumentParser(description="v2 文档自动生成器")
    parser.add_argument("--refresh-all", action="store_true", help="一次性刷新所有 phase 的 docs")
    parser.add_argument("--phase", choices=list(PHASE_TO_DOC.keys()), help="刷新单个 phase")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_JSONL))
    args = parser.parse_args()

    ledger_path = Path(args.ledger)

    if args.refresh_all:
        refresh_all_docs(ledger_path)
    elif args.phase:
        refresh_phase_doc(args.phase, ledger_path)
        print(f"refreshed: docs/{PHASE_TO_DOC[args.phase]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
