#!/usr/bin/env python3
"""§4.6 全对比表 generator (v2, 2026-10-14).

读所有 scene × group 的 5-seed 数据 + DN baseline, 生成论文表 4.6 用的全对比 markdown 表.

设计:
  - 双场景: bicycle, vr_room
  - 多 group: lpips_ms_w005 (主), 也可加其它 (rgb_ssim, lpips_w005)
  - 数据源: 本地 ledger (主) + wandb (交叉)
  - 输出: docs/SECTION_4_6_FULL_COMPARISON.md (含主表 + per-scene 详细表 + gate 检查)
  - 不阻塞: 缺数据时 graceful skip + 标记

用法:
  python scripts/build_section_4_6.py
  python scripts/build_section_4_6.py --output docs/my_4_6.md
  python scripts/build_section_4_6.py --no-wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp4_summarize_v2 import (  # noqa: E402
    WINNER_GATE,
    collect_seed_metrics,
    fetch_wandb_runs,
    cross_check_ledger_vs_wandb,
    evaluate_winner_gate,
    multiseed_summary_text,
)


# 默认扫描配置
DEFAULT_SCENES = [
    {
        "scene": "bicycle",
        "tags_dir": "output/bicycle_e4_lpips_ms_w005",
        "dn_dir": "output/bicycle_dn_splatter",
        "dn_renders": "output/bicycle_dn_renders/eval",
        "tags_group": "lpips_ms_w005",
    },
    {
        "scene": "vr_room",
        "tags_dir": "output/vr_room_e4_lpips_ms_w005",
        "dn_dir": "output/vr_room_dn_splatter",
        "dn_renders": "output/vr_room_dn_renders/eval",
        "tags_group": "lpips_ms_w005",
    },
]


def _read_dn_row(dn_dir: Path) -> dict:
    """读 DN-Splatter 的 results.json (或 fallback 到 test/dn-splatter/results.json)."""
    candidates = [
        dn_dir / "test" / "dn-splatter" / "results.json",
        dn_dir / "test" / "results.json",
        dn_dir / "results.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                results = json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                continue
            block = None
            for key in ("dn-splatter", "ours_30000", "ours_1000"):
                v = results.get(key)
                if isinstance(v, dict) and "PSNR" in v:
                    block = v
                    break
            if block is None:
                for v in results.values():
                    if isinstance(v, dict) and "PSNR" in v:
                        block = v
                        break
            if block:
                # num_points: 尝试从 PLY 算
                from scripts.exp4_summarize_v2 import _count_ply_vertices, _find_ply
                ply = _find_ply(dn_dir)
                n_pts = _count_ply_vertices(ply) if ply else None
                return {
                    "psnr": float(block["PSNR"]),
                    "ssim": float(block["SSIM"]),
                    "lpips": float(block["LPIPS"]),
                    "num_points": n_pts,
                }
    return {}


def build_full_table(scenes: list, project: str, entity: str, group: str,
                     use_wandb: bool = True) -> str:
    """构建 §4.6 全对比 markdown 字符串."""
    lines = []
    lines.append("# §4.6 全对比表 (TAGS vs DN-Splatter, 双场景 × 5-seed)")
    lines.append("")
    lines.append(f"> **生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> **数据源**: 本地 ledger (`output/<scene>/<group>/train/seed<N>/`) + wandb cross-check")
    lines.append(f"> **wandb project**: `{entity}/{project}` (group=`{group or '(all)'}`)")
    lines.append(f"> **winner_gate**: Δpsnr ≥ +{WINNER_GATE['min_delta_psnr']} dB, Δlpips ≤ +{WINNER_GATE['max_delta_lpips']}")
    lines.append("")

    # ============ 0) TL;DR 跨场景主表 ============
    lines.append("## 0) TL;DR: 跨场景主表")
    lines.append("")
    lines.append("| Scene | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Points | n_seeds | Gate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")

    scene_data = []
    for s in scenes:
        scene = s["scene"]
        tags_dir = ROOT / s["tags_dir"]
        dn_dir = ROOT / s["dn_dir"]
        tags_group = s["tags_group"]

        if not tags_dir.is_dir():
            lines.append(f"| {scene} | TAGS_{tags_group} | **缺数据** (`{tags_dir}`) | | | | | |")
            scene_data.append({"scene": scene, "skip": True, "reason": f"tags_dir 不存在: {tags_dir}"})
            continue

        tags_local = collect_seed_metrics(tags_dir)
        dn = _read_dn_row(dn_dir)

        if not tags_local.get("per_seed"):
            lines.append(f"| {scene} | TAGS_{tags_group} | **无 seed 数据** | | | | 0 | |")
            scene_data.append({"scene": scene, "skip": True, "reason": "无 seed 数据"})
            continue

        # wandb cross-check
        wandb_runs = []
        if use_wandb and entity and project:
            try:
                wandb_runs = fetch_wandb_runs(project, entity, group or tags_group, scene)
            except Exception:
                pass
        cross = cross_check_ledger_vs_wandb(tags_local, wandb_runs)
        gate = evaluate_winner_gate(tags_local, dn)

        m = tags_local["mean"]; st = tags_local["std"]
        gate_mark = "✅ pass" if gate.get("passed") else ("❌ fail" if gate.get("passed") is False else "—")
        dn_pts = dn.get("num_points") or 0
        dn_pts_str = f"{int(dn_pts):,}" if dn_pts else "—"

        lines.append(
            f"| {scene} | **TAGS_{tags_group}** | "
            f"**{m.get('psnr', 0):.3f} ± {st.get('psnr', 0):.3f}** | "
            f"**{m.get('ssim', 0):.4f} ± {st.get('ssim', 0):.4f}** | "
            f"**{m.get('lpips', 0):.4f} ± {st.get('lpips', 0):.4f}** | "
            f"**{int(m.get('num_points', 0)):,}** | "
            f"{tags_local['n_seeds']} | **{gate_mark}** |"
        )
        if dn:
            lines.append(
                f"| {scene} | DN-Splatter | {dn.get('psnr', 0):.3f} | {dn.get('ssim', 0):.4f} "
                f"| {dn.get('lpips', 0):.4f} | {dn_pts_str} | 1 | — |"
            )
            # delta 行
            if gate.get("psnr_delta") is not None:
                lines.append(
                    f"| {scene} | **Δ (TAGS-DN)** | "
                    f"**{gate['psnr_delta']:+.3f}** | "
                    f"**{gate['ssim_delta']:+.4f}** | "
                    f"**{gate['lpips_delta']:+.4f}** | "
                    f"× {(m.get('num_points', 1) / dn_pts) if dn_pts else 0:.2f} | — | "
                    f"{'PSNR✓' if gate.get('psnr_pass') else 'PSNR✗'}/"
                    f"{'LPIPS✓' if gate.get('lpips_pass') else 'LPIPS✗'} |"
                )
        scene_data.append({
            "scene": scene,
            "tags_local": tags_local,
            "dn": dn,
            "wandb_runs": wandb_runs,
            "cross": cross,
            "gate": gate,
            "tags_group": tags_group,
        })

    lines.append("")
    lines.append("> **论文主表候选**: 第 1 行 + 第 3 行 (TAGS_mean ± std / Δ vs DN), 5-seed 完整版")
    lines.append("")

    # ============ 1) Per-scene 详细 (含 per-seed) ============
    lines.append("## 1) Per-scene 详细数据 (含 5-seed per-seed + wandb cross)")
    lines.append("")
    for sd in scene_data:
        if sd.get("skip"):
            continue
        text = multiseed_summary_text(
            scene=sd["scene"],
            tags_group=sd["tags_group"],
            tags_local=sd["tags_local"],
            dn=sd.get("dn", {}),
            gate=sd.get("gate", {}),
            cross=sd.get("cross", {}),
        )
        lines.append(text)
        lines.append("")

    # ============ 2) wandb dashboard 链接 ============
    if entity and project:
        lines.append("## 2) wandb dashboard 链接")
        lines.append("")
        for s in scenes:
            lines.append(f"- **{s['scene']}**: https://wandb.ai/{entity}/{project}/groups/{s['tags_group']}?workspace=user-{entity}")
        lines.append("")

    # ============ 3) 复现指令 ============
    lines.append("## 3) 复现指令")
    lines.append("")
    lines.append("```powershell")
    lines.append("# 1) 跑 5-seed (bicycle §4.5)")
    lines.append("$env:E4_SEEDS = \"0,1,2,3,4\"")
    lines.append("$env:WANDB_ENABLED = \"1\"")
    lines.append("$env:WANDB_PROJECT = \"tags-ablation\"")
    lines.append("$env:WANDB_ENTITY = \"your_username\"")
    lines.append(".\\scripts\\run_bicycle_e4_winner.ps1")
    lines.append("")
    lines.append("# 2) 5-seed 完成后跑这个 generator")
    lines.append("python scripts\\build_section_4_6.py")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append(f"**自动生成**: `scripts/build_section_4_6.py` v2, 跑完 5-seed 后一键再生")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="§4.6 全对比表 generator (v2)")
    parser.add_argument("--scenes-json", help="覆盖默认 scenes 配置, 路径指向 JSON")
    parser.add_argument("--output", default="docs/SECTION_4_6_FULL_COMPARISON.md",
                        help="输出 markdown 路径")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "tags-ablation"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", ""))
    parser.add_argument("--no-wandb", action="store_true", help="跳过 wandb cross-check")
    args = parser.parse_args()

    scenes = DEFAULT_SCENES
    if args.scenes_json and Path(args.scenes_json).is_file():
        scenes = json.loads(Path(args.scenes_json).read_text(encoding="utf-8"))

    print(f"[build_4_6] {len(scenes)} scene(s): {[s['scene'] for s in scenes]}")
    print(f"[build_4_6] wandb: project={args.wandb_project} entity={args.wandb_entity} group={args.wandb_group}")
    print(f"[build_4_6] output: {args.output}")

    md = build_full_table(
        scenes=scenes,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        use_wandb=not args.no_wandb,
    )
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[build_4_6] Wrote {out_path} ({len(md)} chars)")


if __name__ == "__main__":
    main()
