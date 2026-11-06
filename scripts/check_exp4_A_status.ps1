# Check exp4 A 组进度
# 用法: .\scripts\check_exp4_A_status.ps1
$py = "C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe"
& $py -c @'
import json
from pathlib import Path
root = Path("output/exp4v3_A_bicycle/bicycle")
groups = ["A.A1_Raw3DGS", "A.A2_TAGS_Geo", "A.A3_TAGS_SPA", "A.A4_Full_TAGS"]
seeds = [0, 1, 2]
print(f"{'group':<20} {'seed':<5} {'status':<10} {'psnr':<8} {'iter':<6} {'ply?':<5} {'cfg_resolution':<6}")
print("-" * 80)
for g in groups:
    for s in seeds:
        seed_dir = root / g / "train" / f"seed{s}"
        summary_path = root / g / "summary.json"
        ply_30k = seed_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        cfg_path = seed_dir / "cfg_args"

        if not seed_dir.exists():
            status = "NOT_RUN"; psnr = ""; it = ""; ply_ok = ""; cfg = ""
        else:
            ply_ok = "Y" if ply_30k.is_file() else "N"
            if summary_path.is_file():
                j = json.loads(summary_path.read_text(encoding="utf-8"))
                seed_info = j.get("seeds", {}).get(str(s), {})
                psnr = f"{seed_info.get('psnr', 0):.2f}" if seed_info.get('psnr') else ""
                it = str(seed_info.get('iterations', 0))
                status = "DONE" if (ply_ok == "Y" and psnr) else "STALE"
            else:
                status = "TRAIN"; psnr = ""; it = ""
            cfg = ""
            if cfg_path.is_file():
                txt = cfg_path.read_text(encoding="utf-8")
                import re
                m = re.search(r"resolution=([\d\.]+)", txt)
                if m:
                    cfg = m.group(1)
        print(f"{g:<20} {s:<5} {status:<10} {psnr:<8} {it:<6} {ply_ok:<5} {cfg:<6}")
print()
print("STALE = summary 有 PSNR 但 final ply 不在 (会被新 launcher 重新跑)")
print("NOT_RUN = train 目录都没建")
print("TRAIN = 在跑或卡了")
'@
