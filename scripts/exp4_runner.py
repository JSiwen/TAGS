#!/usr/bin/env python3
"""v2 通用 TAGS 训练+渲染+评分执行器.

供各 run_exp4_*.py 脚本调用, 封装:
  1) 训练 train_optimized.py (按 -Scene -Seed -GroupId 隔离输出目录)
  2) 渲染 render.py
  3) 评分 metrics.py
  4) 解析 results.json 写 summary.json

输出目录约定:
  output/exp4_{scene}/{group_id}/train/seed{seed}/
  output/exp4_{scene}/{group_id}/renders/eval_seed{seed}/
  output/exp4_{scene}/{group_id}/summary.json  (本脚本汇总)

调用:
  python scripts/exp4_runner.py \
    --scene bicycle --group 4_2_C_Reference --seed 0 \
    --extra-args "--prune_ratio2 0.04" \
    --output-root output/exp4_bicycle
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# 基础训练参数 (v3 默认: 1/2 分辨率 images_2, 旧版 images_4)
# 可在 train_full() 中通过 images_dir 参数覆盖
DEFAULT_BASE_ARGS = [
    "--eval",
    "--llffhold", "8",
    "--images", "images_2",
    "--iterations", "30000",
]


def run_subprocess(cmd, cwd=None, env=None) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, env=env, check=False
    )
    return result.returncode


# =====================================================================
#  Live→wandb watcher 集成 (v3, 2026-10-17)
#  每个 train 启动一个后台 watcher 进程, 监控 train_iter_metrics.jsonl,
#  把 train/psnr, train/ssim, train/lpips_proxy, train/loss, scene/total_points
#  实时推到 wandb (run_name = "<scene>__<group>__seed<seed>_live").
#  设计: 不侵入 train_optimized.py, 只观察 live_logs/ 文件.
# =====================================================================

_LIVE_WATCHER_PROC = None  # 当前 live watcher 子进程 (Popen 或 None)
_LIVE_WATCHER_PIDFILE: Optional[Path] = None  # 用 pidfile 跨进程追踪


def _live_wandb_watcher_pidfile(train_dir: Path) -> Path:
    return train_dir / "live_wandb_watcher.pid"


def start_live_wandb_watcher(
    scene: str, group: str, seed: int,
    train_dir: Path, python: str, env: Optional[Dict] = None,
    poll_interval: float = 10.0,  # 10s poll (训练每 ~2-5s 一个 iter, 10s 够细)
) -> Optional[subprocess.Popen]:
    """启动 live→wandb watcher 后台进程.

    Returns:
        Popen 对象 (已 detach), 失败返回 None.

    Notes:
        - 如果 WANDB_ENABLED != 1, 直接 return None (no-op)
        - 如果 train_dir/live_logs/ 不存在, 延迟等 (polling)
        - 如果已有 watcher (pidfile 存在且进程在), 不会重复启
    """
    global _LIVE_WATCHER_PROC, _LIVE_WATCHER_PIDFILE
    if os.environ.get("WANDB_ENABLED", "0").strip() not in ("1", "true", "True", "yes"):
        return None
    if env is None:
        env = os.environ.copy()
    # 找 jsonl: train_dir/live_logs/train_iter_metrics.jsonl
    jsonl = train_dir / "live_logs" / "train_iter_metrics.jsonl"
    # 早期 jsonl 还没建, 最多等 60s
    waited = 0.0
    while not jsonl.is_file() and waited < 60.0:
        time.sleep(2.0)
        waited += 2.0
    if not jsonl.is_file():
        print(f"  [live-wandb] warn: {jsonl} 60s 内未出现, 跳过 watcher")
        return None
    pidfile = _live_wandb_watcher_pidfile(train_dir)
    # 检查是否已有 watcher
    if pidfile.is_file():
        try:
            old_pid = int(pidfile.read_text().strip())
            import psutil  # type: ignore
            if psutil.pid_exists(old_pid):
                print(f"  [live-wandb] watcher 已存在 (pid={old_pid}), 复用")
                return None
        except Exception:
            pass
        # stale pidfile, 删
        try:
            pidfile.unlink()
        except Exception:
            pass
    cmd = [
        python, "scripts/exp4_live_to_wandb.py",
        "--jsonl", str(jsonl),
        "--scene", scene,
        "--group", group,
        "--seed", str(seed),
        "--poll-interval", str(poll_interval),
        "--stall-polls", "4",  # 4*10s = 40s 没新行才 finish (给 render 留时间)
    ]
    print(f"  [live-wandb] start watcher: {jsonl.relative_to(REPO_ROOT)}")
    creationflags = 0
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP: 让 child 不和 launcher 共享 console (避免信号串扰)
        creationflags = 0x00000200
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags if os.name == "nt" else 0,
        )
    except Exception as exc:
        print(f"  [live-wandb] start failed: {exc!r}")
        return None
    _LIVE_WATCHER_PROC = proc
    _LIVE_WATCHER_PIDFILE = pidfile
    # 写 pidfile (跨进程追踪)
    try:
        pidfile.write_text(str(proc.pid), encoding="utf-8")
    except Exception:
        pass
    return proc


def stop_live_wandb_watcher(graceful_timeout: float = 60.0) -> None:
    """停 live→wandb watcher.

    策略:
      1) 优雅: 让 watcher 自己 finish (再发 SIGTERM/ terminate 触发 stop)
      2) 强制: 超过 graceful_timeout 直接 kill
    """
    global _LIVE_WATCHER_PROC, _LIVE_WATCHER_PIDFILE
    proc = _LIVE_WATCHER_PROC
    pidfile = _LIVE_WATCHER_PIDFILE
    if proc is None and pidfile is not None and pidfile.is_file():
        # 从 pidfile 恢复
        try:
            import psutil  # type: ignore
            old_pid = int(pidfile.read_text().strip())
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
        except Exception:
            proc = None
    if proc is None:
        # 没 watcher, 清 stale pidfile
        if pidfile is not None:
            try:
                pidfile.unlink(missing_ok=True)
            except Exception:
                pass
        return
    print(f"  [live-wandb] stop watcher (pid={getattr(proc, 'pid', '?')}, graceful={graceful_timeout}s)")
    try:
        proc.terminate()  # SIGTERM
        try:
            proc.wait(timeout=graceful_timeout)
            print(f"  [live-wandb] watcher gracefully stopped")
        except subprocess.TimeoutExpired:
            print(f"  [live-wandb] watcher 超时 {graceful_timeout}s, force kill")
            try:
                proc.kill()
                proc.wait(timeout=10.0)
            except Exception:
                pass
    except Exception as exc:
        print(f"  [live-wandb] stop warn: {exc!r}")
    _LIVE_WATCHER_PROC = None
    if pidfile is not None:
        try:
            pidfile.unlink(missing_ok=True)
        except Exception:
            pass


# =====================================================================
# 断点续跑辅助 (v2, 2026-10-14)
# =====================================================================

def find_latest_checkpoint(model_dir: Path) -> Optional[Dict]:
    """在 model_dir 下找最新的 chkpnt<N>.pth.

    Returns:
        {"path": Path, "iteration": int, "size_mb": float} 或 None
    """
    if not model_dir.is_dir():
        return None
    best_path = None
    best_iter = -1
    for p in model_dir.glob("chkpnt*.pth"):
        if not p.is_file():
            continue
        m = re.match(r"^chkpnt(\d+)$", p.stem)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_iter:
            best_iter = it
            best_path = p
    if best_path is None:
        return None
    return {
        "path": best_path,
        "iteration": best_iter,
        "size_mb": best_path.stat().st_size / (1024 * 1024),
    }


def is_run_completed(model_dir: Path, iterations: int) -> bool:
    """判定 run 是否已完成 (final ply 存在)."""
    final_ply = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if final_ply.is_file():
        return True
    # 备选: 直接放 model_dir 顶层 (老格式)
    return (model_dir / "point_cloud.ply").is_file()


def should_skip_run(model_dir: Path, iterations: int, skip_completed: bool = True) -> bool:
    """若 skip_completed 且 run 已完成, 返回 True (上层应 continue).

    决策依据: model_dir 下有 `point_cloud/iteration_<iterations>/point_cloud.ply`,
    即已生成最终 PLY, 视为训练完成.
    """
    if not skip_completed:
        return False
    return is_run_completed(model_dir, iterations)


def parse_psnr_ssim_lpips(results_json: Path) -> Optional[Dict[str, float]]:
    """解析 metrics.py 生成的 results.json.

    标准格式: ``{"<method_name>": {"SSIM": ..., "PSNR": ..., "LPIPS": ...}, ...}``.
    顶层每个 key 是一个训练方法, 值是 SSIM/PSNR/LPIPS 三项的 dict.
    """
    if not results_json.is_file():
        return None
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        return None
    out = {}
    for method_name, metrics in payload.items():
        if not isinstance(metrics, dict):
            # 防御: 若某项不是 dict, 跳过
            continue
        psnr = metrics.get("PSNR")
        ssim = metrics.get("SSIM")
        lpips = metrics.get("LPIPS")
        if psnr is not None:
            out["psnr"] = float(psnr)
        if ssim is not None:
            out["ssim"] = float(ssim)
        if lpips is not None:
            out["lpips"] = float(lpips)
        out["method"] = method_name
    return out


def parse_runtime(train_dir: Path) -> Optional[Dict[str, float]]:
    """解析 train_optimized.py 在训练结束前写的 runtime.json (B 改动, 2026-10-19).

    格式: {"wall_time_sec": float, "peak_vram_mb": float, "iterations": int, "seed": int}

    缺失字段不抛异常, 调用方按 None 处理.
    """
    rt_path = train_dir / "runtime.json"
    if not rt_path.is_file():
        return None
    try:
        data = json.loads(rt_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    out: Dict[str, float] = {}
    if "wall_time_sec" in data:
        out["wall_time_sec"] = float(data["wall_time_sec"])
    if "peak_vram_mb" in data:
        out["peak_vram_mb"] = float(data["peak_vram_mb"])
    return out


def count_ply_points(model_dir: Path) -> Optional[int]:
    """从 point_cloud/iteration_30000/point_cloud.ply 读 vertex 数.

    高效版: 用 grep 找 "element vertex N", 不读整个 binary body.
    PLY header 是 ASCII, 通常前 1-2 KB 就够.
    """
    candidates = [
        model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        model_dir / "point_cloud.ply",
        model_dir / "splat.ply",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            # 读前 64 KB (header 通常远小于这个)
            with path.open("rb") as f:
                head = f.read(64 * 1024)
            # 找 "element vertex <N>"
            for raw in head.split(b"\n"):
                txt = raw.decode("ascii", errors="ignore").strip()
                if txt.startswith("element vertex"):
                    parts = txt.split()
                    if len(parts) >= 3 and parts[-1].isdigit():
                        return int(parts[-1])
            # 备选: 解析整个 header (一直读到 "end_header")
            with path.open("rb") as f:
                header = b""
                while b"end_header" not in header:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    header += chunk
                    if len(header) > 4 * 1024 * 1024:  # 4 MB cap
                        break
            for raw in header.split(b"\n"):
                txt = raw.decode("ascii", errors="ignore").strip()
                if txt.startswith("element vertex"):
                    parts = txt.split()
                    if len(parts) >= 3 and parts[-1].isdigit():
                        return int(parts[-1])
        except Exception:
            return None
    return None


def run_full(
    scene: str,
    group: str,
    seed: int,
    extra_args: list,
    output_root: Path,
    iterations: int = 30000,
    hold: int = 8,
    python: str = "python",
    skip_train: bool = False,
    skip_render: bool = False,
    skip_metrics: bool = False,
    skip_completed: bool = True,
    auto_resume: bool = True,
    env_vars: Optional[Dict[str, str]] = None,
    images_dir: str = "images_2",  # v3: 默认 1/2 物理分辨率
    resolution: int = 2,  # v3: 默认 resolution=2 (主阶段 1/4 物理像素, 配合 images_2)
) -> Dict:
    """对单个 (scene, group, seed) 跑完整 train + render + metrics 流水线.

    续跑机制 (v2, 2026-10-14):
      - skip_completed=True: 检测到 final ply 直接返回已有结果, 不重跑
      - auto_resume=True:    检测到 chkpnt*.pth 自动注入 --start_checkpoint 续跑
      - 续跑时在 log 里追加 "==== RESUME ====" 标记, 便于审计

    返回: dict {psnr, ssim, lpips, num_points, train_exit, render_exit, metrics_exit,
                resumed_from: int|None, skipped: bool}
    """
    train_dir = output_root / scene / group / "train" / f"seed{seed}"
    train_dir.mkdir(parents=True, exist_ok=True)

    # 准备环境变量
    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)
    # 必须设置 PYTHONIOENCODING (与现有 run_vr_room_dn_other_env.ps1 一致)
    run_env["PYTHONIOENCODING"] = "utf-8"
    run_env["PYTHONUTF8"] = "1"

    info = {
        "scene": scene,
        "group": group,
        "seed": seed,
        "train_dir": str(train_dir),
        "iterations": iterations,
        "images_dir": images_dir,
        "resolution": resolution,
        "extra_args": list(extra_args),
        "resumed_from": None,
        "skipped": False,
    }

    # ===== wandb: 每个 (scene, group, seed) 初始化一次 run =====
    # 这样 train.py 里的 _get_wandb() 能拿到 wandb.run 进行 per-iter log
    _wb = None
    if os.environ.get("WANDB_ENABLED", "0").strip() in ("1", "true", "True", "yes"):
        try:
            from scripts.exp4_wandb_logger import _ensure_run  # type: ignore
            wb_project = os.environ.get("WANDB_PROJECT", "TAGSs_exp4")
            wb_entity = os.environ.get("WANDB_ENTITY", "") or None
            wb_group_env = os.environ.get("WANDB_RUN_GROUP", "")
            wb_tags_env = os.environ.get("WANDB_TAGS", "")
            wb_tags = [t.strip() for t in wb_tags_env.split(",") if t.strip()] + [scene, group, f"seed{seed}"]
            run_name = f"{scene}__{group}__seed{seed}"
            cfg = {
                "scene": scene,
                "group": group,
                "seed": seed,
                "iterations": iterations,
                "images_dir": images_dir,
                "resolution": resolution,
                "extra_args": list(extra_args),
            }
            _wb = _ensure_run(
                run_name=run_name,
                config=cfg,
                project=wb_project,
                entity=wb_entity,
                group=wb_group_env or None,
                tags=wb_tags,
            )
            if _wb is not None:
                print(f"  [wandb] init OK: {run_name} (project={wb_project}, group={wb_group_env or '-'})")
        except Exception as exc:
            print(f"  [wandb] init failed: {exc!r} (continue without wandb)")
            _wb = None

    # ===== 续跑检测 1: 跳过已完成 =====
    if not skip_train and should_skip_run(train_dir, iterations, skip_completed):
        print(f"  [skip] {scene}/{group}/seed{seed} 已完成 (final ply 在), 跳过训练/渲染/评分")
        # 仍读 results.json 把分数写回
        info["skipped"] = True
        info["train_exit"] = 0
        info["render_exit"] = 0
        info["metrics_exit"] = 0
        # 直接解析已有结果
        existing_results = train_dir / "results.json"
        if not existing_results.is_file():
            # 备选: test/ours_30000/results.json
            existing_results = train_dir / "test" / "ours_30000" / "results.json"
        metrics = parse_psnr_ssim_lpips(existing_results)
        if metrics:
            info["psnr"] = metrics.get("psnr")
            info["ssim"] = metrics.get("ssim")
            info["lpips"] = metrics.get("lpips")
            info["method"] = metrics.get("method")
        points = count_ply_points(train_dir)
        if points is not None:
            info["num_points"] = points
        # 写 summary 然后返回
        _write_summary(output_root, scene, group, seed, info)
        _log_to_ledger_and_wandb(scene, group, seed, info)
        return info

    # ===== 续跑检测 2: 找最新 chkpnt 用于 resume =====
    resume_ckpt = None
    if not skip_train and auto_resume:
        resume_ckpt = find_latest_checkpoint(train_dir)
        if resume_ckpt is not None:
            print(f"  [resume] {scene}/{group}/seed{seed} 找到 chkpnt{resume_ckpt['iteration']}"
                  f" ({resume_ckpt['size_mb']:.1f} MB), 续跑 from iter {resume_ckpt['iteration']}")
            info["resumed_from"] = resume_ckpt["iteration"]
        else:
            print(f"  [start] {scene}/{group}/seed{seed} 从头跑 (无 chkpnt)")

    # 1) 训练
    if not skip_train:
        train_cmd = [
            python, "train_optimized.py",
            "-s", f"data/{scene}",
            "-m", str(train_dir),
            "--seed", str(seed),
            "--iterations", str(iterations),
            "--llffhold", str(hold),
            "--images", images_dir,
            "--resolution", str(resolution),
            "--eval",
        ] + list(extra_args)
        # 注入 --start_checkpoint (如果有 chkpnt)
        if resume_ckpt is not None:
            train_cmd += ["--start_checkpoint", str(resume_ckpt["path"])]
        # log 文件追加 RESUME 标记
        log_path = train_dir / "train_console.log"
        if resume_ckpt is not None and log_path.is_file():
            from datetime import datetime
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n==== RESUME {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                        f"from chkpnt{resume_ckpt['iteration']} ====\n")
        # 启 live→wandb watcher (后台, 监控 train_iter_metrics.jsonl, 推 psnr/ssim/loss 曲线)
        start_live_wandb_watcher(
            scene=scene, group=group, seed=seed,
            train_dir=train_dir, python=python, env=run_env,
        )
        exit_code = run_subprocess(train_cmd, cwd=REPO_ROOT, env=run_env)
        # 停 live→wandb watcher (不论 train 成功/失败)
        stop_live_wandb_watcher()
        info["train_exit"] = exit_code
        if exit_code != 0:
            return info
    else:
        info["train_exit"] = 0  # 假定已通过

    # 2) 渲染
    if not skip_render:
        render_cmd = [
            python, "render.py",
            "-s", f"data/{scene}",
            "-m", str(train_dir),
            "--iteration", str(iterations),
            "--skip_train",
        ]
        exit_code = run_subprocess(render_cmd, cwd=REPO_ROOT, env=run_env)
        info["render_exit"] = exit_code
        if exit_code != 0:
            return info
    else:
        info["render_exit"] = 0

    # 3) 评分
    if not skip_metrics:
        metrics_cmd = [python, "metrics.py", "-m", str(train_dir)]
        exit_code = run_subprocess(metrics_cmd, cwd=REPO_ROOT, env=run_env)
        info["metrics_exit"] = exit_code
        if exit_code != 0:
            return info
    else:
        info["metrics_exit"] = 0

    # 4) 解析
    metrics = parse_psnr_ssim_lpips(train_dir / "results.json")
    if metrics:
        info["psnr"] = metrics.get("psnr")
        info["ssim"] = metrics.get("ssim")
        info["lpips"] = metrics.get("lpips")
        info["method"] = metrics.get("method")
    points = count_ply_points(train_dir)
    if points is not None:
        info["num_points"] = points
    # ===== v3 Group A: 读取训练元数据 (B 改动, 2026-10-19) =====
    # runtime.json 由 train_optimized.py 在 "Training complete." 前写
    runtime = parse_runtime(train_dir)
    if runtime:
        info["wall_time_sec"] = runtime.get("wall_time_sec")
        info["peak_vram_mb"] = runtime.get("peak_vram_mb")
        print(f"  [runtime] wall={runtime.get('wall_time_sec')}s peak_vram={runtime.get('peak_vram_mb')}MB")

    # 写 summary
    _write_summary(output_root, scene, group, seed, info)

    # 5) 立刻记录到 ledger + wandb + 文档 (v2 高可靠性新功能)
    _log_to_ledger_and_wandb(scene, group, seed, info)

    # 6) finish 当前 wandb run (per-iter log 已累积, 末尾 log 写完, 关闭)
    if _wb is not None:
        try:
            import wandb as _w  # type: ignore
            if getattr(_w, "run", None) is not None:
                _w.finish()
        except Exception:
            pass

    return info


def _write_summary(output_root: Path, scene: str, group: str, seed: int, info: Dict) -> None:
    """写 summary.json (per (scene, group), seeds dict)."""
    summary_path = output_root / scene / group / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = load_summary(summary_path)
    seeds = summary.setdefault("seeds", {})
    seeds[str(seed)] = info
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _log_to_ledger_and_wandb(scene: str, group: str, seed: int, info: Dict) -> None:
    """记录到 ledger + wandb + 文档 (best-effort, 失败不阻塞主流程).

    跳过策略: 如果 info['skipped']=True, 只写 doc (本地落档), 不调 wandb.
    wandb 末尾 log 由 backfill 脚本统一补, 避免在 launcher 主体里 reinit/finish 撞状态.
    """
    try:
        from scripts.exp4_doc_writer import write_run_result_to_doc  # type: ignore
        config_dict = {
            "scene": scene, "group": group, "seed": seed,
            "iterations": info.get("iterations"),
            "extra_args": info.get("extra_args", []),
            "resumed_from": info.get("resumed_from"),
        }
        write_run_result_to_doc(
            scene=scene, group=group, seed=seed,
            metrics=info, config=config_dict, extra=None,
        )
    except Exception as exc:
        print(f"  [warn] exp4_doc_writer failed: {exc!r}")

    # wandb 末尾 log: 跳过路径不调 (避免 reinit/finish 撞状态)
    if info.get("skipped"):
        return
    try:
        from scripts.exp4_wandb_logger import log_run_result  # type: ignore
        config_dict = {
            "scene": scene, "group": group, "seed": seed,
            "iterations": info.get("iterations"),
            "extra_args": info.get("extra_args", []),
            "resumed_from": info.get("resumed_from"),
        }
        try:
            log_run_result(
                scene=scene, group=group, seed=seed,
                metrics=info, config=config_dict,
            )
        except Exception as exc:
            print(f"  [warn] wandb log failed: {exc!r}")
    except ImportError as exc:
        print(f"  [warn] exp4_wandb_logger not importable: {exc!r}")


def load_summary(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="v2 通用 TAGS train+render+metrics runner")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--hold", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true",
                        help="禁用跳过已完成 case (默认: 跳过)")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="禁用 chkpnt 自动续跑 (默认: 续跑)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="强制从头跑 (跳过 chkpnt, 删 final ply). 慎用!")
    parser.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, default=[],
        help="传给 train_optimized.py 的剩余参数, 全部在 -- 后面"
    )
    parser.add_argument(
        "--images-dir", default="images_2",
        help="v3: 主实验用 images_2 (1/2 分辨率), 标准分辨率验证用 images, 备选 images_4"
    )
    args = parser.parse_args()

    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)

    # 强制重训: 删 final ply + chkpnt (慎用, 不会删已 log 的 summary)
    if args.force_retrain:
        train_dir = output_root / args.scene / args.group / "train" / f"seed{args.seed}"
        if train_dir.is_dir():
            import shutil
            for name in ["point_cloud", "test"]:
                p = train_dir / name
                if p.is_dir():
                    print(f"  [force] rm {p}")
                    shutil.rmtree(p, ignore_errors=True)
            for ck in train_dir.glob("chkpnt*.pth"):
                print(f"  [force] rm {ck}")
                ck.unlink(missing_ok=True)

    info = run_full(
        scene=args.scene,
        group=args.group,
        seed=args.seed,
        extra_args=args.extra_args,
        output_root=output_root,
        iterations=args.iterations,
        hold=args.hold,
        python=args.python,
        skip_train=args.skip_train,
        skip_render=args.skip_render,
        skip_metrics=args.skip_metrics,
        skip_completed=not args.no_skip_completed,
        auto_resume=not args.no_auto_resume,
        images_dir=getattr(args, "images_dir", "images_2"),
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
