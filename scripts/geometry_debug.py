"""Small, bounded geometry debug pipeline. CPU evaluation needs numpy/scipy only.

All coordinates are COLMAP world coordinates; depth is camera Z in scene units.
No monocular prior is treated as truth. See docs/GEOMETRY_DEBUG.md.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import traceback
import os
import platform
import shlex
import time
import uuid
from datetime import datetime, timezone

try:
    import numpy as np
except ImportError:
    np = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_colmap(repo, source, cfg):
    # Import the existing pure-numpy loader without scene/__init__ CUDA imports.
    spec = importlib.util.spec_from_file_location("colmap_cpu", str(repo / "scene/colmap_loader.py"))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    sparse = source / "sparse/0"
    cameras = loader.read_intrinsics_binary(str(sparse / "cameras.bin"))
    images = loader.read_extrinsics_binary(str(sparse / "images.bin"))
    points = {}
    with open(str(sparse / "points3D.bin"), "rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            p = struct.unpack("<QdddBBBd", f.read(43))
            n = struct.unpack("<Q", f.read(8))[0]
            f.seek(8 * n, 1)
            if n >= cfg["sfm_min_track"] and p[7] <= cfg["sfm_max_error"]:
                points[int(p[0])] = np.array(p[1:4])
    ordered = sorted(images.values(), key=lambda im: im.name)
    candidates = ordered[::cfg["holdout"]]
    if len(candidates) < cfg["max_views"]:
        raise ValueError("Not enough holdout cameras")
    # Fixed anchor, then strongest track overlap among holdout candidates.
    tracks = {im.id: set(int(i) for i in im.point3D_ids if int(i) in points) for im in candidates}
    anchor = candidates[0]
    selected = [anchor] + sorted(candidates[1:], key=lambda im: (-len(tracks[anchor.id] & tracks[im.id]), im.name))[:cfg["max_views"] - 1]
    centers = np.array([-im.qvec2rotmat().T.dot(im.tvec) for im in ordered])
    radius = float(np.quantile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 0.9))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Degenerate camera scale")
    views = []
    for im in selected:
        cam = cameras[im.camera_id]
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            fx, cx, cy = cam.params
            fy = fx
        else:
            raise ValueError("Undistorted pinhole cameras required: " + cam.model)
        scale = min(1., cfg["max_side"] / max(cam.width, cam.height))
        w, h = int(round(cam.width * scale)), int(round(cam.height * scale))
        sx, sy = w / cam.width, h / cam.height
        # COLMAP image coordinates use pixel centers at .5. Arrays use integers.
        K = np.array([[fx * sx, 0, cx * sx - .5], [0, fy * sy, cy * sy - .5], [0, 0, 1.]])
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = im.qvec2rotmat(), im.tvec
        ids = sorted(tracks[im.id])
        views.append(dict(id=im.id, name=im.name, width=w, height=h, K=K.tolist(), w2c=w2c.tolist(), observed_ids=ids))
    return views, points, radius, dict(registered_images=len(images), total_sparse_points=count, filtered_sparse_points=len(points))


def backproject(depth, K, w2c, mask, stride=1):
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    valid = mask[v, u] & np.isfinite(depth[v, u]) & (depth[v, u] > 0)
    uv = np.stack([u[valid], v[valid]], axis=1)
    rays = np.column_stack([uv, np.ones(len(uv))]).dot(np.linalg.inv(K).T)
    camera_xyz = rays * depth[v[valid], u[valid], None]
    xyz = (camera_xyz - w2c[:3, 3]).dot(w2c[:3, :3])
    return xyz, uv


def project(xyz, K, w2c):
    p = xyz.dot(w2c[:3, :3].T) + w2c[:3, 3]
    q = p.dot(K.T)
    uv = q[:, :2] / np.where(abs(q[:, 2:3]) > 1e-12, q[:, 2:3], 1e-12)
    return uv, p[:, 2]


def sample(depth, mask, uv, z):
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    rounded = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(np.int64)
    u, v = rounded.T
    inside = finite & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    idx = np.flatnonzero(inside)
    idx = idx[mask[v[idx], u[idx]]]
    return idx, depth[v[idx], u[idx]]


def consistency(a, b, cfg):
    xyz, source_uv = backproject(a["depth"], a["K"], a["w2c"], a["mask"], cfg["stride"])
    uv, z = project(xyz, b["K"], b["w2c"])
    idx, target = sample(b["depth"], b["mask"], uv, z)
    err = abs(z[idx] - target) / target
    tol = cfg["relative_depth_tolerance"]
    # Points behind the target depth are occluded; do not score their residuals.
    visible = z[idx] <= target * (1 + tol)
    e = err[visible]
    target_uv = np.rint(uv[idx])
    rays = np.column_stack([target_uv, np.ones(len(idx))]).dot(np.linalg.inv(b["K"]).T)
    target_xyz = (rays * target[:, None] - b["w2c"][:3, 3]).dot(b["w2c"][:3, :3])
    cycle_uv, _ = project(target_xyz, a["K"], a["w2c"])
    cycle = np.linalg.norm(cycle_uv - source_uv[idx], axis=1)[visible]
    consistent = (e <= tol) & (cycle <= cfg.get("cycle_pixel_tolerance", 1.))
    return dict(source_samples=len(xyz), target_valid=len(idx), occluded=int((~visible).sum()),
                scored=int(visible.sum()), consistent=int(consistent.sum()),
                consistent_fraction=float(consistent.mean()) if len(e) else None,
                cycle_pixel_error_mean=float(cycle.mean()) if len(cycle) else None,
                visible_absrel=float(e.mean()) if len(e) else None,
                all_target_valid_absrel=float(err.mean()) if len(err) else None,
                target_valid_fraction=len(idx) / len(xyz) if len(xyz) else None)


def sparse_metrics(bundle, view, points, radius, cfg):
    ids = view["observed_ids"]
    xyz = np.array([points[i] for i in ids], dtype=float).reshape(-1, 3)
    uv, z = project(xyz, bundle["K"], bundle["w2c"])
    idx, pred = sample(bundle["depth"], bundle["mask"], uv, z)
    e = abs(pred - z[idx])
    # Sparse reference -> surface samples only; no dense accuracy/F-score claim.
    surface, _ = backproject(bundle["depth"], bundle["K"], bundle["w2c"], bundle["mask"], cfg["stride"])
    from scipy.spatial import cKDTree
    distances = cKDTree(surface).query(xyz[idx])[0] if len(surface) and len(idx) else np.array([])
    return dict(observed_filtered_points=len(ids), scored=len(idx),
                valid_fraction=len(idx) / len(ids) if len(ids) else None,
                depth_absrel=float((e / z[idx]).mean()) if len(e) else None,
                depth_rmse_scene_units=float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
                sparse_to_surface_mean_over_radius=float(distances.mean() / radius) if len(distances) else None,
                sparse_to_surface_coverage={str(t): float((distances <= t * radius).mean()) if len(distances) else None for t in cfg["radius_thresholds"]})


def render_exports(repo, checkpoint, views, cfg, out):
    sys.path.insert(0, str(repo))
    import torch
    from gaussian_renderer import GaussianModel
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from scene.cameras import MiniCam
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU geometry tests are independent")
    gaussians = GaussianModel(cfg["sh_degree"])
    gaussians.load_ply(str(checkpoint))
    bg = torch.zeros(3, device="cuda")
    white = torch.ones_like(gaussians.get_xyz)
    calibration = None
    with torch.no_grad():
        for view in views:
            K, w2c = np.array(view["K"]), np.array(view["w2c"])
            w, h = view["width"], view["height"]
            P = torch.zeros((4, 4), device="cuda", dtype=torch.float32)
            P[0, 0], P[1, 1] = 2 * K[0, 0] / w, 2 * K[1, 1] / h
            P[0, 2], P[1, 2] = 2 * (K[0, 2] + .5) / w - 1, 2 * (K[1, 2] + .5) / h - 1
            P[2, 2], P[2, 3], P[3, 2] = 100 / 99.99, -1 / 99.99, 1
            W = torch.tensor(w2c, device="cuda", dtype=torch.float32).T.contiguous()
            camera = MiniCam(w, h, 2 * math.atan(h / (2 * K[1, 1])), 2 * math.atan(w / (2 * K[0, 0])), .01, 100., W, W @ P.T)
            camera.image_name = Path(view["name"]).stem
            settings = GaussianRasterizationSettings(image_height=h, image_width=w,
                tanfovx=w / (2 * K[0, 0]), tanfovy=h / (2 * K[1, 1]), bg=bg,
                scale_modifier=1., viewmatrix=W, projmatrix=camera.full_proj_transform,
                sh_degree=0, campos=camera.camera_center, prefiltered=False, debug=False, antialiasing=False)
            rasterizer = GaussianRasterizer(raster_settings=settings)
            def raster(xyz, opacity, scales, rotations, colors):
                # None is required here: the repository render wrapper passes an
                # empty SH tensor, rejected by the installed Python extension.
                return rasterizer(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
                    colors_precomp=colors, opacities=opacity, scales=scales,
                    rotations=rotations, cov3D_precomp=None)
            if calibration is None:
                xyz = (np.array([[0., 0., 2.]]) - w2c[:3, 3]).dot(w2c[:3, :3])
                rgb_test, _, inv_test = raster(torch.tensor(xyz, device="cuda", dtype=torch.float32),
                    torch.full((1, 1), .5, device="cuda"), torch.full((1, 3), .1, device="cuda"),
                    torch.tensor([[1., 0., 0., 0.]], device="cuda"), torch.ones((1, 3), device="cuda"))
                valid = rgb_test[0] > .01
                if not valid.any():
                    raise RuntimeError("Rasterizer depth calibration produced no pixels")
                recovered = rgb_test[0][valid] / inv_test[0][valid]
                error = float((recovered - 2).abs().max().item())
                if not math.isfinite(error) or error > 1e-3:
                    raise RuntimeError("Installed rasterizer inverse-depth semantics failed calibration")
                calibration = dict(known_Z=2., max_absolute_error=error, valid_pixels=int(valid.sum().item()))
            rgb, _, raw_inv = raster(gaussians.get_xyz, gaussians.get_opacity,
                gaussians.get_scaling, gaussians.get_rotation, white)
            inv = raw_inv[0].cpu().numpy()
            alpha = rgb[0].cpu().numpy()
            if inv.shape != (h, w):
                raise ValueError("Unexpected rasterizer depth shape")
            mask = np.isfinite(inv) & np.isfinite(alpha) & (inv > 1e-8) & (alpha >= cfg["alpha_min"])
            depth = np.zeros_like(inv)
            depth[mask] = alpha[mask] / inv[mask]
            np.savez_compressed(str(out / (str(view["id"]) + ".npz")), depth=depth, alpha=alpha, weighted_invdepth=inv, mask=mask, K=K, w2c=w2c)
    return dict(torch_version=torch.__version__, cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(0), gaussians=int(gaussians.get_xyz.shape[0]), depth_calibration=calibration)


def evaluate(out, views, points, radius, cfg, export_dir=None):
    bundles, rows, clouds = {}, [], []
    for view in views:
        with np.load(str((export_dir or out) / (str(view["id"]) + ".npz"))) as f:
            b = {key: f[key] for key in f.files}
        bundles[view["id"]] = b
        cloud, _ = backproject(b["depth"], b["K"], b["w2c"], b["mask"], cfg["stride"])
        clouds.append(cloud)
        rows.append(dict(id=view["id"], name=view["name"], valid_pixels=int(b["mask"].sum()),
                         valid_pixel_fraction=float(b["mask"].mean()), surface_samples=len(cloud),
                         sfm=sparse_metrics(b, view, points, radius, cfg)))
    pairs = []
    for a in views:
        for b in views:
            if a["id"] == b["id"]:
                continue
            shared = len(set(a["observed_ids"]) & set(b["observed_ids"]))
            row = dict(source=a["id"], target=b["id"], shared_tracks=shared)
            if shared < cfg["min_shared_tracks"]:
                row["status"] = "skipped_insufficient_overlap"
            else:
                row.update(consistency(bundles[a["id"]], bundles[b["id"]], cfg))
                row["status"] = "ok" if row["scored"] else "no_valid_samples"
            pairs.append(row)
    cloud = np.concatenate(clouds)
    np.save(str(out / "surface_samples.npy"), cloud)
    with open(str(out / "surface_samples.ply"), "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % len(cloud))
        np.savetxt(f, cloud, fmt="%.7g")
    return dict(views=rows, directed_pairs=pairs, surface_points=len(cloud),
                depth_semantics="alpha / sum(T_i * alpha_i / Z_i): reciprocal conditional mean inverse Z",
                caveat="SfM sparse reference is training-correlated; consistency is not absolute accuracy; no dense-GT Chamfer/F-score")


def create_run_dir(base, stage):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + stage + "_" + uuid.uuid4().hex[:8]
    out = Path(base) / run_id
    out.mkdir(parents=True, exist_ok=False)
    return out


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(result):
    views = result["views"]
    pairs = result["directed_pairs"]
    ok = [p for p in pairs if p["status"] == "ok"]
    n = sum(v["sfm"]["scored"] for v in views)
    candidates = sum(v["sfm"]["observed_filtered_points"] for v in views)
    scored = sum(p["scored"] for p in ok)
    target_n = sum(p["target_valid"] for p in ok)
    def sfm_mean(key):
        available = [v for v in views if v["sfm"][key] is not None and v["sfm"]["scored"]]
        denom = sum(v["sfm"]["scored"] for v in available)
        return sum(v["sfm"][key] * v["sfm"]["scored"] for v in available) / denom if denom else None
    return dict(views=len(views), surface_points=result["surface_points"],
        depth_valid_pixel_fraction=sum(v["valid_pixel_fraction"] for v in views) / len(views) if views else None,
        sfm_observations=n, sfm_candidates=candidates, sfm_valid_fraction=n / candidates if candidates else None,
        sfm_depth_absrel_observation_weighted=sfm_mean("depth_absrel"),
        sfm_depth_rmse_scene_units=math.sqrt(sum(v["sfm"]["depth_rmse_scene_units"] ** 2 * v["sfm"]["scored"] for v in views if v["sfm"]["scored"]) / n) if n else None,
        sparse_to_surface_mean_over_radius=sfm_mean("sparse_to_surface_mean_over_radius"),
        sparse_coverage={key: sum(v["sfm"]["sparse_to_surface_coverage"][key] * v["sfm"]["scored"] for v in views if v["sfm"]["scored"]) / n if n else None for key in (views[0]["sfm"]["sparse_to_surface_coverage"] if views else [])},
        total_directed_pairs=len(pairs), evaluated_directed_pairs=len(ok),
        skipped_pairs=sum(p["status"] == "skipped_insufficient_overlap" for p in pairs),
        no_valid_pairs=sum(p["status"] == "no_valid_samples" for p in pairs),
        scored_pair_samples=scored, consistent_pair_samples=sum(p["consistent"] for p in ok),
        consistency_fraction_sample_weighted=sum(p["consistent"] for p in ok) / scored if scored else None,
        visible_depth_absrel=sum(p["visible_absrel"] * p["scored"] for p in ok) / scored if scored else None,
        all_target_valid_depth_absrel=sum(p["all_target_valid_absrel"] * p["target_valid"] for p in ok) / target_n if target_n else None,
        occlusion_candidate_fraction=sum(p["occluded"] for p in ok) / target_n if target_n else None,
        cycle_pixel_error_mean=sum(p["cycle_pixel_error_mean"] * p["scored"] for p in ok) / scored if scored else None)


def save_measurements(out, result):
    write_json(out / "metrics.json", result)
    summary = summarize(result)
    write_json(out / "summary.json", dict(status="ok", metrics=summary))
    rows = []
    for view in result["views"]:
        row = {k: v for k, v in view.items() if k != "sfm"}
        row.update({"sfm_" + k: v for k, v in view["sfm"].items() if k != "sparse_to_surface_coverage"})
        row.update({"sfm_coverage_" + k: v for k, v in view["sfm"]["sparse_to_surface_coverage"].items()})
        rows.append(row)
    write_csv(out / "per_view.csv", rows)
    write_csv(out / "per_pair.csv", result["directed_pairs"])
    text = ["# Geometry evaluation result", "", "Status: ok", "", "| Metric | Value |", "|---|---:|"]
    text.extend("| {} | {} |".format(k, v) for k, v in summary.items())
    text.extend(["", result["caveat"], "", "Depth: " + result["depth_semantics"],
                 "", "Counts are observations/samples, not unique SfM points. RMSE is in COLMAP scene units.",
                 "Compare only runs with identical camera selection, scale, masks, resolution and thresholds."])
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def validate_config(cfg):
    for key in ["max_views", "holdout", "stride", "max_side", "sfm_min_track", "sh_degree"]:
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool):
            raise ValueError("Integer required: " + key)
    if cfg["max_views"] < 2 or cfg["holdout"] < 1 or cfg["stride"] < 1 or cfg["max_side"] < 8:
        raise ValueError("Invalid view/resolution/stride bounds")
    if not 0 < cfg["alpha_min"] <= 1 or cfg["sfm_min_track"] < 1 or cfg["sh_degree"] < 0:
        raise ValueError("Invalid alpha, track length or SH degree")
    for key in ["relative_depth_tolerance", "cycle_pixel_tolerance", "sfm_max_error"]:
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("Nonnegative finite value required: " + key)
    if not cfg["radius_thresholds"] or any(not math.isfinite(t) or t <= 0 for t in cfg["radius_thresholds"]):
        raise ValueError("Positive finite radius thresholds required")


def execute(args, repo, cfg, out):
    if np is None:
        raise RuntimeError("numpy is missing; install CPU requirements")
    validate_config(cfg)
    source, checkpoint = repo / cfg["source_path"], repo / cfg["checkpoint"]
    if args.stage in ("preflight", "run") and not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(checkpoint))
    print("Loading SfM cameras and sparse reference...", flush=True)
    views, points, radius, counts = load_colmap(repo, source, cfg)
    manifest = dict(config=cfg, repo=str(repo), source=str(source), checkpoint=str(checkpoint),
        checkpoint_exists=checkpoint.is_file(), radius_scene_units=radius, counts=counts,
        views=views, python=sys.version, script_sha256=sha256(Path(__file__)),
        pixel_convention="integer array coordinates; COLMAP principal point scaled then minus 0.5",
        inputs_sha256={name: sha256(source / "sparse/0" / name) for name in ["cameras.bin", "images.bin", "points3D.bin"]})
    export_dir = out
    if args.stage == "evaluate":
        if not args.input_run:
            raise ValueError("--input-run is required for evaluate; use the exact export run directory")
        export_dir = args.input_run.resolve()
        original = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
        for key in ["inputs_sha256", "views"]:
            if original[key] != manifest[key]:
                raise ValueError("Export manifest mismatch: " + key)
        # Paths may differ after migration; all measurement settings must match.
        for key in cfg:
            if key not in ("method", "source_path", "checkpoint", "output") and original["config"].get(key) != cfg[key]:
                raise ValueError("Export measurement setting mismatch: " + key)
        manifest["input_run"] = str(export_dir)
        manifest["export_manifest_sha256"] = sha256(export_dir / "manifest.json")
        manifest["export_checkpoint_sha256"] = original.get("checkpoint_sha256")
        manifest["export_runtime"] = original.get("runtime")
        manifest["depth_files_sha256"] = {str(v["id"]): sha256(export_dir / (str(v["id"]) + ".npz")) for v in views}
    else:
        manifest["checkpoint_sha256"] = sha256(checkpoint)
    write_json(out / "manifest.json", manifest)
    if args.stage == "run":
        print("Rendering {} selected views...".format(len(views)), flush=True)
        manifest["runtime"] = render_exports(repo, checkpoint, views, cfg, out)
        write_json(out / "manifest.json", manifest)
    summary = None
    if args.stage != "preflight":
        print("Evaluating sparse reference and directed view pairs...", flush=True)
        summary = save_measurements(out, evaluate(out, views, points, radius, cfg, export_dir))
    return dict(counts=counts, selected_views=[v["name"] for v in views], metrics=summary)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=["preflight", "run", "evaluate"], default="run")
    p.add_argument("--output", type=Path, help="Output BASE directory; a unique run subdirectory is always created")
    p.add_argument("--input-run", type=Path, help="Existing exported run directory for CPU evaluate")
    p.add_argument("--source-path", help="Override config source_path (absolute or relative to --repo)")
    p.add_argument("--checkpoint", help="Override config checkpoint (absolute or relative to --repo)")
    p.add_argument("--method", help="Override method label")
    p.add_argument("--max-views", type=int)
    p.add_argument("--max-side", type=int)
    arguments = list(sys.argv[1:] if argv is None else argv)
    parse_error = None
    try:
        args = p.parse_args(arguments)
    except SystemExit as exc:
        if exc.code == 0:  # --help is not an experiment
            return 0
        parse_error = ValueError("Invalid command-line arguments; use --help. argv=" + repr(arguments))
        args = argparse.Namespace(repo=Path.cwd(), config=Path("missing-config"), output=Path.cwd() / "geometry_records", stage="cli_error")
    repo = args.repo.resolve()
    cfg, config_error = {}, None
    if parse_error is None:
        try:
            cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
            if not isinstance(cfg, dict):
                raise ValueError("Config must be a JSON object")
            for key in ["source_path", "checkpoint", "method", "max_views", "max_side"]:
                value = getattr(args, key)
                if value is not None:
                    cfg[key] = value
        except Exception as exc:
            config_error = exc
            cfg = {}
    try:
        base = args.output or Path(cfg.get("output", "output/geometry_records"))
    except (TypeError, ValueError) as exc:
        config_error = exc
        base = repo / "output/geometry_records"
    if not base.is_absolute():
        base = repo / base
    out = create_run_dir(base, args.stage)
    started = time.monotonic()
    record = dict(schema_version=2, run_id=out.name, stage=args.stage, status="running",
        started_utc=datetime.now(timezone.utc).isoformat(), output_dir=str(out.resolve()),
        repo=str(repo), cwd=str(Path.cwd()), argv=arguments,
        command_posix=" ".join(shlex.quote(x) for x in [sys.executable, str(Path(__file__).resolve())] + arguments),
        hostname=platform.node(), platform=platform.platform(), python=sys.version,
        executable=sys.executable, config_path=str(args.config.resolve()), script_sha256=sha256(Path(__file__)))
    write_json(out / "record.json", record)
    write_json(out / "config_snapshot.json", cfg)
    write_json(out / "summary.json", dict(status="running", metrics=None))
    code = 0
    with (out / "console.log").open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("RUN_DIR=" + str(out.resolve()), flush=True)
            try:
                if parse_error is not None:
                    raise parse_error
                if args.config.is_file():
                    record["config_sha256"] = sha256(args.config)
                    (out / "config_original.json").write_bytes(args.config.read_bytes())
                if config_error is not None:
                    raise config_error
                record.update(execute(args, repo, cfg, out))
                record["status"] = "ok"
            except BaseException as exc:
                code = 130 if isinstance(exc, KeyboardInterrupt) else 1
                record.update(status="interrupted" if code == 130 else "failed",
                    error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
            finally:
                record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, exit_code=code)
                if record["status"] != "ok" or args.stage == "preflight":
                    write_json(out / "summary.json", dict(status=record["status"], metrics=None, error=record.get("error")))
                write_json(out / "record.json", record)
                write_json(out / (args.stage + "_status.json"), record)
                print(json.dumps(record, indent=2), flush=True)
    artifacts = [dict(path=f.name, bytes=f.stat().st_size, sha256=sha256(f)) for f in sorted(out.iterdir()) if f.is_file()]
    write_json(out / "artifacts.json", dict(run_id=out.name, files=artifacts))
    return code


if __name__ == "__main__":
    sys.exit(main())


