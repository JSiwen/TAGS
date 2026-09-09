#
# Re-export point_cloud.ply from an existing trained folder (e.g. recover truncated saves).
# Avoids loading COLMAP cameras — only reads/writes the Gaussian PLY.
#

import os
import shutil
from argparse import ArgumentParser

from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state


def _iteration_numbers(point_cloud_root):
    out = []
    for fname in os.listdir(point_cloud_root):
        if fname.startswith("iteration_") and os.path.isdir(os.path.join(point_cloud_root, fname)):
            try:
                out.append(int(fname.split("_")[-1]))
            except ValueError:
                continue
    return sorted(out, reverse=True)


def _ply_expected_filesize_min(path):
    """Lower-bound expected file size from ASCII header (detect truncated binary PLY without scanning vertices)."""
    with open(path, "rb") as f:
        first = f.readline()
        if not first.startswith(b"ply"):
            return None
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                return None
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header_end = f.tell()
    text = b"".join(header_lines).decode("ascii", errors="replace").splitlines()
    n_vert = None
    n_float_props = 0
    in_vertex = False
    for line in text:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            if parts[1] == "vertex":
                in_vertex = True
                n_vert = int(parts[2])
            else:
                in_vertex = False
        elif in_vertex and parts[0] == "property":
            if any(t in parts for t in ("float", "float32", "double")):
                n_float_props += 1
    if n_vert is None or n_float_props == 0:
        return None
    bytes_per_row = n_float_props * 4
    return header_end + n_vert * bytes_per_row


def latest_readable_iteration(model_path):
    pc_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_root):
        raise RuntimeError(f"No point_cloud directory: {pc_root}")
    for it in _iteration_numbers(pc_root):
        p = os.path.join(pc_root, f"iteration_{it}", "point_cloud.ply")
        if not os.path.isfile(p):
            continue
        expected = _ply_expected_filesize_min(p)
        sz = os.path.getsize(p)
        if expected is not None and sz + 4096 < expected:
            print(f"[export_ply] Skip truncated (size {sz} < ~{expected}): iteration_{it}")
            continue
        return it
    raise RuntimeError(f"No plausible complete point_cloud.ply under {pc_root}")


def main():
    parser = ArgumentParser(description="Re-export point_cloud.ply from iteration_* (safe tmp + replace).")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument(
        "--load_iteration",
        type=int,
        default=-1,
        help="-1: pick highest iteration with a complete-sized PLY; else load that iteration explicitly.",
    )
    parser.add_argument(
        "--save_iteration",
        type=int,
        default=-1,
        help="-1: save into the same iteration folder as loaded; else write point_cloud/iteration_<n>/",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--prune_truncated",
        action="store_true",
        help="Remove point_cloud/iteration_* folders whose PLY is smaller than header implies (e.g. disk-full truncation).",
    )
    args = get_combined_args(parser)

    dataset = model.extract(args)
    safe_state(args.quiet)

    pc_root = os.path.join(dataset.model_path, "point_cloud")
    if args.prune_truncated and os.path.isdir(pc_root):
        for it in list(_iteration_numbers(pc_root)):
            ddir = os.path.join(pc_root, f"iteration_{it}")
            p = os.path.join(ddir, "point_cloud.ply")
            if not os.path.isfile(p):
                continue
            expected = _ply_expected_filesize_min(p)
            sz = os.path.getsize(p)
            if expected is not None and sz + 4096 < expected:
                print(f"[export_ply] Removing truncated iteration_{it} ({sz} B vs ~{expected} B expected)")
                shutil.rmtree(ddir)

    load_it = args.load_iteration
    if load_it < 0:
        load_it = latest_readable_iteration(dataset.model_path)
        print(f"[export_ply] Using latest readable iteration: {load_it}")

    save_it = args.save_iteration if args.save_iteration >= 0 else load_it
    if save_it != load_it:
        print(
            f"[export_ply] Note: loading iteration_{load_it}, saving as iteration_{save_it} "
            "(folder name does not change model quality)."
        )

    du = shutil.disk_usage(dataset.model_path)
    free_gb = du.free / (1024**3)
    if free_gb < 1.0:
        print(f"[export_ply] Warning: only {free_gb:.2f} GiB free here — need enough space for full PLY.")

    ply_in = os.path.join(dataset.model_path, "point_cloud", f"iteration_{load_it}", "point_cloud.ply")
    if not os.path.isfile(ply_in):
        raise FileNotFoundError(ply_in)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_in, dataset.train_test_exp)

    out_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{save_it}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "point_cloud.ply")
    tmp_path = out_path + ".tmp"

    if os.path.isfile(tmp_path):
        os.remove(tmp_path)

    gaussians.save_ply(tmp_path)
    os.replace(tmp_path, out_path)

    mb = os.path.getsize(out_path) / (1024**2)
    print(f"[export_ply] Done: {out_path} ({mb:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
