import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from read_write_model import Camera, read_model, write_model

scene = Path(r"D:\TAGSs\TAGSs_core\data\vr_room")
sparse = scene / "sparse" / "0"
src_test = Path(
    r"D:\DN-Splatter\dn-splatter-main\dn-splatter-main\datasets\room_datasets\vr_room\iphone\long_capture\test.txt"
)

ext = ".bin" if (sparse / "cameras.bin").exists() else ".txt"
cameras, images, points3D = read_model(str(sparse), ext=ext)
print("cameras", len(cameras), "images", len(images), "points3D", len(points3D))
print("cam models", sorted({c.model for c in cameras.values()}))
print("example image name", next(iter(images.values())).name)

new_cams = {}
for cid, cam in cameras.items():
    if cam.model in ("PINHOLE", "SIMPLE_PINHOLE"):
        new_cams[cid] = cam
        continue
    if cam.model == "OPENCV":
        dist = cam.params[4:]
        if dist.max() > 1e-8:
            raise SystemExit(f"camera {cid} OPENCV distortion not zero: {dist}")
        new_cams[cid] = Camera(
            id=cam.id,
            model="PINHOLE",
            width=cam.width,
            height=cam.height,
            params=cam.params[:4],
        )
        continue
    raise SystemExit(f"unsupported camera model: {cam.model}")

write_model(new_cams, images, points3D, str(sparse), ext=".bin")
write_model(new_cams, images, points3D, str(sparse), ext=".txt")
print("wrote PINHOLE cameras.bin/txt")

names = {Path(im.name).stem: im.name for im in images.values()}
test_lines = []
for raw in src_test.read_text(encoding="utf-8").splitlines():
    stem = raw.strip()
    if not stem:
        continue
    if stem not in names:
        raise SystemExit(f"test stem not in COLMAP images: {stem}")
    test_lines.append(names[stem])
(sparse / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
print("test views", len(test_lines))

depth_params = {Path(im.name).stem: {"scale": 1.0, "offset": 0.0} for im in images.values()}
(sparse / "depth_params.json").write_text(json.dumps(depth_params, indent=2), encoding="utf-8")
print("depth_params", len(depth_params))
print("test.txt head", test_lines[:3])
