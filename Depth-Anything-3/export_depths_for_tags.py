import glob
import os

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

IMAGES_DIR = r"D:\TAGSs\TAGSs_core\data\bicycle\images"
OUT_DIR = r"D:\TAGSs\TAGSs_core\data\bicycle\depth_any_out"
MODEL_ID = "depth-anything/DA3MONO-LARGE"
BATCH_SIZE = 4

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda")
model = DepthAnything3.from_pretrained(MODEL_ID).to(device).eval()

paths = sorted(
    p for p in glob.glob(os.path.join(IMAGES_DIR, "*"))
    if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png"}
)

for i in range(0, len(paths), BATCH_SIZE):
    batch = paths[i:i + BATCH_SIZE]
    pred = model.inference(batch)
    for path, depth in zip(batch, pred.depth):
        raw = cv2.imread(path)
        if raw is None:
            raise FileNotFoundError(path)
        orig_h, orig_w = raw.shape[:2]

        depth = np.asarray(depth, dtype=np.float32)
        inv = 1.0 / np.clip(depth, 1e-6, None)
        inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)
        gray = (inv * 255.0).astype(np.uint8)
        # TAGSs make_depth_scale expects depth PNG size to match COLMAP image size (same as V2).
        if gray.shape[:2] != (orig_h, orig_w):
            gray = cv2.resize(gray, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}.png"), gray)
        print("saved", stem)

print("done:", len(paths))