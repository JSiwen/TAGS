#!/usr/bin/env python3
"""Export scalar Depth Anything V2 PNGs with input/model provenance."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, default=Path(__file__).resolve().parents[1] / 'Depth-Anything-V2')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', choices=('vits', 'vitb', 'vitl'), default='vitl')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Refusing nonempty output directory: {}'.format(args.output))
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files or len({p.stem for p in files}) != len(files):
        raise SystemExit('Image set is empty or has duplicate stems')
    sys.path.insert(0, str(args.model_root.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2
    configs = {
        'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(encoder=args.encoder, **configs[args.encoder])
    model.load_state_dict(torch.load(str(args.weights), map_location='cpu'))
    model = model.to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'status': 'running', 'encoder': args.encoder, 'input_size': args.input_size,
        'weights': str(args.weights.resolve()), 'weights_sha256': sha256(args.weights),
        'model_source_sha256': {str(p.relative_to(args.model_root)): sha256(p) for p in sorted((args.model_root / 'depth_anything_v2').rglob('*.py'))},
        'torch': torch.__version__, 'device': args.device,
        'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        'encoding': 'uint8 monocular inverse depth, per-image min/max normalized; scalar PNG',
        'command': sys.argv, 'files': [],
    }
    manifest_path = args.output / 'generation_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with torch.no_grad():
        for i, path in enumerate(files):
            rgb = cv2.imread(str(path))
            if rgb is None:
                raise ValueError('Unreadable image: {}'.format(path))
            depth = model.infer_image(rgb, args.input_size)
            if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or depth.max() <= depth.min():
                raise ValueError('Invalid depth prediction: {}'.format(path))
            scalar = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            target = args.output / (path.stem + '.png')
            if not cv2.imwrite(str(target), scalar):
                raise IOError('Could not write {}'.format(target))
            manifest['files'].append({'image': path.name, 'image_sha256': sha256(path),
                                      'depth': target.name, 'depth_sha256': sha256(target),
                                      'shape_hw': list(scalar.shape)})
            print('{}/{} {}'.format(i + 1, len(files), path.name), flush=True)
    manifest['status'] = 'completed'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
