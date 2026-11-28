"""Frozen Delivery Area follow-up; preparation is separate from expensive training.

Python 3.7 for TAGS stages; Python 3.10 and modern torch for depth/normals.
See docs/DELIVERY_AREA_EXECUTION_20261128.md. Never uses scan GT for priors.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import re
import shlex
import subprocess
import sys
import statistics
import tarfile

ROOT = Path(os.environ.get('ROOT', '/autodl-fs/data/tags_architecture_ablation_20261112'))
CODE = ROOT / 'code'
FOLLOW = ROOT / 'audit/followup_20261116'
OLD = Path('/root/autodl-tmp/TAGS/experiments/playground_projection_v2')
DATA = Path('/autodl-fs/data/eth3d_delivery_area')
RAW = DATA / 'raw/delivery_area'
SCENE = DATA / 'tags_delivery_area'
DEPTH = DATA / 'delivery_area_mono_prior'
OUT = ROOT / 'audit/delivery_area_projection_v2_confirm_20261128'
HERE = Path(__file__).resolve().parent
TOOLS = Path('/autodl-fs/data/eth3d_courtyard/tools/courtyard_commands_20261124')
PY = '/root/miniconda3/envs/TAGS/bin/python'
GROUPS = ['3dgs', 'tags_nogeo', 'tags_depth_only', 'tags_depth_dsine_w1']
WEIGHTS = {'depth_anything_v2_vitl.pth': 'a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345',
           'dsine.pt': '629472c66ea9ec83bab6ba7c957ad7d983bb9c91063a54ad74ff5b8229cbf22f'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(cmd, log=None, env=None, cwd=None):
    print(' '.join(shlex.quote(str(x)) for x in cmd), flush=True)
    if log:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as f:
            subprocess.run(list(map(str, cmd)), check=True, stdout=f,
                           stderr=subprocess.STDOUT, env=env, cwd=cwd)
    else:
        subprocess.run(list(map(str, cmd)), check=True, env=env, cwd=cwd)


def colmap():
    # Avoid importing TAGS's utils package into the DSINE process.
    spec = importlib.util.spec_from_file_location('delivery_area_colmap', str(CODE / 'utils/read_write_model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model():
    return colmap().read_model(str(SCENE / 'sparse/0'), '.bin')


def files_hashes(paths):
    return {str(p): digest(p) for p in sorted(paths)}


def verify_files(mapping):
    for name, expected in mapping.items():
        if digest(name) != expected:
            raise RuntimeError('Changed input: ' + name)


def prepare():
    from PIL import Image
    marker = SCENE / 'preparation.json'
    if marker.exists():
        verify_files(load(marker)['files'])
        print('Prepared scene verified; reuse.')
        return
    if SCENE.exists() and any(SCENE.iterdir()):
        raise RuntimeError('Partial/foreign scene exists; inspect it before retry: ' + str(SCENE))
    reader = colmap()
    cams, images, points = reader.read_model(str(RAW / 'dslr_calibration_undistorted'), '.txt')
    assert len(images) == 44 and all(c.model == 'PINHOLE' for c in cams.values())
    names = [Path(i.name).name for i in images.values()]
    assert len(set(names)) == len(names)
    (SCENE / 'images_4').mkdir(parents=True)
    (SCENE / 'sparse/0').mkdir(parents=True)
    rewritten = {}
    sources = []
    for key, im in images.items():
        src = RAW / 'images' / im.name
        cam = cams[im.camera_id]
        with Image.open(str(src)) as image:
            assert image.size == (cam.width, cam.height)
            size = (round(cam.width / 4), round(cam.height / 4))
            image.convert('RGB').resize(size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS).save(
                str(SCENE / 'images_4' / Path(im.name).name), quality=95, subsampling=0)
        sources.append(src)
        rewritten[key] = im._replace(name=Path(im.name).name)
    reader.write_model(cams, rewritten, points, str(SCENE / 'sparse/0'), '.bin')
    names.sort()
    outputs = list((SCENE / 'images_4').iterdir()) + list((SCENE / 'sparse/0').iterdir())
    save(marker, dict(count=len(names), train=[n for i, n in enumerate(names) if i % 8],
                      test=names[::8], resize='round(W/4),round(H/4); PIL LANCZOS; JPEG q95 subsampling0',
                      files=files_hashes(outputs), source_files=files_hashes(sources)))
    print('Prepared 44 cameras; 38 train / 6 test. Original native COLMAP K preserved.')


def weight(name):
    p = Path('/autodl-fs/data/eth3d_courtyard/tools/weights') / name
    assert digest(p) == WEIGHTS[name], 'Weight hash mismatch: ' + str(p)
    return p


def encode_depth(raw):
    import numpy as np
    assert raw.ndim == 2 and np.isfinite(raw).all()
    lo, hi = float(raw.min()), float(raw.max())
    assert hi > lo, 'Degenerate prediction'
    return np.rint((raw - lo) / (hi - lo) * 65535).astype(np.uint16)


def depth():
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(CODE / 'Depth-Anything-V2'))
    from depth_anything_v2.dpt import DepthAnythingV2
    assert not (OUT / 'audit/frozen_protocol.json').exists(), 'Protocol frozen: do not regenerate priors.'
    w = weight('depth_anything_v2_vitl.pth')
    assert torch.cuda.is_available()
    network = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    network.load_state_dict(torch.load(str(w), map_location='cpu', weights_only=True), strict=True)
    network.cuda().eval()
    for d in ['raw', 'scalar']:
        (DEPTH / d).mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for src in sorted((SCENE / 'images_4').iterdir()):
            raw = network.infer_image(cv2.imread(str(src)), input_size=518).astype(np.float32)
            png = encode_depth(raw)
            np.save(str(DEPTH / 'raw' / (src.stem + '.npy')), raw)
            dest = DEPTH / 'scalar' / (src.stem + '.png')
            assert cv2.imwrite(str(dest), png)
            rows.append(dict(image=src.name, image_sha256=digest(src), png_sha256=digest(dest), shape=list(raw.shape)))
            print(src.name, raw.shape, flush=True)
    save(DEPTH / 'prediction.json', dict(weights_sha256=WEIGHTS[w.name], input_size=518,
         encoding='rint per-image minmax uint16 / 65535; larger=nearer', rows=rows,
         source_files=files_hashes((CODE / 'Depth-Anything-V2/depth_anything_v2').rglob('*.py')),
         independent_scan_used=False, torch=torch.__version__))


def fit_scale(predicted, reference):
    import numpy as np
    assert len(predicted) >= 10 and np.isfinite(predicted).all() and np.isfinite(reference).all()
    a, b = np.median(predicted), np.median(reference)
    spread_a, spread_b = np.mean(np.abs(predicted - a)), np.mean(np.abs(reference - b))
    assert spread_a > 1e-8 and spread_b > 1e-8
    scale = float(spread_b / spread_a)
    return dict(scale=scale, offset=float(b - scale * a))


def calibrate():
    import cv2
    import numpy as np
    assert not (OUT / 'audit/frozen_protocol.json').exists(), 'Protocol frozen: do not recalibrate.'
    cams, images, points = model()
    params, rows = {}, []
    for im in sorted(images.values(), key=lambda x: x.name):
        cam = cams[im.camera_id]
        encoded = cv2.imread(str(DEPTH / 'scalar' / (Path(im.name).stem + '.png')), cv2.IMREAD_UNCHANGED)
        assert encoded.dtype == np.uint16
        mono = encoded.astype(np.float32) / 65535.0
        h, w = mono.shape
        keep = [i for i, pid in enumerate(im.point3D_ids) if pid in points and points[pid].error <= 2 and len(points[pid].image_ids) >= 3]
        xyz = np.array([points[im.point3D_ids[i]].xyz for i in keep])
        assert len(xyz) >= 10, im.name
        z = (xyz.dot(im.qvec2rotmat().T) + im.tvec)[:, 2]
        xy = im.xys[keep] * [w / cam.width, h / cam.height] - 0.5
        valid = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
        xy = xy[valid].astype(np.float32)
        sample = cv2.remap(mono, xy[:, 0, None], xy[:, 1, None], cv2.INTER_LINEAR).ravel()
        ref = 1.0 / z[valid]
        param = fit_scale(sample, ref)
        params[Path(im.name).stem] = param
        rows.append(dict(image=im.name, sparse_samples=len(ref), positive_fraction=float((mono * param['scale'] + param['offset'] > 0).mean()),
                         sparse_inverse_depth_mae=float(np.abs(sample * param['scale'] + param['offset'] - ref).mean())))
    save(DEPTH / 'depth_params_corrected.json', params)
    save(DEPTH / 'calibration_audit.json', dict(rows=rows, fit='median + mean absolute deviation',
        coordinates='COLMAP x*sx-0.5, y*sy-0.5; sx and sy independent', sparse_error_max=2, min_track=3,
        independent_scan_used=False, note='SfM alignment protocol; not independent geometric ground truth'))


def normals():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import geffnet
    from PIL import Image
    sys.path.insert(0, str(TOOLS / 'dsine_source'))
    from models.dsine.v02 import DSINE_v02
    assert not (OUT / 'audit/frozen_protocol.json').exists(), 'Protocol frozen: do not regenerate priors.'
    args = argparse.Namespace(NNET_encoder_B=5, NNET_decoder_NF=2048, NNET_decoder_BN=False,
        NNET_decoder_down=8, NNET_learned_upsampling=True, NRN_prop_ps=5, NRN_num_iter_train=5,
        NRN_num_iter_test=5, NRN_ray_relu=True, NNET_output_dim=3, NNET_feature_dim=64, NNET_hidden_dim=64)
    w = weight('dsine.pt')
    create = geffnet.create_model
    # Full checkpoint includes the encoder: avoid an unused network download.
    geffnet.create_model = lambda name, **kw: create(name, **dict(kw, pretrained=False))
    try:
        network = DSINE_v02(args).cuda().eval()
    finally:
        geffnet.create_model = create
    state = torch.load(str(w), map_location='cpu', weights_only=False)['model']
    network.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    cams, images, _ = model()
    dest = SCENE / 'normals_dsine'
    dest.mkdir(exist_ok=True)
    mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)
    rows = []
    with torch.inference_mode():
        for im in sorted(images.values(), key=lambda x: x.name):
            src = SCENE / 'images_4' / im.name
            rgb = np.array(Image.open(str(src)).convert('RGB'), dtype=np.float32) / 255
            h, w_img = rgb.shape[:2]
            dh, dw = (-h) % 32, (-w_img) % 32
            top, left = dh // 2, dw // 2
            cam = cams[im.camera_id]
            fx, fy, cx, cy = cam.params
            # DSINE defines pixel centres as x+.5, not integer x.
            k = torch.tensor([[fx*w_img/cam.width, 0, cx*w_img/cam.width+left],
                              [0, fy*h/cam.height, cy*h/cam.height+top], [0, 0, 1]],
                             dtype=torch.float32, device='cuda')[None]
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda()
            tensor = (F.pad(tensor, (left, dw-left, top, dh-top), value=0) - mean) / std
            pred = network(tensor, intrins=k, mode='test')[-1][0, :, top:top+h, left:left+w_img]
            pred = F.normalize(pred, dim=0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            assert pred.shape == (h, w_img, 3) and np.isfinite(pred).all()
            path = dest / (Path(im.name).stem + '.npy')
            np.save(str(path), pred)
            rows.append(dict(image=im.name, image_sha256=digest(src), normal_sha256=digest(path)))
            print(im.name, pred.shape, flush=True)
    save(dest / 'manifest.json', dict(weights_sha256=WEIGHTS[w.name], rows=rows,
         source_files=files_hashes((TOOLS / 'dsine_source').rglob('*.py')),
         coordinates='camera xyz; signed; DSINE raw scaled COLMAP K; no flip', torch=torch.__version__))


def identity():
    v = load(OLD / 'audit/projection_validation.json')
    assert v['pass'] is True
    for label, repo in [('tags', CODE), ('baseline', ROOT / '3dgs_seeded')]:
        head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
        patch = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD'])
        assert head == v[label + '_commit'], label + ' commit changed'
        assert hashlib.sha256(patch).hexdigest() == v[label + '_patch_sha256'], label + ' patch changed'
    return v


def audit():
    import cv2
    import numpy as np
    from PIL import Image
    identity()
    verify_files(load(SCENE / 'preparation.json')['files'])
    for manifest in [DEPTH / 'prediction.json', SCENE / 'normals_dsine/manifest.json']:
        verify_files(load(manifest)['source_files'])
    dp_record = load(DEPTH / 'prediction.json')
    nr_record = load(SCENE / 'normals_dsine/manifest.json')
    assert dp_record['weights_sha256'] == WEIGHTS['depth_anything_v2_vitl.pth']
    assert nr_record['weights_sha256'] == WEIGHTS['dsine.pt']
    depth_rows = {r['image']: r for r in dp_record['rows']}
    normal_rows = {r['image']: r for r in nr_record['rows']}
    cams, images, _ = model()
    params = load(DEPTH / 'depth_params_corrected.json')
    median = np.median([p['scale'] for p in params.values()])
    rows, paths = [], [SCENE / 'preparation.json', DEPTH / 'depth_params_corrected.json',
                       DEPTH / 'prediction.json', DEPTH / 'calibration_audit.json', SCENE / 'normals_dsine/manifest.json']
    assert len(images) == 44 and len(params) == 44
    for im in sorted(images.values(), key=lambda x: x.name):
        stem = Path(im.name).stem
        img = SCENE / 'images_4' / im.name
        dp, npy = DEPTH / 'scalar' / (stem + '.png'), SCENE / 'normals_dsine' / (stem + '.npy')
        assert digest(img) == depth_rows[im.name]['image_sha256'] == normal_rows[im.name]['image_sha256'], im.name
        assert digest(dp) == depth_rows[im.name]['png_sha256'], im.name
        assert digest(npy) == normal_rows[im.name]['normal_sha256'], im.name
        width, height = Image.open(str(img)).size
        depth_arr = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        n = np.load(str(npy), allow_pickle=False)
        assert depth_arr.dtype == np.uint16 and depth_arr.shape == (height, width), im.name
        assert n.shape == (height, width, 3) and np.isfinite(n).all(), im.name
        error = float(np.max(np.abs(np.linalg.norm(n, axis=-1) - 1)))
        assert error < .01, (im.name, error)
        p = params[stem]
        assert np.isfinite([p['scale'], p['offset']]).all()
        assert .2 * median <= p['scale'] <= 5 * median, (im.name, 'loader would disable depth')
        positive = float((depth_arr.astype(np.float32)/65535*p['scale'] + p['offset'] > 0).mean())
        assert positive > .5, (im.name, 'less than half valid positive inverse depth')
        rows.append(dict(image=im.name, shape=[height, width], normal_unit_error=error, positive_depth_fraction=positive))
        paths += [img, dp, npy]
    paths += list((SCENE / 'sparse/0').glob('*.bin'))
    save(OUT / 'audit/input_audit.json', dict(passed=True, rows=rows, train=38, test=6, files=files_hashes(paths)))
    print('PASS: 44 complete aligned inputs, 38/6 split, finite unit normals, accepted depth scales.')


def projection():
    identity()
    for label, repo, python in [('tags', CODE, PY), ('baseline', ROOT / '3dgs_seeded', '/root/miniconda3/envs/gs_env/bin/python')]:
        env = dict(os.environ, PROJ_REPO=str(repo), SCENE_PATH=str(SCENE))
        for script in ['rasterizer_k_projection_smoke.py', 'test_camera_k_flow.py']:
            run([python, OLD / 'audit' / script], OUT / 'audit' / (label + '_' + script + '.log'), env, str(repo))
    save(OUT / 'audit/delivery_area_projection.json', dict(passed=True, identity=identity()))


def launcher():
    os.environ.update(ROOT=str(ROOT), FOLLOW=str(FOLLOW))
    module = runpy.run_path(str(FOLLOW / 'a800_followup.py'))
    def reference(group):
        name = 'tags_depth_dsine_w1' if group == 'tags_depth_dsine' else group
        return OLD / 'garden' / (name + '_seed0')
    module['main'].__globals__.update(GARDEN=SCENE, REPAIRED=DEPTH, OUT=OUT, reference=reference)
    return module


def planned(seeds):
    args = argparse.Namespace(batch='garden', seeds=seeds,
        groups=['3dgs', 'tags_nogeo', 'tags_depth_only', 'tags_depth_dsine'], scales=[1.0], late_normal=False, execute=False)
    return args, list(launcher()['make_plan'](args))


def freeze():
    identity()
    assert load(OUT / 'audit/input_audit.json')['passed']
    verify_files(load(OUT / 'audit/input_audit.json')['files'])
    assert load(OUT / 'audit/delivery_area_projection.json')['passed']
    args, plans = planned([0, 1, 2])
    rows = []
    allowed = {'-m', '-s', '--seed', '--depths', '--depth_params_path', '--normal_maps_dir'}
    for path, ref, command in plans:
        old = load(ref / 'experiment_manifest.json')
        assert old['status'] == 'trained', str(ref)
        original = old['command']
        def normalize(cmd):
            result, i = [], 0
            while i < len(cmd):
                if cmd[i] in allowed:
                    i += 2
                else:
                    result.append(cmd[i]); i += 1
            return result
        assert normalize(command) == normalize(original), 'Unexpected non-path training change: ' + str(path)
        rows.append(dict(model=str(path), reference=str(ref), command=command))
    cfg = load(OLD / 'surface_config.json')
    cfg.update(source_path=str(SCENE), max_views=44, output=str(OUT / 'geometry'))
    save(OUT / 'surface_config.json', cfg)
    sources = [FOLLOW / 'a800_followup.py', FOLLOW / 'evaluate_normals.py', CODE / 'scripts/geometry_debug.py',
               Path(__file__).resolve(), OUT / 'surface_config.json', OUT / 'audit/input_audit.json',
               OUT / 'audit/delivery_area_projection.json', OLD / 'audit/projection_validation.json']
    sources += sorted(set(ref / 'experiment_manifest.json' for _, ref, _ in plans))
    sources += list((CODE / 'Depth-Anything-V2/depth_anything_v2').rglob('*.py'))
    sources += list((TOOLS / 'dsine_source').rglob('*.py'))
    sources.append(ROOT / 'audit/paper_review_20261117T112522Z/geometry_trial_playground/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation')
    for folder in ['dslr_scan_eval', 'occlusion']:
        sources += [p for p in (RAW / folder).rglob('*') if p.is_file()]
    snapshot = dict(protocol='Delivery Area confirmatory v1; no result-dependent retuning', primary='scan F1@2cm',
                    projection=identity(), commands=rows, files=files_hashes(sources))
    frozen = OUT / 'audit/frozen_protocol.json'
    if frozen.exists():
        assert load(frozen) == snapshot, 'Frozen protocol already exists and differs; inspect, do not overwrite.'
    else:
        save(frozen, snapshot)
    for row in rows:
        print(' '.join(shlex.quote(x) for x in row['command']))
    print('FROZEN', frozen)


def gate():
    identity()
    verify_files(load(OUT / 'audit/frozen_protocol.json')['files'])
    verify_files(load(OUT / 'audit/input_audit.json')['files'])


def train(seeds):
    gate()
    args, _ = planned(seeds)
    args.execute = True
    launcher()['train'](args)


def models(seeds):
    return [OUT / 'garden' / ('%s_seed%d' % (g, s)) for s in seeds for g in GROUPS]


def evaluate(seeds):
    gate()
    launcher()['evaluate'](argparse.Namespace(scene='garden', step=30000, models=list(map(str, models(seeds)))))
    for m in models(seeds):
        assert load(m / 'followup_eval_30000/complete.json')['views'] == 6, str(m)


def scan_f1(path):
    lines = re.findall(r'^F1-scores:\s*(.*)$', Path(path).read_text(), re.MULTILINE)
    assert len(lines) == 1, 'Expected exactly one F1-scores row: ' + str(path)
    values = list(map(float, lines[0].split()))
    assert len(values) == 6 and all(0 <= x <= 1 for x in values), str(path)
    return values


def geometry(seeds):
    gate()
    evaluator = ROOT / 'audit/paper_review_20261117T112522Z/geometry_trial_playground/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation'
    gt = RAW / 'dslr_scan_eval/scan_alignment.mlp'
    for m in models(seeds):
        ply = m / 'point_cloud/iteration_30000/point_cloud.ply'
        parent = OUT / 'geometry' / m.name
        marker = parent / 'complete.json'
        if marker.exists():
            verify_files(load(marker)['files'])
            print('SKIP verified geometry', m.name)
            continue
        if parent.exists() and any(parent.iterdir()):
            raise RuntimeError('Partial geometry output: inspect before retry: ' + str(parent))
        parent.mkdir(parents=True, exist_ok=True)
        run([PY, CODE / 'scripts/geometry_debug.py', '--repo', CODE, '--config', OUT / 'surface_config.json',
             '--stage', 'run', '--checkpoint', ply, '--method', m.name, '--output', parent], parent / 'export.log')
        surfaces = list(parent.glob('*/surface_samples.ply'))
        assert len(surfaces) == 1, str(surfaces)
        log = parent / 'scan.log'
        run([evaluator, '--reconstruction_ply_path', surfaces[0], '--ground_truth_mlp_path', gt,
             '--tolerances', '0.01,0.02,0.05,0.1,0.2,0.5', '--voxel_size', '0.01'], log)
        scan_f1(log)
        save(marker, dict(files=files_hashes([ply, surfaces[0], gt, evaluator, log]), model=str(m)))


def normal_eval(playground=False):
    identity()
    if not playground:
        gate()
    dest = OUT / 'audit' / ('playground_normals' if playground else 'delivery_area_normals')
    paths = [str((OLD if playground else OUT) / 'garden' / ('%s_seed%d' % (g, s))) for s in range(3) for g in GROUPS]
    for p in paths:
        assert (Path(p) / 'point_cloud/iteration_30000/point_cloud.ply').is_file(), p
    save(dest / 'models.json', paths)
    assert not (dest / 'normal_angles.json').exists(), 'Normal output exists; preserve previous evidence.'
    run([PY, FOLLOW / 'evaluate_normals.py', '--code', CODE, '--models', dest / 'models.json', '--iteration', '30000',
         '--output', dest / 'normal_angles.json', '--diagnostic-dir', dest / 'diagnostics'], dest / 'evaluate.log')


def summarize():
    gate()
    rows = []
    for seed in range(3):
        for group in GROUPS:
            m = OUT / 'garden' / ('%s_seed%d' % (group, seed))
            manifest = load(m / 'experiment_manifest.json')
            assert manifest['status'] == 'trained'
            metrics = load(m / 'followup_eval_30000/complete.json')
            assert metrics['views'] == 6
            scan = OUT / 'geometry' / m.name
            verify_files(load(scan / 'complete.json')['files'])
            row = dict(group=group, seed=seed, F1_2cm_percent=100*scan_f1(scan / 'scan.log')[1],
                       points=metrics['points'], wall_minutes=manifest['wall_seconds']/60)
            row.update({k: float(metrics['rgb'][k]) for k in ['PSNR', 'SSIM', 'LPIPS']})
            rows.append(row)
    with (OUT / 'run_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    columns = ['F1_2cm_percent', 'PSNR', 'SSIM', 'LPIPS', 'points', 'wall_minutes']
    groups = {g: {k: dict(mean=statistics.mean(r[k] for r in rows if r['group'] == g),
                          sample_sd=statistics.stdev(r[k] for r in rows if r['group'] == g))
                  for k in columns} for g in GROUPS}
    paired = {}
    index = {(r['group'], r['seed']): r for r in rows}
    for candidate, baseline in [('tags_depth_only', 'tags_nogeo'), ('tags_depth_dsine_w1', 'tags_depth_only'), ('tags_depth_dsine_w1', '3dgs')]:
        pair = {}
        for k in columns:
            delta = [index[candidate, s][k] - index[baseline, s][k] for s in range(3)]
            pair[k] = dict(seed_deltas=delta, mean=statistics.mean(delta), sample_sd=statistics.stdev(delta))
        paired[candidate + '_minus_' + baseline] = pair
    save(OUT / 'summary.json', dict(n_seeds=3, groups=groups, paired=paired,
         note='F1 differences are percentage points. Normal angles are separately reported as DSINE consistency.'))
    print(json.dumps(dict(groups=groups, paired=paired), indent=2))


def pack():
    gate()
    archive = OUT.parent / (OUT.name + '_evaluations.tar.gz')
    assert not archive.exists(), 'Archive exists; preserve it.'
    for m in models([0, 1, 2]):
        assert load(m / 'experiment_manifest.json')['status'] == 'trained'
        assert (m / 'followup_eval_30000/complete.json').is_file()
        assert (OUT / 'geometry' / m.name / 'complete.json').is_file()
    assert (OUT / 'audit/delivery_area_normals/normal_angles.json').is_file()
    assert (OUT / 'summary.json').is_file()
    # Keep metrics, manifests and source snapshots; trained PLYs remain on server.
    files = [p for p in OUT.rglob('*') if p.is_file() and not p.is_symlink()
             and (p.name == 'cfg_args' or p.suffix.lower() in {'.json', '.log', '.txt', '.csv', '.patch', '.py', '.md', '.sha256'})
             and p.stat().st_size < 50*1024*1024 and p.name not in {'evidence_sha256.json', 'pack_result.txt'}]
    save(OUT / 'evidence_sha256.json', files_hashes(files))
    files.append(OUT / 'evidence_sha256.json')
    with tarfile.open(str(archive), 'w:gz') as tar:
        for p in files:
            tar.add(str(p), arcname=str(p.relative_to(OUT)), recursive=False)
    print(archive, digest(archive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'depth', 'calibrate', 'normals', 'audit', 'projection', 'freeze',
                                        'train', 'evaluate', 'geometry', 'normal-eval', 'playground-normals', 'summarize', 'pack'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], choices=[0, 1, 2])
    args = parser.parse_args()
    actions = {'prepare': prepare, 'depth': depth, 'calibrate': calibrate, 'normals': normals, 'audit': audit,
               'projection': projection, 'freeze': freeze, 'train': lambda: train(args.seeds),
               'evaluate': lambda: evaluate(args.seeds), 'geometry': lambda: geometry(args.seeds),
               'normal-eval': normal_eval, 'playground-normals': lambda: normal_eval(True), 'summarize': summarize, 'pack': pack}
    actions[args.stage]()


if __name__ == '__main__':
    main()
