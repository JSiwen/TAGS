#!/usr/bin/env python3
"""Repair completed A800 baseline evaluations without changing training artifacts.

Waits for the specified original queue to exit, then evaluates only baseline
runs with recorded evaluation failures. Every retry has a separate evidence
directory and manifest; original failure manifests and logs remain unchanged.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def queue_alive(pid, expected_script):
    cmdline = Path('/proc') / str(pid) / 'cmdline'
    return cmdline.exists() and str(expected_script).encode() in cmdline.read_bytes().split(b'\0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--queue-script', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    out = args.root / 'output/a800_bicycle_images4_native_20261112'
    status_path = args.root / 'audit/evaluation_repair_status.json'
    state = {'status': 'waiting_for_training_queue', 'queue_pid': args.wait_pid,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'runs': []}
    write(status_path, state)
    while queue_alive(args.wait_pid, args.queue_script):
        time.sleep(30)
    if not (out / 'orchestrator_status.json').is_file():
        state['status'] = 'blocked_queue_exited_without_completion_record'
        write(status_path, state)
        raise SystemExit(state['status'])
    state['status'] = 'repairing_evaluations'
    write(status_path, state)
    for seed in (0, 1, 2):
        model = out / ('3dgs_seed{}'.format(seed))
        original = model / 'experiment_manifest.json'
        m = json.loads(original.read_text())
        ply = model / 'point_cloud/iteration_30000/point_cloud.ply'
        if m.get('train_exit_code') != 0 or not ply.is_file():
            raise SystemExit('Baseline training incomplete: {}'.format(model))
        if m.get('status') == 'completed':
            continue
        recovery = model / 'evaluation_recovery_v1'
        recovery.mkdir(exist_ok=False)
        (recovery / 'original_experiment_manifest.json').write_bytes(original.read_bytes())
        # Evaluate in a separate model directory, retaining the initial failed
        # logs and the manually recovered seed-0 outputs byte for byte.
        (recovery / 'cfg_args').write_bytes((model / 'cfg_args').read_bytes())
        (recovery / 'point_cloud').symlink_to(model / 'point_cloud', target_is_directory=True)
        source = Path(m['source_path'])
        common = ['-m', str(recovery), '-s', str(source), '--images', 'images_4',
                  '--depths', str(source / 'depth_any_out'), '--depth_input_encoding', 'monocular_inverse',
                  '--depth_params_path', str(args.root / 'audit/depth_params_corrected.json'),
                  '--normal_maps_dir', '', '--resolution', '1', '--llffhold', '8', '--eval',
                  '--iteration', '30000', '--quiet']
        commands = {
            'render': [args.python, str(args.root / 'code/render.py')] + common + ['--skip_train'],
            'metrics': [args.python, str(args.root / 'code/metrics.py'), '-m', str(recovery), '--split', 'test'],
            'depth_consistency': [args.python, str(args.root / 'code/scripts/evaluate_depth_per_view.py')] + common + ['--split', 'test'],
        }
        record = {'status': 'running', 'run_id': m['run_id'], 'training_manifest_sha256': digest(original),
                  'final_ply_sha256': digest(ply), 'final_ply': str(ply), 'commands': commands,
                  'reason': 'Original 3DGS cfg_args has no llffhold; explicitly supply hold 8.', 'exit_codes': {}}
        write(recovery / 'recovery_manifest.json', record)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        for name, command in commands.items():
            with (recovery / (name + '.log')).open('w') as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            record['exit_codes'][name] = code
            write(recovery / 'recovery_manifest.json', record)
            if code:
                raise SystemExit('Recovery failed: {} {}'.format(model, name))
        record['status'] = 'completed'
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(recovery / 'recovery_manifest.json', record)
        state['runs'].append({'run_id': m['run_id'], 'recovery': str(recovery), 'status': 'completed'})
        write(status_path, state)
    state['status'] = 'completed'
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write(status_path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
