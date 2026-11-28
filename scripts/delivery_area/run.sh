#!/usr/bin/env bash
# Absolute paths make this independent of cwd, shell startup, and conda activation.
set -euo pipefail
export ROOT=/autodl-fs/data/tags_architecture_ablation_20261112
export FOLLOW="$ROOT/audit/followup_20261116"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
DA_DATA=/autodl-fs/data/eth3d_delivery_area
DA_OUT="$ROOT/audit/delivery_area_projection_v2_confirm_20261128"
DA_PY=/root/miniconda3/envs/TAGS/bin/python
DA_PRIOR=/root/miniconda3/envs/dsine/bin/python
DA_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$DA_OUT/audit"
stage=${1:?Usage: tagsda preflight|download|prepare|depth|calibrate|normals|audit|projection|freeze|train|evaluate|geometry|normal-eval|playground-normals|summarize|pack}
shift
case "$stage" in
preflight)
  for f in "$DA_PY" "$DA_PRIOR" "$ROOT/code/train_optimized.py" "$FOLLOW/a800_followup.py" "$FOLLOW/evaluate_normals.py" /root/autodl-tmp/TAGS/experiments/playground_projection_v2/audit/projection_validation.json; do test -f "$f" || { echo "Missing: $f"; exit 1; }; done
  nvidia-smi
  df -h "$DA_DATA" "$ROOT"
  "$DA_PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda)'
  "$DA_PRIOR" -c 'import torch,cv2,geffnet; assert torch.cuda.is_available(); print(torch.__version__)'
  "$DA_PY" -c 'import hashlib,pathlib; p=pathlib.Path("/autodl-fs/data/tags_architecture_ablation_20261112/audit/followup_20261116"); expected={"a800_followup.py":"f1e8a20f3a02451a5943c9520eb0bd0e7da38450a02ded40b4bc291273fea538","evaluate_normals.py":"c34b18cba95ace72980601ce39b69b78f5f4b0ed6598bca7df28ef8b0572c948"}; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in expected.items()), "Dependency drift: audit before running"'
  exit ;;
download)
  command -v 7z
  mkdir -p "$DA_DATA/downloads" "$DA_DATA/raw"
  for name in delivery_area_dslr_undistorted delivery_area_dslr_scan_eval delivery_area_dslr_occlusion; do
    archive="$DA_DATA/downloads/$name.7z"
    if [[ ! -f "$archive" ]]; then
      curl --fail --location --retry 3 --output "$archive.part" "https://www.eth3d.net/data/$name.7z"
      7z t "$archive.part"
      mv -- "$archive.part" "$archive"
    fi
    7z t "$archive"
    7z x -aos "-o$DA_DATA/raw" "$archive"
    sha256sum "$archive"
  done
  test -f "$DA_DATA/raw/delivery_area/dslr_calibration_undistorted/cameras.txt"
  test -f "$DA_DATA/raw/delivery_area/dslr_scan_eval/scan_alignment.mlp"
  exit ;;
depth|normals) interpreter="$DA_PRIOR" ;;
*) interpreter="$DA_PY" ;;
esac
if [[ "$stage" == train ]]; then
  exec 9>"$DA_OUT/train.lock"
  flock -n 9 || { echo 'Another matrix process holds the lock'; exit 1; }
  "$DA_PY" -c 'import shutil; assert shutil.disk_usage("/autodl-fs/data").free >= 80*2**30, "Need 80 GiB planning headroom; also check personal quota"'
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo 'GPU has active compute jobs; inspect before training'; exit 1; fi
fi
# Refuse to reuse a broken evaluation link (the previous Courtyard failure).
if [[ "$stage" == evaluate ]]; then
  "$DA_PY" - "$DA_OUT" <<'PY'
import pathlib,sys
for p in pathlib.Path(sys.argv[1]).glob('garden/*/followup_eval_30000/point_cloud'):
    assert p.is_symlink() and p.resolve()==p.parent.parent/'point_cloud' and p.is_dir(), 'Inspect evaluation link: '+str(p)
PY
fi
log="$DA_OUT/audit/${stage}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
"$interpreter" -u "$DA_DIR/pipeline.py" "$stage" "$@" 2>&1 | tee "$log"
