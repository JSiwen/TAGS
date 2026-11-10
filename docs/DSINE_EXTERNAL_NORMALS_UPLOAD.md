# DSINE external-normal patch: AutoDL upload and smoke run

This patch adds the first external-normal supervision path for the Bicycle
`images_4` experiment. The existing depth-derived normal path remains the
default (`--normal_prior_source depth`). External DSINE maps are float32 `.npy`
files in camera coordinates `(right, down, front)`.

## Build and install the upload bundle

From the TAGS repository root, build the ZIP:

```bash
python scripts/package_dsine_external_normals_patch.py
```

The output is `output/dsine_external_normals_patch.zip`. Upload and extract it on
AutoDL, then install it into the code checkout used by `run_tags`:

```bash
python install_dsine_external_normals_patch.py \
  --target /root/autodl-tmp/TAGS_github
```

The installer backs up the 11 files it replaces into a timestamped
`backup_dsine_external_normals_*` directory. It does not alter data, weights, or
the `run_tags` shell function.

## Verify the installed interface

```bash
cd /root/autodl-tmp/TAGS_github
python train_optimized.py --help 2>&1 | grep -E 'normal_prior_source|normal_maps_dir'
```

Both flags must be present. The data directory remains:

```bash
SOURCE=/root/autodl-tmp/TAGS/TAGSs_bicycle_v14/data/bicycle
NORMAL_DIR="$SOURCE/normals_dsine"
```

Before training, validate the uploaded normal maps:

```bash
python scripts/validate_dsine_normals.py --source "$SOURCE" \
  --normal-dir "$NORMAL_DIR"
```

## Short smoke run

Use a new output name because `run_tags` refuses existing model directories.
The smoke run overrides the common 30k iteration and save schedule:

```bash
export LOGROOT="$OUTROOT/logs"
mkdir -p "$OUTROOT" "$LOGROOT"

run_tags v16_dsine_smoke_seed0_r1 0 \
  --enable_geo_prior --geo_prior_depth --geo_prior_normal \
  --normal_prior_source external \
  --normal_maps_dir "$NORMAL_DIR" \
  --normal_prior_weight_init 0.02 \
  --normal_prior_weight_final 0.02 \
  --normal_prior_tv_weight 0.0 \
  --iterations 64 \
  --save_iterations 64 \
  --test_iterations 64
```

Confirm the log reports external source, loaded normal maps, nonzero valid-mask
fraction, and a finite nonzero `geo/normal_loss`. Only then run a full paired
depth-normal versus DSINE-normal comparison. This patch has not been tested by a
full training run.
