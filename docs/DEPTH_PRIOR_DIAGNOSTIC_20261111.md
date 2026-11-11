# Bicycle Depth Prior Diagnostic — 2026-11-11

## Scope

This is a local, short diagnostic on the `data/bicycle` copy with seed 0, 25 held-out test cameras, resolution setting `2` (about 618 × 411), and a 4,000-iteration budget. All four runs use the same initialization and optimization settings; the intended independent variable is depth decoding / explicit depth and current-depth-normal supervision. The `noGeo` condition disables the explicit depth/normal prior, while the geometry controller and sparse optimization remain enabled. These runs diagnose the pipeline; they are not full-budget or multi-seed paper results.

The pre-change code and selected experiment files were backed up before editing:

- `backups/geo-prior-diagnosis-pre-20261111.tar.gz`
- Expanded snapshot and checksums: `backups/geo-prior-diagnosis-pre-20261111/`

## Finding

The Bicycle depth PNGs are `uint8` monocular **inverse-depth** maps. The previous loader selected its decoding branch using the maximum value of the whole image. Since these images normally reach 255, the loader treated every pixel as millimeter sensor depth and computed `1 / (raw × 0.001)`. This reversed the ordering of the inverse-depth signal and produced values in the wrong scale. The per-image scale/offset calibration had been computed from normalized inverse depth, so applying that calibration after the wrong decoding did not repair the mismatch.

For the audited image `_DSC8679.png`, after applying the same saved calibration, the old decoder produced inverse-depth percentiles about `[3.32, 12.44, 397.10]` (p01/p50/p99), while decoding the image as `raw / 255` produced `[0.031, 0.227, 0.779]`. The median differs by about 55× and the order is reversed. The regenerated calibration from all 194 local images matched the existing calibration to the reported precision, so calibration fitting is not the cause of this observed mismatch.

The previous per-image `max > 50` heuristic could also choose different meanings for images from the same dataset if their maxima differed. It did not switch meaning at individual pixels.

## Results

### Image metrics at iteration 4,000

| Condition | Decode | Explicit prior | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---:|---:|---:|
| Legacy depth | old threshold / millimeter inversion | depth | 24.9326 | 0.71819 | 0.3120 |
| Corrected noGeo | uint8 normalized inverse depth | none | **24.9950** | **0.72056** | **0.3114** |
| Corrected depth | uint8 normalized inverse depth | depth | 24.9806 | 0.71823 | 0.3146 |
| Corrected depth + current normal | uint8 normalized inverse depth | depth + normal derived from the same depth | 24.9858 | 0.71785 | 0.3147 |

The corrected depth prior improves slightly over the legacy-depth run in PSNR, but it does not beat corrected `noGeo` on this short image-quality comparison. Adding normals derived from the current depth provides no image-metric improvement over depth-only. The effect sizes are small at 4,000 steps; this is evidence against claiming that the decoder fix alone restores visual quality.

### Depth-prior consistency at iteration 4,000

Evaluated on the same 25 test cameras. These metrics measure agreement with the loaded monocular prior, not absolute depth accuracy.

| Condition | Mean Pearson ↑ | Mean Spearman ↑ | Mean log-L1 ↓ |
|---|---:|---:|---:|
| Legacy depth | −0.5089 | −0.8964 | 3.8804 |
| Corrected noGeo | +0.9415 | +0.9498 | 0.1750 |
| Corrected depth | **+0.9681** | **+0.9826** | 0.1520 |
| Corrected depth + current normal | +0.9671 | +0.9818 | **0.1357** |

The decoder correction flips the predicted/prior relationship from strongly negative to strongly positive. Enabling corrected depth further improves consistency over `noGeo`, but not the short-run image metrics.

An earlier 1,200-step smoke comparison showed the same pattern: legacy-depth Pearson/Spearman were −0.5139/−0.8975; corrected noGeo were +0.9398/+0.9437; corrected depth were +0.9655/+0.9772; corrected depth + current normal were +0.9654/+0.9774. At this shorter budget, image metrics were too close to distinguish a reliable visual effect.

## Code changes

- Added `utils/depth_encoding.py` with explicit `auto`, `monocular_inverse`, `sensor_mm`, and `legacy_threshold` decoding modes. `auto` interprets `uint8` as normalized monocular inverse depth and retains the previous `uint16` compatibility behavior.
- Wired `--depth_input_encoding` and `--depth_params_path` through argument parsing, scene loading, and camera construction. Camera loading prints one audit line with dtype and raw/decoded ranges.
- Corrected the calibration utility to sample every COLMAP coordinate and added an optional output path so calibration can be regenerated without overwriting the dataset file.
- Documented the input encodings and recommended explicit Bicycle setting in `docs/DEPTH_MAP_USAGE.md` and `docs/TAGS_EXPERIMENT_OPTIMIZATION_PROTOCOL.md`.
- Added focused decoder and sampling unit tests in `tests/test_depth_input_decoding.py`.

## Interpretation and next step

The confirmed primary defect is a depth **encoding and polarity mismatch** in loading. It explains the negative depth correlation and makes the old depth-derived normal target unreliable as well. However, the short controlled runs do not establish that the decoder mismatch was the sole cause of image-quality decline: even with the corrected signal, depth and current-depth-normal supervision trail `noGeo` slightly on the 4,000-step image metrics. The prior weights, timing, normal construction, and the short budget may each contribute; the present experiment does not separate those effects statistically.

Before starting Stage 4, freeze the corrected decoder and run a matched longer comparison that changes one factor at a time: corrected noGeo, corrected depth-only, and corrected depth + current normal. Use at least the same seeds and image/evaluator protocol as the existing Stage 3 comparisons. Evaluate both RGB quality and depth consistency, and only retain the normal condition if it adds measurable benefit beyond depth-only. Do not interpret these single-seed short runs as a final Stage 3 ranking.

## Reproduction artifacts

Training logs, `cfg_args`, training curves, camera manifests, and per-view depth evaluation JSONs are under `output/geo_prior_diagnostic/`. The four 4,000-step run directories are:

- `legacy_depth_4k`
- `auto_nogeo_4k`
- `auto_depth_4k`
- `auto_depth_normal_4k`

The run outputs are local diagnostics and are not part of the source commit.
