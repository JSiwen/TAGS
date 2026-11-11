# Local capped 3DGS vs. corrected TAGS — 2026-11-11

## Protocol

- Hardware/runtime: local RTX 3080 10 GB, `gaussian_splatting` environment, PyTorch 1.12.1+cu116.
- Scene: local `data/bicycle`, `images_4`, `--resolution 2` (rendered test images are 618 × 411), `--eval`, LLFF hold 8; the three matched seeds use the same 25 test views.
- Budget: 4,000 iterations, 400,000 maximum live Gaussians, seed 0/1/2. Initial COLMAP point count was 54,275.
- Matched core settings: same image scale, SH degree, default Adam, DSSIM weight, densification start (500), densification end (3,000), opacity reset interval (3,000), gradient threshold (0.00016), train/test split, and initialization seed.
- Evaluation: rendered all final PLYs with the same root `render.py`, then evaluated the resulting PNGs with the same root `metrics.py` (VGG LPIPS). All four seed-0 runs had identical 25 GT files; all run pairs used the same camera filenames.

The 3DGS checkout is Graphdeco commit `54c035f7834b564019656c3e3fcc3646292f727d`. Before this experiment, its local working tree already contained an uncommitted CUDA OOM guard in `scene/gaussian_model.py`. I preserved that edit. For the comparison I temporarily added only a hard point ceiling and configurable RNG seed, then restored the baseline files byte-for-byte to their pre-experiment state. The cap selected the highest-gradient eligible clone/split candidates that fit the remaining budget. The exact experiment patch and pre-existing local patch are retained under `output/geo_prior_diagnostic/3dgs_cap_baseline/`.

At 4,000 steps, the TAGS geometry controller (starts at 9,000) and SPA pruning (starts at 13,000) have not activated. `noGeo` therefore means no explicit depth/normal supervision; it does not mean that all geometry-related mechanisms are disabled for a longer run.

## Three-seed RGB results

Values are the mean and sample standard deviation across three seed-level metrics; each seed metric averages the same 25 test views.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Final Gaussians by seed |
|---|---:|---:|---:|---|
| Original 3DGS with 400k cap | 24.8883 ± 0.0811 | 0.71845 ± 0.00227 | **0.31093 ± 0.00064** | 399,954 / 399,952 / 399,966 |
| Corrected TAGS noGeo | 24.8874 ± 0.0846 | 0.71837 ± 0.00173 | 0.31148 ± 0.00047 | 400,000 / 400,000 / 400,000 |
| Corrected TAGS depth prior | **24.8911 ± 0.0653** | 0.71608 ± 0.00174 | 0.31509 ± 0.00089 | 400,000 / 400,000 / 400,000 |

Paired seed differences (second method minus first):

| Pair | ΔPSNR (mean ± SD) | ΔSSIM (mean ± SD) | ΔLPIPS (mean ± SD) |
|---|---:|---:|---:|
| TAGS noGeo − capped 3DGS | −0.00088 ± 0.00995 | −0.00008 ± 0.00089 | +0.00055 ± 0.00017 |
| TAGS depth − TAGS noGeo | +0.00368 ± 0.03139 | −0.00228 ± 0.00021 | +0.00362 ± 0.00042 |

TAGS noGeo and capped 3DGS are effectively tied on this short test: the PSNR differences change sign across seeds, SSIM is nearly the same, and LPIPS is slightly worse for TAGS in all three seeds. This experiment does not show an image-quality advantage for the modified TAGS noGeo path over the original 3DGS baseline.

The corrected depth prior has a reproducible image-metric tradeoff: SSIM and LPIPS are worse than noGeo in all three seeds. PSNR is mixed (one seed improves, two decline), so its mean change is not a stable gain. The 4,000-step depth-normal run exists only for seed 0; against depth-only it changes PSNR by +0.00523 dB, SSIM by −0.00038, and LPIPS by +0.00013. That single-seed result does not establish that current-depth normals improve image quality.

## Depth-prior consistency

These values measure agreement with the monocular input, not absolute geometry accuracy. Across seeds, values below are the mean and sample standard deviation of the seed-level 25-view means.

| Condition | Pearson ↑ | Spearman ↑ | log-L1 ↓ |
|---|---:|---:|---:|
| Corrected TAGS noGeo | 0.94137 ± 0.00060 | 0.94913 ± 0.00133 | 0.17424 ± 0.01101 |
| Corrected TAGS depth prior | **0.96821 ± 0.00051** | **0.98265 ± 0.00028** | **0.14676 ± 0.00599** |

Depth-prior consistency improved in every matched seed. Together with the repeatable SSIM/LPIPS regression, this supports a concrete interpretation: the corrected prior is now correctly supervised, and the current loss/weight/schedule trades image similarity for agreement with that prior. The decoder fix solved the inverted-input defect; it did not solve the image-quality tradeoff.

## Remaining limitations and risks

1. **Short training only.** Four thousand iterations at 618 × 411 is a diagnostic, not a substitute for the 30,000-step native-scale protocol. Geometry control and SPA have not yet started in these TAGS runs.
2. **Single scene and reused test views.** Bicycle and its 25 holdout cameras were used in prior candidate analysis. These are exploratory comparisons, not independent confirmation or a publication headline result.
3. **Cap policies are close but not identical.** Both cap at 400,000 and prioritize high-gradient candidates, but they are separate code paths. TAGS reaches the cap exactly in these runs; 3DGS ends slightly below it after pruning. Their densification trajectories are therefore not perfectly identical.
4. **Local baseline caveat.** The 3DGS checkout had a pre-existing OOM guard. No OOM or guard-trigger was observed here, and the experiment set `TAGGS_DENSIFY_MIN_FREE_MB=0`, but the baseline should still be described as the local Graphdeco checkout with the documented cap/seed patch and pre-existing guard.
5. **Normal branch is under-sampled.** Depth + current-depth-normal was run for seed 0 only. Its small metric differences are inconclusive.
6. **No independent geometry ground truth.** The depth consistency gain only demonstrates agreement with the monocular prior. It does not establish lower geometric error against sensor/LiDAR ground truth.

## Conclusion and next experiment

On the local machine, a controlled point ceiling makes short TAGS and original 3DGS runs feasible. Under the matched 4,000-step protocol, corrected TAGS noGeo does not outperform capped 3DGS. Corrected depth supervision consistently improves agreement with the monocular prior, while degrading SSIM and LPIPS; PSNR is seed-dependent. The current-depth-normal branch has not shown a reliable additional benefit.

Before stage 4, the most informative next test is to freeze the corrected input decoder and compare noGeo against depth-only at a longer budget on the same three seeds, then decide whether the RGB/geometry tradeoff is acceptable. Retain depth + normal as a separate ablation until it has at least three matched seeds. For any paper claim, add a scene not used in the earlier test-set selection and an independent geometry reference.

## Artifacts

All runs and raw outputs remain under `output/geo_prior_diagnostic/`; the final PLYs, logs, `cfg_args`, `results.json`, per-view RGB metrics, depth-evaluation JSON, and rendered test images are retained there. The side-by-side seed-0 render sheet is `output/geo_prior_diagnostic/comparison_contact_sheet.png`. Temporary 3DGS source backups and cap/seed patches are under `output/geo_prior_diagnostic/3dgs_cap_baseline/`. No dataset, PLY, or generated experiment output is included in the source commit.
