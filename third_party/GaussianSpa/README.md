# GaussianSpa: An “Optimizing-Sparsifying” Simplification Framework for Compact and High-Quality 3D Gaussian Splatting (CVPR 2025)
<p align="center">
Yangming Zhang*, Wenqi Jia*, Wei Niu, Miao Yin
</p>
<p align="center">
<a href="https://noodle-lab.github.io/gaussianspa"><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
<a href="https://arxiv.org/pdf/2411.06019"><img src="https://img.shields.io/badge/Arxiv-Page-FF0000"></a>
</p>
GaussianSpa formulates simplification as an optimization problem and introduces an “optimizing-sparsifying” solution that alternately solves two independent sub-problems, gradually imposing strong sparsity onto the Gaussians in the training process. GaussianSpa significantly reduces the number of Gaussians while maintaining high rendering quality.

# Update log
**Feb 2025:** Results evaluated by metrics.py across all scenes are available now.

**Mar 2025:** We apply our "sparisifying-optimizing" framework to vanilla 3DGS and Mini-Splatting. Release the codes and scripts.

# Usasge
## 1. Clone the repository
The repository contains submodules which are not compatible with newest 3DGS , thus please check it out with 
```shell
# HTTPS
git clone https://github.com/noodle-lab/GaussianSpa.git --recursive
```
## 2. Setup environment
```shell
conda create -n gaussian_spa python=3.7
conda activate gaussian_spa
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## 3. Download datasets
Download datasets [Mip-360](https://jonbarron.info/mipnerf360/) and [Tanks&Temples and Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Then put the scenes to Dataset folder for running scripts perfectly. The expected folder structure is:

```txt
GaussianSpa
├──train_opacity.sh
├──train_imp_score.sh
└── ...
Dataset
├── drjohnson
├── playroom
├── bicycle
├── bonsai
├── counter
├── flowers
├── garden
├── kitchen
├── room
├── stump
├── treehill
├── train
└── truck

```
## 4. Running
We provide two different criteria to sparisify auxiliary variable **Z** (details please refer our paper [**5.2. Generality of GaussianSpa**](https://arxiv.org/pdf/2411.06019)): opacity in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file) and importance score proposed by [Mini-Splatting](https://github.com/fatPeter/mini-splatting). 
```shell
# Metric opacity for vanilla 3DGS
chmod +x train_opa.sh
bash train_opa.sh
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train_op.py</span></summary>

#### --optimizing_spa
Flag to enable optimizing and spasifying
#### --prune_ratio2
Ratios for spasifying/pruning points at the sparisifying stop iteration.
#### --optimizing_spa_interval
Interval to perform the “sparsifying” step every fixed number of iterations
</details>

```shell
# Metric opacity for Mini-Splatting
chmod +x train_opacity.sh
bash train_opacity.sh

# Metric importance score for Mini-Splatting
chmod +x train_imp_score.sh
bash train_imp_score.sh
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train_opacity.py or train_imp_score.py</span></summary>

#### --optimizing_spa
Flag to enable optimizing and spasifying
#### --prune_ratio1
Ratios for pruning points at the simplifying iteration1.
#### --prune_ratio2
Ratios for spasifying/pruning points at the sparisifying stop iteration.
#### --optimizing_spa_interval
Interval to perform the “sparsifying” step every fixed number of iterations
</details>

Other arguments are similar to offical 3DGS and Mini-Splatting.

<details>
 <summary><span style="font-weight: bold;">Recommend Arguments Setting for Evaluation</span></summary>
 
 |   Scenes  |   Method  | Pruning_ratio1 | Pruning_ratio2 |
 |:---------:|:---------:|:--------------:|:--------------:|
 | drjohnson | imp_score |       75       |       50       |
 |  playroom |  opacity  |       50       |       77       |
 |  bicycle  | imp_score |       50       |       72       |
 |   bonsai  | imp_score |       50       |       80       |
 |  counter  |  opacity  |       60       |       50       |
 |  flowers  |  opacity  |       50       |       70       |
 |   garden  | imp_score |       60       |       50       |
 |  kitchen  |  opacity  |       40       |       80       |
 |    room   |  opacity  |       50       |       80       |
 |   stump   | imp_score |       65       |       70       |
 |  treehill |  opacity  |       40       |       75       |
 |   train   |  opacity  |       50       |       80       |
 |   trunk   |  opacity  |       62       |       80       |
 
 </details>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h1 class="title">Citation</h1>
    <h4>Consider citing our paper if you find its insights and contributions somewhat valuable.</p>
    <pre><code>@misc{zhang2024gaussianspa,
        title={GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting}, 
        author={Yangming Zhang and Wenqi Jia and Wei Niu and Miao Yin},
        year={2024},
        eprint={2411.06019},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2411.06019}, 
}</code></pre>
  </div>
</section>
