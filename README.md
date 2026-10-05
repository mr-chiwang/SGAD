<div align="center">

<h1>SGAD: Semantic and Geometric-aware Descriptor for Local Feature Matching</h1>

<p>
  Xiangzeng Liu<sup>*</sup>, Chi Wang<sup>*†</sup>, Guanglu Shi, Xiaodong Zhang, Qiguang Miao<sup>†</sup>, Miao Fan
</p>
<p>
  <em><sup>*</sup>Equal Contribution;&nbsp;&nbsp;<sup>†</sup>Corresponding author.</em>
</p>

<div style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://arxiv.org/pdf/2508.02278" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/ICCV_2025-Highlight-blue" alt="ICCV Highlight"></a>
  <a href="https://arxiv.org/abs/2508.02278" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2508.02278-b31b1b" alt="arXiv"></a>
  <a href="https://mr-chiwang.github.io/SGAD/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
</div>

</div>

---

This repository provides **SGAD** and **SGAD++**, with an Area-to-Point Matching
(A2PM) pipeline that combines learned region matching with existing point matchers.

[Model checkpoints](https://huggingface.co/MrWangChi/A2PM-SGAD)
| [Data assets](https://huggingface.co/datasets/MrWangChi/A2PM-SGAD-data)

**SGAD** learns area descriptors using DINOv2 features, geometric positional
encoding, and classification/ranking supervision. Its Hierarchical Containment
Redundancy Filter (HCRF) reduces overlapping area matches.

**SGAD++** extends the region matcher with DINOv3 features, RoPE region attention,
and soft-overlap supervision, with BCE and Focal variants.

The `sgad/` subproject provides standalone training, inference, and region
evaluation. The repository-level A2PM pipeline integrates these models with
LoFTR, RoMa, DKM, or SuperPoint+SuperGlue. See [SGAD usage](sgad/README.md)
for standalone training, inference, and results.

## Installation

### Clone the repository

```bash
git clone --recursive https://github.com/mr-chiwang/SGAD.git
cd SGAD
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Basic dependencies

```bash
conda create -n mv_sgad python=3.12.12
conda activate mv_sgad
pip install -r requirements.txt
pip install -e sgad
```

## Data

Download the released bbox packages and count metadata using
[Data Preparation](docs/data_preparation.md#released-assets), then arrange images
and scene indices in the documented layout. If you want to prepare your own
boxes, see [Custom Data Preparation](docs/data_preparation.md#custom-data-preparation).

## Pretrained Models

Prepare the selected point matcher, the SGAD / SGAD++ checkpoint, and its DINO
backbone. The point matchers are official Git submodules; weight downloads and
default paths are listed below.

### Point-matcher weights

The `indoor` configurations are used for ScanNet, and the `outdoor`
configurations for MegaDepth. Prepare the files for your selected dataset.

| Point matcher | Official download entry | Required filenames | Default directory |
|---|---|---|---|
| LoFTR (DS) | [LoFTR download instructions](https://github.com/zju3dv/LoFTR#installation) | Indoor: `indoor_ds_new.ckpt`; outdoor: `outdoor_ds.ckpt` | `point_matchers/LoFTR/weights/` |
| DKMv3 | [Official DKMv3 release](https://github.com/Parskatt/storage/releases/tag/dkmv3) | Indoor: `DKMv3_indoor.pth`; outdoor: `DKMv3_outdoor.pth` | `point_matchers/DKM/weights/` |
| RoMa | [Official RoMa release](https://github.com/Parskatt/storage/releases/tag/roma) | Indoor: `roma_indoor.pth`; outdoor: `roma_outdoor.pth` | `point_matchers/RoMa/weights/` |
| RoMa DINOv2 backbone (both datasets) | [Official DINOv2 ViT-L/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | `dinov2_vitl14_pretrain.pth` | `point_matchers/RoMa/weights/` |
| SuperPoint + SuperGlue | [Official weights directory](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights) | `superpoint_v1.pth`, `superglue_indoor.pth`, `superglue_outdoor.pth` | `point_matchers/SuperGluePretrainedNetwork/models/weights/` |

SuperPoint and SuperGlue weights are already tracked by their official
submodule and are available after `git submodule update --init --recursive`.
Its YAML `weights: indoor` or `weights: outdoor` selects the SuperGlue model.
For LoFTR, DKM, and RoMa, the configurations in `conf/point_matcher/` load local
files from the table above. To store them elsewhere, override
`point_matcher.weights` and, for RoMa, `point_matcher.dinov2_weights`.

### Region-matcher backbones

`A2PM_PRETRAINED_ROOT` defaults to `<repo-root>/pretrained/`. The directories
below are relative to that root.

| Model | Official download entry | Required format | Default location | Used by |
|---|---|---|---|---|
| DINOv2 ViT-L/14 | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | Complete Transformers model directory | `dinov2_large/` | SGAD |
| DINOv3 ViT-L/16, LVD-1689M | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | Complete Transformers model directory | `dinov3-vitl16-pretrain-lvd1689m/` | SGAD++ BCE and Focal |

Download each SGAD backbone's configuration and parameters into its model
directory. RoMa uses the DINOv2 `.pth` file listed in the point-matcher table.

With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
installed, run the relevant command from the repository root:

```bash
# SGAD backbone.
hf download facebook/dinov2-large --local-dir pretrained/dinov2_large

# SGAD++ backbone: first obtain access on the official model page, then log in.
hf auth login
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir pretrained/dinov3-vitl16-pretrain-lvd1689m
```

If `A2PM_PRETRAINED_ROOT` points elsewhere, use that root for the download
destinations. The A2PM adapter also accepts individual backbone-directory
overrides through `A2PM_DINOV2_PATH` and `A2PM_DINOV3_PATH`.

If you already have the official DINOv3 `.pth`, the
[conversion tool](sgad/tools/README.md#dinov3-checkpoint-conversion) can produce
the local Transformers directory instead.

Download [SGAD / SGAD++ matching checkpoints](sgad/README.md#checkpoints).

## A2PM Evaluation

Run an experiment from `conf/experiment/` using `+experiment=<config-name>`.
For example, LoFTR on MegaDepth and RoMa on ScanNet:

```bash
export A2PM_SGAD_WEIGHTS=/path/to/weights/sgad_indoor_scannet.ckpt
export A2PM_SGAD_MEGADEPTH_WEIGHTS=/path/to/weights/sgad_outdoor_megadepth.ckpt
mkdir -p res

run_dir=$(mktemp -d "$PWD/res/loftr-md640.XXXXXX")
python scripts/test.py \
  +experiment=a2pm_sgad_egam_loftr_megadepth_640_area640_sgad \
  out_path="$run_dir/results" hydra.run.dir="$run_dir/hydra" verbose=0

run_dir=$(mktemp -d "$PWD/res/roma-sn640.XXXXXX")
python scripts/test.py \
  +experiment=a2pm_sgad_egam_roma_scannet_640 \
  out_path="$run_dir/results" hydra.run.dir="$run_dir/hydra" verbose=0
```

Experiment parameters are defined in the YAML files. Use
`pair_file_path=/path/to/pairs.txt` for a custom pair list. Batch metrics can be
aggregated with `metric/eval_ratios.py`.

This release fixes geometric refinement (GAM/EGAM) and evaluates all 1,500
pairs. Failed PM and A2PM estimates receive a 180° error independently,
while valid estimates retain their measured errors. The paper followed
MESA's evaluation on the intersection of recorded PM and A2PM results,
excluding A2PM pipeline failures. These changes affect absolute AUC;
compare relative improvements with the paper.

[Evaluation results](docs/a2pm_evaluation_results.md)

## Citation

If you use SGAD, please cite its ICCV 2025 paper:

```bibtex
@InProceedings{Liu_2025_ICCV,
  author = {Liu, Xiangzeng and Wang, Chi and Shi, Guanglu and Zhang, Xiaodong and Miao, Qiguang and Fan, Miao},
  title = {SGAD: Semantic and Geometric-aware Descriptor for Local Feature Matching},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2025},
  pages = {27095-27104}
}
```

## Acknowledgements

Our code builds on [A2PM-MESA](https://github.com/Easonyesheng/A2PM-MESA) and
[LoFTR](https://github.com/zju3dv/LoFTR). We thank their authors and the authors
of the following repositories for their work:

- [SAM](https://github.com/facebookresearch/segment-anything)
- [SuperPoint+SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [DKM](https://github.com/Parskatt/DKM)
- [RoMa](https://github.com/Parskatt/RoMa)
- [SGAM](https://github.com/Easonyesheng/SGAM)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [RoMa V2](https://github.com/Parskatt/RoMaV2)

## License

Our original code is licensed under the [MIT License](LICENSE). Incorporated
third-party code retains its upstream licenses; see
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
