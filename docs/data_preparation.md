# Data Preparation

The release provides A2PM and SGAD bounding boxes, per-image counts, and
pair-count metadata. Download these assets and arrange the source datasets
using the layouts below.

| Task | Required data |
|---|---|
| A2PM evaluation | LoFTR 1500-pair test data and A2PM evaluation boxes |
| SGAD training and evaluation | Source datasets, LoFTR scene indices, SGAD boxes, and count metadata |

## Released Assets

Download the prepared bbox packages and count metadata from the
[A2PM-SGAD collection](https://huggingface.co/collections/MrWangChi/a2pm-sgad-6abfb178b1940477ad5cbf0d):

| Asset | Download | Extract or link to |
|---|---|---|
| `a2pm_evaluation_bboxes.tar.gz` | [Download](https://huggingface.co/datasets/MrWangChi/A2PM-SGAD-data/resolve/main/a2pm_evaluation_bboxes.tar.gz?download=true) | `assets/a2pm_bboxes/{scannet1500/640x480,megadepth1500/1296x968}/` |
| `sgad_scannet_bboxes.tar.gz` | [Download](https://huggingface.co/datasets/MrWangChi/A2PM-SGAD-data/resolve/main/sgad_scannet_bboxes.tar.gz?download=true) | `sgad/data/scannet_bboxes/` |
| `sgad_megadepth_bboxes.tar.gz` | [Download](https://huggingface.co/datasets/MrWangChi/A2PM-SGAD-data/resolve/main/sgad_megadepth_bboxes.tar.gz?download=true) | `sgad/data/megadepth_bboxes/` |
| `sgad_data_metadata.tar.gz` | [Download](https://huggingface.co/datasets/MrWangChi/A2PM-SGAD-data/resolve/main/sgad_data_metadata.tar.gz?download=true) | `sgad/assets/pair_bbox_counts/` and per-image count indices under `sgad/data/` |

Extract the packages for your task from the repository root:

```bash
# A2PM evaluation.
tar -xzf /path/to/a2pm_evaluation_bboxes.tar.gz

# SGAD / SGAD++: metadata and boxes for the selected dataset.
tar -xzf /path/to/sgad_data_metadata.tar.gz
tar -xzf /path/to/sgad_scannet_bboxes.tar.gz
tar -xzf /path/to/sgad_megadepth_bboxes.tar.gz
```

## Images and Scene Indices

Follow the original licenses and access requirements of each dataset.

| Data | Source | Used by |
|---|---|---|
| ScanNet RGB-D scenes | [ScanNet official guide](https://github.com/ScanNet/ScanNet#scannet-data) | A2PM, SGAD, SGAD++ |
| MegaDepth depth maps (`phoenix`) | [MegaDepth_v1](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz) | SGAD, SGAD++ |
| D2-Net-preprocessed MegaDepth images (`Undistorted_SfM`) | [D2-Net MegaDepth data](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB) | A2PM, SGAD, SGAD++ |
| LoFTR training indices and 1500-pair test data | [LoFTR data release](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf) | A2PM, SGAD, SGAD++ |

For ScanNet, use the Python exporter from the official ScanNet SensReader. Each
exported scene must contain `color/`, `depth/`, `intrinsic/`, and `pose/`.

MegaDepth training uses two upstream roots together: depth maps under `phoenix`
and D2-Net-preprocessed images under `Undistorted_SfM`.

Unpack the LoFTR archives for your task:

```bash
# For ZIP downloads.
unzip downloaded-file.zip

# Full-dataset indices for SGAD / SGAD++ training.
tar xf train-data/megadepth_indices.tar
tar xf train-data/scannet_indices.tar

# Compact benchmark datasets for A2PM evaluation.
tar xf testdata/megadepth_test_1500.tar
tar xf testdata/scannet_test_1500.tar
```

For A2PM evaluation, use the images extracted from the LoFTR test archives directly.

## Bbox Format

Released boxes are `N x 4` NumPy arrays in `xywh` format. A2PM uses `float32`
arrays; SGAD loaders accept `.npy` or `.json` files.

| Bbox package | Coordinate space | Directory |
|---|---|---|
| A2PM ScanNet | 640 × 480 | `assets/a2pm_bboxes/scannet1500/640x480/` |
| A2PM MegaDepth | 1296 × 968 | `assets/a2pm_bboxes/megadepth1500/1296x968/` |
| SGAD ScanNet | Original image coordinates | `sgad/data/scannet_bboxes/` |
| SGAD MegaDepth | Original image coordinates | `sgad/data/megadepth_bboxes/` |

## A2PM Dataset Layout

The 1500-pair lists are included in `scripts/`.

### Expected layout

```text
SGAD/
├── data/
│   └── testdata/
│       ├── scannet_test_1500/
│       │   └── scene0707_00/
│       │       ├── color/
│       │       ├── depth/
│       │       ├── intrinsic/
│       │       └── pose/
│       └── megadepth_test_1500/
│           ├── Undistorted_SfM/
│           │   └── <scene>/images/
│           └── scene_info_val_1500/
│               └── <scene-index>.npz
└── assets/
    ├── pairs/
    │   ├── scannet1500_pairs.txt
    │   └── megadepth1500_pairs.txt
    └── a2pm_bboxes/
        ├── scannet1500/
        │   └── 640x480/<scene>/<frame>.npy
        └── megadepth1500/
            └── 1296x968/<scene>/<image>.npy
```

The default dataset and bbox paths follow the layout above. To use data stored
elsewhere, override the relevant environment variables before evaluation:

```bash
export A2PM_SGAD_ROOT=/path/to/SGAD
export A2PM_SCANNET1500_ROOT="$A2PM_SGAD_ROOT/data/testdata/scannet_test_1500"
export A2PM_MEGADEPTH1500_ROOT="$A2PM_SGAD_ROOT/data/testdata/megadepth_test_1500"
export A2PM_SCANNET1500_BBOX_ROOT="$A2PM_SGAD_ROOT/assets/a2pm_bboxes/scannet1500/640x480"
export A2PM_MEGADEPTH1500_BBOX_ROOT="$A2PM_SGAD_ROOT/assets/a2pm_bboxes/megadepth1500/1296x968"
```

## SGAD Dataset Layout

Create the following layout under `sgad/`. Run the link commands from the
repository root.

### Complete layout

```text
sgad/
├── data/
│   ├── scannet/
│   │   ├── train/<scene>/{color,depth,intrinsic,pose}/
│   │   ├── test/<scene>/{color,depth,intrinsic,pose}/
│   │   └── index/
│   │       ├── intrinsics.npz
│   │       ├── statistics.json
│   │       └── scene_data/
│   │           ├── train/<scene>.npz
│   │           └── train_list/scannet_all.txt
│   ├── scannet_bboxes/
│   │   ├── train/<scene>/bboxes/<frame>.npy
│   │   └── test/<scene>/bboxes/<frame>.npy
│   ├── megadepth/
│   │   ├── train/{phoenix,Undistorted_SfM}/
│   │   ├── test/{phoenix,Undistorted_SfM}/
│   │   └── index/
│   │       ├── scene_info_0.1_0.7/<scene-index>.npz
│   │       └── trainvaltest_list/{train,val,test}_list.txt
│   └── megadepth_bboxes/
│       ├── train/{phoenix,Undistorted_SfM}/.../bboxes/<image>.npy
│       └── test/{phoenix,Undistorted_SfM}/.../bboxes/<image>.npy
└── assets/
    ├── scannet_test_1500/
    │   ├── test.npz
    │   ├── intrinsics.npz
    │   ├── statistics.json
    │   └── scannet_test.txt
    ├── megadepth_test_1500_scene_info/
    │   ├── <scene-index>.npz
    │   └── megadepth_test_1500.txt
    └── pair_bbox_counts/
        ├── scannet/
        │   ├── train/<scene>.npy
        │   └── test1500/test.npy
        └── megadepth/
            ├── train/<scene-index>.npy
            └── test1500/<scene-index>.npy
```

MegaDepth `train/` and `test/` share the same `phoenix` and `Undistorted_SfM`
data. Bbox roots use `<bbox-root>/<split>`.

### Link the upstream datasets

Replace `/path/to/...` with your local dataset and index paths:

```bash
export A2PM_SGAD_ROOT=/path/to/SGAD
export SGAD_DIR="$A2PM_SGAD_ROOT/sgad"

mkdir -p "$SGAD_DIR/data/scannet"/{train,test,index}
ln -s /path/to/scannet_train/* "$SGAD_DIR/data/scannet/train/"
ln -s /path/to/scannet_test/* "$SGAD_DIR/data/scannet/test/"
ln -s /path/to/scannet_indices/intrinsics.npz "$SGAD_DIR/data/scannet/index/intrinsics.npz"
ln -s /path/to/scannet_indices/statistics.json "$SGAD_DIR/data/scannet/index/statistics.json"
ln -s /path/to/scannet_indices/scene_data "$SGAD_DIR/data/scannet/index/scene_data"

mkdir -p "$SGAD_DIR/data/megadepth"/{train,test,index}
ln -s /path/to/megadepth/phoenix "$SGAD_DIR/data/megadepth/train/phoenix"
ln -s /path/to/megadepth_d2net/Undistorted_SfM "$SGAD_DIR/data/megadepth/train/Undistorted_SfM"
ln -s /path/to/megadepth/phoenix "$SGAD_DIR/data/megadepth/test/phoenix"
ln -s /path/to/megadepth_d2net/Undistorted_SfM "$SGAD_DIR/data/megadepth/test/Undistorted_SfM"
ln -s /path/to/megadepth_indices/scene_info_0.1_0.7 "$SGAD_DIR/data/megadepth/index/scene_info_0.1_0.7"
ln -s /path/to/megadepth_indices/trainvaltest_list "$SGAD_DIR/data/megadepth/index/trainvaltest_list"
```

Link the benchmark indices from `point_matchers/LoFTR/assets/`:

```bash
mkdir -p "$SGAD_DIR/assets"
ln -s ../../point_matchers/LoFTR/assets/scannet_test_1500 \
  "$SGAD_DIR/assets/scannet_test_1500"
ln -s ../../point_matchers/LoFTR/assets/megadepth_test_1500_scene_info \
  "$SGAD_DIR/assets/megadepth_test_1500_scene_info"
```

### Bbox and count metadata

Each dataset bbox root contains `train/` and `test/`. The MegaDepth archive uses
a relative `test -> train` link to share bbox storage. The metadata package
contains pair sidecars under `sgad/assets/pair_bbox_counts/` and the per-image
indices `sgad/data/{scannet,megadepth}_bboxes/*_bboxes_counts.npz`.

## Custom Data Preparation

Use this workflow for new images or replacement bbox assets. Run SAM commands
from the repository root on a GPU machine.

### Generate Bounding Boxes

Download [SAM ViT-H](https://github.com/facebookresearch/segment-anything#model-checkpoints)
(`sam_vit_h_4b8939.pth`) and set its path:

```bash
export A2PM_SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
```

Generate boxes at the original image resolution:

```bash
segmentor/sam_seg.sh /path/to/image.jpg /path/to/output
```

Set the coordinate space explicitly for resized images:

```bash
segmentor/sam_seg.sh /path/to/image.jpg /path/to/output \
  --resize-width 640 --resize-height 480
```

For A2PM benchmarks, run the dataset batch scripts with the data paths set in
[A2PM Dataset Layout](#a2pm-dataset-layout):

```bash
SCANNET_INPUT_ROOT="$A2PM_SCANNET1500_ROOT" \
SCANNET_OUTPUT_ROOT="$A2PM_SCANNET1500_BBOX_ROOT" \
bash segmentor/batch_seg_scannet.sh

MEGADEPTH_INPUT_ROOT="$A2PM_MEGADEPTH1500_ROOT/Undistorted_SfM" \
MEGADEPTH_OUTPUT_ROOT="$A2PM_MEGADEPTH1500_BBOX_ROOT" \
bash segmentor/batch_seg_megadepth.sh
```

The batch scripts use 640 × 480 for ScanNet and 1296 × 968 for MegaDepth.
Standalone SGAD boxes use original image coordinates and the SGAD directory
layout above.

To extract boxes from existing SAM object-array files:

```bash
python segmentor/extract_bboxes.py \
  --input-root /path/to/full_sam_results \
  --output-root /path/to/bboxes
```

### Pair Bbox Counts

From the repository root, enter `sgad/` and scan the replacement boxes:

```bash
cd sgad
python tools/bbox_data/scan_bbox_counts.py \
  --bbox_root data/megadepth_bboxes --relative_to data \
  --output data/megadepth_bboxes/megadepth_bboxes_counts.npz
python tools/bbox_data/scan_bbox_counts.py \
  --bbox_root data/scannet_bboxes --relative_to data \
  --output data/scannet_bboxes/scannet_bboxes_counts.npz
```

Each scene index has an `int32` `.npy` sidecar of shape `[N, 2]`, with the same
basename and rows aligned to `name` (ScanNet) or `pair_infos` (MegaDepth).
Columns contain bbox counts for images 0 and 1; `-1` denotes missing metadata.
Dataset loading retains a pair when both counts meet the configured threshold.

#### MegaDepth Training

```bash
python tools/bbox_data/build_pair_bbox_counts.py \
  --dataset megadepth \
  --data_root data/megadepth/train \
  --npz_root data/megadepth/index/scene_info_0.1_0.7 \
  --list_path data/megadepth/index/trainvaltest_list/train_list.txt \
  --bbox_root data/megadepth_bboxes \
  --counts_npz data/megadepth_bboxes/megadepth_bboxes_counts.npz \
  --count_rel_base data \
  --out_root assets/pair_bbox_counts/megadepth/train
```

#### MegaDepth 1500-Pair Evaluation

```bash
python tools/bbox_data/build_pair_bbox_counts.py \
  --dataset megadepth \
  --data_root data/megadepth/test \
  --npz_root assets/megadepth_test_1500_scene_info \
  --list_path assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt \
  --bbox_root data/megadepth_bboxes \
  --counts_npz data/megadepth_bboxes/megadepth_bboxes_counts.npz \
  --count_rel_base data \
  --out_root assets/pair_bbox_counts/megadepth/test1500
```

#### ScanNet Training

```bash
python tools/bbox_data/build_pair_bbox_counts.py \
  --dataset scannet \
  --data_root data/scannet/train \
  --npz_root data/scannet/index/scene_data/train \
  --list_path data/scannet/index/scene_data/train_list/scannet_all.txt \
  --bbox_root data/scannet_bboxes \
  --counts_npz data/scannet_bboxes/scannet_bboxes_counts.npz \
  --count_rel_base data \
  --out_root assets/pair_bbox_counts/scannet/train
```

#### ScanNet 1500-Pair Evaluation

```bash
python tools/bbox_data/build_pair_bbox_counts.py \
  --dataset scannet \
  --data_root data/scannet/test \
  --npz_root assets/scannet_test_1500 \
  --list_path assets/scannet_test_1500/scannet_test.txt \
  --bbox_root data/scannet_bboxes \
  --counts_npz data/scannet_bboxes/scannet_bboxes_counts.npz \
  --count_rel_base data \
  --out_root assets/pair_bbox_counts/scannet/test1500
```

For upstream dataset details, see LoFTR's
[`docs/TRAINING.md`](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md).
