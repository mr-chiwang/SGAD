# SGAD Tools

Run these tools from the repository root after
[installation](../../README.md#installation).

| Tool | Purpose |
|---|---|
| `bbox_data/verify_bboxes.py` | Validate bbox files and save visualizations. |
| `bbox_data/scan_bbox_counts.py` | Build per-image bbox-count indices. |
| `bbox_data/analyze_bbox_counts.py` | List images below bbox-count thresholds. |
| `bbox_data/build_pair_bbox_counts.py` | Build pair-count metadata. |
| `bbox_data/list_images.py` | List dataset images. |
| `model_conversion/convert_dinov3_vit_pth_to_hf.py` | Convert DINOv3 weights to Transformers format. |

BBox generation and pair-count commands are in
[Custom Data Preparation](../../docs/data_preparation.md#custom-data-preparation).
Model inference and checkpoint export are in the
[SGAD README](../README.md).

## Count Analysis

```bash
python sgad/tools/bbox_data/analyze_bbox_counts.py \
  --counts_npz /path/to/bbox_counts.npz \
  --thresholds 1 3 5 --format txt --workers 4
```

## Bbox Validation

```bash
python sgad/tools/bbox_data/verify_bboxes.py \
  --root_dir /path/to/dataset_images \
  --output_root /path/to/dataset_bboxes \
  --max_files 100 --scan_workers 32 \
  --viz_dir sgad/tools/viz --viz_samples 20
```

The tool finds images in `color`, `imgs`, or `images` directories and loads
matching `.npy` files from parallel `bboxes` directories.

## DINOv3 Checkpoint Conversion

Convert an official DINOv3 ViT-L/16 LVD-1689M `.pth` checkpoint to a local
Transformers model directory:

```bash
python sgad/tools/model_conversion/convert_dinov3_vit_pth_to_hf.py \
  --src /path/to/dinov3_vitl16_pretrain.pth \
  --dst pretrained/dinov3-vitl16-pretrain-lvd1689m
```

Download sources are in the
[backbone guide](../../README.md#region-matcher-backbones). Use the same
output directory name under `A2PM_PRETRAINED_ROOT` when overriding that root.
