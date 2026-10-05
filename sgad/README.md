# SGAD / SGAD++ Region Matching

This subproject provides SGAD and SGAD++ models for matching image regions.

| Model variant | Backbone | Attention and positional encoding | Matching objective | Ranking objective |
|---|---|---|---|---|
| `sgad` | DINOv2 | alternating attention + MLP positional encoding | hard focal | ListMLE |
| `sgad_plus_plus_bce` | DINOv3 | alternating attention + RoPE | soft BCE | ListMLE |
| `sgad_plus_plus_focal` | DINOv3 | alternating attention + RoPE | soft focal | ListMLE |

## Setup

Follow the [installation guide](../README.md#installation), prepare the
[released data](../docs/data_preparation.md#released-assets) and
[DINO backbone](../README.md#region-matcher-backbones), then enter `sgad/`:

```bash
cd sgad
```

## Configurations

| Dataset | SGAD | SGAD++ BCE | SGAD++ Focal |
|---|---|---|---|
| ScanNet | `configs/scannet_sgad.py` | `configs/scannet_sgad_plus_plus_bce.py` | `configs/scannet_sgad_plus_plus_focal.py` |
| MegaDepth train | `configs/megadepth_train_sgad.py` | `configs/megadepth_train_sgad_plus_plus_bce.py` | `configs/megadepth_train_sgad_plus_plus_focal.py` |
| MegaDepth test | `configs/megadepth_test_sgad.py` | `configs/megadepth_test_sgad_plus_plus_bce.py` | `configs/megadepth_test_sgad_plus_plus_focal.py` |

The released SGAD checkpoints use a fixed positional scale of `1.0` with these
configurations:

- `configs/scannet_sgad_fixed_pos_scale.py`
- `configs/megadepth_train_sgad_fixed_pos_scale.py`
- `configs/megadepth_test_sgad_fixed_pos_scale.py`

## Checkpoints

Download the released Lightning checkpoints from
[Hugging Face](https://huggingface.co/MrWangChi/A2PM-SGAD/tree/main/weights):

| Checkpoint | Model | Dataset |
|---|---|---|
| `sgad_indoor_scannet.ckpt` | SGAD, fixed positional scale | ScanNet |
| `sgad_outdoor_megadepth.ckpt` | SGAD, fixed positional scale | MegaDepth |
| `sgadplusplus_bce_indoor_scannet.ckpt` | SGAD++ BCE | ScanNet |
| `sgadplusplus_bce_outdoor_megadepth.ckpt` | SGAD++ BCE | MegaDepth |

To export model parameters:

```bash
python tools/legacy/convert_checkpoint.py /path/to/weights/sgad_indoor_scannet.ckpt sgad.pt \
  --model-variant sgad
```

## Training

Train SGAD on MegaDepth:

```bash
bash scripts/reproduce_train/megadepth.sh configs/megadepth_train_sgad_fixed_pos_scale.py
```

For SGAD++ BCE:

```bash
bash scripts/reproduce_train/megadepth.sh configs/megadepth_train_sgad_plus_plus_bce.py
```

The training script uses 8 GPUs with batch size 32 per GPU.

## Evaluation

Evaluate SGAD on ScanNet or MegaDepth:

```bash
bash scripts/reproduce_test/scannet.sh \
  configs/scannet_sgad_fixed_pos_scale.py /path/to/weights/sgad_indoor_scannet.ckpt
bash scripts/reproduce_test/megadepth.sh \
  configs/megadepth_test_sgad_fixed_pos_scale.py /path/to/weights/sgad_outdoor_megadepth.ckpt
```

The [region matching metric framework](../docs/sgad_region_evaluation.md)
defines match quality, recall, coverage, and aggregation.

### Released-checkpoint evaluation

The released checkpoints are evaluated with `sgad/test.py` on the 1500-pair
lists. Both models use the same images, boxes, and region-filtering settings
specified in the commands below.

ScanNet uses 640 × 480 images. MegaDepth preserves original image coordinates
and pads images to 2000 × 2000. SGAD extracts features from region crops;
SGAD++ pools regions from full-image features.

For MegaDepth, SGAD++ extracts full-image features with DINOv3 after padding
images to 2000 × 2000, making training and inference very slow. We strongly
recommend adjusting the image resize and padding sizes to suit your hardware
and task.

Evaluation uses FP32, batch size 1, and a prediction confidence threshold of 0.2.
MP thresholds refer to ground-truth overlap. ScanNet input filtering retains
1497 pairs; 3 have empty regions after bbox filtering, yielding 1494 evaluated
pairs. MegaDepth yields 1500 evaluated pairs.

### Results

Values come from the evaluator's `summary.json` and are expressed as
percentages; higher is better. Differences are in percentage points.

| Dataset | Metric | SGAD | SGAD++ | Difference |
|---|---|---:|---:|---:|
| ScanNet | HMQR | 50.72 | 58.46 | +7.74 |
| ScanNet | MOM | 51.39 | 63.34 | +11.96 |
| ScanNet | BMR | 55.12 | 58.61 | +3.49 |
| ScanNet | MP@0.2 | 69.38 | 84.32 | +14.94 |
| ScanNet | MP@0.4 | 63.16 | 78.00 | +14.84 |
| ScanNet | MP@0.6 | 53.27 | 66.24 | +12.97 |
| ScanNet | Coverage | 55.66 | 47.51 | −8.15 |
| ScanNet | AUC-ORC | 62.32 | 66.66 | +4.34 |
| MegaDepth | HMQR | 39.77 | 53.27 | +13.50 |
| MegaDepth | MOM | 49.88 | 64.95 | +15.06 |
| MegaDepth | BMR | 36.71 | 48.84 | +12.13 |
| MegaDepth | MP@0.2 | 60.21 | 78.59 | +18.37 |
| MegaDepth | MP@0.4 | 56.72 | 74.09 | +17.37 |
| MegaDepth | MP@0.6 | 51.40 | 67.34 | +15.93 |
| MegaDepth | Coverage | 39.78 | 41.91 | +2.12 |
| MegaDepth | AUC-ORC | 46.78 | 60.69 | +13.91 |

### Reproduce the results

```bash
WEIGHTS=/path/to/weights
mkdir -p ../res
RESULTS=$(mktemp -d "$PWD/../res/sgad-region.XXXXXX")
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for variant in sgad sgadplusplus_bce; do
  if [ "$variant" = sgad ]; then
    sn_config=scannet_sgad_fixed_pos_scale
    md_config=megadepth_test_sgad_fixed_pos_scale
  else
    sn_config=scannet_sgad_plus_plus_bce
    md_config=megadepth_test_sgad_plus_plus_bce
  fi
  python test.py "configs/$sn_config.py" \
    --ckpt_path "$WEIGHTS/${variant}_indoor_scannet.ckpt" \
    --output_dir "$RESULTS/${variant}_scannet" \
    --devices 1 --batch_size 1 --num_workers 4 --precision 32 \
    --opts MODEL.BBOX.MIN_AREA_SIZE 2000 MODEL.BBOX.MIN_WH 28 \
    MODEL.BBOX.MIN_COUNT 10 MODEL.BBOX.IOU_THR 0.4 \
    MODEL.BBOX.TEST_ENABLE_FILTER True MODEL.BBOX.TEST_ENABLE_EXPAND False \
    MODEL.BBOX.TEST_ENABLE_RANDOM_GEN False DATASET.TEST_MIN_PAIR_BBOX_COUNT 1 \
    DATASET.MIN_OVERLAP_SCORE_TEST 0.0
  python test.py "configs/$md_config.py" \
    --ckpt_path "$WEIGHTS/${variant}_outdoor_megadepth.ckpt" \
    --output_dir "$RESULTS/${variant}_megadepth" \
    --devices 1 --batch_size 1 --num_workers 4 --precision 32 \
    --opts MODEL.BBOX.MIN_AREA_SIZE 4000 MODEL.BBOX.MIN_WH 28 \
    MODEL.BBOX.MIN_COUNT 10 MODEL.BBOX.IOU_THR 0.2 \
    MODEL.BBOX.TEST_ENABLE_FILTER True MODEL.BBOX.TEST_ENABLE_EXPAND False \
    MODEL.BBOX.TEST_ENABLE_RANDOM_GEN False DATASET.TEST_MIN_PAIR_BBOX_COUNT 1 \
    DATASET.MIN_OVERLAP_SCORE_TEST 0.0
done
```

## Inference on Image Pairs

For paired `input1` and `input2` image folders with precomputed bboxes:

```bash
python tools/inference/batch_sgad_bbox_pairs.py \
  --data_roots /path/to/paired-data \
  --output_root /path/to/output \
  --weights /path/to/weights/sgad_outdoor_megadepth.ckpt
```

Each data root can contain multiple pairs of sibling `input1`/`input2`
directories. The command writes matched regions to `.npz` files under
`--output_root`.

See [Data Preparation](../docs/data_preparation.md#custom-data-preparation) to
prepare your own boxes and [Tools](tools/README.md) for data analysis and
backbone conversion.
