import os
from pathlib import Path

from yacs.config import CfgNode as CN
from .model_variants import (
    ALTERNATING_ATTENTION_ROPE,
    DEFAULT_MODEL_VARIANT,
)
_CN = CN()

_A2PM_ROOT = Path(os.environ.get("A2PM_SGAD_ROOT", Path(__file__).resolve().parents[4]))
_PRETRAINED_ROOT = Path(os.environ.get("A2PM_PRETRAINED_ROOT", _A2PM_ROOT / "pretrained"))

# Model configuration
_CN.MODEL = CN()
_CN.MODEL.NAME = 'sgad_plus_plus'
_CN.MODEL.VARIANT = DEFAULT_MODEL_VARIANT.value
_CN.MODEL.BACKBONE_TYPE = 'dinov3'  # options: ['dinov2', 'dinov3']
_CN.MODEL.RESOLUTION_BBOX = 2

# Backbone configuration
_CN.MODEL.DINOV2 = CN()
_CN.MODEL.DINOV2.INITIAL_DIM = 1024
_CN.MODEL.DINOV2.PATH = str(_PRETRAINED_ROOT / "dinov2_large")

_CN.MODEL.DINOV3 = CN()
_CN.MODEL.DINOV3.INITIAL_DIM = 1024  # output dimension (1024 for vitl16)
_CN.MODEL.DINOV3.PATH = str(_PRETRAINED_ROOT / "dinov3-vitl16-pretrain-lvd1689m")  # local path
_CN.MODEL.DINOV3.LAYER_IDX = None  # None=last layer only, or list like [11, 17] for multi-layer
_CN.MODEL.DINOV3.FROZEN = True  # freeze weights
# When FROZEN is false, positive values train the final LayerNorm and trailing encoder blocks;
# zero leaves the complete backbone trainable.
_CN.MODEL.DINOV3.TRAINABLE_LAYERS = 0
_CN.MODEL.DINOV3.NORMALIZE_FEATS = False  # L2 normalize output features

# Region-matching configuration
_CN.MODEL.BBOX = CN()
_CN.MODEL.BBOX.LAYER_NAMES = ['self', 'cross'] * 4
_CN.MODEL.BBOX.MIN_AREA_SIZE = 4000
_CN.MODEL.BBOX.MIN_WH = 28  # minimum width/height for bbox filtering
_CN.MODEL.BBOX.MIN_COUNT = 3  # minimum number of bboxes per image
_CN.MODEL.BBOX.IOU_THR = 0.2

# Default region preprocessing by data phase; dataset-family configs may override these values.
# Training may expand rejected boxes or synthesize boxes to meet the minimum count.
_CN.MODEL.BBOX.TRAIN_ENABLE_FILTER = True      # filter by area/wh during training
_CN.MODEL.BBOX.TRAIN_ENABLE_EXPAND = True      # expand rejected bboxes during training
_CN.MODEL.BBOX.TRAIN_ENABLE_RANDOM_GEN = True  # generate random bboxes if count insufficient

# Validation defaults to filtering without expansion or synthetic regions.
_CN.MODEL.BBOX.VAL_ENABLE_FILTER = True        # filter by area/wh during validation
_CN.MODEL.BBOX.VAL_ENABLE_EXPAND = False       # no expansion during validation
_CN.MODEL.BBOX.VAL_ENABLE_RANDOM_GEN = False   # no random gen during validation

# Testing defaults to filtering without expansion or synthetic regions.
_CN.MODEL.BBOX.TEST_ENABLE_FILTER = True       # filter by area/wh during testing
_CN.MODEL.BBOX.TEST_ENABLE_EXPAND = False      # no expansion during testing
_CN.MODEL.BBOX.TEST_ENABLE_RANDOM_GEN = False  # no random gen during testing

_CN.MODEL.BBOX.FROM_IMG = True  # extract features from full image then ROI pool by bbox regions

# Region-transformer architecture
_CN.MODEL.BBOX.ATTENTION_TYPE = ALTERNATING_ATTENTION_ROPE
_CN.MODEL.BBOX.NHEAD = 8  # number of attention heads
_CN.MODEL.BBOX.FFN_RATIO = 4.0  # FFN hidden dimension ratio
_CN.MODEL.BBOX.QKV_BIAS = True  # whether to use bias in QKV projection
_CN.MODEL.BBOX.DROPOUT = 0.0  # dropout rate
_CN.MODEL.BBOX.ATTN_DROPOUT = 0.0  # attention dropout rate
_CN.MODEL.BBOX.POS_SCALE_LEARNABLE = True  # learn the SGAD MLP positional-encoding scale

# RoPE settings for SGAD++ alternating attention.
_CN.MODEL.BBOX.ROPE = CN()
_CN.MODEL.BBOX.ROPE.BASE = 100.0  # frequency base for RoPE
_CN.MODEL.BBOX.ROPE.SHIFT_COORDS = None  # training augmentation: shift coordinates
_CN.MODEL.BBOX.ROPE.JITTER_COORDS = None  # training augmentation: jitter coordinates
_CN.MODEL.BBOX.ROPE.RESCALE_COORDS = None  # training augmentation: rescale coordinates

# Loss configuration
_CN.MODEL.LOSS = CN()
_CN.MODEL.LOSS.FOCAL_ALPHA = 0.25
_CN.MODEL.LOSS.FOCAL_GAMMA = 2.0
_CN.MODEL.LOSS.POS_WEIGHT = 1.0
_CN.MODEL.LOSS.NEG_WEIGHT = 1.0

# Soft-overlap supervision avoids thresholding overlap targets into binary labels.
_CN.MODEL.LOSS.USE_SOFT_OVERLAP_LOSS = True
_CN.MODEL.LOSS.SOFT_OVERLAP_LOSS_TYPE = 'bce'  # Default for SGAD++; SGAD++-Focal selects focal_soft.
_CN.MODEL.LOSS.SOFT_OVERLAP_LOSS_WEIGHT = 1.0

# Dataset configuration
_CN.DATASET = CN()
# Dataset paths for training and validation
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.TRAIN_BBOX_ROOT = None  # (optional directory for precomputed bboxes)
_CN.DATASET.TRAIN_PAIR_BBOX_COUNTS_ROOT = None
_CN.DATASET.TRAIN_MIN_PAIR_BBOX_COUNT = 0
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None  # Path to the validation scene list.
_CN.DATASET.VAL_INTRINSIC_PATH = None
_CN.DATASET.VAL_BBOX_ROOT = None  # (optional directory for precomputed bboxes)
_CN.DATASET.VAL_PAIR_BBOX_COUNTS_ROOT = None
_CN.DATASET.VAL_MIN_PAIR_BBOX_COUNT = 0
# Dataset paths for testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None  # Path to the test scene list.
_CN.DATASET.TEST_INTRINSIC_PATH = None
_CN.DATASET.TEST_IMGSIZE = None
_CN.DATASET.TEST_BBOX_ROOT = None  # (optional directory for precomputed bboxes)
_CN.DATASET.TEST_PAIR_BBOX_COUNTS_ROOT = None
_CN.DATASET.TEST_MIN_PAIR_BBOX_COUNT = 0

# General dataset options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = 'dark'  # options: [None, 'dark']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 2000  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# Trainer configuration
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
# Optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.
_CN.TRAINER.ADAMW_DECAY = 0.01

# Step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 1760


# Learning-rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# Training and validation visualization
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 10     # number of validation pairs to plot

# Training data sampler
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'
# Scene-balanced sampler options
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # Sample within each scene with replacement.
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # Shuffle the combined scene samples within each epoch.
_CN.TRAINER.SB_REPEAT = 1  # Repeat the sampled indices this many times.
# Gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# Global seed for reproducibility, scene assignment, and RandomConcatSampler order.
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Return an independent clone of the default SGAD configuration."""
    return _CN.clone()
