from sgad.config.default import get_cfg_defaults
from sgad.config.model_variants import ModelVariant, apply_model_variant


def _apply_model(cfg, variant: ModelVariant):
    apply_model_variant(cfg, variant)
    if variant is not ModelVariant.SGAD:
        cfg.MODEL.BBOX.ROPE.BASE = 100.0
        cfg.MODEL.BBOX.ROPE.SHIFT_COORDS = 0.1
        cfg.MODEL.BBOX.ROPE.JITTER_COORDS = 1.2
        cfg.MODEL.BBOX.ROPE.RESCALE_COORDS = 1.5
        cfg.MODEL.LOSS.SOFT_OVERLAP_LOSS_WEIGHT = 1.0


def make_scannet_cfg(variant: ModelVariant):
    cfg = get_cfg_defaults()
    _apply_model(cfg, variant)

    min_covisibility_train = 10
    min_covisibility_val = 1
    min_covisibility_test = 1

    train_base_path = "data/scannet/index/scene_data"
    cfg.DATASET.TRAINVAL_DATA_SOURCE = "ScanNet"
    cfg.DATASET.TRAIN_DATA_ROOT = "data/scannet/train"
    cfg.DATASET.TRAIN_NPZ_ROOT = f"{train_base_path}/train"
    cfg.DATASET.TRAIN_LIST_PATH = f"{train_base_path}/train_list/scannet_all.txt"
    cfg.DATASET.TRAIN_INTRINSIC_PATH = "data/scannet/index/intrinsics.npz"
    cfg.DATASET.TRAIN_BBOX_ROOT = "data/scannet_bboxes"
    cfg.DATASET.TRAIN_PAIR_BBOX_COUNTS_ROOT = "assets/pair_bbox_counts/scannet/train"
    cfg.DATASET.TRAIN_MIN_PAIR_BBOX_COUNT = min_covisibility_train

    val_base_path = "assets/scannet_test_1500"
    cfg.DATASET.VAL_NPZ_ROOT = val_base_path
    cfg.DATASET.VAL_LIST_PATH = f"{val_base_path}/scannet_test.txt"
    cfg.DATASET.VAL_PAIR_BBOX_COUNTS_ROOT = "assets/pair_bbox_counts/scannet/test1500"
    cfg.DATASET.VAL_MIN_PAIR_BBOX_COUNT = min_covisibility_val

    test_base_path = "assets/scannet_test_1500"
    cfg.DATASET.TEST_NPZ_ROOT = test_base_path
    cfg.DATASET.TEST_LIST_PATH = f"{test_base_path}/scannet_test.txt"
    cfg.DATASET.TEST_PAIR_BBOX_COUNTS_ROOT = "assets/pair_bbox_counts/scannet/test1500"
    cfg.DATASET.TEST_MIN_PAIR_BBOX_COUNT = min_covisibility_test
    cfg.DATASET.TEST_DATA_SOURCE = "ScanNet"
    cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/scannet/test"
    cfg.DATASET.VAL_INTRINSIC_PATH = cfg.DATASET.TEST_INTRINSIC_PATH = (
        "assets/scannet_test_1500/intrinsics.npz"
    )
    cfg.DATASET.VAL_BBOX_ROOT = cfg.DATASET.TEST_BBOX_ROOT = "data/scannet_bboxes"
    cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

    cfg.MODEL.BBOX.MIN_AREA_SIZE = 2000
    cfg.MODEL.BBOX.MIN_WH = 28
    cfg.MODEL.BBOX.MIN_COUNT = 10
    cfg.MODEL.BBOX.IOU_THR = 0.4
    cfg.MODEL.BBOX.TRAIN_ENABLE_FILTER = True
    cfg.MODEL.BBOX.TRAIN_ENABLE_EXPAND = True
    cfg.MODEL.BBOX.TRAIN_ENABLE_RANDOM_GEN = True
    cfg.MODEL.BBOX.VAL_ENABLE_FILTER = True
    cfg.MODEL.BBOX.VAL_ENABLE_EXPAND = False
    cfg.MODEL.BBOX.VAL_ENABLE_RANDOM_GEN = False
    cfg.MODEL.BBOX.TEST_ENABLE_FILTER = True
    cfg.MODEL.BBOX.TEST_ENABLE_EXPAND = False
    cfg.MODEL.BBOX.TEST_ENABLE_RANDOM_GEN = False

    cfg.TRAINER.CANONICAL_LR = 1e-3
    cfg.TRAINER.WARMUP_STEP = 473
    cfg.TRAINER.WARMUP_RATIO = 0.1
    cfg.TRAINER.MSLR_MILESTONES = [4, 8, 12, 16, 20, 25, 30]
    cfg.TRAINER.OPTIMIZER = "adamw"
    cfg.TRAINER.ADAMW_DECAY = 0.1
    cfg.TRAINER.N_SAMPLES_PER_SUBSET = 20

    return cfg


def make_megadepth_cfg(variant: ModelVariant, *, test: bool = False):
    cfg = get_cfg_defaults()
    _apply_model(cfg, variant)

    min_covisibility_train = 10
    min_covisibility_val = 1
    min_covisibility_test = 1

    train_base_path = "data/megadepth/index"
    cfg.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"
    cfg.DATASET.TRAIN_DATA_ROOT = "data/megadepth/train"
    cfg.DATASET.TRAIN_NPZ_ROOT = f"{train_base_path}/scene_info_0.1_0.7"
    cfg.DATASET.TRAIN_LIST_PATH = f"{train_base_path}/trainvaltest_list/train_list.txt"
    cfg.DATASET.TRAIN_BBOX_ROOT = "data/megadepth_bboxes"
    cfg.DATASET.TRAIN_PAIR_BBOX_COUNTS_ROOT = "assets/pair_bbox_counts/megadepth/train"
    cfg.DATASET.TRAIN_MIN_PAIR_BBOX_COUNT = min_covisibility_train
    cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.01

    val_base_path = "assets/megadepth_test_1500_scene_info"
    cfg.DATASET.VAL_NPZ_ROOT = val_base_path
    cfg.DATASET.VAL_LIST_PATH = f"{val_base_path}/megadepth_test_1500.txt"
    cfg.DATASET.VAL_PAIR_BBOX_COUNTS_ROOT = "assets/pair_bbox_counts/megadepth/test1500"
    cfg.DATASET.VAL_MIN_PAIR_BBOX_COUNT = min_covisibility_val
    test_base_path = "assets/megadepth_test_1500_scene_info"
    cfg.DATASET.TEST_NPZ_ROOT = test_base_path
    cfg.DATASET.TEST_LIST_PATH = f"{test_base_path}/megadepth_test_1500.txt"
    cfg.DATASET.TEST_PAIR_BBOX_COUNTS_ROOT = "assets/pair_bbox_counts/megadepth/test1500"
    cfg.DATASET.TEST_MIN_PAIR_BBOX_COUNT = min_covisibility_test
    cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
    cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
    cfg.DATASET.VAL_BBOX_ROOT = cfg.DATASET.TEST_BBOX_ROOT = "data/megadepth_bboxes"
    cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
    cfg.DATASET.MGDPT_IMG_RESIZE = 2000
    cfg.DATASET.MGDPT_IMG_PAD = True
    cfg.DATASET.MGDPT_DF = 16

    cfg.MODEL.BBOX.MIN_AREA_SIZE = 6000 if test else 4000
    cfg.MODEL.BBOX.MIN_WH = 28
    cfg.MODEL.BBOX.MIN_COUNT = 3 if test else 10
    cfg.MODEL.BBOX.IOU_THR = 0.4 if test else 0.2
    cfg.MODEL.BBOX.TRAIN_ENABLE_FILTER = True
    cfg.MODEL.BBOX.TRAIN_ENABLE_EXPAND = True
    cfg.MODEL.BBOX.TRAIN_ENABLE_RANDOM_GEN = True
    cfg.MODEL.BBOX.VAL_ENABLE_FILTER = True
    cfg.MODEL.BBOX.VAL_ENABLE_EXPAND = False
    cfg.MODEL.BBOX.VAL_ENABLE_RANDOM_GEN = False
    cfg.MODEL.BBOX.TEST_ENABLE_FILTER = True
    cfg.MODEL.BBOX.TEST_ENABLE_EXPAND = test
    cfg.MODEL.BBOX.TEST_ENABLE_RANDOM_GEN = False

    cfg.TRAINER.CANONICAL_LR = 1e-4
    cfg.TRAINER.WARMUP_STEP = 864
    cfg.TRAINER.WARMUP_RATIO = 0.1
    cfg.TRAINER.MSLR_MILESTONES = [4, 8, 12, 16, 20, 25, 30]
    cfg.TRAINER.OPTIMIZER = "adamw"
    cfg.TRAINER.ADAMW_DECAY = 0.1
    cfg.TRAINER.N_SAMPLES_PER_SUBSET = 50

    return cfg
