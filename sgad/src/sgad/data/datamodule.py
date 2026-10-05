from os import path as osp
from collections import abc

import pytorch_lightning as pl
from loguru import logger
from torch import distributed as dist
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    DistributedSampler,
)
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from sgad.data.augmentation import build_augmentor
from sgad.data.datasets.megadepth import MegaDepthDataset
from sgad.data.datasets.scannet import ScanNetDataset
from sgad.data.sampler import RandomConcatSampler
from sgad.data.sampling import get_local_split
class SGADDataModule(pl.LightningDataModule):
    """Build scene-level datasets and dataloaders for SGAD.

    Training scenes are partitioned across ranks before dataset construction to
    reduce per-process memory use.
    """
    def __init__(self, args, config):
        super().__init__()

        # Dataset sources and paths
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.val_data_source = self.trainval_data_source
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        
        # Training and validation paths
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.train_bbox_root = config.DATASET.TRAIN_BBOX_ROOT  # (optional)
        self.train_pair_bbox_counts_root = config.DATASET.TRAIN_PAIR_BBOX_COUNTS_ROOT
        self.train_min_pair_bbox_count = config.DATASET.TRAIN_MIN_PAIR_BBOX_COUNT

        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        self.val_bbox_root = config.DATASET.VAL_BBOX_ROOT  # (optional)
        self.val_pair_bbox_counts_root = config.DATASET.VAL_PAIR_BBOX_COUNTS_ROOT
        self.val_min_pair_bbox_count = config.DATASET.VAL_MIN_PAIR_BBOX_COUNT
        # Test paths
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH
        self.test_bbox_root = config.DATASET.TEST_BBOX_ROOT  # (optional)
        self.test_pair_bbox_counts_root = config.DATASET.TEST_PAIR_BBOX_COUNTS_ROOT
        self.test_min_pair_bbox_count = config.DATASET.TEST_MIN_PAIR_BBOX_COUNT

        # Dataset behavior
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None or 'dark'

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD
        self.mgdpt_df = config.DATASET.MGDPT_DF
        # Downsampling factor corresponding to RESOLUTION_BBOX.
        self.bbox_scale = 1 / config.MODEL.RESOLUTION_BBOX

        # DataLoader options
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'collate_fn': custom_collate_fn
        }
                
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'collate_fn': custom_collate_fn
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            # 'prefetch_factor': args.prefetch_factor,
            'pin_memory': True,
            'collate_fn': custom_collate_fn
        }

        # Scene-balanced sampler options
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT

        # Shared region-filtering thresholds
        self.min_bbox_area = config.MODEL.BBOX.MIN_AREA_SIZE
        self.min_bbox_wh = config.MODEL.BBOX.MIN_WH
        self.min_bbox_count = config.MODEL.BBOX.MIN_COUNT

        # Training-phase filtering behavior
        self.train_bbox_enable_filter = config.MODEL.BBOX.TRAIN_ENABLE_FILTER
        self.train_bbox_enable_expand = config.MODEL.BBOX.TRAIN_ENABLE_EXPAND
        self.train_bbox_enable_random_gen = config.MODEL.BBOX.TRAIN_ENABLE_RANDOM_GEN

        # Validation-phase filtering behavior
        self.val_bbox_enable_filter = config.MODEL.BBOX.VAL_ENABLE_FILTER
        self.val_bbox_enable_expand = config.MODEL.BBOX.VAL_ENABLE_EXPAND
        self.val_bbox_enable_random_gen = config.MODEL.BBOX.VAL_ENABLE_RANDOM_GEN

        # Test-phase filtering behavior
        self.test_bbox_enable_filter = config.MODEL.BBOX.TEST_ENABLE_FILTER
        self.test_bbox_enable_expand = config.MODEL.BBOX.TEST_ENABLE_EXPAND
        self.test_bbox_enable_random_gen = config.MODEL.BBOX.TEST_ENABLE_RANDOM_GEN

        # Sampling seed
        self.seed = config.TRAINER.SEED

    def setup(self, stage=None):
        """
        Build the datasets required by the requested Lightning stage.

        Args:
            stage: One of ``fit``, ``validate``, or ``test``.
        """

        assert stage in ['fit', 'validate', 'test'], "stage must be either fit or test"

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0

        if stage == 'fit':

            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode='train',
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root,
                bbox_root=self.train_bbox_root,
                pair_bbox_counts_root=self.train_pair_bbox_counts_root,
                min_pair_bbox_count=self.train_min_pair_bbox_count)

            # Build one validation loader per configured validation subset.
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root,
                        bbox_root=self.val_bbox_root,
                        pair_bbox_counts_root=self.val_pair_bbox_counts_root,
                        min_pair_bbox_count=self.val_min_pair_bbox_count))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root,
                    bbox_root=self.val_bbox_root,
                    pair_bbox_counts_root=self.val_pair_bbox_counts_root,
                    min_pair_bbox_count=self.val_min_pair_bbox_count)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')

        elif stage == 'validate':
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root,
                        bbox_root=self.val_bbox_root,
                        pair_bbox_counts_root=self.val_pair_bbox_counts_root,
                        min_pair_bbox_count=self.val_min_pair_bbox_count))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root,
                    bbox_root=self.val_bbox_root,
                    pair_bbox_counts_root=self.val_pair_bbox_counts_root,
                    min_pair_bbox_count=self.val_min_pair_bbox_count)
            logger.info(f'[rank:{self.rank}] Val Dataset loaded!')
        else:  # stage == 'test'
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root,
                bbox_root=self.test_bbox_root,
                pair_bbox_counts_root=self.test_pair_bbox_counts_root,
                min_pair_bbox_count=self.test_min_pair_bbox_count)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       intri_path,
                       mode='train',
                       min_overlap_score=0.,
                       pose_dir=None,
                       bbox_root=None,
                       pair_bbox_counts_root=None,
                       min_pair_bbox_count=0):
        """Build a concatenated per-scene dataset for the current rank and phase."""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
        else:
            local_npz_names = npz_names
        logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')

        dataset_builder = self._build_concat_dataset
        return dataset_builder(data_root, local_npz_names, split_npz_root, intri_path,
                                mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir,
                                bbox_root=bbox_root, pair_bbox_counts_root=pair_bbox_counts_root,
                                min_pair_bbox_count=min_pair_bbox_count)

    def _build_concat_dataset(
        self,
        data_root,
        npz_names,
        npz_dir,
        intrinsic_path,
        mode,
        min_overlap_score=0.,
        pose_dir=None,
        bbox_root=None,
        pair_bbox_counts_root=None,
        min_pair_bbox_count=0
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None

        if mode == 'train':
            data_source = self.trainval_data_source
        elif mode == 'val':
            data_source = self.val_data_source
        else:
            data_source = self.test_data_source

        # Select filtering behavior for the current data phase.
        if mode == 'train':
            bbox_enable_filter = self.train_bbox_enable_filter
            bbox_enable_expand = self.train_bbox_enable_expand
            bbox_enable_random_gen = self.train_bbox_enable_random_gen
        elif mode == 'val':
            bbox_enable_filter = self.val_bbox_enable_filter
            bbox_enable_expand = self.val_bbox_enable_expand
            bbox_enable_random_gen = self.val_bbox_enable_random_gen
        else:  # test
            bbox_enable_filter = self.test_bbox_enable_filter
            bbox_enable_expand = self.test_bbox_enable_expand
            bbox_enable_random_gen = self.test_bbox_enable_random_gen

        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        for npz_name in tqdm(npz_names,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            # Dataset construction eagerly loads pair metadata from each NPZ file.
            npz_path = osp.join(npz_dir, npz_name)
            pair_bbox_counts_path = None
            if pair_bbox_counts_root is not None:
                counts_name = f"{osp.splitext(npz_name)[0]}.npy"
                pair_bbox_counts_path = osp.join(pair_bbox_counts_root, counts_name)
            if data_source == 'ScanNet':
                dataset = ScanNetDataset(data_root,
                                         npz_path,
                                         intrinsic_path,
                                         mode=mode,
                                         min_overlap_score=min_overlap_score,
                                         augment_fn=augment_fn,
                                         pose_dir=pose_dir,
                                         bbox_root=bbox_root,
                                         min_bbox_area=self.min_bbox_area,
                                         min_bbox_wh=self.min_bbox_wh,
                                         min_bbox_count=self.min_bbox_count,
                                         bbox_enable_filter=bbox_enable_filter,
                                         bbox_enable_expand=bbox_enable_expand,
                                         bbox_enable_random_gen=bbox_enable_random_gen,
                                         pair_bbox_counts_path=pair_bbox_counts_path,
                                         min_pair_bbox_count=min_pair_bbox_count)
            elif data_source == 'MegaDepth':
                dataset = MegaDepthDataset(data_root,
                                           npz_path,
                                           mode=mode,
                                           min_overlap_score=min_overlap_score,
                                           img_resize=self.mgdpt_img_resize,
                                           df=self.mgdpt_df,
                                           img_padding=self.mgdpt_img_pad,
                                           depth_padding=self.mgdpt_depth_pad,
                                           augment_fn=augment_fn,
                                           bbox_scale=self.bbox_scale,
                                           bbox_root=bbox_root,
                                           min_bbox_area=self.min_bbox_area,
                                           min_bbox_wh=self.min_bbox_wh,
                                           min_bbox_count=self.min_bbox_count,
                                           bbox_enable_filter=bbox_enable_filter,
                                           bbox_enable_expand=bbox_enable_expand,
                                           bbox_enable_random_gen=bbox_enable_random_gen,
                                           pair_bbox_counts_path=pair_bbox_counts_path,
                                           min_pair_bbox_count=min_pair_bbox_count)
            else:
                raise NotImplementedError()
            if len(dataset) > 0:
                datasets.append(dataset)
            elif self.rank == 0:
                logger.debug(f"Skipping empty {mode} scene after pair filtering: {npz_name}")
        if not datasets:
            raise RuntimeError(f"No {mode} pairs remain after applying dataset filters")
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """Build the scene-balanced training DataLoader."""
        assert self.data_sampler in ['scene_balance']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                        self.n_samples_per_subset,
                                        self.subset_replacement,
                                        self.shuffle, self.repeat, self.seed)
        else:
            sampler = None
        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return dataloader
    
    def val_dataloader(self):
        """Build one distributed validation DataLoader per validation subset."""
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.world_size > 1 else None
            return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                sampler = DistributedSampler(dataset, shuffle=False) if self.world_size > 1 else None
                dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.world_size > 1 else None
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    elem = batch[0]
    batch_data = {}

    for key in elem:
        if key in ['bboxes0', 'bboxes1']:
            # Region counts vary by sample, so keep region tensors in a list.
            batch_data[key] = [d[key] for d in batch]
        else:
            # Stack all fixed-shape fields with PyTorch's default collation.
            batch_data[key] = default_collate([d[key] for d in batch])

    return batch_data
