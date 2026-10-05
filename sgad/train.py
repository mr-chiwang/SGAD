import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.strategies import DDPStrategy

from sgad.config.default import get_cfg_defaults
from sgad.utils.misc import get_rank_zero_only_logger
from sgad.utils.profiler import build_profiler
from sgad.data.datamodule import SGADDataModule
from sgad.training.lightning_module import SGADLightningModule
from sgad.training.checkpointing import build_checkpoint_callbacks

loguru_logger = get_rank_zero_only_logger(loguru_logger)


class EpochSummaryCallback(Callback):
    """Print one concise rank-zero validation summary per epoch during checkpointing."""

    def __init__(self):
        super().__init__()
        self._last_logged_epoch = -1
        # Track the best validation metrics.
        self._best_MOM = 0.0
        self._best_BMR = 0.0
        self._best_HMQR = 0.0
        self._best_Coverage = 0.0

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Update and print the validation summary at most once per epoch."""
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        # Lightning may invoke this hook more than once for the same epoch.
        if epoch == self._last_logged_epoch:
            return
        self._last_logged_epoch = epoch

        metrics = trainer.callback_metrics

        # Read the validation metrics published by the LightningModule.
        MOM = metrics.get('MOM_avg_val(all)', None)
        BMR = metrics.get('BMR_avg_val(all)', None)
        HMQR = metrics.get('HMQR_avg_val(all)', None)
        Coverage = metrics.get('Coverage_avg_val(all)', None)

        # Skip until the validation metric is available.
        if HMQR is None:
            return

        # Convert scalar tensors to plain values for formatting and comparisons.
        MOM_val = float(MOM) if MOM is not None else 0.0
        BMR_val = float(BMR) if BMR is not None else 0.0
        HMQR_val = float(HMQR) if HMQR is not None else 0.0
        Coverage_val = float(Coverage) if Coverage is not None else 0.0

        self._best_MOM = max(self._best_MOM, MOM_val)
        self._best_BMR = max(self._best_BMR, BMR_val)
        self._best_HMQR = max(self._best_HMQR, HMQR_val)
        self._best_Coverage = max(self._best_Coverage, Coverage_val)

        summary = (
            f"Epoch {epoch} | "
            f"HMQR: {HMQR_val:.4f} | "
            f"MOM: {MOM_val:.4f} | "
            f"BMR: {BMR_val:.4f} | "
            f"Coverage: {Coverage_val:.4f}\n"
            f"    best | "
            f"HMQR: {self._best_HMQR:.4f} | "
            f"MOM: {self._best_MOM:.4f} | "
            f"BMR: {self._best_BMR:.4f} | "
            f"Coverage: {self._best_Coverage:.4f}"
        )
        print(f"\n{summary}\n")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'main_cfg_path', type=str, nargs='?',
        default='configs/scannet_sgad_plus_plus_bce.py',
        help='model and dataset config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='optional SGAD-family initialization checkpoint')
    parser.add_argument(
        '--resume_from_checkpoint_path', type=str,
        default=None,
        help='resume from checkpoint path')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='use inference profiling, or leave it unset')
    parser.add_argument(
        '--pred_score_thr', type=float, default=0.5, help='bbox matching threshold.')

    # Expose the trainer options supported by this entry point.
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--fast_dev_run', type=lambda x: bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)
    parser.add_argument('--benchmark', type=lambda x: bool(strtobool(x)), nargs='?', const=True, default=False,
                        help='Enable cudnn benchmark mode (Lightning 2.x: set via torch.backends.cudnn.benchmark)')

    return parser.parse_args()


def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # Load the defaults and apply the selected experiment configuration.
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # Reproduce initialization and sampling.

    # Configure cuDNN benchmark mode from the CLI option.
    torch.backends.cudnn.benchmark = args.benchmark

    # Derive runtime batch-size metadata used by the optimizer configuration.
    if args.devices == -1:
        _n_gpus = torch.cuda.device_count()
    elif args.devices > 0:
        _n_gpus = args.devices
    else:
        raise ValueError("--devices must be a positive integer or -1")
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR

    # Build the Lightning module.
    profiler = build_profiler(args.profiler_name)
    model = SGADLightningModule(
        config,
        pretrained_ckpt=(
            args.ckpt_path
            if args.resume_from_checkpoint_path is None
            else None
        ),
        profiler=profiler,
        pred_score_threshold=args.pred_score_thr,
    )

    loguru_logger.info(f"Model LightningModule initialized!")

    # Build the data module.
    data_module = SGADDataModule(args, config)
    loguru_logger.info(f"Model DataModule initialized!")

    # Configure TensorBoard logging.
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)

    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    # Configure checkpointing and epoch-summary callbacks.
    best_ckpt_callback, latest_ckpt_callback = build_checkpoint_callbacks(ckpt_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    epoch_summary = EpochSummaryCallback()
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.extend([best_ckpt_callback, latest_ckpt_callback])
    callbacks.append(epoch_summary)

    # Configure the trainer.
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=args.num_sanity_val_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 1,
        use_distributed_sampler=False,  # Keep the data module's custom sampler.
        reload_dataloaders_every_n_epochs=0,
        profiler=profiler
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")

    # Start training, optionally restoring the complete trainer state.
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint_path)


if __name__ == '__main__':
    main()
