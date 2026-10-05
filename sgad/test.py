import torch
import pytorch_lightning as pl
import argparse
import pprint
from pathlib import Path
from loguru import logger as loguru_logger

from sgad.config.default import get_cfg_defaults
from sgad.utils.profiler import build_profiler

from sgad.data.datamodule import SGADDataModule
from sgad.training.lightning_module import SGADLightningModule

def parse_args():
    # Define the evaluation CLI, including the supported Lightning trainer options.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='use inference profiling, or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, choices=[1], default=1,
        help='one image pair per batch for region evaluation')
    parser.add_argument(
        '--num_workers', type=int, default=8)
    # Expose the trainer options supported by this entry point.
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output_dir', type=Path, default=None,
                        help='new directory for resolved config and per-pair test results')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='YACS KEY VALUE overrides, placed last on the command line')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pprint.pprint(vars(args))

    # Load the defaults and apply the selected experiment configuration.
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_list(args.opts)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / 'config.yaml').write_text(config.dump())
    # config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # Reproduce initialization and sampling.

    loguru_logger.info(f"Args and config initialized!")

    # Build the Lightning module.
    profiler = build_profiler(args.profiler_name)
    model = SGADLightningModule(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    model.test_output_dir = args.output_dir
    loguru_logger.info(f"Model-lightning initialized!")

    # Build the data module.
    data_module = SGADDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # Configure the evaluation trainer.
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        strategy=args.strategy,
        benchmark=args.benchmark,
        use_distributed_sampler=False,  # Keep the data module's custom sampler.
        logger=False,
        inference_mode=False,
        profiler=profiler
    )

    loguru_logger.info(f"Start testing!")
    with torch.no_grad():
        trainer.test(model, datamodule=data_module, verbose=False)
