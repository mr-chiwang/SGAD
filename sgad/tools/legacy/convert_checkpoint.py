"""Convert a legacy Lightning checkpoint to the portable SGAD model format."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from sgad.config.default import get_cfg_defaults
from sgad.config.model_variants import apply_model_variant
from sgad.models import build_sgad_model
from sgad.models.checkpoints import infer_checkpoint_model_variant, make_portable_checkpoint
from sgad.utils.misc import lower_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model-variant",
        choices=("sgad", "sgad_plus_plus_bce", "sgad_plus_plus_focal"),
        default=None,
        help="Override automatic legacy-config inference.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YACS config override. Otherwise the checkpoint training config is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(args.input, **load_kwargs)
    checkpoint_config = checkpoint.get("hyper_parameters", {}).get("config")
    if checkpoint_config is None:
        checkpoint_config = checkpoint.get("config")
    inferred_variant = infer_checkpoint_model_variant(checkpoint)
    if (
        args.config is None
        and checkpoint_config is None
        and args.model_variant is None
        and inferred_variant is None
    ):
        raise ValueError(
            "A checkpoint without model metadata requires --model-variant or --config."
        )

    if args.config:
        config = get_cfg_defaults()
        config.merge_from_file(str(args.config))
    else:
        config = checkpoint_config
        if config is None:
            config = get_cfg_defaults()

    variant = args.model_variant or (inferred_variant.value if inferred_variant is not None else None)
    if variant is not None:
        apply_model_variant(config, variant)
    model = build_sgad_model(lower_config(config)["model"])
    portable = make_portable_checkpoint(
        checkpoint,
        model=model,
        model_variant=args.model_variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, args.output)


if __name__ == "__main__":
    main()
