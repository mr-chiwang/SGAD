from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from sgad.config.model_variants import ModelVariant, coerce_model_variant, resolve_model_variant


PORTABLE_CHECKPOINT_FORMAT = "sgad_model_v1"


@dataclass(frozen=True)
class CheckpointLoadReport:
    model_variant: ModelVariant
    loaded_keys: int
    model_keys: int
    skipped_keys: int
    mismatched_keys: tuple[str, ...]
    strict: bool


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for raw_key, value in state_dict.items():
        key = raw_key
        if key.startswith("module."):
            key = key.removeprefix("module.")
        if key.startswith("matcher."):
            key = key.removeprefix("matcher.")
        if key.startswith("loftr_bbox."):
            key = "region_transformer." + key.removeprefix("loftr_bbox.")
        elif key.startswith("bbox_transformer."):
            key = "region_transformer." + key.removeprefix("bbox_transformer.")
        if key in normalized:
            raise ValueError(f"Duplicate state-dict key after prefix normalization: {key!r}")
        normalized[key] = value
    return normalized


def infer_checkpoint_model_variant(checkpoint: Mapping[str, Any]) -> ModelVariant | None:
    """Infer a variant only when checkpoint contents identify it unambiguously."""
    if checkpoint.get("model_variant"):
        return coerce_model_variant(checkpoint["model_variant"])
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    config = hyper_parameters.get("config") if isinstance(hyper_parameters, Mapping) else None
    if config is None:
        config = checkpoint.get("config")
    if config is not None:
        return resolve_model_variant(config)

    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    if isinstance(raw_state_dict, Mapping):
        for raw_key in raw_state_dict:
            key = raw_key.removeprefix("module.").removeprefix("matcher.")
            if key.startswith("loftr_bbox."):
                return ModelVariant.SGAD

    # The RoPE architecture is shared by SGAD++-BCE and SGAD++-Focal, so its
    # weights alone cannot identify the complete model variant.
    return None


def checkpoint_model_variant(
    checkpoint: Mapping[str, Any],
    *,
    fallback: str | ModelVariant | None = None,
) -> ModelVariant:
    inferred = infer_checkpoint_model_variant(checkpoint)
    if inferred is not None:
        return inferred
    return coerce_model_variant(fallback) if fallback is not None else ModelVariant.SGAD


def make_portable_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: nn.Module,
    model_variant: str | ModelVariant | None = None,
) -> dict[str, Any]:
    report = load_sgad_checkpoint(model, checkpoint)
    if report.loaded_keys != report.model_keys or report.mismatched_keys:
        missing = report.model_keys - report.loaded_keys
        raise ValueError(
            "Legacy checkpoint cannot be exported as a strict portable model: "
            f"missing={missing}, mismatched={len(report.mismatched_keys)}"
        )

    return {
        "format": PORTABLE_CHECKPOINT_FORMAT,
        "model_variant": report.model_variant.value,
        "state_dict": model.state_dict(),
    }


def load_sgad_checkpoint(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
) -> CheckpointLoadReport:
    expected_variant = getattr(model, "model_variant", None)
    variant = checkpoint_model_variant(checkpoint, fallback=expected_variant)
    if (
        expected_variant is not None
        and infer_checkpoint_model_variant(checkpoint) is not None
        and variant is not coerce_model_variant(expected_variant)
    ):
        raise ValueError(
            f"Checkpoint variant {variant.value!r} does not match model variant "
            f"{coerce_model_variant(expected_variant).value!r}"
        )
    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")
    state_dict = normalize_state_dict_keys(raw_state_dict)
    model_state = model.state_dict()

    if checkpoint.get("format") == PORTABLE_CHECKPOINT_FORMAT:
        model.load_state_dict(state_dict, strict=True)
        return CheckpointLoadReport(
            model_variant=variant,
            loaded_keys=len(state_dict),
            model_keys=len(model_state),
            skipped_keys=0,
            mismatched_keys=(),
            strict=True,
        )

    compatible = {}
    mismatched = []
    skipped = 0
    for key, value in state_dict.items():
        if key not in model_state:
            skipped += 1
            continue

        target_shape = model_state[key].shape
        fixed_value = value
        if value.shape != target_shape:
            if value.ndim == 3 and value.shape[1] == 1 and value.squeeze(1).shape == target_shape:
                fixed_value = value.squeeze(1)
            elif value.ndim == 2 and value.unsqueeze(1).shape == target_shape and target_shape[1] == 1:
                fixed_value = value.unsqueeze(1)

        if fixed_value.shape == target_shape:
            compatible[key] = fixed_value
        else:
            mismatched.append(key)

    if not compatible:
        raise ValueError("Legacy checkpoint does not contain any compatible SGAD model weights")

    model.load_state_dict(compatible, strict=False)
    return CheckpointLoadReport(
        model_variant=variant,
        loaded_keys=len(compatible),
        model_keys=len(model_state),
        skipped_keys=skipped,
        mismatched_keys=tuple(mismatched),
        strict=False,
    )
