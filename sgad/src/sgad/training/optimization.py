from __future__ import annotations

from typing import Any


def compute_linear_warmup_lr(
    *,
    global_step: int,
    warmup_step: int,
    warmup_ratio: float,
    true_lr: float,
) -> float:
    base_lr = warmup_ratio * true_lr
    return base_lr + (global_step / warmup_step) * abs(true_lr - base_lr)


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def apply_lr_warmup(
    optimizer: Any,
    trainer_cfg: Any,
    *,
    global_step: int,
) -> float | None:
    warmup_step = trainer_cfg.WARMUP_STEP
    if global_step >= warmup_step:
        return None

    warmup_type = trainer_cfg.WARMUP_TYPE
    if warmup_type == "linear":
        lr = compute_linear_warmup_lr(
            global_step=global_step,
            warmup_step=warmup_step,
            warmup_ratio=trainer_cfg.WARMUP_RATIO,
            true_lr=trainer_cfg.TRUE_LR,
        )
        set_optimizer_lr(optimizer, lr)
        return lr

    if warmup_type == "constant":
        return None

    raise ValueError(f"Unknown lr warm-up strategy: {warmup_type}")


def run_optimizer_step(
    optimizer: Any,
    trainer_cfg: Any,
    *,
    global_step: int,
    optimizer_closure: Any = None,
) -> float | None:
    applied_lr = apply_lr_warmup(
        optimizer,
        trainer_cfg,
        global_step=global_step,
    )
    optimizer.step(closure=optimizer_closure)
    optimizer.zero_grad()
    return applied_lr
