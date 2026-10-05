from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callbacks(ckpt_dir: str | Path) -> tuple[ModelCheckpoint, ModelCheckpoint]:
    """Build independent callbacks for the best and latest training states."""
    dirpath = str(ckpt_dir)
    best_checkpoint = ModelCheckpoint(
        monitor="HMQR_avg_val(all)",
        verbose=False,
        save_top_k=1,
        save_last=False,
        mode="max",
        dirpath=dirpath,
        filename="{epoch}-{HMQR_avg_val(all):.3f}",
    )
    latest_checkpoint = ModelCheckpoint(
        monitor=None,
        verbose=False,
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        dirpath=dirpath,
        filename="last",
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    return best_checkpoint, latest_checkpoint
