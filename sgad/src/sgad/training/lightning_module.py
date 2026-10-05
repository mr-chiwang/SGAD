
import json
from loguru import logger
from pathlib import Path

import torch
import pytorch_lightning as pl

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from sgad.models import build_sgad_model
from sgad.models.checkpoints import load_sgad_checkpoint
from sgad.losses.bbox_loss import Loss
from sgad.optimizers import build_optimizer, build_scheduler
from sgad.evaluation.visualization import make_bbox_matching_figures
from sgad.utils.comm import gather
from sgad.utils.misc import lower_config
from sgad.utils.profiler import PassThroughProfiler

from sgad.training.optimization import run_optimizer_step
from sgad.evaluation.matching import (
    aggregate_test_outputs,
    assign_bbox_matches,
    compute_all_matches_by_threshold,
    compute_sim_auc_by_overlap_threshold,
    compute_universal_metrics_for_batch,
    create_empty_universal_metrics,
    deduplicate_sample_outputs,
    evaluation_sample_key,
)
from sgad.evaluation.inference import run_test_inference
from sgad.training.inference import run_trainval_inference
from sgad.training.validation import (
    build_pending_validation_metrics,
    process_validation_epoch_outputs,
)
from sgad.evaluation.reporting import build_test_log_entry, log_test_results
from sgad.training.epoch_reporting import (
    log_training_step_metrics,
    process_train_epoch_outputs,
)



class SGADLightningModule(pl.LightningModule):
    def __init__(
        self,
        config,
        pretrained_ckpt=None,
        profiler=None,
        pred_score_threshold=0.2,
    ):
        super().__init__()
        # Capture serializable initialization arguments as Lightning hyperparameters; exclude the profiler.
        self.save_hyperparameters(ignore=["profiler"])
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher
        self.pred_score_threshold = pred_score_threshold
        self.matcher = build_sgad_model(config=_config['model'])

        self.train_bboxes0_format = "xywh"
        self.train_bboxes1_format = "xywh"

        self.val_bboxes0_format = "xywh"
        self.val_bboxes1_format = "xywh"
        self.test_bboxes0_format = "xywh"
        self.test_bboxes1_format = "xywh"

        self.loss = Loss(_config)

        if pretrained_ckpt:
            # PyTorch >=2.6 defaults torch.load(weights_only=True), which can fail
            # for older checkpoints that contain yacs config objects.
            try:
                ckpt = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(pretrained_ckpt, map_location='cpu')

            report = load_sgad_checkpoint(self.matcher, ckpt)
            logger.info(
                f"Loaded {report.model_variant.value} initialization weights from "
                f"'{pretrained_ckpt}': {report.loaded_keys}/{report.model_keys} keys "
                f"(strict={report.strict}, mismatched={len(report.mismatched_keys)})"
            )

        # Lightning 2.x epoch-end hooks consume these explicitly retained outputs.
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Lazily initialized rank-zero JSON log path.
        self._log_file_path = None
        # Validation metrics awaiting the next training-epoch log entry.
        self._pending_val_metrics = None

    def on_save_checkpoint(self, checkpoint):
        """Attach a stable SGAD-family identity to Lightning checkpoints."""
        checkpoint["model_variant"] = self.matcher.model_variant.value

    def _get_log_file_path(self):
        """Return the rank-zero JSON log path after the trainer is attached."""
        if self._log_file_path is None and self.trainer.global_rank == 0:
            if hasattr(self.trainer, 'logger') and self.trainer.logger is not None:
                log_dir = Path(self.trainer.logger.log_dir)
                self._log_file_path = log_dir / "train.log"
        return self._log_file_path

    def _write_log_entry(self, entry: dict):
        """Append one JSON Lines record on rank 0."""
        if self.trainer.global_rank != 0:
            return
        log_file = self._get_log_file_path()
        if log_file is None:
            return
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def configure_optimizers(self):
        # FIXME: Verify scheduler-state restoration with --resume_from_checkpoint_path.
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        run_optimizer_step(
            optimizer,
            self.config.TRAINER,
            global_step=self.trainer.global_step,
            optimizer_closure=optimizer_closure,
        )
    
    def _trainval_inference(self, batch, is_train=True):
        """Run training or validation inference for a prepared batch.

        Returns:
            ``False`` when any sample has an empty region list and the entire
            batch is skipped; otherwise ``True``.
        """
        return run_trainval_inference(
            batch,
            config=self.config,
            profiler=self.profiler,
            matcher=self.matcher,
            loss=self.loss,
            is_train=is_train,
        )

    def assign_matching(
        self,
        batch,
        pred_score_threshold: float = 0.5,
    ):
        """Assign mutual-nearest region pairs above the prediction threshold.

        Args:
            batch: Batch containing the region confidence matrix.
            pred_score_threshold: Strict score cutoff applied in both directions.
        """
        assign_bbox_matches(
            batch,
            pred_score_threshold=pred_score_threshold,
        )

    def _compute_universal_metrics(self, batch, pred_score_threshold=0.5):
        """Compute model-independent region-matching metrics for one batch.

        Args:
            batch: Batch containing ``overlap_matrix`` and region confidences.
            pred_score_threshold: Prediction-score cutoff used to derive fresh
                mutual-nearest match indices.

        Returns:
            Macro-averaged metrics for the batch.
        """
        return compute_universal_metrics_for_batch(
            batch,
            pred_score_threshold=pred_score_threshold,
        )

    def _compute_sim_auc(self, batch, overlap_thresholds=None):
        return compute_sim_auc_by_overlap_threshold(
            batch,
            overlap_thresholds=overlap_thresholds,
        )

    def training_step(self, batch, batch_idx):
        batch['bboxes0_format'] = self.train_bboxes0_format
        batch['bboxes1_format'] = self.train_bboxes1_format
        self._trainval_inference(batch, is_train=True)

        # On rank 0, log loss, sample-level MOM, and optional visualizations.
        log_training_step_metrics(
            batch,
            is_rank_zero=self.trainer.global_rank == 0,
            global_step=self.global_step,
            log_every_n_steps=self.trainer.log_every_n_steps,
            experiment=self.logger.experiment if self.trainer.global_rank == 0 else None,
            compute_universal_metrics=self._compute_universal_metrics,
            pred_score_threshold=self.pred_score_threshold,
            enable_plotting=self.config.TRAINER.ENABLE_PLOTTING,
            compute_sim_auc=self._compute_sim_auc,
        )

        output = {'loss': batch['loss']}
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        is_rank_zero = self.trainer.global_rank == 0
        result = process_train_epoch_outputs(
            outputs,
            current_epoch=self.current_epoch,
            is_rank_zero=is_rank_zero,
            experiment=self.logger.experiment if is_rank_zero else None,
            pending_val_metrics=self._pending_val_metrics,
        )
        if result is not None and is_rank_zero:
            self._write_log_entry(result["log_entry"])
            if result["consumed_pending_val_metrics"]:
                self._pending_val_metrics = None

        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        batch['bboxes0_format'] = self.val_bboxes0_format
        batch['bboxes1_format'] = self.val_bboxes1_format
        has_valid_bboxes = self._trainval_inference(batch, is_train=False)

        # Compute model-independent region-matching metrics.
        if has_valid_bboxes:
            universal_metrics = self._compute_universal_metrics(batch, pred_score_threshold=self.pred_score_threshold)
            bbox_sim_auc = self._compute_sim_auc(batch, overlap_thresholds=[0.2])[0.2]
        else:
            # NaN metrics exclude this sample from metric aggregation; loss remains.
            universal_metrics = self._create_empty_universal_metrics()
            bbox_sim_auc = float("nan")

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)

        figures = {'top1': [], 'top3': []}
        if (
            self.config.TRAINER.ENABLE_PLOTTING
            and batch_idx % val_plot_interval == 0
            and has_valid_bboxes
        ):
            figures = make_bbox_matching_figures(batch)

        output = {
            'sample_key': evaluation_sample_key(batch),
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
            'universal_metrics': universal_metrics,
            'sim_auc': bbox_sim_auc,
        }
        self.validation_step_outputs.append(output)
        return output

    def _create_empty_universal_metrics(self):
        """Return NaN metrics that are ignored during metric aggregation."""
        return create_empty_universal_metrics()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        summary = process_validation_epoch_outputs(
            outputs,
            current_epoch=self.trainer.current_epoch,
            ckpt_path=self.trainer.ckpt_path,
            sanity_checking=self.trainer.sanity_checking,
            is_rank_zero=self.trainer.global_rank == 0,
            experiment=self.logger.experiment if self.trainer.global_rank == 0 else None,
            close_figures=lambda: plt.close('all'),
        )
        self.validation_step_outputs.clear()

        # Aggregation and TensorBoard writes stay on rank 0, but checkpoint
        # callbacks on every rank need the same globally aggregated metrics.
        summary = self.trainer.strategy.broadcast(summary, src=0)
        if summary is None:
            return

        # Publish the broadcast metrics on every rank; HMQR is the checkpoint monitor.
        MOM_val = summary["MOM"]
        BMR_val = summary["BMR"]
        HMQR_val = summary["HMQR"]
        Coverage_val = summary["Coverage"]
        AUC_ORC_val = summary["AUC_ORC"]
        sim_auc_val = summary["sim_auc"]
        self.log('HMQR_avg_val(all)', torch.tensor(HMQR_val), sync_dist=False)
        self.log('MOM_avg_val(all)', torch.tensor(MOM_val), sync_dist=False)
        self.log('BMR_avg_val(all)', torch.tensor(BMR_val), sync_dist=False)
        self.log('Coverage_avg_val(all)', torch.tensor(Coverage_val), sync_dist=False)
        self.log('AUC_ORC_avg_val(all)', torch.tensor(AUC_ORC_val), sync_dist=False)
        self.log('sim_auc_avg_val(all)', torch.tensor(sim_auc_val), sync_dist=False)

        # Retain rank-zero validation metrics for the next epoch log entry.
        if self.trainer.global_rank == 0:
            self._pending_val_metrics = build_pending_validation_metrics(summary)

    def test_step(self, batch, batch_idx):
        batch['bboxes0_format'] = self.test_bboxes0_format
        batch['bboxes1_format'] = self.test_bboxes1_format
        sample_key = evaluation_sample_key(batch)

        try:
            result = run_test_inference(
                batch,
                config=self.config,
                profiler=self.profiler,
                matcher=self.matcher,
                compute_all_matches=self._compute_all_matches,
            )
            self.test_step_outputs.append({"sample_key": sample_key, "result": result})
            return result

        except Exception as e:
            logger.error(f"test_step error: {batch_idx}")
            logger.error(f"error: {e}")
            self.test_step_outputs.append({"sample_key": sample_key, "result": None})
            return None

    def _compute_all_matches(self, batch, pred_score_thresholds):
        """Compute region matches independently at each prediction threshold."""
        return compute_all_matches_by_threshold(
            batch,
            pred_score_thresholds,
        )

    def on_test_epoch_end(self):
        """Gather rank-local outputs, deduplicate, and macro-average metrics on rank 0."""
        rank_outputs = gather(self.test_step_outputs)
        gathered_outputs = (
            [output for outputs in rank_outputs for output in outputs]
            if self.trainer.global_rank == 0
            else []
        )
        self.test_step_outputs.clear()

        if self.trainer.global_rank != 0:
            return None, 0

        keyed_outputs = deduplicate_sample_outputs(gathered_outputs)
        outputs = [output["result"] for output in keyed_outputs]
        if len(outputs) == 0:
            return None, 0

        universal_final, sim_auc_final, avg_time_matcher, valid_outputs_count = aggregate_test_outputs(outputs)

        output_dir = getattr(self, 'test_output_dir', None)
        if output_dir is not None:
            # Save per-pair results, including undefined metrics.
            def scalar(value):
                return value.item()

            with (output_dir / 'pairs.jsonl').open('w') as stream:
                for output in keyed_outputs:
                    stream.write(json.dumps(output, default=scalar) + '\n')
            summary = {
                'total_outputs': len(outputs),
                'failed_outputs': len(outputs) - valid_outputs_count,
                'valid_outputs': valid_outputs_count,
                'universal_metrics': universal_final,
                'sim_auc': sim_auc_final,
                'avg_matcher_time': avg_time_matcher,
            }
            (output_dir / 'summary.json').write_text(json.dumps(summary, default=scalar, indent=2) + '\n')

        # Log the aggregated test results on rank 0.
        log_test_results(
            universal_final,
            sim_auc_final,
            avg_time_matcher,
            valid_outputs_count,
            log_info=logger.info,
        )

        self._write_log_entry(
            build_test_log_entry(
                universal_final,
                sim_auc_final,
                avg_time_matcher,
                valid_outputs_count,
            )
        )

        return universal_final, avg_time_matcher
