"""DINOv3 patch-token extractor backed by a local Transformers checkpoint."""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel
from loguru import logger


class Dinov3Extractor(nn.Module):
    """Extract DINOv3 patch tokens with optional layer selection and normalization.

    Args:
        config: Configuration dict containing:
            - path: Local path to the pretrained model
            - layer_idx: List of layer indices to extract features from (for multi-layer extraction)
            - frozen: Whether to freeze the model weights
            - trainable_layers: Number of trailing encoder blocks to train along
              with the final LayerNorm. Zero leaves the full model trainable when
              ``frozen`` is false.
            - normalize_feats: Whether to L2-normalize output tokens.
        device: Device to load model on
    """

    def __init__(self, config, device=None):
        super().__init__()

        # Keep the default construction on CPU so distributed launchers can
        # place each process on its local device without touching GPU 0 first.
        self.device = torch.device("cpu" if device is None else device)

        # Resolve model and feature-extraction options.
        default_model_path = os.environ.get("A2PM_DINOV3_PATH", None)
        if default_model_path is None:
            a2pm_root = Path(os.environ.get("A2PM_SGAD_ROOT", Path(__file__).resolve().parents[5]))
            pretrained_root = Path(os.environ.get("A2PM_PRETRAINED_ROOT", a2pm_root / "pretrained"))
            default_model_path = str(pretrained_root / "dinov3-vitl16-pretrain-lvd1689m")
        self.model_path = config.get("path", default_model_path)
        self.layer_idx = config.get('layer_idx', None)  # None means use last layer only
        self.frozen = config.get('frozen', True)
        self.trainable_layers = config.get('trainable_layers', 0)
        self.normalize_feats = config.get('normalize_feats', False)

        # Load the model from the local checkpoint.
        logger.info(f"Loading DINOv3 model from local path: {self.model_path}")

        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True).to(self.device)

        # Cache the architecture metadata needed for token extraction.
        self.patch_size = self.model.config.patch_size
        self.hidden_size = self.model.config.hidden_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 4)  # Model-family fallback.

        # Apply the requested trainability policy.
        if self.frozen:
            self._freeze_model()
        elif self.trainable_layers > 0:
            self.set_trainable_layers(self.trainable_layers)

        logger.info(f"DINOv3 model loaded successfully:")
        logger.info(f"  - Path: {self.model_path}")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Num layers: {self.num_hidden_layers}")
        logger.info(f"  - Num register tokens: {self.num_register_tokens}")
        logger.info(f"  - Frozen: {self.frozen}")
        self._print_param_stats()

    def _freeze_model(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("DINOv3 model frozen")

    def _print_param_stats(self):
        """Print parameter statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    def set_trainable_layers(self, num_layers: int = 2):
        """Train the final LayerNorm and the requested trailing encoder blocks.

        Args:
            num_layers: Number of layers from the end to make trainable
        """
        # Freeze the full model before selecting trainable modules.
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the final LayerNorm.
        for param in self.model.norm.parameters():
            param.requires_grad = True
        logger.info("LayerNorm set to trainable")

        # Unfreeze the trailing encoder blocks.
        total_layers = len(self.model.layer)
        start_idx = max(0, total_layers - num_layers)

        for i in range(start_idx, total_layers):
            for param in self.model.layer[i].parameters():
                param.requires_grad = True
            logger.info(f"Layer {i} set to trainable")

        self._print_param_stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3 spatial features.

        Args:
            x: Preprocessed pixel values with shape ``[B, C, H, W]``.

        Returns:
            Patch tokens with shape ``[B, num_patches, feature_dim]``. The
            feature dimension is multiplied by the number of selected layers.
        """
        if self.layer_idx is not None:
            # Extract and concatenate the requested hidden states.
            outputs = self.model(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of [B, seq_len, hidden_size]

            selected_features = [hidden_states[idx] for idx in self.layer_idx]

            # Remove the CLS and register-token prefix from each hidden state.
            patch_features = [feat[:, 1 + self.num_register_tokens:, :] for feat in selected_features]
            out = torch.cat(patch_features, dim=-1)  # [B, num_patches, hidden_size * num_layers]
        else:
            # Use the final hidden state when no intermediate layers are requested.
            outputs = self.model(pixel_values=x)
            last_hidden_state = outputs.last_hidden_state  # [B, 1 + num_register + num_patches, hidden_size]

            # Remove the CLS and register-token prefix.
            out = last_hidden_state[:, 1 + self.num_register_tokens:, :]  # [B, num_patches, hidden_size]

        # Normalize each patch token independently when requested.
        if self.normalize_feats:
            out = F.normalize(out, dim=-1)

        return out
