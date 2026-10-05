import torch
from transformers import AutoModel
import torch.nn as nn
from loguru import logger

class Dinov2Extractor(nn.Module):
    def __init__(self, config, device=None, trainable_layers=0):
        super().__init__()
        # Keep the default construction on CPU so distributed launchers can
        # place each process on its local device without touching GPU 0 first.
        self.device = torch.device("cpu" if device is None else device)

        dinov2_path = config['path']
        self.model = AutoModel.from_pretrained(dinov2_path, local_files_only=True).to(self.device)
        self.device = next(self.model.parameters()).device

        self.set_trainable_layers(trainable_layers)
        
    def set_trainable_layers(self, num_layers=2):
        """Freeze all parameters, then unfreeze the final LayerNorm and trailing blocks.

        Args:
            num_layers: Number of encoder blocks to unfreeze from the end.
        """
        for param in self.model.parameters():
            param.requires_grad = False
            
        if hasattr(self.model, 'layernorm'):
            for param in self.model.layernorm.parameters():
                param.requires_grad = True
                logger.info("LayerNorm layer set to trainable")
        
        total_layers = len(self.model.encoder.layer)
        start_layer = max(0, total_layers - num_layers)
        
        for i in range(start_layer, total_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True
            logger.info(f"Layer {i} set to trainable")
            
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    def forward(self, x):
        """Extract DINOv2 token features.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Token features with shape ``[B, 1 + num_patches, D]``. Position 0
            is the CLS token and ``num_patches = (H / 14) * (W / 14)``.
        """
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state
