import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """MLP encoding of pooled pairwise bbox geometry."""

    def __init__(self, embed_dim, dropout_rate=0.5):
        super().__init__()
        self.relative_encoding = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, patch_coords, mask=None):
        relative_geometry = self.compute_relative_geometry(patch_coords, mask=mask)
        return self.relative_encoding(relative_geometry)

    def compute_relative_geometry(self, patch_coords, mask=None):
        diff_x = patch_coords[:, :, None, 0] - patch_coords[:, None, :, 0]
        diff_y = patch_coords[:, :, None, 1] - patch_coords[:, None, :, 1]

        distances = torch.sqrt(diff_x**2 + diff_y**2)
        relative_angles = torch.atan2(diff_y, diff_x)
        relative_geometry = torch.stack(
            [distances, torch.sin(relative_angles), torch.cos(relative_angles)],
            dim=-1,
        )
        if mask is None:
            return relative_geometry.mean(dim=2)

        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        relative_geometry = relative_geometry * pair_mask.unsqueeze(-1)
        valid_counts = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
        return relative_geometry.sum(dim=2) / valid_counts
