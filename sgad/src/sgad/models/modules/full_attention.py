# Copyright SenseTime. All Rights Reserved.
# Licensed under Apache-2.0; see THIRD_PARTY_LICENSES at the repository root.
# Adapted from LoFTR's FullAttention with SGAD masking and dropout settings.

import torch
from torch.nn import Module, Dropout


class FullAttention(Module):
    def __init__(self, use_dropout=True, attention_dropout=0.3):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Apply masked multi-head scaled dot-product attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute scaled logits and mask invalid query or key positions.
        QK = torch.einsum("nlhd,nshd->nlsh", queries / queries.size(3)**.5, keys)
        N, L, S, H = QK.shape
        if q_mask is not None:
            QK.masked_fill_(~(q_mask.bool().reshape(N, L, 1, 1)), -1e9)
        if kv_mask is not None:
            QK.masked_fill_(~(kv_mask.bool().reshape(N, 1, S, 1)), -1e9)

        # Normalize attention weights and aggregate values.
        softmax_temp = 1. #/ queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()
