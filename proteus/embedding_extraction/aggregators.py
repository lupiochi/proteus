"""
proteus/embedding_extraction/aggregators.py

Pooling modules that aggregate per-position embeddings [B, L, D] into
sequence-level representations [B, D] (mean, max, attention, CLS, weighted-mean,
hierarchical, and per-conformation aggregation). Used for downstream tasks that
need a single vector per protein rather than the full per-residue profile.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
from torch import nn


class EmbeddingAggregator(nn.Module, ABC):
    """
    Base class for embedding aggregators.

    Aggregators convert per-position embeddings [B, L, D] to sequence-level [B, D].
    """

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate embeddings.

        Args:
            embeddings: Per-position embeddings [B, L, D] or [B, N, D]
            mask: Valid position mask [B, L] or [B, N], 1 for valid, 0 for padding

        Returns:
            Aggregated embedding [B, D]
        """
        pass


class MeanAggregator(EmbeddingAggregator):
    """
    Mean pooling over sequence positions.
    Respects padding mask if provided.
    """

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return embeddings.mean(dim=1)

        # Expand mask for broadcasting [B, L] -> [B, L, 1]
        mask_expanded = mask.unsqueeze(-1).float()

        # Masked mean
        summed = (embeddings * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)

        return summed / count


class MaxAggregator(EmbeddingAggregator):
    """
    Max pooling over sequence positions.
    Respects padding mask if provided.
    """

    def __init__(self, fill_value: float = -1e9):
        super().__init__()
        self.fill_value = fill_value

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return embeddings.max(dim=1).values

        # Fill masked positions with very negative values
        mask_expanded = mask.unsqueeze(-1).float()
        masked_embeddings = (
            embeddings * mask_expanded + (1 - mask_expanded) * self.fill_value
        )

        return masked_embeddings.max(dim=1).values


class AttentionAggregator(EmbeddingAggregator):
    """
    Attention-based pooling with learnable query.

    Computes attention weights over positions and returns weighted sum.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = embeddings.shape[0]

        # Expand query for batch [1, 1, D] -> [B, 1, D]
        query = self.query.expand(B, -1, -1)

        # Create key padding mask (True = ignore position)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        # Attention: query attends to all positions
        output, _ = self.attention(
            query=query,
            key=embeddings,
            value=embeddings,
            key_padding_mask=key_padding_mask,
        )

        return output.squeeze(1)  # [B, D]


class WeightedMeanAggregator(EmbeddingAggregator):
    """
    Weighted mean pooling with learnable position weights.

    Learns position-specific importance weights that are applied
    before mean pooling. Useful for capturing position-dependent
    importance (e.g., N-terminal vs C-terminal regions).
    """

    def __init__(
        self,
        hidden_size: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature

        # Project embeddings to scalar importance
        self.importance_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute importance scores [B, L, 1]
        importance = self.importance_proj(embeddings)

        if mask is not None:
            # Mask out padding positions with very negative values
            mask_expanded = mask.unsqueeze(-1).float()
            importance = importance * mask_expanded + (1 - mask_expanded) * (-1e9)

        # Softmax over positions to get weights
        weights = torch.softmax(importance / self.temperature, dim=1)

        # Weighted sum
        return (embeddings * weights).sum(dim=1)


class CLSTokenAggregator(EmbeddingAggregator):
    """
    Use a learned CLS token prepended to the sequence.

    Similar to BERT-style pooling where a special token
    aggregates information through self-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Transformer layers for CLS interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = embeddings.shape[0]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings_with_cls = torch.cat([cls_tokens, embeddings], dim=1)

        # Create mask for CLS + sequence
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            src_key_padding_mask = ~full_mask.bool()
        else:
            src_key_padding_mask = None

        # Forward through transformer
        output = self.transformer(
            embeddings_with_cls,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Return CLS token representation
        return output[:, 0, :]


class ConformationAggregator(nn.Module):
    """
    Aggregates embeddings across multiple conformations.

    Supports various strategies for combining information
    from different structural conformations.
    """

    def __init__(
        self,
        hidden_size: int,
        method: Literal["mean", "max", "attention", "gated"] = "mean",
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.method = method

        if method == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True,
            )
            self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
            nn.init.normal_(self.query, std=0.02)

        elif method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        conformation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate across conformations.

        Args:
            embeddings: [num_conf, B, D] or [num_conf, B, L, D]
            conformation_mask: [num_conf] or [num_conf, B] indicating valid conformations

        Returns:
            Aggregated embeddings [B, D] or [B, L, D]
        """
        if self.method == "mean":
            if conformation_mask is not None:
                mask = conformation_mask.float()
                while mask.dim() < embeddings.dim():
                    mask = mask.unsqueeze(-1)
                return (embeddings * mask).sum(0) / mask.sum(0).clamp(min=1)
            return embeddings.mean(dim=0)

        elif self.method == "max":
            return embeddings.max(dim=0).values

        elif self.method == "attention":
            # Reshape for attention: [B, num_conf, D] or [B*L, num_conf, D]
            orig_shape = embeddings.shape
            if embeddings.dim() == 4:
                # [num_conf, B, L, D] -> [B*L, num_conf, D]
                num_conf, B, L, D = embeddings.shape
                embeddings = embeddings.permute(1, 2, 0, 3).reshape(B * L, num_conf, D)
            else:
                # [num_conf, B, D] -> [B, num_conf, D]
                embeddings = embeddings.permute(1, 0, 2)

            B_eff = embeddings.shape[0]
            query = self.query.expand(B_eff, -1, -1)

            output, _ = self.attention_pool(query, embeddings, embeddings)
            output = output.squeeze(1)

            if len(orig_shape) == 4:
                output = output.reshape(orig_shape[1], orig_shape[2], orig_shape[3])

            return output

        elif self.method == "gated":
            # Compute gates per conformation
            gates = self.gate(embeddings)  # [num_conf, B, *, 1]
            gates = torch.softmax(gates, dim=0)
            return (embeddings * gates).sum(dim=0)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class HierarchicalAggregator(nn.Module):
    """
    Two-stage aggregation: position -> sequence, then conformation -> final.

    Useful for getting a single embedding from multiple conformations
    of a protein sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        position_aggregator: EmbeddingAggregator,
        conformation_aggregator: ConformationAggregator,
    ):
        super().__init__()
        self.position_aggregator = position_aggregator
        self.conformation_aggregator = conformation_aggregator

    def forward(
        self,
        embeddings: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        conformation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate embeddings hierarchically.

        Args:
            embeddings: [num_conf, B, L, D]
            position_mask: [B, L] or [num_conf, B, L]
            conformation_mask: [num_conf] or [num_conf, B]

        Returns:
            Final embedding [B, D]
        """
        num_conf, B, L, D = embeddings.shape

        # First: aggregate over positions for each conformation
        seq_embeddings = []
        for i in range(num_conf):
            mask = position_mask
            if position_mask is not None and position_mask.dim() == 3:
                mask = position_mask[i]
            seq_emb = self.position_aggregator(embeddings[i], mask)
            seq_embeddings.append(seq_emb)

        # Stack: [num_conf, B, D]
        seq_embeddings = torch.stack(seq_embeddings, dim=0)

        # Second: aggregate over conformations
        return self.conformation_aggregator(seq_embeddings, conformation_mask)
