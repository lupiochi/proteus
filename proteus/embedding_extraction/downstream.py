"""
proteus/embedding_extraction/downstream.py

Task-specific prediction heads (PPI, contact, function, localization, binding
site, stability, mutation effect) that consume SimpleFold embeddings, plus a
configurable DownstreamModel wrapper. Provided for reuse; not required by the
core PROTEUS plasticity score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import torch
from torch import nn


@dataclass
class DownstreamTaskConfig:
    """Configuration for downstream task heads."""

    hidden_size: int
    num_classes: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True


class PPIHead(nn.Module):
    """
    Protein-Protein Interaction prediction head.

    Takes embeddings from two proteins and predicts interaction probability.

    Supports multiple interaction modeling strategies:
    - concat: Concatenate embeddings
    - bilinear: Bilinear interaction
    - dot: Dot product (similarity)
    - combined: All of the above
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
        interaction_type: Literal["concat", "bilinear", "dot", "combined"] = "combined",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.interaction_type = interaction_type

        # Compute input dimension based on interaction type
        if interaction_type == "concat":
            input_dim = embedding_dim * 2
        elif interaction_type == "bilinear":
            input_dim = hidden_dim
            self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, hidden_dim)
        elif interaction_type == "dot":
            input_dim = 1
        elif interaction_type == "combined":
            input_dim = embedding_dim * 2 + hidden_dim + 1
            self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict interaction between two proteins.

        Args:
            emb_a: Embedding of protein A [B, D]
            emb_b: Embedding of protein B [B, D]

        Returns:
            Interaction logits [B, num_classes]
        """
        if self.interaction_type == "concat":
            features = torch.cat([emb_a, emb_b], dim=-1)

        elif self.interaction_type == "bilinear":
            features = self.bilinear(emb_a, emb_b)

        elif self.interaction_type == "dot":
            features = (emb_a * emb_b).sum(dim=-1, keepdim=True)

        elif self.interaction_type == "combined":
            concat = torch.cat([emb_a, emb_b], dim=-1)
            bilinear = self.bilinear(emb_a, emb_b)
            dot = (emb_a * emb_b).sum(dim=-1, keepdim=True)
            features = torch.cat([concat, bilinear, dot], dim=-1)

        return self.classifier(features)


class ContactPredictionHead(nn.Module):
    """
    Contact prediction head for predicting residue-residue contacts.

    Takes per-residue embeddings and predicts contact map.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Project embeddings
        self.proj = nn.Linear(embedding_dim, hidden_dim)

        # Outer product + processing
        self.pair_embed = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual blocks for pair processing
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_blocks)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_blocks)]
        )

        # Output projection
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict contact map.

        Args:
            embeddings: Per-residue embeddings [B, L, D]
            mask: Valid residue mask [B, L]

        Returns:
            Contact logits [B, L, L]
        """
        B, L, _ = embeddings.shape

        # Project
        x = self.proj(embeddings)  # [B, L, H]

        # Outer concatenation for pair features
        x_i = x.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, H]
        x_j = x.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, H]
        pair = torch.cat([x_i, x_j], dim=-1)  # [B, L, L, 2H]

        # Process pairs
        pair = self.pair_embed(pair)  # [B, L, L, H]

        # Residual blocks
        for block, norm in zip(self.blocks, self.norms):
            pair = pair + block(norm(pair))

        # Output
        logits = self.output(pair).squeeze(-1)  # [B, L, L]

        # Symmetrize
        logits = (logits + logits.transpose(-1, -2)) / 2

        # Apply mask if provided
        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            logits = logits.masked_fill(~mask_2d, float("-inf"))

        return logits


class FunctionPredictionHead(nn.Module):
    """
    Protein function prediction head (e.g., EC number, GO terms).

    Multi-label classification with optional hierarchical structure.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        hierarchical: bool = False,
        hierarchy_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hierarchical = hierarchical

        layers = []
        prev_dim = embedding_dim

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        if hierarchical and hierarchy_levels:
            # Separate heads for each hierarchy level
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dim, n_classes) for n_classes in hierarchy_levels]
            )
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict function labels.

        Args:
            embeddings: Protein embeddings [B, D]

        Returns:
            Function logits [B, num_classes] or list of [B, n] for hierarchical
        """
        x = self.encoder(embeddings)

        if self.hierarchical and hasattr(self, "heads"):
            return [head(x) for head in self.heads]

        return self.head(x)


class LocalizationHead(nn.Module):
    """
    Subcellular localization prediction head.

    Predicts where a protein localizes in the cell.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_locations: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        multi_location: bool = True,
    ):
        super().__init__()
        self.multi_location = multi_location

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_locations),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict localization.

        Args:
            embeddings: Protein embeddings [B, D]

        Returns:
            Localization logits [B, num_locations]
        """
        return self.classifier(embeddings)


class BindingSiteHead(nn.Module):
    """
    Binding site prediction head.

    Predicts which residues are involved in binding.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = embedding_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict binding sites.

        Args:
            embeddings: Per-residue embeddings [B, L, D]
            mask: Valid residue mask [B, L]

        Returns:
            Binding site logits [B, L]
        """
        logits = self.classifier(embeddings).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        return logits


class StabilityHead(nn.Module):
    """
    Protein stability prediction head.

    Predicts stability metrics (e.g., melting temperature, ddG).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_outputs: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict stability.

        Args:
            embeddings: Protein embeddings [B, D]

        Returns:
            Stability predictions [B, num_outputs]
        """
        return self.regressor(embeddings)


class MutationEffectHead(nn.Module):
    """
    Mutation effect prediction head.

    Predicts the effect of single-point mutations.
    Can use wild-type and mutant embeddings or delta embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        use_delta: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_delta = use_delta

        input_dim = embedding_dim if use_delta else embedding_dim * 2

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        wt_embedding: torch.Tensor,
        mut_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict mutation effect.

        Args:
            wt_embedding: Wild-type embedding [B, D]
            mut_embedding: Mutant embedding [B, D]

        Returns:
            Effect score [B, 1]
        """
        if self.use_delta:
            x = mut_embedding - wt_embedding
        else:
            x = torch.cat([wt_embedding, mut_embedding], dim=-1)

        return self.predictor(x)


class DownstreamModel(nn.Module):
    """
    Wrapper that combines SimpleFold embeddings with a task-specific head.

    This is a convenience class for training downstream models.
    """

    def __init__(
        self,
        embedding_pipeline,
        task_head: nn.Module,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding_pipeline = embedding_pipeline
        self.task_head = task_head
        self.freeze_embeddings = freeze_embeddings

        if freeze_embeddings:
            for param in self.embedding_pipeline.parameters():
                param.requires_grad = False

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        flow,
        **task_kwargs,
    ):
        """
        Extract embeddings and run task head.

        Args:
            batch: Feature dictionary
            flow: Flow process
            **task_kwargs: Additional arguments for task head

        Returns:
            Task head output
        """
        if self.freeze_embeddings:
            with torch.no_grad():
                embeddings = self.embedding_pipeline(batch, flow)
        else:
            embeddings = self.embedding_pipeline(batch, flow)

        return self.task_head(embeddings, **task_kwargs)
