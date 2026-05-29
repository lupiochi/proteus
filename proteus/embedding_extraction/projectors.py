"""
proteus/embedding_extraction/projectors.py

Projection modules that map SimpleFold embeddings to alternative
dimensionalities (linear, MLP, bottleneck, multi-scale) and combine ESM layers.
Used to adapt the raw trunk embeddings for downstream task heads.
"""

from __future__ import annotations

from typing import Optional, Literal, List

import torch
from torch import nn


class EmbeddingProjector(nn.Module):
    """
    Projects embeddings to different dimensions.

    Supports multiple projection methods:
    - linear: Simple linear projection
    - mlp: Multi-layer perceptron with activation
    - bottleneck: Compress then expand
    - pca: PCA-like learned projection
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        method: Literal["linear", "mlp", "bottleneck", "residual"] = "linear",
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method

        act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }[activation]

        if method == "linear":
            self.proj = nn.Linear(input_dim, output_dim)

        elif method == "mlp":
            hidden_dims = hidden_dims or [input_dim * 2]
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.proj = nn.Sequential(*layers)

        elif method == "bottleneck":
            bottleneck_dim = min(input_dim, output_dim) // 2
            bottleneck_dim = max(bottleneck_dim, 64)

            self.proj = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim) if layer_norm else nn.Identity(),
                act_fn,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(bottleneck_dim, output_dim),
            )

        elif method == "residual":
            # Use residual connection if dimensions match
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim) if layer_norm else nn.Identity(),
                act_fn,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(output_dim, output_dim),
            )
            if input_dim == output_dim:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Linear(input_dim, output_dim, bias=False)

        else:
            raise ValueError(f"Unknown method: {method}")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "residual":
            return self.proj(x) + self.skip(x)
        return self.proj(x)


class MultiScaleProjector(nn.Module):
    """
    Creates embeddings at multiple scales/dimensions.

    Useful for tasks that benefit from different granularities
    or for comparing embedding sizes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        method: str = "mlp",
        shared_backbone: bool = True,
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims

        if shared_backbone:
            backbone_dim = backbone_dim or input_dim
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, backbone_dim),
                nn.LayerNorm(backbone_dim),
                nn.GELU(),
            )
            self.heads = nn.ModuleDict(
                {
                    str(dim): EmbeddingProjector(backbone_dim, dim, method=method)
                    for dim in output_dims
                }
            )
        else:
            self.backbone = None
            self.heads = nn.ModuleDict(
                {
                    str(dim): EmbeddingProjector(input_dim, dim, method=method)
                    for dim in output_dims
                }
            )

    def forward(
        self,
        x: torch.Tensor,
        output_dim: Optional[int] = None,
    ) -> dict:
        """
        Project to multiple dimensions.

        Args:
            x: Input embeddings [B, L, D] or [B, D]
            output_dim: If specified, only return this dimension

        Returns:
            Dictionary mapping dimension to projected embeddings
        """
        if self.backbone is not None:
            x = self.backbone(x)

        if output_dim is not None:
            return {output_dim: self.heads[str(output_dim)](x)}

        return {int(dim): head(x) for dim, head in self.heads.items()}


class LayerCombiner(nn.Module):
    """
    Combines embeddings from multiple layers/extraction points.

    Supports various combination strategies:
    - concat: Concatenate along feature dimension
    - weighted: Learnable weighted sum
    - attention: Cross-attention to combine
    - last_n: Average of last N layers
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        method: Literal["concat", "weighted", "attention", "last_n"] = "weighted",
        output_dim: Optional[int] = None,
        last_n: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.method = method
        self.last_n = last_n

        output_dim = output_dim or hidden_size

        if method == "concat":
            self.proj = nn.Linear(hidden_size * num_layers, output_dim)

        elif method == "weighted":
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
            self.proj = (
                nn.Linear(hidden_size, output_dim)
                if hidden_size != output_dim
                else nn.Identity()
            )

        elif method == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True,
            )
            self.proj = (
                nn.Linear(hidden_size, output_dim)
                if hidden_size != output_dim
                else nn.Identity()
            )
            nn.init.normal_(self.query, std=0.02)

        elif method == "last_n":
            self.proj = (
                nn.Linear(hidden_size, output_dim)
                if hidden_size != output_dim
                else nn.Identity()
            )

    def forward(
        self,
        layer_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine embeddings from multiple layers.

        Args:
            layer_embeddings: List of [B, L, D] tensors, one per layer

        Returns:
            Combined embedding [B, L, output_dim]
        """
        if self.method == "concat":
            combined = torch.cat(layer_embeddings, dim=-1)
            return self.proj(combined)

        elif self.method == "weighted":
            weights = torch.softmax(self.layer_weights, dim=0)
            stacked = torch.stack(layer_embeddings, dim=0)  # [num_layers, B, L, D]
            combined = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
            return self.proj(combined)

        elif self.method == "attention":
            B, L, D = layer_embeddings[0].shape
            # Stack layers: [B*L, num_layers, D]
            stacked = torch.stack(layer_embeddings, dim=2).reshape(B * L, -1, D)

            query = self.query.expand(B * L, -1, -1)
            combined, _ = self.attention(query, stacked, stacked)
            combined = combined.squeeze(1).reshape(B, L, D)
            return self.proj(combined)

        elif self.method == "last_n":
            # Average last N layers
            last_layers = layer_embeddings[-self.last_n :]
            combined = torch.stack(last_layers, dim=0).mean(dim=0)
            return self.proj(combined)


class ESMLayerCombiner(nn.Module):
    """
    Specifically for combining ESM2 layer representations.

    ESM2-3B has 37 layers (36 transformer + final embedding).
    Different layers capture different properties:
    - Early layers: Surface/structural
    - Middle layers: Functional
    - Late layers: Evolutionary
    """

    def __init__(
        self,
        esm_dim: int = 2560,
        num_esm_layers: int = 37,
        output_dim: Optional[int] = None,
        method: Literal["learned", "fixed", "task_specific"] = "learned",
        layer_groups: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.num_esm_layers = num_esm_layers
        self.method = method

        output_dim = output_dim or esm_dim

        if method == "learned":
            # Learnable weights for each layer
            self.layer_weights = nn.Parameter(torch.zeros(num_esm_layers))
            self.proj = EmbeddingProjector(esm_dim, output_dim, method="mlp")

        elif method == "fixed":
            # Pre-defined layer groups (from ProstT5 paper insights)
            self.layer_groups = layer_groups or [
                list(range(0, 12)),  # Early: structural
                list(range(12, 24)),  # Middle: functional
                list(range(24, 37)),  # Late: evolutionary
            ]
            self.group_weights = nn.Parameter(torch.ones(len(self.layer_groups)))
            self.proj = EmbeddingProjector(esm_dim, output_dim, method="mlp")

        elif method == "task_specific":
            # Multiple task-specific combinations
            self.structural_weights = nn.Parameter(torch.zeros(num_esm_layers))
            self.functional_weights = nn.Parameter(torch.zeros(num_esm_layers))
            self.evolutionary_weights = nn.Parameter(torch.zeros(num_esm_layers))

            # Initialize with prior knowledge
            with torch.no_grad():
                self.structural_weights[:12] = 1.0
                self.functional_weights[12:24] = 1.0
                self.evolutionary_weights[24:] = 1.0

            self.proj = EmbeddingProjector(esm_dim * 3, output_dim, method="mlp")

    def forward(
        self,
        esm_embeddings: torch.Tensor,
        task: Optional[Literal["structural", "functional", "evolutionary"]] = None,
    ) -> torch.Tensor:
        """
        Combine ESM layer embeddings.

        Args:
            esm_embeddings: [B, L, num_layers, D]
            task: Optional task hint for task_specific method

        Returns:
            Combined embedding [B, L, output_dim]
        """
        if self.method == "learned":
            weights = torch.softmax(self.layer_weights, dim=0)
            combined = (weights.unsqueeze(0).unsqueeze(0) @ esm_embeddings).squeeze(2)
            return self.proj(combined)

        elif self.method == "fixed":
            group_embeds = []
            for group in self.layer_groups:
                group_embed = esm_embeddings[:, :, group, :].mean(dim=2)
                group_embeds.append(group_embed)

            weights = torch.softmax(self.group_weights, dim=0)
            combined = sum(w * e for w, e in zip(weights, group_embeds))
            return self.proj(combined)

        elif self.method == "task_specific":
            s_weights = torch.softmax(self.structural_weights, dim=0)
            f_weights = torch.softmax(self.functional_weights, dim=0)
            e_weights = torch.softmax(self.evolutionary_weights, dim=0)

            structural = (s_weights.unsqueeze(0).unsqueeze(0) @ esm_embeddings).squeeze(
                2
            )
            functional = (f_weights.unsqueeze(0).unsqueeze(0) @ esm_embeddings).squeeze(
                2
            )
            evolutionary = (
                e_weights.unsqueeze(0).unsqueeze(0) @ esm_embeddings
            ).squeeze(2)

            combined = torch.cat([structural, functional, evolutionary], dim=-1)
            return self.proj(combined)


class StructureAwareProjector(nn.Module):
    """
    Projects embeddings while incorporating structural information.

    Uses predicted coordinates or pLDDT to weight/modulate embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        coord_dim: int = 3,
        use_plddt: bool = True,
        plddt_bins: int = 50,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_plddt = use_plddt

        # Coordinate encoder
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

        if use_plddt:
            # pLDDT modulation
            self.plddt_encoder = nn.Sequential(
                nn.Linear(plddt_bins, 64),
                nn.GELU(),
                nn.Linear(64, input_dim),
            )

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 2 if not use_plddt else input_dim * 3, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        plddt_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project with structural awareness.

        Args:
            embeddings: [B, L, D]
            coords: [B, N, 3] atom coordinates (will be aggregated to residue level)
            plddt_logits: [B, L, 50] optional pLDDT logits

        Returns:
            Projected embeddings [B, L, output_dim]
        """
        # Encode coordinates (assuming already at residue level or will aggregate)
        # For simplicity, we assume coords are residue-level here
        # In practice, you might need to aggregate from atom to residue
        coord_feat = self.coord_encoder(coords)

        features = [embeddings, coord_feat]

        if self.use_plddt and plddt_logits is not None:
            plddt_feat = self.plddt_encoder(plddt_logits)
            features.append(plddt_feat)

        combined = torch.cat(features, dim=-1)
        return self.proj(combined)
