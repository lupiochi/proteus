"""
proteus/embedding_extraction/pipeline.py

High-level pipeline that ties together the extractor, aggregators, and
projectors to produce ready-to-use embeddings for downstream tasks from a
SimpleFold model and input batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal

import torch
from torch import nn

from .extractor import (
    SimpleFoldEmbeddingExtractor,
    ExtractionConfig,
    ExtractionPoint,
)
from .aggregators import (
    MeanAggregator,
    MaxAggregator,
    AttentionAggregator,
    ConformationAggregator,
)
from .projectors import (
    EmbeddingProjector,
    ESMLayerCombiner,
)


@dataclass
class EmbeddingPipelineConfig:
    """Configuration for the embedding pipeline."""

    # Extraction points to use
    extraction_points: List[str] = field(default_factory=lambda: ["trunk_out"])

    # If using trunk layers, which ones (0-indexed)
    trunk_layers: Optional[List[int]] = None

    # Sequence aggregation method
    sequence_aggregation: Literal["mean", "max", "attention", "cls", "none"] = "mean"

    # Conformation handling
    num_conformations: int = 1
    conformation_aggregation: Literal["mean", "max", "attention", "gated", "none"] = (
        "mean"
    )

    # Output dimension (None = keep original)
    output_dim: Optional[int] = None

    # Projection method
    projection_method: Literal["linear", "mlp", "bottleneck", "residual"] = "mlp"

    # ESM layer combination (if using ESM embeddings)
    esm_combination: Literal["learned", "fixed", "task_specific", "none"] = "learned"

    # Sampling parameters
    num_timesteps: int = 500
    tau: float = 0.3

    # Seeds for reproducibility
    seeds: Optional[List[int]] = None


class SimpleFoldEmbeddingPipeline(nn.Module):
    """
    High-level pipeline for extracting SimpleFold embeddings.

    This pipeline handles:
    1. Extracting embeddings from specified points in SimpleFold
    2. Combining multiple extraction points
    3. Aggregating across sequence positions
    4. Aggregating across multiple conformations
    5. Projecting to desired output dimension

    Example usage:
        # For PPI prediction
        pipeline = SimpleFoldEmbeddingPipeline.for_ppi(
            model=folding_model,
            output_dim=512,
            num_conformations=5,
        )
        embeddings = pipeline(batch, flow)

        # For contact prediction (per-residue)
        pipeline = SimpleFoldEmbeddingPipeline.for_residue_level(
            model=folding_model,
            extraction_points=["trunk_out", "esm_combined"],
        )
        residue_embeddings = pipeline(batch, flow)
    """

    def __init__(
        self,
        model: nn.Module,
        config: EmbeddingPipelineConfig,
        plddt_latent_model: Optional[nn.Module] = None,
        plddt_out_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.plddt_latent_model = plddt_latent_model
        self.plddt_out_model = plddt_out_model

        # Build extraction config
        extraction_points = [
            ExtractionPoint(p) if isinstance(p, str) else p
            for p in config.extraction_points
        ]
        extraction_config = ExtractionConfig(
            points=extraction_points,
            trunk_layers=config.trunk_layers,
        )

        # Create extractor
        self.extractor = SimpleFoldEmbeddingExtractor(
            model=model,
            config=extraction_config,
            plddt_latent_model=plddt_latent_model,
            plddt_out_model=plddt_out_model,
        )

        # Determine hidden sizes for each extraction point
        self.hidden_sizes = self._get_hidden_sizes()

        # Build aggregators and projectors
        self._build_modules()

    def _get_hidden_sizes(self) -> Dict[str, int]:
        """Get hidden size for each extraction point."""
        # Based on SimpleFold architecture
        sizes = {
            "esm_raw": 2560,
            "esm_combined": 2560,
            "esm_projected": self.model.hidden_size,
            "atom_encoder_in": self.model.hidden_size,
            "atom_encoder_out": self.model.atom_hidden_size_enc,
            "token_grouped": self.model.hidden_size,
            "token_with_esm": self.model.hidden_size,
            "trunk_out": self.model.hidden_size,
            "atom_ungrouped": self.model.hidden_size,
            "atom_decoder_in": self.model.atom_hidden_size_dec,
            "atom_decoder_out": self.model.atom_hidden_size_dec,
            "plddt_latent": self.model.hidden_size,
            "plddt_logits": 50,
        }

        # Add trunk layers
        if self.config.trunk_layers:
            for layer_idx in self.config.trunk_layers:
                sizes[f"trunk_layer_{layer_idx}"] = self.model.hidden_size

        return sizes

    def _build_modules(self):
        """Build aggregators and projectors based on config."""
        config = self.config

        # Sequence aggregators (per extraction point)
        self.seq_aggregators = nn.ModuleDict()
        if config.sequence_aggregation != "none":
            for point in config.extraction_points:
                point_name = (
                    point.value if isinstance(point, ExtractionPoint) else point
                )
                hidden_size = self.hidden_sizes.get(point_name, self.model.hidden_size)

                if config.sequence_aggregation == "mean":
                    agg = MeanAggregator()
                elif config.sequence_aggregation == "max":
                    agg = MaxAggregator()
                elif config.sequence_aggregation == "attention":
                    agg = AttentionAggregator(hidden_size)
                else:
                    agg = MeanAggregator()  # Default

                self.seq_aggregators[point_name] = agg

        # Conformation aggregator
        if config.num_conformations > 1 and config.conformation_aggregation != "none":
            # Use size of primary extraction point
            primary_point = config.extraction_points[0]
            primary_name = (
                primary_point.value
                if isinstance(primary_point, ExtractionPoint)
                else primary_point
            )
            hidden_size = self.hidden_sizes.get(primary_name, self.model.hidden_size)

            self.conf_aggregator = ConformationAggregator(
                hidden_size=hidden_size,
                method=config.conformation_aggregation,
            )
        else:
            self.conf_aggregator = None

        # Point combiner (if multiple extraction points)
        if len(config.extraction_points) > 1:
            total_dim = sum(
                self.hidden_sizes.get(
                    p.value if isinstance(p, ExtractionPoint) else p,
                    self.model.hidden_size,
                )
                for p in config.extraction_points
            )
            target_dim = config.output_dim or self.model.hidden_size
            self.point_combiner = EmbeddingProjector(
                input_dim=total_dim,
                output_dim=target_dim,
                method=config.projection_method,
            )
        else:
            self.point_combiner = None

        # Output projector
        if config.output_dim is not None:
            primary_point = config.extraction_points[0]
            primary_name = (
                primary_point.value
                if isinstance(primary_point, ExtractionPoint)
                else primary_point
            )
            input_dim = self.hidden_sizes.get(primary_name, self.model.hidden_size)

            if self.point_combiner is not None:
                self.output_projector = None  # Already handled by combiner
            else:
                self.output_projector = EmbeddingProjector(
                    input_dim=input_dim,
                    output_dim=config.output_dim,
                    method=config.projection_method,
                )
        else:
            self.output_projector = None

        # ESM layer combiner
        if config.esm_combination != "none" and any(
            "esm" in (p.value if isinstance(p, ExtractionPoint) else p)
            for p in config.extraction_points
        ):
            self.esm_combiner = ESMLayerCombiner(
                esm_dim=2560,
                num_esm_layers=37,
                output_dim=config.output_dim,
                method=config.esm_combination,
            )
        else:
            self.esm_combiner = None

    @torch.no_grad()
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        flow,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract and process embeddings.

        Args:
            batch: Feature dictionary from ProteinDataProcessor
            flow: LinearPath flow process
            return_all: If True, return all intermediate results

        Returns:
            Final embedding tensor or dictionary of results
        """
        config = self.config

        if config.num_conformations > 1:
            # Multi-conformation extraction
            result = self.extractor.extract_multi_conformation(
                batch=batch,
                flow=flow,
                num_conformations=config.num_conformations,
                num_timesteps=config.num_timesteps,
                tau=config.tau,
                seeds=config.seeds,
                aggregate_method="none",  # We'll aggregate ourselves
            )
            all_embeddings = result["conformations"]
        else:
            # Single conformation
            device = batch["coords"].device
            noise = torch.randn_like(batch["coords"])

            # Run sampling
            from model.torch.sampler import EMSampler

            sampler = EMSampler(
                num_timesteps=config.num_timesteps,
                tau=config.tau,
            )
            out = sampler.sample(self.model, flow, noise, batch)

            # Extract at final timestep
            t = torch.ones(batch["coords"].shape[0], device=device)
            extracted = self.extractor.extract_single_forward(
                batch, out["denoised_coords"], t
            )
            all_embeddings = [extracted]

        # Process each conformation
        processed_embeddings = []
        for conf_emb in all_embeddings:
            point_embeddings = []

            for point in config.extraction_points:
                point_name = (
                    point.value if isinstance(point, ExtractionPoint) else point
                )
                emb = conf_emb.get(point_name)

                if emb is None:
                    continue

                mask = conf_emb.get_mask(point_name)

                # Sequence aggregation if configured
                if (
                    config.sequence_aggregation != "none"
                    and point_name in self.seq_aggregators
                ):
                    emb = self.seq_aggregators[point_name](emb, mask)

                point_embeddings.append(emb)

            # Combine extraction points
            if len(point_embeddings) > 1 and self.point_combiner is not None:
                combined = torch.cat(point_embeddings, dim=-1)
                processed = self.point_combiner(combined)
            elif len(point_embeddings) == 1:
                processed = point_embeddings[0]
                if self.output_projector is not None:
                    processed = self.output_projector(processed)
            else:
                raise ValueError("No valid embeddings extracted")

            processed_embeddings.append(processed)

        # Aggregate across conformations
        if config.num_conformations > 1 and self.conf_aggregator is not None:
            stacked = torch.stack(processed_embeddings, dim=0)
            final_embedding = self.conf_aggregator(stacked)
        else:
            final_embedding = processed_embeddings[0]

        if return_all:
            return {
                "embedding": final_embedding,
                "per_conformation": processed_embeddings,
                "raw_extractions": all_embeddings,
            }

        return final_embedding

    @classmethod
    def for_ppi(
        cls,
        model: nn.Module,
        output_dim: int = 512,
        num_conformations: int = 5,
        **kwargs,
    ) -> "SimpleFoldEmbeddingPipeline":
        """
        Create pipeline optimized for PPI prediction.

        Uses trunk output with attention pooling and multi-conformation averaging.
        """
        config = EmbeddingPipelineConfig(
            extraction_points=["trunk_out"],
            sequence_aggregation="attention",
            num_conformations=num_conformations,
            conformation_aggregation="mean",
            output_dim=output_dim,
            projection_method="mlp",
            **kwargs,
        )
        return cls(model=model, config=config)

    @classmethod
    def for_residue_level(
        cls,
        model: nn.Module,
        extraction_points: Optional[List[str]] = None,
        output_dim: Optional[int] = None,
        num_conformations: int = 1,
        **kwargs,
    ) -> "SimpleFoldEmbeddingPipeline":
        """
        Create pipeline for residue-level tasks (contact prediction, etc.).

        Keeps per-residue embeddings without sequence aggregation.
        """
        config = EmbeddingPipelineConfig(
            extraction_points=extraction_points or ["trunk_out", "esm_projected"],
            sequence_aggregation="none",
            num_conformations=num_conformations,
            conformation_aggregation="mean" if num_conformations > 1 else "none",
            output_dim=output_dim,
            projection_method="mlp",
            **kwargs,
        )
        return cls(model=model, config=config)

    @classmethod
    def for_structure_aware(
        cls,
        model: nn.Module,
        plddt_latent_model: nn.Module,
        plddt_out_model: nn.Module,
        output_dim: int = 512,
        num_conformations: int = 3,
        **kwargs,
    ) -> "SimpleFoldEmbeddingPipeline":
        """
        Create pipeline that incorporates structure confidence (pLDDT).

        Uses both trunk embeddings and pLDDT information.
        """
        config = EmbeddingPipelineConfig(
            extraction_points=["trunk_out", "plddt_latent"],
            sequence_aggregation="attention",
            num_conformations=num_conformations,
            conformation_aggregation="gated",  # Weight by confidence
            output_dim=output_dim,
            projection_method="mlp",
            **kwargs,
        )
        return cls(
            model=model,
            config=config,
            plddt_latent_model=plddt_latent_model,
            plddt_out_model=plddt_out_model,
        )

    @classmethod
    def for_esm_baseline(
        cls,
        model: nn.Module,
        output_dim: int = 512,
        esm_combination: str = "learned",
        **kwargs,
    ) -> "SimpleFoldEmbeddingPipeline":
        """
        Create pipeline using only ESM embeddings (no structure prediction).

        Useful as a baseline to compare structure-aware embeddings against.
        """
        config = EmbeddingPipelineConfig(
            extraction_points=["esm_raw"],
            sequence_aggregation="mean",
            num_conformations=1,  # No structure sampling needed
            output_dim=output_dim,
            esm_combination=esm_combination,
            projection_method="mlp",
            **kwargs,
        )
        return cls(model=model, config=config)

    @classmethod
    def for_multi_scale(
        cls,
        model: nn.Module,
        trunk_layers: List[int] = [6, 12, 18, 24, 30, 35],
        output_dim: int = 512,
        num_conformations: int = 1,
        **kwargs,
    ) -> "SimpleFoldEmbeddingPipeline":
        """
        Create pipeline extracting from multiple trunk layers.

        Captures hierarchical information at different processing depths.
        """
        extraction_points = ["trunk_out"] + [f"trunk_layer_{i}" for i in trunk_layers]

        config = EmbeddingPipelineConfig(
            extraction_points=extraction_points,
            trunk_layers=trunk_layers,
            sequence_aggregation="mean",
            num_conformations=num_conformations,
            conformation_aggregation="mean" if num_conformations > 1 else "none",
            output_dim=output_dim,
            projection_method="mlp",
            **kwargs,
        )
        return cls(model=model, config=config)
