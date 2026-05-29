"""
proteus/embedding_extraction/__init__.py

SimpleFold embedding-extraction utilities.

Tools for extracting per-residue and per-conformation embeddings from the
SimpleFold flow-matching structure predictor. PROTEUS uses these to build the
sequence-only (zero-coordinate) and structure-converged representations along
the denoising trajectory. Optional aggregation, projection, and downstream-task
heads are exposed for reusing the embeddings in other settings.
"""

from __future__ import annotations

from .extractor import (
    SimpleFoldEmbeddingExtractor,
    ExtractionConfig,
    ExtractionPoint,
    ExtractedEmbeddings,
)

from .aggregators import (
    EmbeddingAggregator,
    MeanAggregator,
    MaxAggregator,
    AttentionAggregator,
    WeightedMeanAggregator,
    CLSTokenAggregator,
    ConformationAggregator,
    HierarchicalAggregator,
)

from .projectors import (
    EmbeddingProjector,
    MultiScaleProjector,
    LayerCombiner,
    ESMLayerCombiner,
    StructureAwareProjector,
)

from .pipeline import (
    SimpleFoldEmbeddingPipeline,
    EmbeddingPipelineConfig,
)

from .downstream import (
    DownstreamTaskConfig,
    PPIHead,
    ContactPredictionHead,
    FunctionPredictionHead,
    LocalizationHead,
    BindingSiteHead,
    StabilityHead,
    MutationEffectHead,
    DownstreamModel,
)

__all__ = [
    # Extractor
    "SimpleFoldEmbeddingExtractor",
    "ExtractionConfig",
    "ExtractionPoint",
    "ExtractedEmbeddings",
    # Aggregators
    "EmbeddingAggregator",
    "MeanAggregator",
    "MaxAggregator",
    "AttentionAggregator",
    "WeightedMeanAggregator",
    "CLSTokenAggregator",
    "ConformationAggregator",
    "HierarchicalAggregator",
    # Projectors
    "EmbeddingProjector",
    "MultiScaleProjector",
    "LayerCombiner",
    "ESMLayerCombiner",
    "StructureAwareProjector",
    # Pipeline
    "SimpleFoldEmbeddingPipeline",
    "EmbeddingPipelineConfig",
    # Downstream
    "DownstreamTaskConfig",
    "PPIHead",
    "ContactPredictionHead",
    "FunctionPredictionHead",
    "LocalizationHead",
    "BindingSiteHead",
    "StabilityHead",
    "MutationEffectHead",
    "DownstreamModel",
]
