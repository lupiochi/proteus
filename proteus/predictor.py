"""
proteus/predictor.py

Abstract interface for extracting PROTEUS embeddings from a structure predictor.

PROTEUS was developed and validated using SimpleFold. This interface defines the
minimal contract that any embedding extractor must satisfy to plug into the scoring
pipeline. Whether the approach generalises to other flow-matching predictors has not
been tested and is left for future work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PredictorOutput:
    """Output from a single flow-matching forward pass.

    Attributes:
        seq_emb:     (L, D) trunk embedding extracted at flow t=1 (zero coordinate
                     input, sequence-only prior). Represents the protein sequence
                     in the predictor's learned representation.
        struct_embs: K arrays of shape (L, D), each the trunk embedding from one
                     independent sampler run at flow t=0 (converged structure).
                     Multiple runs use different random seeds.
        protein_id:  Optional identifier (e.g. FASTA header, PDB accession).
    """
    seq_emb: np.ndarray
    struct_embs: list[np.ndarray]
    protein_id: Optional[str] = None


class FlowMatchingPredictor(ABC):
    """Abstract base class for embedding extractors compatible with PROTEUS scoring.

    PROTEUS was validated on SimpleFold only. This base class exists to cleanly
    separate the scoring logic (proteus.score, proteus.stats) from the embedding
    extraction so that the former can be used independently of any predictor.

    The only obligation is to implement `embed()`. The `score()` convenience method
    is already implemented and calls `embed()` followed by the PROTEUS scoring
    functions from `proteus.score`.

    Example
    -------
    >>> from proteus.adapters.simplefold import SimpleFoldPredictor
    >>> predictor = SimpleFoldPredictor.from_pretrained("path/to/weights")
    >>> output = predictor.embed("MKTAYIAKQRQISFVKSHFSRQ...", n_conformations=10)
    >>> from proteus import compute_all_features
    >>> features = compute_all_features(output.seq_emb, output.struct_embs)
    """

    @abstractmethod
    def embed(
        self,
        sequence: str,
        n_conformations: int = 10,
        protein_id: Optional[str] = None,
    ) -> PredictorOutput:
        """Extract PROTEUS embeddings for a single protein sequence.

        Args:
            sequence:        Amino-acid sequence string (single-letter codes).
            n_conformations: Number of independent sampler runs (K). Recommended: 10.
            protein_id:      Optional label attached to the output.

        Returns:
            PredictorOutput with seq_emb (L, D) and struct_embs (list of K (L, D)).
        """
        ...

    def score(
        self,
        sequence: str,
        n_conformations: int = 10,
        protein_id: Optional[str] = None,
    ) -> dict[str, float]:
        """Convenience wrapper: embed then compute all PROTEUS features.

        Returns:
            Dict with keys: l2_delta_max, l2_delta_mean, l2_delta_p90,
            cos_dist_mean, cos_dist_p90, ensemble_spread, length.
        """
        from .score import compute_all_features
        output = self.embed(sequence, n_conformations=n_conformations, protein_id=protein_id)
        return compute_all_features(output.seq_emb, output.struct_embs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, **kwargs) -> "FlowMatchingPredictor":
        """Load a pretrained predictor from disk.

        Subclasses should override this to load model weights and return a ready
        instance. The base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_pretrained(). "
            "Load the model directly via the constructor."
        )
