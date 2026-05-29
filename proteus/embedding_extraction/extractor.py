"""
proteus/embedding_extraction/extractor.py

Extraction of internal SimpleFold representations.

Captures the model's trunk (and optionally ESM, atom-encoder, and pairwise)
embeddings at requested points along the flow-matching denoising trajectory.
PROTEUS uses the sequence-only point (zero coordinates) and the
structure-converged point (end of the sampler) to compute its plasticity score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Literal, Union

import torch
from torch import nn
from tqdm import tqdm
from einops import repeat

# SimpleFold dependency: resolved from the SimpleFold `utils` package on PYTHONPATH.
from utils.boltz_utils import center_random_augmentation


class ExtractionPoint(Enum):
    """Available extraction points in SimpleFold architecture."""

    # ESM-related
    ESM_RAW = "esm_raw"  # Raw ESM2 embeddings [B, M, 37, 2560]
    ESM_COMBINED = "esm_combined"  # After learnable softmax combination [B, M, 2560]
    ESM_PROJECTED = "esm_projected"  # After projection to hidden size [B, M, 1536]

    # Atom encoder
    ATOM_ENCODER_IN = "atom_encoder_in"  # Input to atom encoder [B, N, 1536]
    ATOM_ENCODER_OUT = "atom_encoder_out"  # Output of atom encoder [B, N, 512]

    # Token-level (residue)
    TOKEN_GROUPED = "token_grouped"  # After grouping atoms to tokens [B, M, 1536]
    TOKEN_WITH_ESM = "token_with_esm"  # After concatenating with ESM [B, M, 1536]

    # Trunk layers
    TRUNK_LAYER = "trunk_layer"  # Intermediate trunk layers [B, M, 1536]
    TRUNK_OUT = "trunk_out"  # Final trunk output [B, M, 1536]

    # Atom decoder
    ATOM_UNGROUPED = "atom_ungrouped"  # After ungrouping to atoms [B, N, 1536]
    ATOM_DECODER_IN = "atom_decoder_in"  # Input to atom decoder [B, N, 512]
    ATOM_DECODER_OUT = "atom_decoder_out"  # Output of atom decoder [B, N, 512]

    # pLDDT-related
    PLDDT_LATENT = "plddt_latent"  # Latent used for pLDDT prediction [B, M, 1536]
    PLDDT_LOGITS = "plddt_logits"  # pLDDT logits [B, M, 50]

    # Trajectory-based (from sampling)
    TRAJECTORY_COORDS = "trajectory_coords"  # Intermediate coordinates during sampling
    TRAJECTORY_VELOCITY = "trajectory_velocity"  # Velocity predictions during sampling


@dataclass
class ExtractionConfig:
    """Configuration for embedding extraction."""

    points: List[ExtractionPoint] = field(
        default_factory=lambda: [ExtractionPoint.TRUNK_OUT]
    )
    trunk_layers: Optional[List[int]] = (
        None  # Which trunk layers to extract (0-indexed)
    )
    trajectory_steps: Optional[List[int]] = (
        None  # Which timesteps to extract during sampling
    )
    include_attention_weights: bool = False


@dataclass
class ExtractedEmbeddings:
    """Container for extracted embeddings."""

    embeddings: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor]
    metadata: Dict[str, any] = field(default_factory=dict)

    def get(self, point: Union[ExtractionPoint, str]) -> Optional[torch.Tensor]:
        key = point.value if isinstance(point, ExtractionPoint) else point
        return self.embeddings.get(key)

    def get_mask(self, point: Union[ExtractionPoint, str]) -> Optional[torch.Tensor]:
        key = point.value if isinstance(point, ExtractionPoint) else point
        return self.masks.get(key)


class SimpleFoldEmbeddingExtractor(nn.Module):
    """
    Extracts embeddings from various points in SimpleFold for downstream tasks.

    This extractor can:
    1. Extract embeddings from different architectural points (ESM, atom encoder, trunk, etc.)
    2. Handle multiple conformations with various aggregation strategies
    3. Extract trajectory information during the sampling process

    Example usage:
        extractor = SimpleFoldEmbeddingExtractor(
            model=folding_model,
            config=ExtractionConfig(
                points=[ExtractionPoint.TRUNK_OUT, ExtractionPoint.ESM_COMBINED],
                trunk_layers=[12, 24, 35],  # Extract specific trunk layers
            )
        )

        embeddings = extractor.extract(batch, num_conformations=5)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ExtractionConfig] = None,
        plddt_latent_model: Optional[nn.Module] = None,
        plddt_out_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config or ExtractionConfig()
        self.plddt_latent_model = plddt_latent_model
        self.plddt_out_model = plddt_out_model

        # Storage for intermediate activations
        self._activations = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks based on extraction config."""
        self._clear_hooks()

        points = self.config.points

        # ESM hooks
        if (
            ExtractionPoint.ESM_COMBINED in points
            or ExtractionPoint.ESM_PROJECTED in points
        ):

            def esm_proj_hook(module, input, output):
                self._activations["esm_projected"] = output
                if len(input) > 0:
                    self._activations["esm_combined"] = input[0]

            hook = self.model.esm_s_proj.register_forward_hook(esm_proj_hook)
            self._hooks.append(hook)

        # Atom encoder hooks
        if ExtractionPoint.ATOM_ENCODER_IN in points:

            def atom_enc_in_hook(module, input, output):
                if len(input) > 0:
                    # Get the 'latents' argument
                    self._activations["atom_encoder_in"] = (
                        input[0]
                        if isinstance(input[0], torch.Tensor)
                        else input[0].get("latents", input[0])
                    )

            hook = self.model.atom_encoder_transformer.register_forward_hook(
                atom_enc_in_hook
            )
            self._hooks.append(hook)

        if ExtractionPoint.ATOM_ENCODER_OUT in points:

            def atom_enc_out_hook(module, input, output):
                self._activations["atom_encoder_out"] = output

            hook = self.model.atom_encoder_transformer.register_forward_hook(
                atom_enc_out_hook
            )
            self._hooks.append(hook)

        # Token grouping hooks
        if (
            ExtractionPoint.TOKEN_GROUPED in points
            or ExtractionPoint.TOKEN_WITH_ESM in points
        ):

            def esm_cat_hook(module, input, output):
                self._activations["token_with_esm"] = output
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    # Input is concatenated [token_grouped, esm_emb]
                    # We need to extract before concatenation
                    pass

            hook = self.model.esm_cat_proj.register_forward_hook(esm_cat_hook)
            self._hooks.append(hook)

        # Trunk layer hooks
        if ExtractionPoint.TRUNK_LAYER in points and self.config.trunk_layers:
            for layer_idx in self.config.trunk_layers:
                if layer_idx < len(self.model.trunk.blocks):

                    def make_trunk_hook(idx):
                        def trunk_layer_hook(module, input, output):
                            self._activations[f"trunk_layer_{idx}"] = output

                        return trunk_layer_hook

                    hook = self.model.trunk.blocks[layer_idx].register_forward_hook(
                        make_trunk_hook(layer_idx)
                    )
                    self._hooks.append(hook)

        # Trunk output hook
        if ExtractionPoint.TRUNK_OUT in points:

            def trunk_out_hook(module, input, output):
                self._activations["trunk_out"] = output

            hook = self.model.trunk.register_forward_hook(trunk_out_hook)
            self._hooks.append(hook)

        # Atom ungrouping/decoder hooks
        if ExtractionPoint.ATOM_UNGROUPED in points:

            def latent2atom_hook(module, input, output):
                self._activations["atom_decoder_in"] = output

            hook = self.model.latent2atom_proj.register_forward_hook(latent2atom_hook)
            self._hooks.append(hook)

        if ExtractionPoint.ATOM_DECODER_OUT in points:

            def atom_dec_out_hook(module, input, output):
                self._activations["atom_decoder_out"] = output

            hook = self.model.atom_decoder_transformer.register_forward_hook(
                atom_dec_out_hook
            )
            self._hooks.append(hook)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    def _compute_esm_combined(self, feats: Dict) -> torch.Tensor:
        """Manually compute combined ESM embedding."""
        esm_s = feats["esm_s"]  # [B, M, 37, 2560]
        weights = self.model.esm_s_combine.softmax(0)  # [37]
        combined = (weights.unsqueeze(0) @ esm_s).squeeze(2)  # [B, M, 2560]
        return combined

    @torch.no_grad()
    def extract_single_forward(
        self,
        batch: Dict[str, torch.Tensor],
        noised_pos: torch.Tensor,
        t: torch.Tensor,
    ) -> ExtractedEmbeddings:
        """
        Extract embeddings from a single forward pass.

        Args:
            batch: Feature dictionary
            noised_pos: Noised atomic coordinates [B, N, 3]
            t: Timestep tensor [B]

        Returns:
            ExtractedEmbeddings with requested extraction points
        """
        self._register_hooks()
        self._activations = {}

        embeddings = {}
        masks = {}

        # Get masks
        token_mask = batch.get("token_pad_mask", None)
        atom_mask = batch.get("atom_pad_mask", None)

        # Store raw ESM if requested
        if ExtractionPoint.ESM_RAW in self.config.points:
            embeddings["esm_raw"] = batch["esm_s"].clone()
            if token_mask is not None:
                masks["esm_raw"] = token_mask

        # Forward pass through model
        out = self.model(noised_pos, t, batch)

        # Collect activations from hooks
        if ExtractionPoint.ESM_COMBINED in self.config.points:
            if "esm_combined" in self._activations:
                embeddings["esm_combined"] = self._activations["esm_combined"]
            else:
                embeddings["esm_combined"] = self._compute_esm_combined(batch)
            if token_mask is not None:
                masks["esm_combined"] = token_mask

        if ExtractionPoint.ESM_PROJECTED in self.config.points:
            if "esm_projected" in self._activations:
                embeddings["esm_projected"] = self._activations["esm_projected"]
            if token_mask is not None:
                masks["esm_projected"] = token_mask

        if ExtractionPoint.ATOM_ENCODER_IN in self.config.points:
            if "atom_encoder_in" in self._activations:
                embeddings["atom_encoder_in"] = self._activations["atom_encoder_in"]
                if atom_mask is not None:
                    masks["atom_encoder_in"] = atom_mask

        if ExtractionPoint.ATOM_ENCODER_OUT in self.config.points:
            if "atom_encoder_out" in self._activations:
                embeddings["atom_encoder_out"] = self._activations["atom_encoder_out"]
                if atom_mask is not None:
                    masks["atom_encoder_out"] = atom_mask

        if ExtractionPoint.TOKEN_WITH_ESM in self.config.points:
            if "token_with_esm" in self._activations:
                embeddings["token_with_esm"] = self._activations["token_with_esm"]
                if token_mask is not None:
                    masks["token_with_esm"] = token_mask

        # Trunk layers
        if (
            ExtractionPoint.TRUNK_LAYER in self.config.points
            and self.config.trunk_layers
        ):
            for layer_idx in self.config.trunk_layers:
                key = f"trunk_layer_{layer_idx}"
                if key in self._activations:
                    embeddings[key] = self._activations[key]
                    if token_mask is not None:
                        masks[key] = token_mask

        if ExtractionPoint.TRUNK_OUT in self.config.points:
            # The model returns latent which is the trunk output
            embeddings["trunk_out"] = out["latent"]
            if token_mask is not None:
                masks["trunk_out"] = token_mask

        if ExtractionPoint.ATOM_DECODER_IN in self.config.points:
            if "atom_decoder_in" in self._activations:
                embeddings["atom_decoder_in"] = self._activations["atom_decoder_in"]
                if atom_mask is not None:
                    masks["atom_decoder_in"] = atom_mask

        if ExtractionPoint.ATOM_DECODER_OUT in self.config.points:
            if "atom_decoder_out" in self._activations:
                embeddings["atom_decoder_out"] = self._activations["atom_decoder_out"]
                if atom_mask is not None:
                    masks["atom_decoder_out"] = atom_mask

        # pLDDT-related extractions
        if (
            ExtractionPoint.PLDDT_LATENT in self.config.points
            or ExtractionPoint.PLDDT_LOGITS in self.config.points
        ):
            if self.plddt_latent_model is not None and self.plddt_out_model is not None:
                plddt_feat = self.plddt_latent_model(noised_pos, t, batch)
                plddt_out = self.plddt_out_model(plddt_feat["latent"], batch)

                if ExtractionPoint.PLDDT_LATENT in self.config.points:
                    embeddings["plddt_latent"] = plddt_feat["latent"]
                    if token_mask is not None:
                        masks["plddt_latent"] = token_mask

                if ExtractionPoint.PLDDT_LOGITS in self.config.points:
                    embeddings["plddt_logits"] = plddt_out["plddt_logits"]
                    if token_mask is not None:
                        masks["plddt_logits"] = token_mask

        self._clear_hooks()

        return ExtractedEmbeddings(
            embeddings=embeddings, masks=masks, metadata={"timestep": t.clone()}
        )

    @torch.no_grad()
    def extract_with_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        flow,
        num_timesteps: int = 500,
        t_start: float = 1e-4,
        tau: float = 0.3,
        log_timesteps: bool = True,
        w_cutoff: float = 0.99,
        extraction_steps: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, ExtractedEmbeddings]:
        """
        Extract embeddings during the sampling trajectory.

        Args:
            batch: Feature dictionary
            flow: Flow process (LinearPath)
            num_timesteps: Number of sampling steps
            t_start: Starting timestep
            tau: Temperature parameter
            log_timesteps: Use logarithmic timestep spacing
            w_cutoff: Diffusion coefficient cutoff
            extraction_steps: Which steps to extract (default: all)
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping step indices to ExtractedEmbeddings
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = batch["coords"].device

        # Initialize timesteps
        if log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            steps = t.clamp(min=t_start, max=1.0).to(device)
        else:
            steps = torch.linspace(t_start, 1.0, steps=num_timesteps + 1).to(device)

        extraction_steps = (
            extraction_steps
            or self.config.trajectory_steps
            or list(range(0, num_timesteps, num_timesteps // 10))
        )

        # Initialize noise
        noise = torch.randn_like(batch["coords"])
        y_sampled = noise

        trajectory_embeddings = {}
        trajectory_coords = []
        trajectory_velocities = []

        for i in tqdm(range(num_timesteps), desc="Sampling with extraction"):
            t_current = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t_current

            # Center coordinates
            y_sampled = center_random_augmentation(
                y_sampled,
                batch["atom_pad_mask"],
                augmentation=False,
                centering=True,
            )

            batched_t = repeat(t_current, " -> b", b=y_sampled.shape[0])

            # Extract embeddings at specified steps
            if i in extraction_steps:
                extracted = self.extract_single_forward(batch, y_sampled, batched_t)
                trajectory_embeddings[i] = extracted

                if ExtractionPoint.TRAJECTORY_COORDS in self.config.points:
                    trajectory_coords.append((i, y_sampled.clone()))

            # Forward pass
            out = self.model(noised_pos=y_sampled, t=batched_t, feats=batch)
            velocity = out["predict_velocity"]

            if (
                ExtractionPoint.TRAJECTORY_VELOCITY in self.config.points
                and i in extraction_steps
            ):
                trajectory_velocities.append((i, velocity.clone()))

            # Euler-Maruyama step
            score = flow.compute_score_from_velocity(velocity, y_sampled, t_current)

            def diffusion_coefficient(t, eps=0.01):
                w = (1.0 - t) / (t + eps)
                return 0.0 if t >= w_cutoff else w

            diff_coeff = diffusion_coefficient(t_current)
            drift = velocity + diff_coeff * score
            mean_y = y_sampled + drift * dt
            eps = torch.randn_like(y_sampled)
            y_sampled = mean_y + torch.sqrt(2.0 * dt * diff_coeff * tau) * eps

        # Add trajectory data to final embeddings
        if trajectory_coords:
            trajectory_embeddings["trajectory_coords"] = trajectory_coords
        if trajectory_velocities:
            trajectory_embeddings["trajectory_velocities"] = trajectory_velocities

        # Add final denoised coordinates
        trajectory_embeddings["final_coords"] = y_sampled

        return trajectory_embeddings

    @torch.no_grad()
    def extract_multi_conformation(
        self,
        batch: Dict[str, torch.Tensor],
        flow,
        num_conformations: int = 5,
        num_timesteps: int = 500,
        t_start: float = 1e-4,
        tau: float = 0.3,
        log_timesteps: bool = True,
        seeds: Optional[List[int]] = None,
        aggregate_method: Literal["none", "mean", "max", "concat"] = "none",
    ) -> Dict[str, Union[List[ExtractedEmbeddings], ExtractedEmbeddings]]:
        """
        Extract embeddings from multiple conformational samples.

        Args:
            batch: Feature dictionary
            flow: Flow process
            num_conformations: Number of conformational samples
            num_timesteps: Sampling steps per conformation
            t_start: Starting timestep
            tau: Temperature
            log_timesteps: Use log timesteps
            seeds: Random seeds for each conformation
            aggregate_method: How to aggregate across conformations

        Returns:
            Dictionary with embeddings for each conformation or aggregated
        """
        seeds = seeds or list(range(num_conformations))
        device = batch["coords"].device

        all_embeddings = []
        all_final_coords = []

        for conf_idx, seed in enumerate(seeds):
            torch.manual_seed(seed)

            # Initialize timesteps
            if log_timesteps:
                t = 1.0 - torch.logspace(-2, 0, num_timesteps + 1).flip(0)
                t = t - torch.min(t)
                t = t / torch.max(t)
                steps = t.clamp(min=t_start, max=1.0).to(device)
            else:
                steps = torch.linspace(t_start, 1.0, steps=num_timesteps + 1).to(device)

            # Sample trajectory
            noise = torch.randn_like(batch["coords"])
            y_sampled = noise

            for i in tqdm(
                range(num_timesteps),
                desc=f"Conformation {conf_idx + 1}/{num_conformations}",
            ):
                t_current = steps[i]
                t_next = steps[i + 1]
                dt = t_next - t_current

                y_sampled = center_random_augmentation(
                    y_sampled,
                    batch["atom_pad_mask"],
                    augmentation=False,
                    centering=True,
                )

                batched_t = repeat(t_current, " -> b", b=y_sampled.shape[0])
                out = self.model(noised_pos=y_sampled, t=batched_t, feats=batch)
                velocity = out["predict_velocity"]

                score = flow.compute_score_from_velocity(velocity, y_sampled, t_current)

                def diffusion_coefficient(t, eps=0.01):
                    w = (1.0 - t) / (t + eps)
                    return 0.0 if t >= 0.99 else w

                diff_coeff = diffusion_coefficient(t_current)
                drift = velocity + diff_coeff * score
                mean_y = y_sampled + drift * dt
                eps = torch.randn_like(y_sampled)
                y_sampled = mean_y + torch.sqrt(2.0 * dt * diff_coeff * tau) * eps

            # Extract final embeddings at t=1
            batched_t = torch.ones(y_sampled.shape[0], device=device)
            extracted = self.extract_single_forward(batch, y_sampled, batched_t)
            extracted.metadata["conformation_idx"] = conf_idx
            extracted.metadata["seed"] = seed

            all_embeddings.append(extracted)
            all_final_coords.append(y_sampled.clone())

        result = {
            "conformations": all_embeddings,
            "final_coords": torch.stack(all_final_coords, dim=0),  # [num_conf, B, N, 3]
        }

        # Aggregate if requested
        if aggregate_method != "none":
            result["aggregated"] = self._aggregate_conformations(
                all_embeddings, aggregate_method
            )

        return result

    def _aggregate_conformations(
        self, embeddings_list: List[ExtractedEmbeddings], method: str
    ) -> ExtractedEmbeddings:
        """Aggregate embeddings across conformations."""
        aggregated = {}
        masks = {}

        # Get all keys from first embedding
        keys = list(embeddings_list[0].embeddings.keys())

        for key in keys:
            tensors = [
                e.embeddings[key] for e in embeddings_list if key in e.embeddings
            ]
            if not tensors:
                continue

            stacked = torch.stack(tensors, dim=0)  # [num_conf, ...]

            if method == "mean":
                aggregated[key] = stacked.mean(dim=0)
            elif method == "max":
                aggregated[key] = stacked.max(dim=0).values
            elif method == "concat":
                # Concatenate along feature dimension
                aggregated[key] = stacked.transpose(0, 1).reshape(
                    stacked.shape[1], stacked.shape[2], -1
                )

            # Copy mask from first embedding
            if key in embeddings_list[0].masks:
                masks[key] = embeddings_list[0].masks[key]

        return ExtractedEmbeddings(
            embeddings=aggregated,
            masks=masks,
            metadata={
                "aggregation_method": method,
                "num_conformations": len(embeddings_list),
            },
        )

    @torch.no_grad()
    def extract_esm_only(self, batch: Dict[str, torch.Tensor]) -> ExtractedEmbeddings:
        """
        Extract only ESM embeddings without running the structure model.
        Useful for fast baseline comparisons.
        """
        embeddings = {}
        masks = {}

        token_mask = batch.get("token_pad_mask", None)

        # Raw ESM
        embeddings["esm_raw"] = batch["esm_s"].clone()

        # Combined ESM
        embeddings["esm_combined"] = self._compute_esm_combined(batch)

        # Projected ESM
        force_drop_ids = batch.get("force_drop_ids", None)
        embeddings["esm_projected"] = self.model.esm_s_proj(
            embeddings["esm_combined"], False, force_drop_ids
        )

        if token_mask is not None:
            masks["esm_raw"] = token_mask
            masks["esm_combined"] = token_mask
            masks["esm_projected"] = token_mask

        return ExtractedEmbeddings(
            embeddings=embeddings, masks=masks, metadata={"esm_only": True}
        )
