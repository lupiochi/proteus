#!/usr/bin/env python
"""
scripts/extract_embeddings.py

Extract per-residue SimpleFold trunk embeddings used by PROTEUS.

This is the production extraction script that produced every embedding in
Piochi, Karami & Khakzad (2026). It runs the SimpleFold flow-matching sampler
and captures the trunk latent inline at the requested flow timesteps.

Time convention (matches the manuscript):
    t = 0.0  -> sequence-only (single forward pass with zero coordinates)
    t = 1.0  -> structure-converged (latent captured at end of sampler trajectory)

Output (one .npz per protein):
    t0.0          -> [L, D] sequence-only embedding
    t1.0_conf{k}  -> [L, D] structural conformation k (k = 0..K-1)
                     (or t1.0_mean / t1.0_var if --aggregate_conformations)

Usage:
    python scripts/extract_embeddings.py \
        --fasta_path proteins.fasta \
        --output_dir simplefold_embeddings/ \
        --simplefold_model simplefold_360M \
        --n_conformations 10 \
        --num_steps 25 \
        --tau 0.3

Requires SimpleFold (Apple Inc.) installed and on PYTHONPATH:
    git clone https://github.com/apple/ml-simplefold
    cd ml-simplefold && pip install -e .

SimpleFold provides the top-level packages `inference`, `model`, `utils`,
`embedding`, and `processor` used below.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from inference import initialize_folding_model
from model.flow import LinearPath
from model.torch.sampler import EMSampler
from utils.esm_utils import esm_registry, _af2_to_esm
from utils import residue_constants

from embedding.extractor import (
    SimpleFoldEmbeddingExtractor,
    ExtractionConfig,
    ExtractionPoint,
)


# ---------------------------------------------------------------------------
# FASTA dataset
# ---------------------------------------------------------------------------

class FastaDataset(Dataset):
    """Streaming FASTA loader sorted by length for efficient batching."""

    def __init__(self, fasta_path: Path, max_length: int = 1024):
        self.sequences: list[str] = []
        self.ids: list[str] = []

        current_id = None
        current_seq: list[str] = []
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None and len("".join(current_seq)) <= max_length:
                        self.ids.append(current_id)
                        self.sequences.append("".join(current_seq))
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id is not None and len("".join(current_seq)) <= max_length:
                self.ids.append(current_id)
                self.sequences.append("".join(current_seq))

        order = sorted(range(len(self.sequences)), key=lambda i: len(self.sequences[i]))
        self.sequences = [self.sequences[i] for i in order]
        self.ids = [self.ids[i] for i in order]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx]


def collate_fn(batch):
    ids, sequences = zip(*batch)
    return list(ids), list(sequences)


# ---------------------------------------------------------------------------
# ESM tokenisation + SimpleFold batch construction
# ---------------------------------------------------------------------------

def encode_sequences_fast(
    sequences: List[str],
    esm_dict,
    af2_to_esm: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """Tokenise a batch of sequences for ESM-2."""
    unk_idx = residue_constants.restype_order_with_x.get("X", 20)

    batch_size = len(sequences)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)

    aatype = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            aatype[i, j] = residue_constants.restype_order_with_x.get(aa, unk_idx)
        mask[i, : len(seq)] = True

    aatype = aatype.to(device)
    mask = mask.to(device)
    af2_to_esm = af2_to_esm.to(device)

    aatype_shifted = (aatype + 1).masked_fill(~mask, 0)
    esmaa = af2_to_esm[aatype_shifted]

    bos_idx = esm_dict.cls_idx
    eos_idx = esm_dict.eos_idx
    pad_idx = esm_dict.padding_idx

    bos = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    eos = torch.full((batch_size, 1), pad_idx, dtype=torch.long, device=device)

    esmaa_with_special = torch.cat([bos, esmaa, eos], dim=1)
    for i, length in enumerate(lengths):
        esmaa_with_special[i, length + 1] = eos_idx

    return esmaa_with_special, lengths


# ---------------------------------------------------------------------------
# ESM-2 3B loading (HuggingFace backend)
# ---------------------------------------------------------------------------

ESM_HF_3B = "facebook/esm2_t36_3B_UR50D"   # HF mirror of fair-esm esm2_t36_3B_UR50D


class _HFAlphabetShim:
    """Fair-esm-alphabet-compatible view over a HuggingFace EsmTokenizer.

    Exposes only the attributes the tokenisation helpers need (cls_idx, eos_idx,
    padding_idx, get_idx) so encode_sequences_fast() and _af2_to_esm() work
    unchanged with the HuggingFace backend. The ESM-2 vocabulary is identical
    between the two implementations (verified by token-id equality), so the
    resulting token tensors match the facebookresearch/esm path exactly.
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.cls_idx = tokenizer.cls_token_id
        self.eos_idx = tokenizer.eos_token_id
        self.padding_idx = tokenizer.pad_token_id

    def get_idx(self, token: str) -> int:
        return self._tok.convert_tokens_to_ids(token)


def load_esm_hf(model_name: str, device: torch.device):
    """Load ESM-2 3B via HuggingFace transformers.

    Uses fp16 on CUDA together with low_cpu_mem_usage to keep peak system RAM far
    below the ~20 GB fp32 facebookresearch/esm load (which OOMs free Colab),
    while reproducing the same per-layer representations. Returns
    (model, alphabet_shim, af2_to_esm) matching the fair-esm path.
    """
    from transformers import AutoTokenizer, EsmModel

    use_fp16 = torch.device(device).type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        low_cpu_mem_usage=True,
        add_pooling_layer=False,
    )
    model = model.to(device).eval()

    esm_dict = _HFAlphabetShim(tokenizer)
    af2_to_esm = _af2_to_esm(esm_dict)
    return model, esm_dict, af2_to_esm


@torch.no_grad()
def extract_esm_combined(
    esmaa: torch.Tensor,
    lengths: List[int],
    esm_model,
    esm_s_combine: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ESM-2 and combine layer representations using the learned softmax weights."""
    device = esmaa.device
    batch_size = len(lengths)
    max_len = max(lengths)

    if hasattr(esm_model, "config"):  # HuggingFace transformers EsmModel
        attn = (esmaa != esm_model.config.pad_token_id).long()
        out = esm_model(input_ids=esmaa, attention_mask=attn,
                        output_hidden_states=True)
        # hidden_states[k] == fair-esm representations[k] for all k (incl. last,
        # both post emb_layer_norm_after); verified numerically on esm2_t6_8M.
        esm_s = torch.stack(out.hidden_states, dim=2)
    else:  # facebookresearch/esm
        res = esm_model(
            esmaa,
            repr_layers=range(esm_model.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
    esm_s = esm_s[:, 1:-1]
    esm_s = esm_s[:, :max_len]

    weights = F.softmax(esm_s_combine.float(), dim=0)
    combined = (esm_s.float() * weights.view(1, 1, -1, 1)).sum(dim=2)

    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return combined, mask


def create_simplefold_batch(
    esm_combined: torch.Tensor,
    sequences: List[str],
    mask: torch.Tensor,
    device: torch.device,
    scale: float = 16.0,
    ref_scale: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Build the input dict expected by SimpleFold from ESM features."""
    batch_size, max_seq_len, _ = esm_combined.shape

    mol_type = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    res_type = torch.zeros(batch_size, max_seq_len, 33, device=device)
    residue_index = (
        torch.arange(max_seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    entity_id = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    asym_id = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    sym_id = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    pocket_feature = torch.zeros(batch_size, max_seq_len, 4, device=device)
    ref_pos = torch.zeros(batch_size, max_seq_len, 3, device=device)
    ref_element = torch.zeros(batch_size, max_seq_len, 128, device=device)
    ref_atom_name_chars = torch.zeros(batch_size, max_seq_len, 4, 64, device=device)
    ref_charge = torch.zeros(batch_size, max_seq_len, device=device)
    ref_space_uid = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    coords = torch.zeros(batch_size, max_seq_len, 3, device=device)
    atom_to_token = torch.zeros(batch_size, max_seq_len, max_seq_len, device=device)
    atom_to_token_idx = (
        torch.arange(max_seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    esm_s = esm_combined.unsqueeze(2).expand(-1, -1, 37, -1).contiguous()

    unk_idx = residue_constants.restype_order_with_x.get("X", 20)
    lengths: list[int] = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        lengths.append(seq_len)
        for j, aa in enumerate(seq):
            idx = residue_constants.restype_order_with_x.get(aa, unk_idx)
            res_type[i, j, min(idx, 32)] = 1.0
        ref_element[i, :seq_len, 6] = 1.0
        atom_to_token[i, :seq_len, :seq_len] = torch.eye(seq_len, device=device)

    return {
        "mol_type": mol_type,
        "res_type": res_type,
        "residue_index": residue_index,
        "entity_id": entity_id,
        "asym_id": asym_id,
        "sym_id": sym_id,
        "pocket_feature": pocket_feature,
        "token_pad_mask": mask,
        "ref_pos": ref_pos / ref_scale,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_charge": ref_charge,
        "atom_to_token": atom_to_token,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_pad_mask": mask.clone(),
        "ref_space_uid": ref_space_uid,
        "coords": coords / scale,
        "esm_s": esm_s,
        "max_num_tokens": torch.tensor(lengths, dtype=torch.long, device=device),
    }


def expand_batch_for_conformations(batch: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    """Replicate every batch tensor n times along dim 0 (one copy per conformation)."""
    return {
        k: v.repeat(n, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_perresidue_at_timesteps(
    simplefold_model,
    batch: Dict[str, torch.Tensor],
    timesteps: List[float],
    extractor: SimpleFoldEmbeddingExtractor,
    device: torch.device,
    flow: LinearPath,
    sampler: EMSampler,
    n_conformations: int = 10,
    save_all_conformations: bool = True,
    conf_batch_size: int = 0,
) -> Dict[str, torch.Tensor]:
    """Extract per-residue trunk embeddings at one or more flow timesteps.

    Time convention (paper-aligned and identical to SimpleFold's internal flow):
        t = 0.0  -> sequence-only  (single forward, zero coordinates)
        t > 0.0  -> structure-conditioned, captured at flow_t = t inside sampler
        t = 1.0  -> fully structure-converged (clean data side of trajectory)

    Args:
        simplefold_model:       SimpleFold model instance.
        batch:                  Input batch (B sequences).
        timesteps:              User-facing timesteps to extract.
        extractor:              SimpleFoldEmbeddingExtractor for the t=0 forward pass.
        device:                 Torch device.
        flow:                   SimpleFold LinearPath flow schedule.
        sampler:                EMSampler instance (Euler-Maruyama).
        n_conformations:        Number of independent sampler runs for t > 0.
        save_all_conformations: If True save each conformation; else save mean/var.
        conf_batch_size:        Conformations per parallel sampler call (0 = all at once).

    Returns:
        Dict of [B, L, D] tensors keyed by:
          't0.0'                if t=0.0 in timesteps
          't{t}_conf{k}'        for each structural timestep, each conformation
                                (or 't{t}_mean' / 't{t}_var' if not save_all_conformations).
    """
    results: Dict[str, torch.Tensor] = {}
    batch_size = batch["coords"].shape[0]
    T_EPS = 1e-4

    # --- Sequence-only regime: t = 0 ------------------------------------------------
    if 0.0 in timesteps:
        coords = torch.zeros_like(batch["coords"])
        t = torch.full((batch_size,), T_EPS, device=device)
        extracted = extractor.extract_single_forward(batch, coords, t)
        emb = extracted.get("trunk_out")
        if emb is not None:
            results["t0.0"] = emb

    # --- Structure-conditioned regimes: t > 0 ---------------------------------------
    struct_timesteps = sorted([tv for tv in timesteps if tv > 0.0])
    if not struct_timesteps:
        return results

    # SimpleFold's flow schedule already runs t=0 (noise) -> t=1 (data),
    # i.e. the same direction as the manuscript. Direct mapping, no inversion.
    capture_flow_ts = list(struct_timesteps)
    confs_by_tval: Dict[float, list] = {tv: [] for tv in struct_timesteps}

    chunk = conf_batch_size if conf_batch_size > 0 else n_conformations
    for chunk_start in range(0, n_conformations, chunk):
        chunk_end = min(chunk_start + chunk, n_conformations)
        chunk_k = chunk_end - chunk_start

        noises = []
        for conf_idx in range(chunk_start, chunk_end):
            torch.manual_seed(conf_idx * 1000)
            noises.append(torch.randn_like(batch["coords"]))
        noise_batched = torch.cat(noises, dim=0)
        batch_expanded = expand_batch_for_conformations(batch, chunk_k)

        out = sampler.sample(
            simplefold_model,
            flow,
            noise_batched,
            batch_expanded,
            capture_at_flow_t=capture_flow_ts,
        )

        denoised_kb = out["denoised_coords"].reshape(
            chunk_k, batch_size, *out["denoised_coords"].shape[1:]
        )
        captured = out["captured_latents"]

        for t_val in struct_timesteps:
            t_key = f"t{t_val}"
            latent_kb = captured.get(t_val)
            if latent_kb is None:
                continue
            latent_4d = latent_kb.reshape(chunk_k, batch_size, *latent_kb.shape[1:])
            for i, conf_idx in enumerate(range(chunk_start, chunk_end)):
                if torch.isnan(denoised_kb[i]).any():
                    continue
                emb = latent_4d[i]
                if torch.isnan(emb).any():
                    continue
                if save_all_conformations:
                    results[f"{t_key}_conf{conf_idx}"] = emb
                else:
                    confs_by_tval[t_val].append(emb)

    if not save_all_conformations:
        for t_val, confs in confs_by_tval.items():
            if not confs:
                continue
            t_key = f"t{t_val}"
            stacked = torch.stack(confs, dim=0)
            results[f"{t_key}_mean"] = stacked.mean(dim=0)
            if len(confs) > 1:
                results[f"{t_key}_var"] = stacked.var(dim=0)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-residue SimpleFold embeddings for PROTEUS scoring."
    )
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--simplefold_model", type=str, default="simplefold_360M",
        choices=[
            "simplefold_100M", "simplefold_360M", "simplefold_700M",
            "simplefold_1.1B", "simplefold_1.6B", "simplefold_3B",
        ],
    )
    parser.add_argument("--ckpt_dir", type=str, default="~/.cache/simplefold")
    parser.add_argument(
        "--timesteps", type=float, nargs="+", default=[0.0, 1.0],
        help="Flow timesteps to extract (paper convention: 0.0=sequence, 1.0=structure).",
    )
    parser.add_argument(
        "--n_conformations", type=int, default=10,
        help="Number of independent sampler runs per protein (paper default: 10).",
    )
    parser.add_argument(
        "--aggregate_conformations", action="store_true", default=False,
        help="Save mean/var across conformations instead of all individual conformations.",
    )
    parser.add_argument("--num_steps", type=int, default=25,
                        help="Euler-Maruyama sampler steps (paper default: 25).")
    parser.add_argument("--tau", type=float, default=0.3,
                        help="Sampling temperature tau (paper default: 0.3).")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--esm_backend", type=str, default="hf", choices=["hf", "fair"],
        help="ESM-2 3B loader: 'hf' (transformers, fp16 on GPU, low CPU RAM) or "
             "'fair' (facebookresearch/esm, fp32). Both produce matching embeddings.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet_sampler", action="store_true", default=False)
    parser.add_argument(
        "--conf_batch_size", type=int, default=0,
        help="Conformations per parallel sampler call (0 = all at once). Reduce if OOM.",
    )

    args = parser.parse_args()
    args.ckpt_dir = os.path.expanduser(args.ckpt_dir)
    args.backend = "torch"
    args.plddt = False
    args.nsample_per_protein = 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("=" * 60)
    print("PROTEUS embedding extraction (SimpleFold)")
    print("=" * 60)
    print(f"Device:                {device}")
    print(f"Model:                 {args.simplefold_model}")
    print(f"Timesteps (paper):     {args.timesteps}")
    print(f"Conformations / t>0:   {args.n_conformations}")
    print(f"Sampler steps:         {args.num_steps}")
    print(f"Sampling temperature:  {args.tau}")
    print("=" * 60)

    # Metadata
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and args.skip_existing:
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "simplefold_model": args.simplefold_model,
            "timesteps": args.timesteps,
            "n_conformations": args.n_conformations,
            "aggregate_conformations": args.aggregate_conformations,
            "embedding_dim": None,
            "proteins": {},
        }
    existing_ids = set(metadata.get("proteins", {}).keys())

    # Dataset
    print(f"\nLoading FASTA: {args.fasta_path}")
    dataset = FastaDataset(Path(args.fasta_path), max_length=args.max_length)
    print(f"Total sequences:    {len(dataset)}")

    if args.skip_existing and existing_ids:
        keep = [i for i in range(len(dataset)) if dataset.ids[i] not in existing_ids]
        dataset.ids = [dataset.ids[i] for i in keep]
        dataset.sequences = [dataset.sequences[i] for i in keep]
        print(f"Skipping cached:    {len(existing_ids)}")
        print(f"To process:         {len(dataset)}")

    if len(dataset) == 0:
        print("Nothing to process.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load SimpleFold
    print(f"\nLoading SimpleFold ({args.simplefold_model})...")
    simplefold_model, _ = initialize_folding_model(args)
    simplefold_model.eval()
    esm_s_combine = simplefold_model.esm_s_combine.data.clone().to(device)

    config = ExtractionConfig(points=[ExtractionPoint("trunk_out")])
    extractor = SimpleFoldEmbeddingExtractor(model=simplefold_model, config=config)

    flow = LinearPath()
    sampler = EMSampler(
        num_timesteps=args.num_steps,
        t_start=1e-4,
        tau=args.tau,
        log_timesteps=True,
    )

    # Load ESM
    if args.esm_backend == "hf":
        print("\nLoading ESM-2 3B via HuggingFace transformers (fp16 on GPU, low CPU RAM)...")
        esm_model, esm_dict, af2_to_esm = load_esm_hf(ESM_HF_3B, device)
    else:
        print("\nLoading ESM-2 3B via facebookresearch/esm (fp32)...")
        esm_model, esm_dict = esm_registry["esm2_3B"]()
        esm_model = esm_model.to(device)
        esm_model.eval()
        af2_to_esm = _af2_to_esm(esm_dict)

    # Process
    print("\nExtracting per-residue embeddings...")
    start_time = time.time()
    processed = 0
    errors = 0
    pbar = tqdm(dataloader, desc="PROTEUS extract")

    with torch.no_grad():
        for batch_ids, batch_seqs in pbar:
            try:
                esmaa, lengths = encode_sequences_fast(batch_seqs, esm_dict, af2_to_esm, device)
                esm_combined, mask = extract_esm_combined(esmaa, lengths, esm_model, esm_s_combine)
                sf_batch = create_simplefold_batch(esm_combined, batch_seqs, mask, device)

                ts_emb = extract_perresidue_at_timesteps(
                    simplefold_model,
                    sf_batch,
                    args.timesteps,
                    extractor,
                    device,
                    flow,
                    sampler,
                    n_conformations=args.n_conformations,
                    save_all_conformations=not args.aggregate_conformations,
                    conf_batch_size=args.conf_batch_size,
                )

                for i, (protein_id, seq) in enumerate(zip(batch_ids, batch_seqs)):
                    seq_len = lengths[i]
                    save_dict: Dict[str, np.ndarray | str | int] = {
                        "sequence": seq,
                        "length": seq_len,
                    }
                    for key, tensor in ts_emb.items():
                        save_dict[key] = tensor[i, :seq_len].cpu().numpy().astype(np.float16)

                    if metadata["embedding_dim"] is None:
                        for v in save_dict.values():
                            if isinstance(v, np.ndarray) and v.ndim == 2:
                                metadata["embedding_dim"] = int(v.shape[-1])
                                break

                    save_path = embeddings_dir / f"{protein_id}.npz"
                    np.savez_compressed(save_path, **save_dict)
                    metadata["proteins"][protein_id] = {"length": seq_len}
                    processed += 1

                pbar.set_postfix({"ok": processed, "err": errors})

            except Exception as e:
                import traceback
                print(f"\nBatch error: {e}")
                traceback.print_exc()
                errors += len(batch_ids)
                continue

            if processed % 100 == 0 and processed > 0:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    metadata["total_proteins"] = len(metadata["proteins"])
    metadata["cache_time_seconds"] = elapsed
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Processed:      {processed}")
    print(f"  Errors:         {errors}")
    print(f"  Time:           {elapsed:.1f}s ({processed / max(elapsed, 1):.2f} proteins/s)")
    print(f"  Output:         {embeddings_dir}")
    print(f"  Embedding dim:  {metadata['embedding_dim']}")
    print("=" * 60)
    print("\nPer-protein NPZ keys:")
    print("  t0.0           [L, D] sequence-only embedding")
    if args.aggregate_conformations:
        print("  t1.0_mean      [L, D] mean across conformations")
        print("  t1.0_var       [L, D] variance across conformations")
    else:
        print("  t1.0_conf{k}   [L, D] structural conformation k (k=0..K-1)")


if __name__ == "__main__":
    main()
