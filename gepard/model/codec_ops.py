"""
Shared codec operations — mixed-radix unfold and FSQ dequantization.

Scoped for preprocessing (`gepard.data.preprocessing.processor`) and
training-time use (`gepard.model.ref_compressor`). Keeps zero runtime
dependency on NeMo so these
helpers are safe to import in lightweight environments. `codec_wrapper.py`
(inference-time, NeMo-backed) continues to carry its own unfold copy — it is
on a different dependency track and we don't want to couple them.

Mixed-radix decomposition (inverse of FSQ packing):
  A packed token `k` encodes `len(fsq_Levels)` per-dimension codes as
      k = code_0 + code_1 * L_0 + code_2 * L_0*L_1 + ...
  so we recover
      code_d = (k // prod(L_0..L_{d-1})) % L_d
  (little-endian mixed base, consistent with the numpy version that produced
  the on-disk dataset).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def unfold_tokens_np(encoded_tokens: np.ndarray, num_levels: Sequence[int]) -> np.ndarray:
    """Mixed-radix decomposition of packed token indices.

    Args:
        encoded_tokens: (C, T) packed indices, C = number of codebook layers.
        num_levels: FSQ levels per dimension within each codebook.

    Returns:
        (C * len(num_levels), T) per-dimension discrete codes.
    """
    levels = np.array(num_levels, dtype=np.int32)
    dim_base = np.cumprod(np.array([1] + list(num_levels[:-1]), dtype=np.int32))

    parts = []
    for cb in range(encoded_tokens.shape[0]):
        idx = encoded_tokens[cb:cb + 1, :]  # (1, T)
        codes = (idx // dim_base[:, None]) % levels[:, None]
        parts.append(codes)

    return np.concatenate(parts, axis=0)


def unfold_tokens(packed: torch.Tensor, num_levels: Sequence[int]) -> torch.Tensor:
    """Mixed-radix decomposition — torch version.

    Args:
        packed: (B, C, T) packed indices, long tensor.
        num_levels: FSQ levels per dimension.

    Returns:
        (B, C * len(num_levels), T) per-dimension discrete codes.
        Channel order matches numpy version: for codebook c and FSQ dim d,
        output channel index is `c * len(num_levels) + d`.
    """
    if packed.dim() != 3:
        raise ValueError(f"unfold_tokens expects [B, C, T], got {tuple(packed.shape)}")
    device = packed.device
    levels = torch.tensor(list(num_levels), device=device, dtype=torch.long)  # [D]
    bases = torch.tensor(
        np.cumprod([1] + list(num_levels[:-1])).tolist(),
        device=device,
        dtype=torch.long,
    )  # [D]

    # packed: [B, C, T]; broadcast to [B, C, D, T]
    B, C, T = packed.shape
    D = levels.shape[0]
    packed_ = packed.unsqueeze(2)                      # [B, C, 1, T]
    bases_ = bases.view(1, 1, D, 1)                    # [1, 1, D, 1]
    levels_ = levels.view(1, 1, D, 1)                  # [1, 1, D, 1]
    codes = (packed_ // bases_) % levels_              # [B, C, D, T]
    return codes.reshape(B, C * D, T)


def dequantize_codes(
    unfolded: torch.Tensor,
    num_levels: Sequence[int],
    num_layers: int,
) -> torch.Tensor:
    """Per-dimension symmetric dequantization of unfolded FSQ codes.

    Applies `(x - L//2) / (L//2)` per channel using the per-channel level
    pattern `[num_levels * num_layers]`. Output lies in `[-1, 1]` by
    construction (codes are in `[0, L-1]`).

    Args:
        unfolded: [..., C_total] int tensor where `C_total = num_layers * len(num_levels)`.
            Typical shapes: [B, T, C_total] (channel-last) or [B, C_total, T] (channel-mid).
            This helper assumes **channel-last** (last dim is channels).
        num_levels: FSQ levels per dimension within each codebook.
        num_layers: Number of codebook layers. Level pattern repeats `num_layers` times.

    Returns:
        Float tensor of the same shape as `unfolded`, roughly in `[-1, 1]`.
    """
    C_total = unfolded.shape[-1]
    expected = num_layers * len(num_levels)
    if C_total != expected:
        raise ValueError(
            f"dequantize_codes: last dim={C_total} but num_layers*len(num_levels)={expected}"
        )
    levels = torch.tensor(
        list(num_levels) * num_layers,
        device=unfolded.device,
        dtype=torch.float32,
    )  # [C_total]
    scale = (levels // 2).clamp_min(1.0)               # [C_total]
    x = unfolded.float()
    return (x - scale) / scale
