"""
Reference voice compressor: Q-Former-style bottleneck that turns a variable-length
stack of codec codes into K learnable "speaker" tokens consumed by the decoder
as a prefix (see `GepardModel.forward`).

Architecture (MODEL_GUIDE §4.1):

    Input pipeline:
        packed_or_unfolded [B, T_ref, C_in]
        ├── (optional) unfold_tokens → [B, T_ref, C_total]  if dataset is packed
        ├── dequantize_codes         → [B, T_ref, C_total]  float in [-1, 1]
        ├── Linear(C_total → d_model)→ [B, T_ref, d_model]
        └── + sinusoidal PE          → ref_feats [B, T_ref, d_model]

    Queries:
        nn.Parameter(K, d_model), init normal(0, queries_init_std)
        batch-expand to [B, K, d_model]  (position-less)

    For each of L Q-Former blocks (pre-norm RMSNorm + SwiGLU FFN):
        q = q + SelfAttn(RMSNorm(q))                     # bidirectional
        q = q + CrossAttn(RMSNorm(q), kv=ref_feats,      # queries ← ref
                          key_padding_mask=ref_mask)
        q = q + SwiGLU_FFN(RMSNorm(q))

    Output: [B, K, d_model] → decoder prefix
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec_ops import dequantize_codes, unfold_tokens

if TYPE_CHECKING:
    # Duck-typed: any object with the CodecConfig attributes
    # (`num_layers`, `fsq_levels`, `do_unfold`) fits.
    from ..config.schema import CodecConfig, CompressorConfig  # noqa: F401


class RMSNorm(nn.Module):
    """Classic RMSNorm. We carry our own copy (rather than torch.nn.RMSNorm) so
    the module works on older PyTorch versions and has stable behavior under
    bf16 autocast (compute norm in fp32 then cast back)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(orig_dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN — two gated projections + one output projection."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SinusoidalPositionalEncoding(nn.Module):
    """Standard Transformer sinusoidal PE, lazily extended as needed."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("_pe", self._build(max_len, d_model), persistent=False)

    @staticmethod
    def _build(length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(length, dim, dtype=torch.float32)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d] → x + PE[:T]"""
        T = x.shape[1]
        if self._pe.shape[0] < T:
            self._pe = self._build(T, self.d_model).to(self._pe.device)
        return x + self._pe[:T].to(dtype=x.dtype, device=x.device)


class MultiHeadAttentionBlock(nn.Module):
    """Thin wrapper around scaled_dot_product_attention with separate q / kv paths.

    Supports both self-attention (pass same tensor as q and kv) and cross-attention
    (pass queries as q, ref_feats as kv). Key-padding masks on kv go through as a
    4D bool tensor (True = keep, False = mask).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout_p = float(dropout)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        q_in: torch.Tensor,                      # [B, Tq, d]
        kv_in: torch.Tensor,                     # [B, Tkv, d]
        kv_mask: Optional[torch.Tensor] = None,  # [B, Tkv] bool, True = keep
    ) -> torch.Tensor:
        B, Tq, _ = q_in.shape
        Tkv = kv_in.shape[1]

        q = self.q_proj(q_in).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA attn_mask: broadcastable bool or float. Use bool: True = attend.
        if kv_mask is not None:
            # [B, 1, 1, Tkv] → broadcasts over heads and query positions.
            attn_mask = kv_mask.view(B, 1, 1, Tkv)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )                                                        # [B, H, Tq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.resid_dropout(self.out_proj(out))


class QFormerBlock(nn.Module):
    """One Q-Former block: SelfAttn(q) → CrossAttn(q, ref) → FFN, all pre-norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_self = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.norm_cross = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout)

    def forward(
        self,
        q: torch.Tensor,                            # [B, K, d]
        ref_feats: torch.Tensor,                    # [B, T_ref, d]
        ref_mask: Optional[torch.Tensor] = None,    # [B, T_ref] bool
    ) -> torch.Tensor:
        # Self-attn on queries (bidirectional, no mask — queries are always valid).
        h = self.norm_self(q)
        q = q + self.self_attn(h, h, kv_mask=None)

        # Cross-attn: queries attend to ref_feats, masked on ref padding.
        h = self.norm_cross(q)
        q = q + self.cross_attn(h, ref_feats, kv_mask=ref_mask)

        # FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class RefCompressor(nn.Module):
    """Q-Former-style compressor of codec codes into K speaker tokens.

    Args:
        codec: Codec geometry (`num_layers`, `fsq_levels`, `do_unfold`).
            Determines `C_total` and whether the forward pass unfolds packed
            layers or assumes already-unfolded input.
        compressor_cfg: Hyperparameters (num_queries K, num_layers L, d_model, ...).
        backbone_hidden_size: Fallback for `d_model` when `compressor_cfg.d_model is None`.
    """

    def __init__(
        self,
        codec: CodecConfig,
        compressor_cfg: CompressorConfig,
        backbone_hidden_size: int,
    ):
        super().__init__()
        self.num_layers_codec = int(codec.num_layers)
        self.fsq_levels: List[int] = list(codec.fsq_levels)
        do_unfold_on_disk = bool(codec.do_unfold)
        # If the dataset is already unfolded on disk, skip unfold in forward.
        self.do_unfold_in_forward = not do_unfold_on_disk

        self.c_total = self.num_layers_codec * len(self.fsq_levels)
        self.d_model = int(compressor_cfg.d_model) if compressor_cfg.d_model else backbone_hidden_size
        self.num_queries = int(compressor_cfg.num_queries)
        self.num_blocks = int(compressor_cfg.num_layers)
        self.num_heads = int(compressor_cfg.num_heads)
        self.ffn_hidden = self.d_model * int(compressor_cfg.ffn_hidden_size_multiplier)
        self.dropout = float(compressor_cfg.dropout)

        self.input_proj = nn.Linear(self.c_total, self.d_model, bias=True)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model)

        self.queries = nn.Parameter(
            torch.randn(self.num_queries, self.d_model) * float(compressor_cfg.queries_init_std)
        )

        self.blocks = nn.ModuleList([
            QFormerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_hidden=self.ffn_hidden,
                dropout=self.dropout,
            )
            for _ in range(self.num_blocks)
        ])

        # Final norm so the prefix's scale matches other inputs into the decoder.
        # RMSNorm alone gives unit RMS → L2 norm ≈ sqrt(d_model), which dwarfs text/audio
        # embeddings (scale ~1). The learnable scalar starts at 1/sqrt(d_model) so the
        # initial output L2 ≈ 1, and the model can rescale upward if useful.
        self.final_norm = RMSNorm(self.d_model)
        # Shape [1], not scalar []: FSDP2's fully_shard rejects 0-dim parameters.
        # numel stays 1, and [1] broadcasts identically over [B, K, d_model].
        self.output_scale = nn.Parameter(torch.tensor([1.0 / math.sqrt(self.d_model)]))

    def forward(
        self,
        ref_codes: torch.Tensor,                    # [B, T_ref, C_in] long
        ref_mask: Optional[torch.Tensor] = None,    # [B, T_ref] bool, True = real frame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (prefix_out, q_normed):
          prefix_out: [B, K, d_model] — output_scale * RMSNorm(q), what the decoder consumes.
          q_normed:   [B, K, d_model] — RMSNorm(q) before output_scale (RMS=1 per token).
                       Used by diversity loss in pre-output_scale space.
        """
        if ref_codes.dim() != 3:
            raise ValueError(f"ref_codes must be [B, T_ref, C_in]; got {tuple(ref_codes.shape)}")

        # 1. Unfold only if the dataset is packed on disk.
        if self.do_unfold_in_forward:
            # unfold_tokens expects [B, C, T], but we have [B, T, C] — transpose.
            x = unfold_tokens(ref_codes.transpose(1, 2), self.fsq_levels).transpose(1, 2)
            # [B, T, C_total]
        else:
            x = ref_codes                                       # already [B, T, C_total]

        # 2. Dequantize to [-1, 1] floats; cast with the module dtype (bf16 under autocast).
        x = dequantize_codes(x, self.fsq_levels, self.num_layers_codec)
        x = x.to(dtype=self.input_proj.weight.dtype)

        # 3. Linear project to d_model, add sinusoidal PE.
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # 4. Expand learnable queries across the batch.
        B = x.shape[0]
        q = self.queries.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)

        # 5. Run Q-Former blocks; queries attend to ref_feats with the ref mask.
        for block in self.blocks:
            q = block(q, ref_feats=x, ref_mask=ref_mask)

        q_normed = self.final_norm(q)                           # RMS=1 per token
        return self.output_scale * q_normed, q_normed
