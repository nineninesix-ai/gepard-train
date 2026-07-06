"""
Supervised Contrastive (SupCon) loss for voice-cloning training.

Implements the supervised contrastive objective from Khosla et al. (2020),
adapted for the Gepard RefCompressor: each batch sample is associated
with a `speaker_int` label, and the loss pulls compressor outputs from the
same speaker together while pushing different-speaker outputs apart.

This is the "path (a)" gradient signal that was missing from the original
voice-cloning setup — without it, the compressor can equally well memorise
speaker → ref fingerprints (path (b)) since both paths yield identical
reconstruction losses. SupCon forces speaker-discriminative + content-
invariant compressor outputs by construction.

Key design choices
------------------
* Operates on the compressor's `q_normed` (RMS=1 per token, shape [B, K_q, d])
  reduced to a single vector per sample by mean-pool over K_q queries.
* Optional 2-layer MLP projection head (SimCLR-style) before normalisation
  and contrast — projection is the standard trick to keep the projection
  space free to be discriminative while the upstream representation stays
  rich.
* Cross-rank `all_gather` for negatives: with FSDP / DDP, each rank computes
  its features locally and gathers across world for the similarity matrix.
  Effective batch grows to `world_size × P*K`. Local features keep their
  gradient; gathered remote features are detached (MoCo-style; differentiable
  all_gather is also supported via the `differentiable_gather` flag, but
  costs more memory).
* Null-ref samples (speaker_int == NULL_SPEAKER_INT) and self-pairs are
  excluded from the loss via mask construction. When a batch has zero
  positives (rare, but possible), the loss returns 0.

Reference
---------
P. Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Integer code used in `speaker_ints` for null-ref samples. Defined here (the
# loss that masks it) as the single source; the data collator imports it.
NULL_SPEAKER_INT = -1


class SupConProjectionHead(nn.Module):
    """2-layer MLP projection from compressor d_model to a SupCon embedding.

    Following SimCLR / SupCon practice: a non-linear projection between the
    representation we care about (compressor output) and the contrastive
    space. Empirically improves separation and prevents the contrastive
    objective from over-constraining the upstream features.

    Default `hidden=projection_dim=128` keeps the head tiny (~260K params for
    d_model=1024).
    """

    def __init__(self, d_model: int, hidden_dim: int = 128, projection_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def _all_gather_with_grad(t: torch.Tensor) -> torch.Tensor:
    """Differentiable all_gather: gradient flows to the local shard only.

    For SupCon we mostly care about the local sample's gradient flowing to
    its own features. The standard non-differentiable `all_gather` would
    break the autograd graph between the local feats and the gathered tensor
    — this helper splices the local-rank shard back into the gathered
    tensor as a gradient-tracking tensor, so the loss-backward into local
    feats still works. Non-local shards stay detached (MoCo-style: we use
    other ranks' features as negatives but don't propagate gradient through
    them, which is the standard trick to scale contrastive batches).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.detach())
    # Replace local-rank shard with the gradient-tracking version so backward
    # at least touches the local feats.
    gathered[rank] = t
    return torch.cat(gathered, dim=0)


def _all_gather_long(t: torch.Tensor) -> torch.Tensor:
    """Non-differentiable all_gather for integer label tensors."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world = dist.get_world_size()
    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=0)


def _all_gather_bool(t: torch.Tensor) -> torch.Tensor:
    """Non-differentiable all_gather for bool tensors (passed via uint8)."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    u = t.to(torch.uint8)
    world = dist.get_world_size()
    gathered = [torch.zeros_like(u) for _ in range(world)]
    dist.all_gather(gathered, u)
    return torch.cat(gathered, dim=0).to(torch.bool)


def supcon_loss(
    feats: torch.Tensor,                  # [B_local, d]
    speaker_ints: torch.Tensor,           # [B_local] long
    force_null: torch.Tensor,             # [B_local] bool
    temperature: float = 0.1,
    gather_across_ranks: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Compute supervised contrastive loss and diagnostic stats.

    Args:
        feats: per-sample feature vectors. Should already be projected and
            L2-normalised (call sites do this).
        speaker_ints: integer speaker labels. `NULL_SPEAKER_INT` for null-ref
            rows; those are excluded from the loss.
        force_null: bool flag per sample. Null-ref rows are excluded from
            anchors AND from positive masks.
        temperature: softmax temperature (smaller → sharper). Typical 0.07-0.1.
        gather_across_ranks: when True (and distributed is initialised),
            features and labels are all_gathered before computing the
            similarity matrix. Effective batch grows by `world_size`.

    Returns:
        (loss, diagnostics) where:
          loss: scalar tensor. Zero (no grad) if no valid (anchor, positive)
                pair exists in the batch.
          diagnostics: dict with `supcon_pos_sim_mean`, `supcon_neg_sim_mean`,
                `supcon_separation`, `supcon_n_anchors`, `supcon_n_positives_avg`.
    """
    device = feats.device
    if gather_across_ranks:
        feats_all = _all_gather_with_grad(feats)
        sid_all = _all_gather_long(speaker_ints.contiguous())
        null_all = _all_gather_bool(force_null.contiguous())
    else:
        feats_all = feats
        sid_all = speaker_ints
        null_all = force_null

    N = feats_all.size(0)
    valid = (~null_all) & (sid_all != NULL_SPEAKER_INT)  # [N] bool

    # Similarity matrix (cosine since feats are normalised) / temperature.
    sim = (feats_all @ feats_all.T) / float(temperature)  # [N, N]

    # Masks.
    eye = torch.eye(N, device=device, dtype=torch.bool)
    same_speaker = sid_all[:, None] == sid_all[None, :]                 # [N, N]
    valid_pair = valid[:, None] & valid[None, :]                        # both not null
    pos_mask = same_speaker & valid_pair & (~eye)                       # [N, N]
    neg_mask = (~same_speaker) & valid_pair                             # [N, N]

    # For each anchor i, denominator is sum over all j != i where j is valid.
    log_denom_mask = valid_pair & (~eye)                                # [N, N]

    # Mask out self for numerical stability before log-softmax.
    sim = sim.masked_fill(eye, float("-inf"))

    # log-softmax over rows: numerator restricted to log_denom_mask, but
    # F.log_softmax sums over all columns. Replace masked-out cols with -inf
    # so they don't contribute to the partition.
    sim_masked = sim.masked_fill(~log_denom_mask, float("-inf"))
    log_prob = F.log_softmax(sim_masked, dim=-1)                        # [N, N]

    # SupCon: −1/|P(i)| Σ_{p∈P(i)} log_prob[i, p]
    pos_count = pos_mask.sum(dim=-1).clamp(min=1).float()
    per_anchor = -(log_prob.masked_fill(~pos_mask, 0.0).sum(dim=-1) / pos_count)

    # Restrict to anchors that have at least one positive and are not null.
    anchor_keep = valid & (pos_mask.sum(dim=-1) > 0)
    n_anchors = int(anchor_keep.sum().item())
    if n_anchors == 0:
        loss = torch.zeros((), device=device, dtype=feats.dtype)
    else:
        loss = per_anchor[anchor_keep].mean()

    # Diagnostics (float, detached). Compute over valid pairs only.
    with torch.no_grad():
        if pos_mask.any():
            pos_sim_mean = sim.masked_fill(~pos_mask, 0.0).sum() / pos_mask.float().sum().clamp(min=1)
            pos_sim_mean = float((pos_sim_mean * temperature).item())  # un-scale by τ for interp
        else:
            pos_sim_mean = 0.0
        if neg_mask.any():
            # Re-mask sim to clean -inf for neg view.
            sim_for_neg = (feats_all @ feats_all.T)  # raw cosine (no τ, no eye fill)
            neg_sim_mean = float((sim_for_neg.masked_fill(~neg_mask, 0.0).sum()
                                  / neg_mask.float().sum().clamp(min=1)).item())
        else:
            neg_sim_mean = 0.0
        avg_positives = float((pos_mask.sum(dim=-1)[anchor_keep].float().mean().item())
                              if n_anchors > 0 else 0.0)

    diagnostics = {
        "supcon_pos_sim_mean": pos_sim_mean,
        "supcon_neg_sim_mean": neg_sim_mean,
        "supcon_separation": pos_sim_mean - neg_sim_mean,
        "supcon_n_anchors": float(n_anchors),
        "supcon_n_positives_avg": avg_positives,
        "supcon_batch_global": float(N),
    }
    return loss, diagnostics


def reduce_queries_to_feats(
    q_normed: torch.Tensor,                # [B, K_q, d]
    projection: Optional[SupConProjectionHead] = None,
) -> torch.Tensor:
    """Reduce compressor queries to a single per-sample feature vector.

    Pipeline:
      1. mean-pool over the K_q query dimension → [B, d]
      2. (optional) 2-layer MLP projection → [B, projection_dim]
      3. L2-normalise so cosine-sim = dot-product.

    Returns: [B, d_out] float32 (loss is computed in fp32 for stability).
    """
    feats = q_normed.float().mean(dim=1)                # [B, d]
    if projection is not None:
        feats = projection(feats.to(projection.fc1.weight.dtype))
    feats = F.normalize(feats.float(), dim=-1)
    return feats
