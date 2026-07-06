"""
DPO core: trajectory log-likelihood and the preference loss.

The policy is a per-frame stochastic process with TWO decisions at every step
(MODEL_GUIDE §7.1):
    s_t ~ Bernoulli(p_stop(h_{t-1}))      — stop or keep talking
    y_t ~ Π_c Categorical(head_c(h_{t-1})) — 32 independent codebook channels

so the full trajectory likelihood is

    log π(y|x) = log Π_c p_c(y_0 | h_bos)
               + Σ_{t=1}^{T-1} [ log(1 - p_t) + log Π_c p_c(y_t | h_{t-1}) ]
               + log p_T                                  (absent if truncated)

The Bernoulli terms are mandatory: without them the preference gradient never
reaches the stop decision — the exact component behind the runaway defect.
Stop probs are clamped to [p_floor, 1-p_floor] because the stop head is
saturated (stop-head probe, MODEL_GUIDE §7.1: p ∈ {~0, 1}) and raw sigmoids would kill the gradient.

Position layout mirrors TTSRunner.generate() exactly:
    inputs = [ prefix(K) | BOS_text ...text... EOT BOS_audio | frame_0 ... frame_{T-1} ]
    frame_t logits  ← hidden at position (K + T_text - 1 + t)   (BOS_audio for t=0)
    stop check at step t ≥ 1 ← hidden of frame_{t-1}
    terminal stop   ← hidden of frame_{T-1}
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


# Legacy defaults, used only when the caller supplies neither a repeater nor
# explicit ids / frame rate. The DPO stages source these from the composed
# `tokens` / `codec` groups (single source of truth, ROADMAP §G5).
BOS_TEXT = 248073
EOT = 248074
BOS_AUDIO = 248070

FRAME_RATE_HZ = 21.5


@dataclass
class TrajectoryLogprob:
    """Per-sequence decomposed log-likelihood (sums, not means)."""
    logp_tokens: torch.Tensor   # [B] Σ over frames and 32 channels
    logp_stop: torch.Tensor     # [B] Σ Bernoulli terms (no-stop chain + terminal)
    n_frames: torch.Tensor      # [B] T (for length normalization)

    def combined(self, stop_term_weight: float = 1.0, length_normalize: bool = True) -> torch.Tensor:
        total = self.logp_tokens + stop_term_weight * self.logp_stop
        if length_normalize:
            total = total / self.n_frames.clamp(min=1)
        return total


def encode_text_ids(tokenizer, text: str, device, repeater=None) -> torch.LongTensor:
    """Build the text-id layout. With `repeater` (a data.text_repetition.TextRepeater,
    e.g. `runner.repeater`) applies the SAME adaptive text repetition as production
    inference (deterministic target_R) — MUST match across sampling/pairs/train and
    the deployed TTSRunner, or DPO optimizes a different policy (MODEL_GUIDE §5.2/§7.4).
    Without a repeater → legacy single-copy layout."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if repeater is not None:
        return torch.tensor(repeater.expand(ids, training=False), dtype=torch.long, device=device)
    return torch.tensor([BOS_TEXT] + ids + [EOT, BOS_AUDIO], dtype=torch.long, device=device)


@torch.no_grad()
def compute_speaker_prefix(model, ref_codes: torch.Tensor) -> torch.Tensor:
    """ref_codes [1, T_ref, C] → frozen Q-Former prefix [K, d] (constant per speaker)."""
    ref_mask = torch.ones(ref_codes.shape[0], ref_codes.shape[1],
                          dtype=torch.bool, device=ref_codes.device)
    prefix, _ = model.ref_compressor(ref_codes, ref_mask)   # [1, K, d]
    return prefix.squeeze(0)


def trajectory_logprobs(
    model,
    text_ids_list: List[torch.LongTensor],   # per sample: [T_text_i] incl. BOS/EOT/BOS_AUDIO
    prefix_list: List[torch.Tensor],          # per sample: [K, d] speaker prefix
    tokens_list: List[torch.LongTensor],      # per sample: [C=32, T_i] generated frames
    truncated_list: List[bool],               # hit frame cap → no terminal stop event
    p_floor: float = 1e-4,
    requires_grad: bool = False,
) -> TrajectoryLogprob:
    """Batched teacher-forced trajectory log-likelihood.

    Right-pads variable-length sequences; gathers logits at the exact positions
    the autoregressive loop would have used. Works for both the frozen reference
    (requires_grad=False) and the trainable policy.
    """
    device = next(model.parameters()).device
    B = len(tokens_list)
    K = prefix_list[0].shape[0]
    d = prefix_list[0].shape[1]
    emb_dtype = model.model.embed_tokens.weight.dtype

    T_texts = [int(t.shape[0]) for t in text_ids_list]
    T_frames = [int(t.shape[1]) for t in tokens_list]
    L = max(K + tt + tf for tt, tf in zip(T_texts, T_frames))
    T_max = max(T_frames)

    inputs_embeds = torch.zeros(B, L, d, device=device, dtype=emb_dtype)
    attn_mask = torch.zeros(B, L, dtype=torch.long, device=device)
    # frames padded with 0 — a valid token id for every FSQ channel; masked out later
    frames = torch.zeros(B, model.num_codebook_heads, T_max, dtype=torch.long, device=device)

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        for i in range(B):
            tt, tf = T_texts[i], T_frames[i]
            text_emb = model.model.embed_tokens(text_ids_list[i].to(device).unsqueeze(0))  # [1, tt, d]
            inputs_embeds[i, :K] = prefix_list[i].to(device=device, dtype=emb_dtype)
            inputs_embeds[i, K:K + tt] = text_emb.squeeze(0)
            frames[i, :, :tf] = tokens_list[i].to(device)
            # audio frame embeddings via the model's stack: [1, tf, d]
            channel_tokens = [frames[i:i + 1, c, :tf] for c in range(model.num_codebook_heads)]
            inputs_embeds[i, K + tt:K + tt + tf] = model._embed_audio(channel_tokens).squeeze(0)
            attn_mask[i, :K + tt + tf] = 1

        out = model.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
        hidden = out.last_hidden_state                       # [B, L, d]

        # Gather hidden states per sample:
        #   pred_h[t]  = hidden[K + tt - 1 + t]  (predicts frame_t),  t = 0..T-1
        #   frame_h[t] = hidden[K + tt + t]      (hidden OF frame_t → stop checks)
        ar = torch.arange(T_max, device=device).unsqueeze(0)              # [1, T_max]
        start = torch.tensor([K + tt for tt in T_texts], device=device).unsqueeze(1)  # [B,1]
        pred_idx = (start - 1 + ar).clamp(max=L - 1)                      # [B, T_max]
        frame_idx = (start + ar).clamp(max=L - 1)
        pred_h = hidden.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, d))   # [B, T_max, d]
        frame_h = hidden.gather(1, frame_idx.unsqueeze(-1).expand(-1, -1, d))

        valid = ar < torch.tensor(T_frames, device=device).unsqueeze(1)   # [B, T_max]

        # ── token terms: Σ_c log p_c(y_t) at pred positions ──
        logp_tokens = torch.zeros(B, device=device, dtype=torch.float32)
        for c, head in enumerate(model.codebook_heads):
            logits = head(pred_h).float()                                 # [B, T_max, V_c]
            logp = F.log_softmax(logits, dim=-1)
            tok_lp = logp.gather(-1, frames[:, c, :].unsqueeze(-1)).squeeze(-1)  # [B, T_max]
            logp_tokens = logp_tokens + (tok_lp * valid).sum(dim=1)

        # ── Bernoulli stop terms at frame hiddens ──
        stop_logits = model.stop_head(frame_h.to(model.stop_head.weight.dtype)).squeeze(-1).float()
        p = torch.sigmoid(stop_logits).clamp(p_floor, 1.0 - p_floor)      # [B, T_max]
        n_frames_t = torch.tensor(T_frames, device=device)
        is_last = ar == (n_frames_t.unsqueeze(1) - 1)
        no_stop = valid & ~is_last                                        # frames 0..T-2
        logp_stop = (torch.log1p(-p) * no_stop).sum(dim=1)
        terminal = torch.tensor([not tr for tr in truncated_list], device=device)
        term_lp = (torch.log(p) * is_last).sum(dim=1)                     # log p at frame T-1
        logp_stop = logp_stop + term_lp * terminal

    return TrajectoryLogprob(
        logp_tokens=logp_tokens,
        logp_stop=logp_stop,
        n_frames=n_frames_t.float(),
    )


def dpo_loss(
    policy_chosen: TrajectoryLogprob,
    policy_rejected: TrajectoryLogprob,
    ref_chosen: torch.Tensor,      # [B] precomputed combined ref logprobs (same normalization!)
    ref_rejected: torch.Tensor,
    beta: float,
    stop_term_weight: float = 1.0,
    length_normalize: bool = True,
    weights: Optional[torch.Tensor] = None,
):
    """Standard DPO on (optionally length-normalized) trajectory logprobs.

    `weights` (per-pair, [B]) — optional reward-magnitude weighting (MODEL_GUIDE
    §7.2): up-weight pairs with a larger chosen−rejected reward gap so the
    high-contrast (content) pairs get more gradient. None → plain mean
    (round 1/2 behaviour, byte-for-byte). Caller is responsible for normalising
    `weights` (we divide by their sum, so mean(w)=1 keeps the loss scale).

    Returns (loss, metrics_dict).
    """
    pi_w = policy_chosen.combined(stop_term_weight, length_normalize)
    pi_l = policy_rejected.combined(stop_term_weight, length_normalize)
    logits = beta * ((pi_w - ref_chosen) - (pi_l - ref_rejected))
    per_pair = -F.logsigmoid(logits)
    if weights is None:
        loss = per_pair.mean()
    else:
        loss = (weights * per_pair).sum() / weights.sum().clamp(min=1e-8)
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "acc": (logits > 0).float().mean().item(),
            "margin": (pi_w - pi_l).mean().item(),
            "pi_chosen": pi_w.mean().item(),
            "pi_rejected": pi_l.mean().item(),
            "stop_lp_chosen": (policy_chosen.logp_stop / policy_chosen.n_frames).mean().item(),
            "stop_lp_rejected": (policy_rejected.logp_stop / policy_rejected.n_frames).mean().item(),
        }
    return loss, metrics


def adaptive_max_frames(text: str, reward_cfg, sampling_cfg,
                        frame_rate_hz: float = FRAME_RATE_HZ) -> int:
    """Frame cap for rollouts: a multiple of the expected duration (see config)."""
    expected_frames = reward_cfg.expected_sec(text) * frame_rate_hz
    cap = int(sampling_cfg.cap_expected_multiple * expected_frames)
    return max(sampling_cfg.cap_min_frames, min(sampling_cfg.cap_max_frames, cap))
