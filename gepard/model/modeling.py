"""
Gepard multihead model.

Architecture:
  - Stock Qwen3.5 transformer backbone (no modifications)
  - N audio embedding tables (one per FSQ channel) → average → single frame embedding
  - N codebook output heads with per-channel vocabulary sizes
  - 1 binary stop head (end-of-speech prediction)
  - No lm_head, no text generation

Audio heads are configured via audio_heads: Dict[str, int] (channel_name → vocab_size),
derived from the codec group (conf/model/gepard.yaml). Channel names must match dataset column names
produced by unfold_tokens_np in the data pipeline.

Voice cloning (optional):
  - When `vc_config.enabled=True` and `codec` is passed, the model owns a
    `RefCompressor` that turns per-sample codec `ref_codes` into K speaker tokens,
    concatenated as a prefix before the text region: `[prefix | text | audio]`.
  - A learnable `null_prefix` is stochastically swapped in with probability
    `cfg_dropout_prob` during training to enable classifier-free-guidance at inference.
  - Only difference in downstream slicing: `audio_hidden = hidden[:, K + T_text:, :]`.
  - When VC is disabled, all new code paths are skipped and the forward matches the
    pre-VC behavior exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Optional, Dict, List
from dataclasses import dataclass
from transformers import AutoConfig
from transformers.modeling_outputs import ModelOutput

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5ForCausalLM,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from .ref_compressor import RefCompressor
from .losses.supcon import SupConProjectionHead, reduce_queries_to_feats, supcon_loss

if TYPE_CHECKING:
    # Duck-typed: any objects with the schema attributes fit.
    from ..config.schema import CodecConfig, VoiceCloningConfig  # noqa: F401


# Aliases people actually write in configs → canonical torch dtypes. A plain
# `getattr(torch, name)` would resolve e.g. "bf16" to nothing and "float" to
# float32 only by accident.
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype(name) -> torch.dtype:
    """Map a config dtype string (or a torch.dtype) to the torch dtype."""
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPE_MAP[str(name).lower().removeprefix("torch.")]
    except KeyError:
        raise ValueError(
            f"unknown model dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}"
        ) from None


@dataclass
class MultiheadTTSOutput(ModelOutput):
    """Output type for GepardModel."""
    loss: Optional[torch.FloatTensor] = None
    logits_audio: Optional[List[torch.FloatTensor]] = None
    logits_stop: Optional[torch.FloatTensor] = None


class GepardModel(nn.Module):
    """
    Multihead TTS model: Qwen3.5 backbone + N codebook heads + 1 stop head.

    Input:
      - text_ids → stock Qwen3.5 embed_tokens
      - level_audio_* → N separate embeddings → average → backbone

    Output:
      - N codebook heads: Linear(d, vocab_size_i) each, vocab sizes from audio_heads config
      - 1 stop head: Linear(d, 1), binary sigmoid + BCE

    Audio channels are identified by name (matching dataset column names).
    The forward pass receives audio tensors via **kwargs keyed by channel name,
    and label tensors as kwargs keyed by f'labels_{channel_name}'.

    The transformer backbone (Qwen3_5TextModel) is completely stock — no overrides.
    """

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        audio_heads: Dict[str, int],
        stop_loss_weight: float = 1.0,
        stop_pos_weight: float = 1.0,
        vc_config: Optional[VoiceCloningConfig] = None,
        codec: Optional[CodecConfig] = None,
        audio_embed_dim: int = 32,
    ):
        super().__init__()
        self.config = config
        self.audio_heads = audio_heads
        self.channel_names: List[str] = list(audio_heads.keys())
        self.vocab_sizes: List[int] = list(audio_heads.values())
        self.num_codebook_heads: int = len(audio_heads)
        self.stop_loss_weight = stop_loss_weight
        self.stop_pos_weight = stop_pos_weight
        hidden_size = config.hidden_size

        # Stock Qwen3.5 backbone
        self.model = Qwen3_5TextModel(config)

        # Audio input: per-codebook embeddings → concat → 2-layer GELU MLP.
        # Each codebook is a 6-8-way FSQ categorical, so a small per-channel
        # dim (audio_embed_dim) is plenty. A plain sum of lookups is only an
        # additive (linear) function of the channels — the MLP's nonlinearity
        # lets the frame embedding model cross-codebook interactions.
        # Config-driven (conf/model/gepard.yaml → audio_embed_dim); default 32.
        self.audio_embed_dim = audio_embed_dim
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.audio_embed_dim)
            for vocab_size in self.vocab_sizes
        ])
        self.audio_embed_proj = nn.Sequential(
            nn.Linear(self.num_codebook_heads * self.audio_embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            # Affine-free LayerNorm pins the frame-embedding scale. The backbone's
            # input RMSNorm makes audio-embedding magnitude a free direction that
            # otherwise drifts unbounded (and pushes the GELU toward degenerating
            # into a linear map). Normalizing here removes that free direction.
            nn.LayerNorm(hidden_size, elementwise_affine=False),
        )
        # Rescales the unit-norm projection output to the text-embedding std so
        # the frame is not OOD for the ref-compressor / diagnostics. Set in
        # _init_audio_embeddings; persisted in state_dict so resume restores it.
        # A frozen PARAMETER, not a buffer: accelerate's FSDP2 state-dict loader
        # requires every state_dict entry to be shardable (persistent buffers
        # crash it), and fully_shard additionally rejects 0-dim tensors — hence
        # shape [1]. Legacy checkpoints store it 0-dim; normalize_scalar_shapes
        # reshapes on load.
        self.audio_embed_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # Output heads: one Linear per channel, each with its own vocab size
        self.codebook_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for vocab_size in self.vocab_sizes
        ])
        self.stop_head = nn.Linear(hidden_size, 1)

        # Storage for per-head losses (read by callback for wandb logging)
        self._last_losses = {}
        # Storage for diagnostics (embedding norms, T_text, etc.)
        self._last_diagnostics = {}
        # Per-layer cosine similarity text↔audio.
        # Populated in forward when _collect_layer_sims=True (set by DiagnosticsCallback).
        # Uses output_hidden_states=True — FSDP-safe, no layer-level hooks needed.
        self._last_layer_cos_sims: dict = {}
        self._collect_layer_sims: bool = False

        # Updated by MultiheadTTSTrainer.training_step before each forward;
        # drives the diversity/SupCon warmup-ramp curricula.
        self._global_step = 0

        # Voice cloning: reference compressor + learnable null prefix for CFG.
        # When disabled, all these attributes are None / zero and the forward
        # takes the legacy path.
        self.ref_compressor: Optional[RefCompressor] = None
        self.null_prefix: Optional[nn.Parameter] = None
        self.cfg_dropout_prob: float = 0.0
        self.prefix_len: int = 0
        # Diversity (variance) loss config — applied on q_normed (pre-output_scale).
        self.diversity_enabled: bool = False
        self.diversity_gamma: float = 0.5
        self.diversity_weight: float = 1.0
        self.diversity_warmup_start: int = 0
        self.diversity_ramp_steps: int = 1
        # SupCon config / projection head.
        self.supcon_enabled: bool = False
        self.supcon_head: Optional[SupConProjectionHead] = None
        self.supcon_weight: float = 0.0
        self.supcon_warmup_start: int = 0
        self.supcon_ramp_steps: int = 1
        self.supcon_temperature: float = 0.1
        self.supcon_gather: bool = True
        if vc_config is not None and vc_config.enabled:
            if codec is None:
                raise ValueError(
                    "GepardModel: vc_config.enabled=True but codec is None; "
                    "voice cloning needs the codec geometry to know how to unfold/dequantize."
                )
            self.ref_compressor = RefCompressor(
                codec=codec,
                compressor_cfg=vc_config.compressor,
                backbone_hidden_size=hidden_size,
            )
            self.prefix_len = int(vc_config.compressor.num_queries)
            self.cfg_dropout_prob = float(vc_config.training.cfg_dropout_prob)
            null_std = float(vc_config.training.null_prefix_init_std)
            self.null_prefix = nn.Parameter(
                torch.randn(self.prefix_len, self.ref_compressor.d_model) * null_std
            )
            div_cfg = vc_config.training.diversity_loss
            self.diversity_enabled = bool(div_cfg.enabled)
            self.diversity_gamma = float(div_cfg.gamma)
            self.diversity_weight = float(div_cfg.weight)
            self.diversity_warmup_start = int(div_cfg.warmup_start)
            self.diversity_ramp_steps = max(1, int(div_cfg.ramp_steps))

            sup_cfg = vc_config.training.supcon
            self.supcon_enabled = bool(sup_cfg.enabled)
            self.supcon_weight = float(sup_cfg.weight)
            self.supcon_warmup_start = int(sup_cfg.warmup_start)
            self.supcon_ramp_steps = max(1, int(sup_cfg.ramp_steps))
            self.supcon_temperature = float(sup_cfg.temperature)
            self.supcon_gather = bool(sup_cfg.gather_across_ranks)
            if self.supcon_enabled and bool(sup_cfg.use_projection):
                self.supcon_head = SupConProjectionHead(
                    d_model=self.ref_compressor.d_model,
                    hidden_dim=int(sup_cfg.projection_hidden_dim),
                    projection_dim=int(sup_cfg.projection_dim),
                )

    def _init_audio_embeddings(self):
        """Initialize the audio embedding stack.

        Frame embedding flows: tables → concat → Linear → GELU → Linear →
        LayerNorm. The trailing affine-free LayerNorm pins the output scale, so
        init only needs to keep each stage's activations ~O(1) for healthy
        gradient flow — no calibration of the absolute output scale is needed.
        `audio_embed_scale` then rescales the unit-norm output to the text
        embedding std (the backbone discards scale via RMSNorm, but matching
        keeps the frame in-distribution for the ref-compressor / diagnostics).
        """
        with torch.no_grad():
            text_std = self.model.embed_tokens.weight.float().std().item()
            self.audio_embed_scale.fill_(text_std)

            # Embedding tables: unit std — final scale is set by LayerNorm.
            for emb in self.audio_embeddings:
                nn.init.normal_(emb.weight, mean=0.0, std=1.0)

            # MLP: fan-in init keeps pre-activations ~unit variance; zero bias.
            lin1, lin2 = self.audio_embed_proj[0], self.audio_embed_proj[2]
            nn.init.normal_(lin1.weight, mean=0.0, std=lin1.in_features ** -0.5)
            nn.init.zeros_(lin1.bias)
            nn.init.normal_(lin2.weight, mean=0.0, std=lin2.in_features ** -0.5)
            nn.init.zeros_(lin2.bias)

    def _embed_audio(self, audio_tensors: List[torch.LongTensor]) -> torch.FloatTensor:
        """Embed N codebook channels → concat → MLP → single frame embedding.

        Replaces the legacy unweighted mean: a sum of per-codebook lookups is
        only an additive (linear) function of the channels and cannot model
        joint cross-codebook effects. The 2-layer GELU MLP can.
        """
        per_channel = [
            self.audio_embeddings[i](audio_tensors[i])
            for i in range(self.num_codebook_heads)
        ]
        concat = torch.cat(per_channel, dim=-1)            # [B, T_audio, N * audio_embed_dim]
        frame = self.audio_embed_proj(concat)              # [B, T_audio, d], unit-norm (LayerNorm)
        return frame * self.audio_embed_scale.to(frame.dtype)

    def _maybe_apply_cfg_dropout(
        self,
        prefix: torch.Tensor,
        force_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float, float]:
        """Per-sample substitute prefix with null_prefix.

        Two independent triggers, OR-combined per sample:
          - Stochastic CFG dropout at rate `cfg_dropout_prob` (training only).
          - `force_mask` ([B] bool): forced swap regardless of training mode,
            used for samples whose speaker_id is the null sentinel
            (singleton_policy=null_prefix path).

        Returns (prefix, observed_random_rate, observed_force_rate). Both
        rates are reported separately so the trainer can tell how much of
        the unconditional signal is drift from sentinel data vs. CFG.
        """
        if self.null_prefix is None:
            return prefix, 0.0, 0.0
        B = prefix.size(0)
        device = prefix.device

        # Stochastic CFG dropout — training only.
        if self.training and self.cfg_dropout_prob > 0.0:
            random_drop = torch.rand(B, device=device) < self.cfg_dropout_prob
        else:
            random_drop = torch.zeros(B, device=device, dtype=torch.bool)

        # Forced null (e.g. null-sentinel samples). Always applied.
        if force_mask is None:
            forced = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            forced = force_mask.to(device=device, dtype=torch.bool)

        drop = random_drop | forced
        if not drop.any():
            return prefix, 0.0, float(forced.float().mean().item())

        null = self.null_prefix.to(dtype=prefix.dtype, device=device)
        null = null.unsqueeze(0).expand(B, -1, -1)
        prefix = torch.where(drop[:, None, None], null, prefix)
        return (
            prefix,
            float(random_drop.float().mean().item()),
            float(forced.float().mean().item()),
        )

    def forward(
        self,
        text_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels_stop: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MultiheadTTSOutput:
        # Extract audio channel tensors and labels by name from kwargs
        audio_tensors = [kwargs[name] for name in self.channel_names]
        labels = [kwargs.get(f"labels_{name}") for name in self.channel_names]

        # Voice cloning inputs (may be absent when feature is off)
        ref_codes = kwargs.get("ref_codes")
        ref_mask = kwargs.get("ref_mask")

        # 1. Embed text and audio
        text_embeds = self.model.embed_tokens(text_ids)          # [B, T_text, d]
        audio_embeds = self._embed_audio(audio_tensors)          # [B, T_audio, d]
        T_text = text_ids.shape[1]

        # Store embedding norms and T_text for diagnostics/hooks (detached, cheap)
        with torch.no_grad():
            self._last_diagnostics["norm_text_emb"] = text_embeds.float().norm(dim=-1).mean().item()
            self._last_diagnostics["norm_audio_emb"] = audio_embeds.float().norm(dim=-1).mean().item()
        self._last_diagnostics["T_text"] = T_text

        # 1b. Voice-cloning prefix: run compressor, maybe swap with null for CFG.
        prefix = None
        K = 0
        diversity_loss = None
        supcon_term = None
        if self.ref_compressor is not None and ref_codes is not None:
            # q_normed: pre-output_scale RMSNorm(q), RMS=1 per token. Used by L_var so
            # γ stays in natural units. prefix_raw is what the decoder consumes.
            prefix_raw, q_normed = self.ref_compressor(ref_codes, ref_mask)    # both [B, K, d]
            K = prefix_raw.size(1)

            # Diagnostics on the PRE-CFG compressor output. Post-CFG mixing with
            # null_prefix contaminates the signal: when drop_rate>0 some samples
            # contribute null_prefix rows, and the observed metric oscillates
            # wildly between compressor diversity and null_prefix diversity
            # depending on the stochastic per-step drop_rate. Measuring here
            # gives a clean trend of actual compressor learning.
            with torch.no_grad():
                prefix_f = prefix_raw.detach().float()
                self._last_diagnostics["norm_prefix_cond"] = prefix_f.norm(dim=-1).mean().item()
                qn = prefix_f / prefix_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos = torch.einsum("bkd,bjd->bkj", qn, qn)                     # [B, K, K]
                eye = torch.eye(K, device=cos.device, dtype=torch.bool).unsqueeze(0)
                cos_off = cos.masked_select(~eye).view(cos.size(0), -1)        # off-diag
                self._last_diagnostics["query_cos_sim_mean"] = cos_off.mean().item()
                self._last_diagnostics["query_cos_sim_max"] = cos_off.max().item()
                self._last_diagnostics["query_std_across_k"] = prefix_f.std(dim=1).mean().item()

                # null_prefix own diversity — computed directly on the parameter,
                # independent of CFG firing. Shows whether the learnable null is
                # collapsing (cos → 1) or staying diverse (cos → 0).
                if self.null_prefix is not None:
                    null_f = self.null_prefix.detach().float()
                    self._last_diagnostics["norm_prefix_null"] = null_f.norm(dim=-1).mean().item()
                    null_n = null_f / null_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    null_cos = null_n @ null_n.T                                # [K, K]
                    null_eye = torch.eye(K, device=null_cos.device, dtype=torch.bool)
                    null_off = null_cos.masked_select(~null_eye)
                    self._last_diagnostics["null_cos_sim_mean"] = null_off.mean().item()

            # Diversity (variance) loss on q_normed (RMS=1 space). Hinge form:
            # L = mean(relu(γ - std)) ∈ [0, γ]. Linear-ramp curriculum prevents
            # noisy push before the compressor has any signal.
            if self.diversity_enabled and self.training:
                step = int(self._global_step)
                if step >= self.diversity_warmup_start:
                    progress = min(
                        1.0, (step - self.diversity_warmup_start) / self.diversity_ramp_steps
                    )
                    cur_w = self.diversity_weight * progress
                    std_BD = q_normed.float().std(dim=1, unbiased=False).clamp(min=1e-4)
                    L_var = F.relu(self.diversity_gamma - std_BD).mean()
                    diversity_loss = cur_w * L_var
                    with torch.no_grad():
                        self._last_diagnostics["query_div_weight"] = cur_w
                        self._last_diagnostics["query_div_std_mean"] = std_BD.mean().item()

            # Supervised contrastive (SupCon) on q_normed. Computed BEFORE
            # cfg_dropout so the loss sees the actual compressor output
            # (not null_prefix). Warmup + linear ramp give CE loss time to
            # establish coarse phoneme structure before the contrastive
            # objective shapes speaker geometry.
            if self.supcon_enabled and self.training:
                speaker_ints = kwargs.get("speaker_ints")
                force_null_t = kwargs.get("force_null")
                if speaker_ints is not None and force_null_t is not None:
                    step = int(self._global_step)
                    if step >= self.supcon_warmup_start:
                        progress = min(
                            1.0,
                            (step - self.supcon_warmup_start) / self.supcon_ramp_steps,
                        )
                        cur_w = self.supcon_weight * progress
                        # reduce + project + normalize.
                        feats = reduce_queries_to_feats(q_normed, self.supcon_head)
                        sup_raw, sup_diag = supcon_loss(
                            feats=feats,
                            speaker_ints=speaker_ints.to(q_normed.device),
                            force_null=force_null_t.to(q_normed.device).bool(),
                            temperature=self.supcon_temperature,
                            gather_across_ranks=self.supcon_gather,
                        )
                        supcon_term = cur_w * sup_raw
                        # Cache for the main loss sum + logging callbacks.
                        self._last_losses["supcon"] = sup_raw.detach()
                        self._last_diagnostics["supcon_weight"] = cur_w
                        for k, v in sup_diag.items():
                            self._last_diagnostics[k] = v
                    else:
                        supcon_term = None
                else:
                    supcon_term = None
            else:
                supcon_term = None

            # Apply CFG dropout (random + forced for null-sentinel rows) AFTER
            # diagnostics so they see the clean compressor output.
            force_null = kwargs.get("force_null")  # [B] bool or None
            prefix, observed_drop_rate, observed_force_rate = self._maybe_apply_cfg_dropout(
                prefix_raw, force_mask=force_null,
            )
            with torch.no_grad():
                self._last_diagnostics["cfg_dropout_rate_observed"] = observed_drop_rate
                self._last_diagnostics["forced_null_rate"] = observed_force_rate
                self._last_diagnostics["T_prefix"] = K

        # 2. Concatenate into single sequence (with prefix if present).
        if prefix is not None:
            inputs_embeds = torch.cat([prefix, text_embeds, audio_embeds], dim=1)
            prefix_mask = torch.ones(
                prefix.size(0), K,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
            full_attention_mask = attention_mask

        # 3. Forward through stock Qwen3.5 backbone
        # _collect_layer_sims is set by DiagnosticsCallback one step before an expensive
        # logging step. output_hidden_states adds ~28 tensors [B,T,d] — only one step/1000.
        collect_sims = self._collect_layer_sims
        self._collect_layer_sims = False  # reset immediately (before call, safe for all ranks)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            output_hidden_states=collect_sims,
        )
        hidden = outputs.last_hidden_state  # [B, K + T_text + T_audio, d]

        if collect_sims and outputs.hidden_states is not None:
            self._last_layer_cos_sims.clear()
            with torch.no_grad():
                # Layer-wise cos-sim between post-prefix text region and audio region.
                for layer_idx, layer_h in enumerate(outputs.hidden_states):
                    try:
                        h_f = layer_h.detach().float()
                        if T_text == 0 or h_f.shape[1] <= K + T_text:
                            continue
                        h_text = h_f[:, K:K + T_text, :].mean(dim=1)    # [B, d]
                        h_audio = h_f[:, K + T_text:, :].mean(dim=1)    # [B, d]
                        t_n = h_text / h_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        a_n = h_audio / h_audio.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        self._last_layer_cos_sims[layer_idx] = (t_n * a_n).sum(dim=-1).mean().item()
                    except Exception:
                        pass

        # 4. Apply output heads only on the audio region of the hidden states.
        # Labels tensors were built by DataCollator to span [text | audio] (length
        # T_text + T_audio) — they do NOT include the prefix. So we must drop the
        # prefix slice before applying the codebook/stop heads. Otherwise the
        # causal-shifted loss alignment breaks.
        hidden_no_prefix = hidden[:, K:, :]                             # [B, T_text + T_audio, d]
        logits_audio = [head(hidden_no_prefix) for head in self.codebook_heads]
        logits_stop = self.stop_head(hidden_no_prefix)

        # 5. Compute loss with causal shift
        loss = None
        if labels[0] is not None:
            total_loss = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)

            # Codebook CE losses — each head has its own vocab size
            for i, (logits, labels_i, vocab_size) in enumerate(zip(logits_audio, labels, self.vocab_sizes)):
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels_i[:, 1:].contiguous()
                # Guard against the all-(-100) microbatch case (MODEL_GUIDE §6.2): with
                # reduction='mean' and ignore_index=-100, a head whose shifted
                # labels are ALL -100 divides 0/0 -> NaN, poisoning total_loss
                # and every subsequent step. Mirror the stop-head guard below.
                # `sum()*0.0` keeps the head in the autograd graph (no DDP
                # unused-param warning) with zero gradient.
                if (shift_labels != -100).any():
                    ch_loss = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                else:
                    ch_loss = shift_logits.sum() * 0.0
                total_loss = total_loss + ch_loss
                self._last_losses[f"level_{i}"] = ch_loss.detach()

            # Stop BCE loss with manual mask
            shift_stop_logits = logits_stop[:, :-1, 0].contiguous()  # [B, T-1]
            shift_stop_labels = labels_stop[:, 1:].contiguous()       # [B, T-1]
            stop_mask = shift_stop_labels != -100
            if stop_mask.any():
                stop_loss = F.binary_cross_entropy_with_logits(
                    shift_stop_logits[stop_mask].float(),
                    shift_stop_labels[stop_mask].float(),
                    pos_weight=torch.tensor(
                        self.stop_pos_weight, device=hidden.device, dtype=torch.float32
                    ),
                )
                total_loss = total_loss + self.stop_loss_weight * stop_loss
                self._last_losses["stop"] = stop_loss.detach()
            else:
                self._last_losses["stop"] = torch.tensor(0.0, device=hidden.device)

            # Diversity (hinge variance) regularization on compressor queries.
            # Curriculum-weighted; diversity_loss already includes the current weight.
            if diversity_loss is not None:
                total_loss = total_loss + diversity_loss
                self._last_losses["diversity"] = diversity_loss.detach()

            # Supervised contrastive (SupCon) on compressor outputs.
            # supcon_term is already curriculum-scaled by `cur_w` in section 1c.
            if supcon_term is not None:
                total_loss = total_loss + supcon_term
                self._last_losses["supcon_weighted"] = supcon_term.detach()

            loss = total_loss
            self._last_losses["total"] = loss.detach()

        return MultiheadTTSOutput(
            loss=loss,
            logits_audio=logits_audio,
            logits_stop=logits_stop,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate to the Qwen3.5 backbone — HF Trainer calls this on `self.model`
        when `gradient_checkpointing=True`. Our top-level wrapper isn't a
        PreTrainedModel, so we forward the call to the inner stock backbone.
        Heads and audio embeddings stay outside the checkpointed region (their
        compute is cheap; activation memory is dominated by FFN intermediates)."""
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Mirror of gradient_checkpointing_enable — forwards to the backbone."""
        self.model.gradient_checkpointing_disable()

    def get_parameter_or_buffer(self, target: str):
        """Look up a parameter OR buffer by its state_dict name.

        This mirrors the transformers `PreTrainedModel` API that accelerate's
        FSDP2 state-dict loader (`fsdp2_load_full_state_dict`) calls for every
        entry. Our wrapper is a plain nn.Module, so without it accelerate falls
        back to `name.rsplit(".", 1)` — which crashes on the dot-less top-level
        names `null_prefix` and `audio_embed_scale`. `nn.Module.get_parameter`
        handles the dot-less case natively (empty module path = self)."""
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        raise AttributeError(f"{target!r} is neither a parameter nor a buffer of GepardModel")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        audio_heads: Dict[str, int],
        stop_loss_weight: float = 1.0,
        stop_pos_weight: float = 1.0,
        vc_config: Optional[VoiceCloningConfig] = None,
        codec: Optional[CodecConfig] = None,
        audio_embed_dim: int = 32,
        partial_rotary_factor: Optional[float] = None,
        **kwargs,
    ):
        """
        Load pretrained Qwen3.5 weights into GepardModel.

        Two-stage loading:
        1. Load base Qwen3_5ForCausalLM (includes lm_head we don't need)
        2. Copy backbone weights into our model, discard lm_head
        3. Initialize audio embeddings from text embedding distribution

        When `vc_config.enabled=True`, `codec` must be passed as well so the
        RefCompressor can be constructed with the right unfold/dequantize path.
        The compressor and null_prefix are initialized from scratch.

        `partial_rotary_factor` (when given) overrides whatever the backbone
        repo ships — forced into the config BEFORE the backbone is built so
        the rotary embedding is computed with it, and written both flat and
        inside `rope_parameters` (see `set_partial_rotary_factor`). RoPE is
        parameter-free, so the weight copy below is unaffected.
        """
        from .configuration import (
            effective_partial_rotary_factor,
            set_partial_rotary_factor,
        )

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if partial_rotary_factor is not None:
            effective = effective_partial_rotary_factor(config)
            if set_partial_rotary_factor(config, partial_rotary_factor):
                print(
                    f"partial_rotary_factor: backbone repo effective value "
                    f"{effective} → configured {float(partial_rotary_factor)} "
                    f"(synced flat + rope_parameters)"
                )

        # Create our model (randomly initialized)
        model = cls(
            config=config,
            audio_heads=audio_heads,
            stop_loss_weight=stop_loss_weight,
            stop_pos_weight=stop_pos_weight,
            vc_config=vc_config,
            codec=codec,
            audio_embed_dim=audio_embed_dim,
        )

        # Rename torch_dtype -> dtype if needed (transformers >=5.x)
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")

        # Load base model weights
        base_model = Qwen3_5ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Copy backbone weights (strict=False: our model has no lm_head, base has no audio layers)
        model.model.load_state_dict(base_model.model.state_dict(), strict=False)

        del base_model
        print(f"Loaded pretrained backbone from {pretrained_model_name_or_path}")

        # Initialize audio embeddings to match text embedding scale
        model._init_audio_embeddings()

        return model


# ─────────────────────────── config ↔ model bridge ───────────────────────────
# `config_from_model` introspects a built model to record every shape-driver;
# `build_model` reconstructs the model from a `GepardConfig` so its `state_dict`
# layout matches. See docs/MODEL_GUIDE.md §10.9 (self-describing checkpoints).


def config_from_model(
    model: GepardModel,
    *,
    special_tokens: Optional[Dict[str, int]] = None,
    text_repetition: Optional[Dict[str, Any]] = None,
    model_dtype: str = "bfloat16",
    codec: Optional[Dict[str, Any]] = None,
) -> "GepardConfig":
    """Capture a self-describing `GepardConfig` from a constructed model.

    Shape-critical fields are read from the live modules (so they always match the
    weights). `special_tokens` / `text_repetition` / `codec` are not (fully) held
    by the model (it is token-agnostic; codec geometry is introspectable only when
    VC is on) — the caller passes them from the tokens/text_layout/codec config at
    save time. Introspected codec shape facts override the passed values so the
    config can never disagree with the weights; the legacy `fsq_Levels` casing is
    normalized to `fsq_levels`.
    """
    from .configuration import (
        GepardConfig,
        effective_partial_rotary_factor,
        set_partial_rotary_factor,
    )

    backbone_config = model.config.to_dict()
    # Record the value the model actually computed RoPE with (the nested
    # rope_parameters copy — the flat attribute can be a stale mirror), and
    # sync both copies in the serialized backbone so downstream consumers
    # (vLLM reads flat, HF reads nested) can't disagree.
    rotary = effective_partial_rotary_factor(model.config)
    set_partial_rotary_factor(backbone_config, rotary)
    audio_heads = {name: int(v) for name, v in zip(model.channel_names, model.vocab_sizes)}

    codec = dict(codec or {})
    if "fsq_Levels" in codec:
        codec["fsq_levels"] = list(codec.pop("fsq_Levels"))

    voice_cloning: Dict[str, Any] = {"enabled": False}
    if getattr(model, "ref_compressor", None) is not None:
        rc = model.ref_compressor
        codec.update(
            num_layers=int(rc.num_layers_codec),
            fsq_levels=list(rc.fsq_levels),
            do_unfold=(not rc.do_unfold_in_forward),
        )
        supcon: Dict[str, Any] = {"enabled": bool(model.supcon_enabled)}
        if model.supcon_head is not None:
            supcon.update(
                use_projection=True,
                projection_hidden_dim=int(model.supcon_head.fc1.out_features),
                projection_dim=int(model.supcon_head.fc2.out_features),
            )
        else:
            supcon["use_projection"] = False
        voice_cloning = {
            "enabled": True,
            "compressor": {
                "num_queries": int(rc.num_queries),
                "num_layers": int(rc.num_blocks),
                "num_heads": int(rc.num_heads),
                "d_model": int(rc.d_model),
                "ffn_hidden_size_multiplier": int(rc.ffn_hidden // rc.d_model),
                "dropout": float(rc.dropout),
            },
            "training": {
                "cfg_dropout_prob": float(model.cfg_dropout_prob),
                "supcon": supcon,
                "diversity_loss": {"enabled": bool(model.diversity_enabled)},
            },
        }

    return GepardConfig(
        backbone_config=backbone_config,
        audio_heads=audio_heads,
        audio_embed_dim=int(model.audio_embed_dim),
        partial_rotary_factor=rotary,
        special_tokens=special_tokens,
        stop_loss_weight=float(model.stop_loss_weight),
        stop_pos_weight=float(model.stop_pos_weight),
        model_dtype=model_dtype,
        codec=codec,
        text_repetition=text_repetition,
        voice_cloning=voice_cloning,
    )


def build_model(
    cfg: "GepardConfig",
    attn_implementation: Optional[str] = None,
) -> GepardModel:
    """Reconstruct the model from a `GepardConfig` (fresh weights; matching shapes).

    Rebuilds the backbone from the nested config (reconciling `partial_rotary_factor`)
    and the schema dataclasses the constructor duck-types against.
    `attn_implementation` overrides the serialized backbone setting (inference may
    want eager where training used flash_attention_2).
    """
    from types import SimpleNamespace

    from ..config.schema import (
        CompressorConfig,
        DiversityLossConfig,
        SupConConfig,
        VCTrainingConfig,
        VoiceCloningConfig,
    )
    from .configuration import reconcile_backbone_config

    backbone_kwargs = dict(cfg.backbone_config)
    if attn_implementation is not None:
        # Drop any serialized attn setting so the override wins regardless of
        # which key form the transformers version wrote into to_dict().
        for k in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
            backbone_kwargs.pop(k, None)
        backbone_kwargs["attn_implementation"] = attn_implementation
    backbone = Qwen3_5TextConfig(**backbone_kwargs)
    reconcile_backbone_config(backbone, cfg)

    vc_config = None
    codec = None
    if cfg.vc_enabled:
        comp = cfg.voice_cloning.get("compressor", {})
        train = cfg.voice_cloning.get("training", {})
        sc = train.get("supcon", {})
        dv = train.get("diversity_loss", {})
        vc_config = VoiceCloningConfig(
            enabled=True,
            compressor=CompressorConfig(
                num_queries=int(comp["num_queries"]),
                num_layers=int(comp["num_layers"]),
                num_heads=int(comp["num_heads"]),
                d_model=comp.get("d_model"),
                ffn_hidden_size_multiplier=int(comp["ffn_hidden_size_multiplier"]),
                dropout=float(comp.get("dropout", 0.1)),
            ),
            training=VCTrainingConfig(
                cfg_dropout_prob=float(train.get("cfg_dropout_prob", 0.15)),
                supcon=SupConConfig(
                    enabled=bool(sc.get("enabled", False)),
                    use_projection=bool(sc.get("use_projection", True)),
                    projection_hidden_dim=int(sc.get("projection_hidden_dim", 128)),
                    projection_dim=int(sc.get("projection_dim", 128)),
                ),
                diversity_loss=DiversityLossConfig(enabled=bool(dv.get("enabled", False))),
            ),
        )
        # Duck-typed stand-in for the codec group: the compressor only reads
        # num_layers / fsq_levels / do_unfold.
        codec = SimpleNamespace(
            num_layers=int(cfg.codec["num_layers"]),
            fsq_levels=list(cfg.codec["fsq_levels"]),
            do_unfold=bool(cfg.codec.get("do_unfold", True)),
        )

    return GepardModel(
        config=backbone,
        audio_heads=dict(cfg.audio_heads),
        stop_loss_weight=cfg.stop_loss_weight,
        stop_pos_weight=cfg.stop_pos_weight,
        vc_config=vc_config,
        codec=codec,
        audio_embed_dim=cfg.audio_embed_dim,
    )
