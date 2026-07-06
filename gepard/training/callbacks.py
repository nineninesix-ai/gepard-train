"""
Training callbacks for Gepard.
"""

import contextlib
import os
from pathlib import Path

import torch
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import TrainerCallback


def _is_main_process() -> bool:
    # Global rank, not LOCAL_RANK: on multi-node runs every node has a
    # LOCAL_RANK==0 process, but only global rank 0 should write files/wandb.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


try:
    from torch.distributed.tensor import DTensor
except ImportError:  # torch too old for FSDP2 — isinstance() is then always False
    class DTensor:  # pragma: no cover
        pass


def _full_tensor(t: torch.Tensor) -> torch.Tensor:
    """Materialize a parameter that FSDP2's `fully_shard` turned into a DTensor.

    `.full_tensor()` is a COLLECTIVE (all-gather): every rank must call it, in
    the same order — so materialize on all ranks and only *use* the result on
    rank 0. Plain tensors (single GPU, DDP, inside an FSDP1 summon context)
    pass through untouched.
    """
    if isinstance(t, DTensor):
        return t.full_tensor()
    return t


class MultiheadLossLogCallback(TrainerCallback):
    """Logs per-head losses from model._last_losses directly to wandb."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        unwrapped = model
        while hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
        if hasattr(unwrapped, '_last_losses') and unwrapped._last_losses:
            extra = {
                f"loss/{k}": v.item() if torch.is_tensor(v) else v
                for k, v in unwrapped._last_losses.items()
            }
            # Log directly to wandb (WandbCallback already fired at this point)
            if _is_main_process() and wandb.run is not None:
                wandb.log(extra, step=state.global_step)
            # Also inject into logs for console output
            if logs is not None:
                logs.update(extra)


class DiagnosticsCallback(TrainerCallback):
    """Logs diagnostic metrics for monitoring training health.

    Cheap metrics (every log step):
      - grad_norm/emb_cb{k}: gradient norm per audio embedding table
      - grad_norm/text_emb: gradient norm of text embedding table
      - norm/text_emb: mean L2 norm of text embeddings
      - norm/audio_emb: mean L2 norm of audio frame embeddings (after averaging)

    Expensive metrics (every `expensive_every` steps):
      - effective_rank/emb_cb{k}: effective rank of each audio embedding table
      - effective_rank/text_emb: effective rank of text embedding (subsampled)
      - cos_to_init/text_emb: row-wise cosine similarity to initial weights
      - drift/text_emb: relative Frobenius drift from initial weights
      - cosine_sim/text_audio: mean cosine similarity between text and audio embeddings
    """

    def __init__(self, expensive_every: int = 1000):
        self.expensive_every = expensive_every
        self._grad_sq: dict = {}       # LOCAL squared grad norms, read at on_pre_optimizer_step
        self._grads_sharded = False    # True once a DTensor grad is seen (FSDP2)

    def _unwrap(self, model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def _record_grad(self, name, grad):
        if isinstance(grad, DTensor):
            # FSDP2: p.grad is the reduce-scattered DTensor — the local shard
            # is this rank's partition of the true global gradient, so
            # Σ_ranks ‖shard‖² = ‖g‖² (recovered by the all_reduce in on_log).
            self._grads_sharded = True
            grad = grad.to_local()
        self._grad_sq[name] = float(grad.float().pow(2).sum().item())

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """Capture grad norms of the tracked params directly from `p.grad`.

        This event fires after gradient accumulation/unscaling and before
        clipping and the step — the only point where grads are guaranteed
        present under every wrapper. Backward Tensor hooks (the previous
        mechanism) NEVER fire under FSDP2: autograd flows through
        fully_shard's unsharded proxies and grads land in `p.grad` without
        touching hooks on the sharded leaf. Reading p.grad here also means
        the values reflect the full accumulated gradient, not just the last
        microbatch. ~50 tiny .item() syncs per step — negligible next to a
        TTS training step (and the project logs every step anyway).
        """
        for name, p in getattr(self, "_tracked_params", {}).items():
            if p.grad is not None:
                self._record_grad(name, p.grad)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Collect the params whose grad norms we track, snapshot text embeddings."""
        if model is None:
            return
        m = self._unwrap(model)

        # Tracked params, read at on_pre_optimizer_step. Per-layer cosine sim
        # is collected via output_hidden_states in model.forward (set by the
        # _collect_layer_sims flag) — no layer-level forward hooks needed.
        self._tracked_params = {}

        for k in range(m.num_codebook_heads):
            w = m.audio_embeddings[k].weight
            if w.requires_grad:
                self._tracked_params[f"emb_{k}"] = w

        # Voice cloning: track gradient norm of the learnable null_prefix (when present).
        if getattr(m, "null_prefix", None) is not None and m.null_prefix.requires_grad:
            self._tracked_params["null_prefix"] = m.null_prefix

        # Per-param tracking for RefCompressor — aggregated on_log as sqrt(Σ ‖g_i‖²).
        if getattr(m, "ref_compressor", None) is not None:
            for name, p in m.ref_compressor.named_parameters():
                if p.requires_grad:
                    self._tracked_params[f"ref_compressor.{name}"] = p

        # Snapshot initial text embedding (subsampled for memory).
        # Stored on rank 0 only, on CPU, fp32. Deterministic row sampling
        # so cos/drift comparisons are consistent across all expensive steps.
        # Materialization runs on ALL ranks: summon_full_params (FSDP1) and
        # _full_tensor (FSDP2) are collectives — rank-gating them would hang.
        fsdp_ctx = (
            FSDP.summon_full_params(model, writeback=False)
            if isinstance(model, FSDP)
            else contextlib.nullcontext()
        )
        with fsdp_ctx, torch.no_grad():
            m_full = self._unwrap(model)
            W = _full_tensor(m_full.model.embed_tokens.weight)
            if _is_main_process():
                V = W.shape[0]
                n_sample = min(8192, V)
                g = torch.Generator().manual_seed(0)
                self._text_emb_idx = torch.randperm(V, generator=g)[:n_sample]
                self._text_emb_init = W[self._text_emb_idx].detach().float().cpu().clone()
                self._text_emb_init_fro = self._text_emb_init.norm().item()

        # Text embedding gradient, tracked like the audio tables.
        if m.model.embed_tokens.weight.requires_grad:
            self._tracked_params["text_emb"] = m.model.embed_tokens.weight

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        m = self._unwrap(model)
        metrics = {}

        # --- Cheap: gradient norms captured at on_pre_optimizer_step ---
        # _grad_sq holds LOCAL squared norms. Under FSDP2 each rank holds the
        # reduce-scattered shard of every gradient, so the global norm is
        # sqrt(Σ_ranks shard²) — recovered with ONE batched all_reduce over
        # all keys. The decision to enter the collective (distributed on +
        # sharded grads + non-empty key set) is rank-uniform, so no rank can
        # hang waiting for the others. Under DDP / single GPU p.grad was the
        # full accumulated gradient — used as-is, no collective.
        distributed_on = torch.distributed.is_available() and torch.distributed.is_initialized()
        grad_sq = dict(self._grad_sq)
        if distributed_on and self._grads_sharded and grad_sq:
            keys = sorted(grad_sq)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            t = torch.tensor([grad_sq[k] for k in keys], device=device, dtype=torch.float32)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            grad_sq = dict(zip(keys, t.tolist()))

        for k in range(m.num_codebook_heads):
            if f"emb_{k}" in grad_sq:
                metrics[f"grad_norm/emb_{k}"] = grad_sq[f"emb_{k}"] ** 0.5

        if "text_emb" in grad_sq:
            metrics["grad_norm/text_emb"] = grad_sq["text_emb"] ** 0.5

        if grad_sq.get("null_prefix", 0.0) > 0:
            metrics["grad_norm/null_prefix"] = grad_sq["null_prefix"] ** 0.5

        # Whole-compressor norm from per-param hook values: ‖g‖ = sqrt(Σ ‖g_i‖²).
        ref_sq = sum(v for k, v in grad_sq.items() if k.startswith("ref_compressor."))
        if ref_sq > 0:
            metrics["grad_norm/ref_compressor"] = ref_sq ** 0.5

        # --- Cheap: embedding / prefix diagnostics (stored in forward) ---
        if hasattr(m, '_last_diagnostics') and m._last_diagnostics:
            for k, v in m._last_diagnostics.items():
                # Internal split markers, not metrics.
                if k in ("T_text", "T_prefix"):
                    continue
                # Route keys into legible wandb namespaces by prefix.
                if k.startswith("norm_"):
                    metrics[f"norm/{k[len('norm_'):]}"] = v
                elif k == "cfg_dropout_rate_observed":
                    metrics["vc/cfg_dropout_rate_observed"] = v
                elif k.startswith("query_") or k.startswith("null_") or k.startswith("supcon_"):
                    metrics[f"vc/{k}"] = v
                elif k == "forced_null_rate":
                    metrics["vc/forced_null_rate"] = v
                else:
                    metrics[k] = v

        # Arm per-layer cosine sim collection so the *next* forward pass captures hidden_states.
        # We need on_log to fire on step N-1 to arm for step N. This is guaranteed only when
        # logging_steps == 1. If logging_steps > 1 and doesn't align, arm on the current step
        # instead (data will appear in wandb one expensive_every interval later, but no skip).
        # Runs on all ranks (not guarded by _is_main_process) so all FSDP shards collect.
        next_is_expensive = state.global_step > 0 and (state.global_step + 1) % self.expensive_every == 0
        cur_is_expensive  = state.global_step > 0 and state.global_step % self.expensive_every == 0
        if next_is_expensive or cur_is_expensive:
            m._collect_layer_sims = True

        # --- Expensive metrics (every N steps, computed on main process) ---
        if state.global_step > 0 and state.global_step % self.expensive_every == 0:
            # FSDP1: summon_full_params gives full weights inside the context.
            # FSDP2: every `_full_tensor` is an all-gather. Both are collectives,
            # so materialization runs on ALL ranks in the same order; only
            # rank 0 computes and logs.
            fsdp_ctx = (
                FSDP.summon_full_params(model, writeback=False)
                if isinstance(model, FSDP)
                else contextlib.nullcontext()
            )
            with fsdp_ctx, torch.no_grad():
                m_full = self._unwrap(model)
                audio_ws = [
                    _full_tensor(m_full.audio_embeddings[k].weight)
                    for k in range(m_full.num_codebook_heads)
                ]
                W_text = _full_tensor(m_full.model.embed_tokens.weight)
                proj_params = {
                    n: _full_tensor(p)
                    for n, p in m_full.audio_embed_proj.named_parameters()
                }
                if _is_main_process():
                    # Effective rank of each embedding table
                    for k, w_k in enumerate(audio_ws):
                        w = w_k.float()
                        try:
                            s = torch.linalg.svdvals(w)
                            p = s / s.sum()
                            p = p[p > 1e-10]  # avoid log(0)
                            eff_rank = torch.exp(-(p * p.log()).sum()).item()
                            metrics[f"effective_rank/emb_{k}"] = eff_rank
                        except Exception:
                            pass

                    # Text embedding degradation metrics (vs init snapshot, same 8192 rows).
                    if hasattr(self, "_text_emb_init"):
                        W_now = W_text[self._text_emb_idx].float().cpu()
                        W_init = self._text_emb_init

                        # Row-wise cosine to init (per-token, averaged).
                        W_now_n  = W_now  / W_now.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        W_init_n = W_init / W_init.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        metrics["cos_to_init/text_emb"] = (W_now_n * W_init_n).sum(dim=-1).mean().item()

                        # Relative Frobenius drift.
                        metrics["drift/text_emb"] = (
                            (W_now - W_init).norm() / max(self._text_emb_init_fro, 1e-8)
                        ).item()

                        # Effective rank — detect subspace collapse.
                        try:
                            s = torch.linalg.svdvals(W_now)
                            p = s / s.sum()
                            p = p[p > 1e-10]
                            metrics["effective_rank/text_emb"] = torch.exp(-(p * p.log()).sum()).item()
                        except Exception:
                            pass

                    # Global cosine similarity between text and audio embeddings.
                    # Audio embedding tables are small per-codebook (d_small), not
                    # hidden-size — the hidden-space audio frame embedding is their
                    # concat passed through audio_embed_proj. Feed the per-channel
                    # mean codes through the projection to get a comparable [d] vector.
                    text_emb_sample = W_text[:1000].float().mean(dim=0)   # [d]
                    proj_dtype = next(iter(proj_params.values())).dtype
                    chan_means = torch.cat(
                        [w.mean(dim=0) for w in audio_ws]
                    ).to(proj_dtype)                                      # [N * d_small]
                    # functional_call swaps in the materialized full weights, so
                    # the Sequential runs on plain tensors even when the module's
                    # registered params are still DTensor shards (FSDP2).
                    audio_emb_mean = torch.func.functional_call(
                        m_full.audio_embed_proj, proj_params, (chan_means,)
                    ).float()                                             # [d]
                    a_n = audio_emb_mean / audio_emb_mean.norm().clamp(min=1e-8)
                    t_n = text_emb_sample / text_emb_sample.norm().clamp(min=1e-8)
                    metrics["cosine_sim/text_audio"] = (t_n * a_n).sum().item()

            # Per-layer text↔audio cosine sim (from forward hooks, already computed as scalars)
            if _is_main_process() and m._last_layer_cos_sims:
                for layer_idx, cos_val in sorted(m._last_layer_cos_sims.items()):
                    metrics[f"cosine_sim/layer_{layer_idx:02d}/text_audio"] = cos_val

        if metrics and _is_main_process():
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
            if logs is not None:
                logs.update(metrics)


class EagerOptimizerStateCallback(TrainerCallback):
    """Materialize optimizer state for EVERY param on the first step.

    AdamW creates per-param state lazily, on the first step where the param
    carries a gradient. Params whose loss term is warmup-gated — supcon_head.*
    until `supcon.warmup_start` — therefore have no entry in the optimizer
    state of early checkpoints. The plain torch resume path tolerated that,
    but FSDP2 resumes through torch.distributed.checkpoint's
    `set_optimizer_state_dict`, which requires state for every param in the
    param groups and KeyErrors on the gap.

    Giving each grad-less param a zero gradient on the FIRST optimizer step
    makes AdamW materialize canonical state (zero moments — a no-op for the
    update; note a nonzero trainer.weight_decay would decay such params once
    on that step, the project runs 0.0). Later steps: Trainer's
    zero_grad(set_to_none=True) returns the grads to None and the optimizer
    skips those params again, but the state persists into every checkpoint.
    """

    def __init__(self):
        self._done = False

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        if self._done or optimizer is None:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
        self._done = True


class TokenizerSaveCallback(TrainerCallback):
    """Saves tokenizer alongside model checkpoints (main process only)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        if not _is_main_process():
            return control
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = Path(args.output_dir) / checkpoint_folder
        if output_dir.exists():
            self.tokenizer.save_pretrained(output_dir)
        return control


class GepardConfigSaveCallback(TrainerCallback):
    """Writes `gepard_config.json` into each periodic `checkpoint-N` dir.

    HF Trainer's own save covers only the state dict (the model is a bare
    nn.Module, not a PreTrainedModel), so without this the periodic checkpoints
    are not self-describing — only the final `save_model` export would be.
    Takes a zero-arg provider (`GepardTrainer.build_gepard_config`) rather
    than a prebuilt config so a LoRA merge or freeze between saves can't stale it.
    The provider returns None while un-merged LoRA adapters are live — a
    vanilla config would misdescribe the adapter state_dict layout, so those
    periodic checkpoints are deliberately NOT stamped (the final merged export
    is).
    """

    def __init__(self, config_provider):
        self.config_provider = config_provider

    def on_save(self, args, state, control, **kwargs):
        if not _is_main_process():
            return control
        import json

        from ..model.configuration import save_gepard_config, set_partial_rotary_factor

        cfg = self.config_provider()
        if cfg is None:
            return control
        output_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if output_dir.exists():
            save_gepard_config(cfg, output_dir)
            # Backbone config.json alongside — serving engines (vLLM) need it,
            # with partial_rotary_factor duplicated flat + in rope_parameters.
            backbone = dict(cfg.backbone_config)
            set_partial_rotary_factor(backbone, cfg.partial_rotary_factor)
            with open(output_dir / "config.json", "w") as f:
                json.dump(backbone, f, indent=2, sort_keys=False)
                f.write("\n")
        return control


class TrainingMetadataCallback(TrainerCallback):
    """Freezes the resolved training config + provenance into each checkpoint.

    Writes ``training_metadata.json`` (stage, step, epoch, UTC timestamp, base
    model, wandb run, and the full resolved ``cfg``) next to the weights on
    every periodic save. Unlike re-reading ``conf/`` at publish time, this
    snapshot captures CLI/experiment overrides and can never drift when the YAML
    tree is edited later. Rank-0 only; a passive dump with no effect on training.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def on_save(self, args, state, control, **kwargs):
        if not _is_main_process():
            return control
        from ..logging.model_card import write_training_metadata

        output_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if output_dir.exists():
            write_training_metadata(
                self.cfg,
                output_dir,
                stage=getattr(self.cfg.run, "stage", "pretrain"),
                global_step=state.global_step,
                epoch=state.epoch,
            )
        return control
