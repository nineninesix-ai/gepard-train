"""Shared training-side model plumbing (ROADMAP Stage 2, §B2.2/§C1).

The SFT trainer and the DPO loop used to each carry their own copy of "load a
TTS checkpoint into a built model" and "freeze everything + inject backbone
LoRA". Both live here now; the engines stay thin.
"""

from typing import List

import torch.nn as nn
from safetensors.torch import load_file

from ..model.checkpoint_io import normalize_scalar_shapes, resolve_safetensors
from ..model.lora import inject_lora


def load_tts_checkpoint(model: nn.Module, checkpoint: str, tag: str = "checkpoint") -> None:
    """Overwrite `model` weights from a TTS checkpoint (local dir/file or HF repo).

    Uses `load_state_dict(strict=False)` so architecture mismatches (e.g. a
    checkpoint from an older build carrying extra aux-module keys) don't
    crash — unexpected keys are ignored and missing keys stay at their
    current init.
    """
    sf_path = resolve_safetensors(checkpoint)
    print(f"[{tag}] Loading TTS checkpoint from {sf_path}")
    state_dict = load_file(str(sf_path), device="cpu")
    reshaped = normalize_scalar_shapes(state_dict, model)
    if reshaped:
        print(f"[{tag}]  legacy 0-dim params reshaped to match model: {reshaped}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{tag}]  missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[{tag}]  unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[{tag}] Checkpoint loaded — {len(state_dict)} tensors")


def inject_backbone_lora(model: nn.Module, lora_cfg, tag: str = "lora") -> List[str]:
    """Freeze the whole model, then inject LoRA adapters into the backbone.

    Everything (audio heads, stop_head, ref_compressor, null_prefix) stays
    frozen — only the injected lora_A/lora_B in the last N backbone layers
    train. The adapter names live under `model.layers.*`, so any backbone
    LR group picks them up automatically. Callers that additionally train
    heads (DPO's stop_head) re-enable those grads afterwards.

    `lora_cfg` is duck-typed: `finetune.lora` and `dpo.training.lora` both fit.
    Returns the list of injected module names.
    """
    model.requires_grad_(False)
    injected = inject_lora(
        model.model,                     # stock Qwen3.5 backbone (has .layers)
        target_modules=list(lora_cfg.target_modules),
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        last_n_layers=lora_cfg.last_n_layers,
    )
    print(
        f"[{tag}] injected {len(injected)} modules "
        f"(r={lora_cfg.rank}, α={lora_cfg.alpha}, dropout={lora_cfg.dropout}, "
        f"last {lora_cfg.last_n_layers} layers, targets={list(lora_cfg.target_modules)})"
    )
    return injected
