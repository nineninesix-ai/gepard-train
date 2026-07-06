"""
Minimal LoRA for the Qwen3.5 backbone — no peft dependency, explicit
inject / save / load / merge so the adapter lifecycle is fully under our
control (DPO keeps the adapter separate; export merges for inference).

LoRA params are kept in fp32 (they are tiny) while the frozen base stays
bf16; forward casts the activations into the adapter dtype and back.
"""

import math
import re
from typing import Dict, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank delta: y = Wx + (alpha/r)·B(Ax)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero → injection is an exact no-op until training moves it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xa = self.dropout(x).to(self.lora_A.dtype)
        delta = (xa @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def merge_into_base(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight += delta.to(self.base.weight.dtype)


def inject_lora(
    backbone: nn.Module,             # model.model (Qwen3_5TextModel with .layers)
    target_modules: List[str],
    rank: int,
    alpha: int,
    dropout: float,
    last_n_layers: int,
) -> List[str]:
    """Replace target Linears in the last N transformer layers with LoRALinear.

    Returns the list of injected module paths (relative to the backbone).
    """
    num_layers = len(backbone.layers)
    first = max(0, num_layers - last_n_layers)
    pat = re.compile(r"^layers\.(\d+)\.")
    injected = []
    for name, module in list(backbone.named_modules()):
        m = pat.match(name)
        if m is None or int(m.group(1)) < first:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in target_modules or not isinstance(module, nn.Linear):
            continue
        parent = backbone.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        injected.append(name)
    if not injected:
        raise RuntimeError(f"inject_lora: no modules matched {target_modules} in last {last_n_layers} layers")
    return injected


def lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    missing = [k for k in state if k not in dict(model.named_parameters())]
    if missing:
        raise KeyError(f"load_lora_state_dict: {len(missing)} keys not found, e.g. {missing[:3]}")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def merge_all_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and swap LoRALinear back to nn.Linear.

    After this the model state_dict has the original key layout (no lora_* keys)
    and can be saved as a plain inference checkpoint.
    """
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.merge_into_base()
                setattr(module, child_name, child.base)
                n += 1
    return n
