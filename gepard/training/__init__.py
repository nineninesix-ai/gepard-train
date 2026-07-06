"""Gepard training layer: SFT (pretrain + finetune) trainer, DPO loop, callbacks.

Re-exports are lazy (PEP 562): `gepard.training.dpo` must be importable in
venv_dpo, which has no wandb — an eager `from .sft import ...` here would pull
it (sft.py imports wandb at module level for the SFT run).
"""

_EXPORTS = {
    "GepardTrainer": "sft",
    "MultiheadTTSTrainer": "sft",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
