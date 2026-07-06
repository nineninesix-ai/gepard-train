"""Gepard model layer: the multihead TTS model, its submodules (ref compressor,
LoRA, codec ops, losses/) and the self-describing `GepardConfig`.
See docs/MODEL_GUIDE.md §10.9 (self-describing checkpoints).

The heavy `modeling` imports (torch / transformers model classes) are lazy so
`from gepard.model import GepardConfig` stays cheap.
"""

from .configuration import (
    GEPARD_CONFIG_NAME,
    GepardConfig,
    load_gepard_config,
    patch_config_json_rotary,
    reconcile_backbone_config,
    save_gepard_config,
    set_partial_rotary_factor,
)

__all__ = [
    "GEPARD_CONFIG_NAME",
    "GepardConfig",
    "load_gepard_config",
    "save_gepard_config",
    "reconcile_backbone_config",
    "set_partial_rotary_factor",
    "patch_config_json_rotary",
    "build_model",
    "config_from_model",
    "GepardModel",
    "RefCompressor",
]


def __getattr__(name):  # lazy re-export of the torch-heavy modules
    if name in ("build_model", "config_from_model", "GepardModel"):
        from . import modeling
        return getattr(modeling, name)
    if name == "RefCompressor":
        from .ref_compressor import RefCompressor
        return RefCompressor
    raise AttributeError(name)
