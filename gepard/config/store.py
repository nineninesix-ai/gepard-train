"""Hydra ConfigStore registration, custom resolvers, and typed loaders.

Public API:
  load_train(overrides)   -> TrainConfig
  load_prepare(overrides) -> PrepareConfig
  load_dpo(overrides)     -> DPOConfig

Each loader composes the `conf/` tree with Hydra, merges it onto the structured
schema (type + required-field checks), converts to a typed dataclass, then runs
the cross-field validator.
"""

import os
from typing import List, Optional, Type, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from . import schema as S
from .validate import validate

# Absolute path to the repo's conf/ tree (gepard/config/ -> repo root -> conf/).
CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conf"))

T = TypeVar("T")


# ── resolvers ───────────────────────────────────────────────────────────────
def _derive_audio_heads(num_layers, fsq_levels) -> dict:
    """32 audio heads = codec `num_layers` × `fsq_levels` pattern (MODEL_GUIDE
    §2.3). Replaces the hand-written 32-entry map."""
    heads, idx = {}, 0
    for _ in range(int(num_layers)):
        for lvl in fsq_levels:
            heads[f"level_audio_{idx}"] = int(lvl)
            idx += 1
    return heads


def register() -> None:
    """Register resolvers + top-level schemas. Idempotent."""
    if not OmegaConf.has_resolver("gepard.audio_heads"):
        OmegaConf.register_new_resolver("gepard.audio_heads", _derive_audio_heads)
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=S.TrainConfig)
    cs.store(name="prepare_schema", node=S.PrepareConfig)
    cs.store(name="dpo_schema", node=S.DPOConfig)


# ── loaders ───────────────────────────────────────────────────────────────
def _compose(config_name: str, overrides: Optional[List[str]]) -> DictConfig:
    register()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def _typed(cfg: DictConfig, schema: Type[T]) -> T:
    """Merge onto the schema (types + MISSING check) and convert to a dataclass."""
    merged = OmegaConf.merge(OmegaConf.structured(schema), cfg)
    return OmegaConf.to_object(merged)  # raises on unresolved MISSING / bad type


def load_train(overrides: Optional[List[str]] = None) -> S.TrainConfig:
    obj = _typed(_compose("train", overrides), S.TrainConfig)
    validate(obj)
    return obj


def load_sft(overrides: Optional[List[str]] = None) -> S.TrainConfig:
    """Compose the SFT/LoRA fine-tune entry (`conf/sft.yaml`). Same TrainConfig
    schema and validator as pretrain — only the entry file's default groups
    differ (finetune trainer/voice_cloning/finetune/text_layout)."""
    obj = _typed(_compose("sft", overrides), S.TrainConfig)
    validate(obj)
    return obj


def load_prepare(overrides: Optional[List[str]] = None) -> S.PrepareConfig:
    obj = _typed(_compose("prepare", overrides), S.PrepareConfig)
    validate(obj)
    return obj


def load_dpo(overrides: Optional[List[str]] = None) -> S.DPOConfig:
    obj = _typed(_compose("dpo", overrides), S.DPOConfig)
    validate(obj)
    return obj
