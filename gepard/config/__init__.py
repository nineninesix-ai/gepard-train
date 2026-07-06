"""Gepard configuration layer (Hydra + structured configs). See ROADMAP §H.

The dataclass schema (`schema`) is the single source for type/required checks,
the typed loaders, and (future) the checkpoint `config.json`. `store` wires the
`conf/` YAML tree through Hydra; `validate` enforces cross-field invariants.
"""

from . import schema
from .store import load_dpo, load_prepare, load_sft, load_train, register
from .validate import ConfigError, require_dataset_built, validate

__all__ = [
    "schema",
    "register",
    "load_train",
    "load_sft",
    "load_prepare",
    "load_dpo",
    "validate",
    "require_dataset_built",
    "ConfigError",
]
