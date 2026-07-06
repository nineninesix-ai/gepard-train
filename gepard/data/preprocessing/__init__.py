"""Dataset preprocessing: HF-source ingestion, tokenization, codec unfolding.

Modules:
  - processor:        DatasetProcessor + per-shard TrainDataPreProcessor
  - prepare:          CLI entry (`python -m gepard.data.preprocessing.prepare`)
  - text_repetition:  adaptive short-text repetition layout (shared with inference)
  - build_keep_index: short-utterance keep-index builder (CLI)

Re-exports are lazy (PEP 562): `text_repetition` is stdlib-only and imported by
the inference runner, so this __init__ must not eagerly pull `processor` (which
needs datasets/omegaconf — absent in the lean inference env).
"""

_EXPORTS = {
    "DatasetProcessor": "processor",
    "ItemDataset": "processor",
    "TrainDataPreProcessor": "processor",
    "load_config": "processor",
    "TextRepeater": "text_repetition",
    "TextRepetitionConfig": "text_repetition",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
