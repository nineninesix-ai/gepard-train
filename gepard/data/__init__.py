"""Gepard data layer: batch collation and reference-voice sampling.

Re-exports are lazy (PEP 562, same pattern as `gepard.model`) so that importing
a light submodule — e.g. `gepard.data.preprocessing.text_repetition` from the
inference runner — does not drag `datasets` into environments that don't
install it (the [inference] extra deliberately omits datasets/wandb).
"""

_EXPORTS = {
    "DataCollator": "collator",
    "NULL_SPEAKER_INT": "collator",
    "NULL_SPEAKER_SENTINEL": "collator",
    "ReferenceSamplingDataset": "sampling",
    "SpeakerBucketBatchSampler": "sampling",
    "prepare_dataset": "sampling",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
