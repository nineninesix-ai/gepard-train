"""Cross-field configuration validation.

`validate(cfg)` runs once after composition at every entry point and raises
`ConfigError` naming the offending keys. Rules that need the backbone/checkpoint
(compressor width, checkpoint token match, LoRA depth) are re-checked at model
build; this pass covers the pure-config invariants. Intra-group rules live in
each dataclass; required fields are enforced by the schema (MISSING). See
docs/MODEL_GUIDE.md for the *why* behind each coupling (batch composition §4.3,
speaker policy §4.2)."""

import os
from typing import Union

from . import schema as S

Config = Union[S.TrainConfig, S.PrepareConfig, S.DPOConfig]


class ConfigError(ValueError):
    """Raised when a cross-field configuration invariant is violated."""


_DTYPES = {"bfloat16", "float16", "float32"}


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def require_dataset_built(cfg) -> None:
    """Runtime guards for the train/SFT entry points — filesystem preconditions,
    checked before the model loads so a typo'd path fails in seconds, not after
    a multi-GB download. Kept out of `validate()` so pure-config checks never
    touch the filesystem.

    Checks: the built dataset (`data.train_dataset_path`), the optional
    `trainer.keep_index_path` (.npy consumed by `prepare_dataset`), and — only
    when it is explicitly path-like — `finetune.checkpoint_path`."""
    path = cfg.data.train_dataset_path
    if not os.path.isdir(path):
        raise ConfigError(
            f"dataset not built at {path!r} — run the prepare step (`make dataset`) first"
        )
    if not os.path.exists(os.path.join(path, "dataset_info.json")):
        raise ConfigError(
            f"{path!r} exists but is not a saved HF dataset (no dataset_info.json)"
        )

    keep_index = cfg.trainer.keep_index_path
    if keep_index and not os.path.isfile(keep_index):
        raise ConfigError(
            f"trainer.keep_index_path {keep_index!r} does not exist "
            "(build it first, or unset for full-dataset training)"
        )

    # `finetune.checkpoint_path` may be an HF Hub repo id ("org/name") — those
    # are resolved by `resolve_checkpoint_file` at load time and must NOT be
    # existence-checked here. Only reject strings that explicitly claim to be
    # local paths (absolute, ./, ../ or ~) yet point at nothing.
    ckpt = cfg.finetune.checkpoint_path
    if ckpt and ckpt.startswith(("/", "./", "../", "~")) and not os.path.exists(
        os.path.expanduser(ckpt)
    ):
        raise ConfigError(
            f"finetune.checkpoint_path {ckpt!r} looks like a local path but does not exist"
        )


# Mirror of gepard.data.preprocessing.processor.SINGLETON_POLICIES — duplicated
# as a literal so this module stays import-light (the processor pulls in the
# datasets stack). The processor re-checks at run time, so drift fails loudly.
_SINGLETON_POLICIES = {"remove", "null_prefix"}


def _validate_common(cfg: Config) -> None:
    codec = cfg.codec
    _check(codec.frame_rate_hz > 0, "codec.frame_rate_hz must be > 0")
    _check(len(codec.fsq_levels) > 0, "codec.fsq_levels must be non-empty")

    tl = cfg.text_layout
    _check(
        0.0 <= tl.mixed_keep_prob <= 1.0,
        f"text_layout.mixed_keep_prob must be in [0, 1], got {tl.mixed_keep_prob}",
    )
    if tl.enabled:
        _check(tl.target_text_tokens >= 1, "text_layout.target_text_tokens must be >= 1")
        _check(tl.max_repeats >= 1, "text_layout.max_repeats must be >= 1")


def _validate_data(data) -> None:
    """Rules for the `data` group (train + prepare both carry it)."""
    _check(
        data.singleton_policy in _SINGLETON_POLICIES,
        f"data.singleton_policy must be one of {sorted(_SINGLETON_POLICIES)}, "
        f"got {data.singleton_policy!r}",
    )
    _check(
        data.min_clips_per_speaker >= 1,
        f"data.min_clips_per_speaker must be >= 1, got {data.min_clips_per_speaker}",
    )
    if data.max_duration_sec is not None:
        _check(
            data.max_duration_sec > 0,
            f"data.max_duration_sec must be > 0 when set, got {data.max_duration_sec}",
        )


def _validate_lora(lora, prefix: str) -> None:
    """Shared sanity for FinetuneConfig.lora and DPOLoraConfig (same fields)."""
    _check(lora.rank >= 1, f"{prefix}.rank must be >= 1, got {lora.rank}")
    _check(lora.alpha > 0, f"{prefix}.alpha must be > 0, got {lora.alpha}")
    _check(
        0.0 <= lora.dropout < 1.0,
        f"{prefix}.dropout must be in [0, 1), got {lora.dropout}",
    )
    _check(len(lora.target_modules) > 0, f"{prefix}.target_modules must be non-empty")
    _check(
        lora.last_n_layers >= 1,
        f"{prefix}.last_n_layers must be >= 1, got {lora.last_n_layers}",
    )


def _validate_model_and_data(cfg) -> None:
    """Rules shared by train + dpo (both carry model/codec/voice_cloning)."""
    codec, model, vc = cfg.codec, cfg.model, cfg.voice_cloning

    # audio_heads must match codec geometry (num_layers × fsq_levels).
    per_frame = len(codec.fsq_levels) if codec.do_unfold else 1
    expected = codec.num_layers * per_frame
    _check(
        len(model.audio_heads) == expected,
        f"model.audio_heads has {len(model.audio_heads)} entries but codec implies "
        f"{expected} (num_layers={codec.num_layers} x {per_frame}).",
    )
    _check(model.dtype in _DTYPES, f"model.dtype must be one of {_DTYPES}")

    # SupCon needs the compressor path (MODEL_GUIDE §4.3).
    if vc.training.supcon.enabled:
        _check(vc.enabled, "supcon.enabled requires voice_cloning.enabled")


def validate(cfg: Config) -> Config:
    if isinstance(cfg, S.PrepareConfig):
        _validate_common(cfg)
        _validate_data(cfg.data)
        # prepare READS the source corpora — an empty list builds nothing.
        _check(
            len(cfg.data.hf_datasets) > 0,
            "data.hf_datasets must be non-empty for the prepare step",
        )
        return cfg

    _validate_common(cfg)

    if isinstance(cfg, S.TrainConfig):
        _validate_model_and_data(cfg)
        _validate_data(cfg.data)
        if cfg.finetune.lora.enabled:
            _validate_lora(cfg.finetune.lora, "finetune.lora")
        vc, data, tr = cfg.voice_cloning, cfg.data, cfg.trainer

        # dtype ↔ mixed precision must agree.
        _check(
            not (tr.bf16 and tr.fp16), "trainer.bf16 and trainer.fp16 cannot both be true"
        )
        _check(
            tr.bf16 == (cfg.model.dtype == "bfloat16"),
            "trainer.bf16 must match model.dtype==bfloat16",
        )

        if vc.enabled:
            # Voice cloning needs speaker-labelled data (MODEL_GUIDE §4.2).
            _check(
                data.add_speaker_id,
                "voice_cloning.enabled requires data.add_speaker_id=true",
            )
            supcon = vc.training.supcon
            if supcon.enabled:
                # Enough clips per speaker to guarantee K positives (MODEL_GUIDE §4.3).
                _check(
                    data.min_clips_per_speaker >= supcon.K,
                    f"data.min_clips_per_speaker ({data.min_clips_per_speaker}) must be "
                    f">= supcon.K ({supcon.K})",
                )
                # Batch composition P·K + M == per-device batch (MODEL_GUIDE §4.3).
                bs = tr.per_device_train_batch_size
                _check(
                    supcon.P * supcon.K + supcon.M == bs,
                    f"per_device_train_batch_size ({bs}) must equal P*K+M "
                    f"({supcon.P}*{supcon.K}+{supcon.M}={supcon.P * supcon.K + supcon.M})",
                )
        return cfg

    if isinstance(cfg, S.DPOConfig):
        _validate_model_and_data(cfg)
        # dpo-train always injects LoRA — no `enabled` gate to check.
        _validate_lora(cfg.training.lora, "dpo.training.lora")
        _check(cfg.training.beta > 0, f"dpo.training.beta must be > 0, got {cfg.training.beta}")
        # Single logprob floor — both stages must agree (MODEL_GUIDE §7.1).
        _check(
            cfg.pairs.p_floor == cfg.p_floor and cfg.training.p_floor == cfg.p_floor,
            "dpo.p_floor, pairs.p_floor and training.p_floor must be equal",
        )
        # Enough reference speakers for the requested fan-out.
        n_refs = len(cfg.sampling.ref_audios)
        if not cfg.sampling.speaker_pool and n_refs:
            _check(
                cfg.sampling.speakers_per_text <= n_refs,
                f"sampling.speakers_per_text ({cfg.sampling.speakers_per_text}) exceeds "
                f"len(ref_audios) ({n_refs})",
            )
        return cfg

    return cfg
