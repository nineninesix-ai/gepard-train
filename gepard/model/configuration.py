"""`GepardConfig` — the self-describing model config (HF `PretrainedConfig`).

This carries everything needed to reconstruct the model for inference without any
training YAML: the nested backbone config plus the TTS shape-drivers (audio heads,
audio_embed_dim, codec geometry, compressor dims, presence of VC /
supcon head). Written to a **separate `gepard_config.json`** so it never collides
with the backbone `config.json` that `AutoConfig` consumes. See docs/MODEL_GUIDE.md
§10.9 and the field roles in gepard/config/schema.py ([B] = model-build).
"""

import json
import os
from typing import Any, Dict, Optional

from transformers import PretrainedConfig

GEPARD_CONFIG_NAME = "gepard_config.json"


class GepardConfig(PretrainedConfig):
    model_type = "gepard"

    def __init__(
        self,
        backbone_config: Optional[Dict[str, Any]] = None,   # nested Qwen3.5 config dict
        audio_heads: Optional[Dict[str, int]] = None,       # ordered {channel: vocab}
        audio_embed_dim: int = 32,
        partial_rotary_factor: float = 1.0,                 # reconciled into the backbone on load
        special_tokens: Optional[Dict[str, int]] = None,    # bos_text/eot/bos_audio/tts_pad/...
        stop_loss_weight: float = 1.0,
        stop_pos_weight: float = 1.0,
        model_dtype: str = "bfloat16",
        codec: Optional[Dict[str, Any]] = None,             # {num_layers, fsq_levels, do_unfold, frame_rate_hz}
        text_repetition: Optional[Dict[str, Any]] = None,
        voice_cloning: Optional[Dict[str, Any]] = None,     # {enabled, compressor{...}, training{...}}
        **kwargs,
    ):
        self.backbone_config = dict(backbone_config or {})
        # JSON-canonical form so to_dict() is stable across save/load round-trips
        # (JSON stringifies dict keys; the backbone constructor re-ints them).
        if isinstance(self.backbone_config.get("id2label"), dict):
            self.backbone_config["id2label"] = {
                str(k): v for k, v in self.backbone_config["id2label"].items()
            }
        self.audio_heads = {k: int(v) for k, v in (audio_heads or {}).items()}
        self.audio_embed_dim = int(audio_embed_dim)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.special_tokens = dict(special_tokens or {})
        self.stop_loss_weight = float(stop_loss_weight)
        self.stop_pos_weight = float(stop_pos_weight)
        self.model_dtype = str(model_dtype)
        self.codec = dict(codec or {})
        self.text_repetition = dict(text_repetition or {})
        self.voice_cloning = dict(voice_cloning or {})
        # Note: extra keys from older files (e.g. the removed `aligner` block)
        # land in **kwargs and are stored as plain attributes by the parent —
        # old gepard_config.json files still load.
        super().__init__(**kwargs)

    def to_json_string(self, use_diff: bool = True) -> str:
        """Serialize WITHOUT sorting keys (parent uses sort_keys=True).

        `audio_heads` order is load-bearing — the model indexes heads
        positionally (§I1), and lexicographic sorting would put
        `level_audio_10` before `level_audio_2`, silently rewiring every head
        on reload.
        """
        import json

        config_dict = self.to_diff_dict() if use_diff else self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=False) + "\n"

    # ── convenience flags (mirror the model's conditional submodules) ──
    @property
    def vc_enabled(self) -> bool:
        return bool(self.voice_cloning.get("enabled", False))

    @property
    def supcon_head_present(self) -> bool:
        sc = (self.voice_cloning.get("training") or {}).get("supcon") or {}
        return self.vc_enabled and bool(sc.get("enabled")) and bool(sc.get("use_projection"))


def save_gepard_config(cfg: "GepardConfig", output_dir: str) -> str:
    """Write `gepard_config.json` into a checkpoint dir; returns the file path.

    `use_diff=False` serializes every field (not just the delta from defaults),
    so the file fully describes the model on its own.
    """
    path = os.path.join(str(output_dir), GEPARD_CONFIG_NAME)
    cfg.to_json_file(path, use_diff=False)
    return path


def load_gepard_config(checkpoint_path: str) -> Optional["GepardConfig"]:
    """Read `gepard_config.json` from a local checkpoint dir or an HF Hub repo.

    Returns None when the checkpoint has no gepard config (a pre-Stage-3
    checkpoint) so callers can fall back to the composed config tree.
    """
    from .checkpoint_io import resolve_checkpoint_file

    path = resolve_checkpoint_file(checkpoint_path, GEPARD_CONFIG_NAME, required=False)
    if path is None:
        return None
    return GepardConfig.from_json_file(path)


def set_partial_rotary_factor(backbone_cfg, value: float) -> bool:
    """Force `partial_rotary_factor` into BOTH places the ecosystem reads it.

    Since transformers 5.x the model computes RoPE from
    `config.rope_parameters["partial_rotary_factor"]` — the flat top-level
    attribute is a legacy mirror the constructor does NOT keep in sync (they
    can silently diverge; the stock backbone repo already ships nested=1.0
    with the flat copy defaulting to 0.25). vLLM, on the other hand, requires
    the flat top-level copy. So every write goes to both:

      - top-level `partial_rotary_factor`  → what vLLM reads
      - `rope_parameters["partial_rotary_factor"]` → what the HF model reads

    Accepts a `PretrainedConfig`/object or a plain dict. The nested key is
    only written when a `rope_parameters` dict already exists (a config
    without one predates the nested scheme and reads the flat copy).
    Returns True if anything changed.
    """
    target = float(value)
    changed = False
    if isinstance(backbone_cfg, dict):
        if backbone_cfg.get("partial_rotary_factor") != target:
            backbone_cfg["partial_rotary_factor"] = target
            changed = True
        rope = backbone_cfg.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("partial_rotary_factor") != target:
            rope["partial_rotary_factor"] = target
            changed = True
        return changed
    if getattr(backbone_cfg, "partial_rotary_factor", None) != target:
        backbone_cfg.partial_rotary_factor = target
        changed = True
    rope = getattr(backbone_cfg, "rope_parameters", None)
    if isinstance(rope, dict) and rope.get("partial_rotary_factor") != target:
        rope["partial_rotary_factor"] = target
        changed = True
    return changed


def effective_partial_rotary_factor(backbone_cfg, default: float = 1.0) -> float:
    """The `partial_rotary_factor` the model actually computes RoPE with.

    Transformers 5.x reads the nested `rope_parameters` copy; the flat
    top-level attribute is an unsynced legacy mirror (the stock backbone repo
    ships nested=1.0 while the flat copy defaults to 0.25). Nested wins,
    flat is the fallback. Accepts a `PretrainedConfig`/object or a dict.
    """
    if isinstance(backbone_cfg, dict):
        rope = backbone_cfg.get("rope_parameters")
        flat = backbone_cfg.get("partial_rotary_factor", default)
    else:
        rope = getattr(backbone_cfg, "rope_parameters", None)
        flat = getattr(backbone_cfg, "partial_rotary_factor", default)
    if isinstance(rope, dict) and "partial_rotary_factor" in rope:
        return float(rope["partial_rotary_factor"])
    return float(flat)


def reconcile_backbone_config(backbone_cfg, gepard_cfg: "GepardConfig") -> bool:
    """Patch the backbone config's `partial_rotary_factor` to our configured value.

    We deliberately intervene in the downloaded backbone `config.json`: our
    full-attention build expects a specific rotary coverage, so on load we force
    the backbone config to match `gepard_cfg.partial_rotary_factor` — flat AND
    nested (see `set_partial_rotary_factor`). Returns True if changed.
    """
    return set_partial_rotary_factor(backbone_cfg, gepard_cfg.partial_rotary_factor)


def patch_config_json_rotary(path: str, value: Optional[float] = None) -> bool:
    """Ensure a written backbone `config.json` carries the vLLM-compatible
    duplicated `partial_rotary_factor` (flat top-level + inside
    `rope_parameters`).

    `value=None` propagates the file's own effective value — the nested one
    when present (that's what the HF model computed with), else the flat one,
    else 1.0. An explicit `value` forces it into both spots. Rewrites the file
    only when something changed; returns True in that case.
    """
    with open(path) as f:
        cfg = json.load(f)
    if value is None:
        rope = cfg.get("rope_parameters") or {}
        value = rope.get("partial_rotary_factor",
                         cfg.get("partial_rotary_factor", 1.0))
    if not set_partial_rotary_factor(cfg, value):
        return False
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=False)
        f.write("\n")
    return True
