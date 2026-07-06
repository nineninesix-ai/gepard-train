"""Training-provenance capture + Hugging Face model-card rendering.

Two halves of one story, kept together so the write side and the read side
never drift:

* ``write_training_metadata(cfg, output_dir, …)`` — called from the training
  **save path** (the periodic-checkpoint callback and the final export). It
  freezes the *resolved* training config — every value Hydra actually composed,
  **including CLI/experiment overrides that never touch the YAML files** — plus
  the stage, step and a UTC timestamp into ``training_metadata.json`` next to
  the weights. Because it is frozen with the checkpoint, editing ``conf/`` later
  can never make it lie: the checkpoint is the fact, not the current tree.

* ``render_model_card(checkpoint_dir, …)`` / ``write_model_card(...)`` — called
  from ``scripts/upload_to_hf.py`` at **publish time**. It reads that metadata
  back (plus the self-describing ``gepard_config.json``) and renders a
  self-contained ``README.md`` with proper Hugging Face YAML front-matter, so an
  engineer landing on the repo months later can see — at a glance and in full —
  what this checkpoint is and exactly how it was trained.

Nothing here changes training behaviour: the write is a passive JSON dump
alongside the config that already gets stamped into every checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TRAINING_METADATA_NAME = "training_metadata.json"
MODEL_CARD_NAME = "README.md"
GEPARD_CONFIG_NAME = "gepard_config.json"
_METADATA_SCHEMA_VERSION = 1


# ── metadata capture (train side) ────────────────────────────────────────────
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> Optional[str]:
    """Short HEAD of the training repo, best-effort (None outside a checkout)."""
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _wandb_run_id() -> Optional[str]:
    try:
        import wandb

        return wandb.run.id if getattr(wandb, "run", None) is not None else None
    except Exception:
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return None


def build_training_metadata(
    cfg,
    *,
    stage: Optional[str] = None,
    global_step: Optional[int] = None,
    epoch: Optional[float] = None,
    saved_at: Optional[str] = None,
) -> dict:
    """Assemble the provenance record for one checkpoint from the *resolved* cfg.

    ``cfg`` is a ``TrainConfig`` dataclass instance (post-Hydra composition), so
    ``dataclasses.asdict`` captures every field at its real value — overrides
    included. Model architecture is intentionally left to ``gepard_config.json``
    (stamped straight from the live model); this file owns the training recipe.
    """
    stage = stage if stage is not None else getattr(cfg.run, "stage", "pretrain")
    config = dataclasses.asdict(cfg)

    return {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "stage": stage,
        "saved_at": saved_at or _utc_now_iso(),
        "global_step": int(global_step) if global_step is not None else None,
        "epoch": round(float(epoch), 4) if epoch is not None else None,
        "base_model": cfg.model.backbone_id,
        # Set only for phases that start from a prior Gepard checkpoint (SFT/DPO).
        "finetuned_from": cfg.finetune.checkpoint_path or None,
        "wandb": {
            "project": cfg.trainer.wandb.project,
            "name": cfg.trainer.wandb.name,
            "entity": cfg.trainer.wandb.entity or None,
            "run_id": _wandb_run_id(),
        },
        "environment": {
            "gepard_git_commit": _git_commit(),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
        },
        "config": config,
    }


def write_training_metadata(cfg, output_dir, **kwargs) -> str:
    """Write ``training_metadata.json`` into ``output_dir``; returns its path."""
    meta = build_training_metadata(cfg, **kwargs)
    return _dump_training_metadata(meta, output_dir)


def build_dpo_training_metadata(
    cfg,
    *,
    global_step: Optional[int] = None,
    saved_at: Optional[str] = None,
) -> dict:
    """Provenance record for a DPO merged checkpoint.

    The DPO analogue of ``build_training_metadata``: DPO runs a manual loop on a
    ``DPOConfig`` (not ``TrainConfig`` / HF Trainer), so it can't go through the
    same builder. ``finetuned_from`` is the **policy checkpoint** DPO started
    from (``cfg.sampling.checkpoint`` — the SFT model), so the card credits the
    real parent and renders the SFT→DPO lineage instead of mislabelling the DPO
    model as a raw fine-tune of the base LM. Architecture stays owned by
    ``gepard_config.json`` (copied straight from that policy checkpoint).
    """
    t = cfg.training
    return {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "stage": "dpo",
        "saved_at": saved_at or _utc_now_iso(),
        "global_step": int(global_step) if global_step is not None else None,
        "epoch": None,
        "base_model": cfg.model.backbone_id,          # architectural base LM
        "finetuned_from": cfg.sampling.checkpoint,    # immediate parent (SFT policy)
        "wandb": {
            "project": t.wandb_project,
            "name": t.wandb_name or cfg.run_name,
            "entity": getattr(t, "entity", None),
            "run_id": _wandb_run_id(),
        },
        "environment": {
            "gepard_git_commit": _git_commit(),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
        },
        "config": dataclasses.asdict(cfg),
    }


def write_dpo_training_metadata(cfg, output_dir, **kwargs) -> str:
    """Write a DPO ``training_metadata.json`` into ``output_dir``; returns its path."""
    meta = build_dpo_training_metadata(cfg, **kwargs)
    return _dump_training_metadata(meta, output_dir)


def _dump_training_metadata(meta: dict, output_dir) -> str:
    path = os.path.join(str(output_dir), TRAINING_METADATA_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    return path


# ── read helpers (publish side) ──────────────────────────────────────────────
def load_training_metadata(checkpoint_dir) -> Optional[dict]:
    path = Path(checkpoint_dir) / TRAINING_METADATA_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── model-card rendering (publish side) ──────────────────────────────────────
def _get(d: Any, *keys, default=None):
    """Nested dict get: ``_get(cfg, 'trainer', 'wandb', 'name')``."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_dt(iso: Optional[str]) -> str:
    if not iso:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(iso)


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "—"


def _kv_table(rows) -> str:
    """Two-column markdown table from ``[(key, value), …]`` (skips None values)."""
    lines = ["| | |", "|---|---|"]
    for k, v in rows:
        if v is None or v == "":
            continue
        lines.append(f"| **{k}** | {v} |")
    return "\n".join(lines)


def _yaml_list(items) -> str:
    return "".join(f"\n  - {it}" for it in items)


def _is_hub_id(x: Optional[str]) -> bool:
    """True for an HF repo id (``org/name``); False for a local path or None."""
    return bool(x) and "/" in x and not x.startswith((".", "/", "~"))


def _hub_link(repo_or_path: Optional[str]) -> str:
    """Markdown link for a hub repo id (``org/name``); plain code for a path."""
    if not repo_or_path:
        return "—"
    if _is_hub_id(repo_or_path):
        return f"[`{repo_or_path}`](https://huggingface.co/{repo_or_path})"
    return f"`{repo_or_path}`"


def render_model_card(
    checkpoint_dir,
    *,
    repo_id: Optional[str] = None,
    uploaded_at: Optional[str] = None,
) -> str:
    """Render a full Hugging Face ``README.md`` for a checkpoint directory.

    Sources: ``training_metadata.json`` (the frozen recipe) and
    ``gepard_config.json`` (the self-describing architecture). Degrades
    gracefully when either is missing — a legacy checkpoint still gets a valid,
    honestly-hedged card rather than a crash.
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta = load_training_metadata(checkpoint_dir) or {}
    gcfg = _load_json(checkpoint_dir / GEPARD_CONFIG_NAME) or {}
    cfg = meta.get("config") or {}

    stage = meta.get("stage") or "unknown"
    stage_pretty = {
        "pretrain": "Pretrain",
        "sft": "SFT (fine-tune)",
        "dpo": "DPO",
    }.get(stage, stage.title())

    base_model = meta.get("base_model") or _get(gcfg, "backbone_config", "_name_or_path")
    finetuned_from = meta.get("finetuned_from")
    step = meta.get("global_step")
    epoch = meta.get("epoch")
    saved_at = _fmt_dt(meta.get("saved_at"))

    # ---- front-matter ----
    tags = ["gepard", "text-to-speech", "tts", "speech", "qwen3"]
    if stage != "unknown":
        tags.append(stage)
    fm_lines = [
        "---",
        "library_name: transformers",
        "license: apache-2.0",
        "pipeline_tag: text-to-speech",
    ]
    # base_model = the immediate parent for the Hub lineage graph: the checkpoint
    # we fine-tuned FROM (SFT/DPO) — NOT the stock backbone, which would misrepresent
    # an SFT model as a raw fine-tune of the base LM. Pretrain (no finetuned_from)
    # points at the backbone. Only a real hub id qualifies; a local base falls back
    # to the backbone with no (false) relation claim.
    if _is_hub_id(finetuned_from):
        fm_lines.append(f"base_model: {finetuned_from}")
        fm_lines.append("base_model_relation: finetune")
    elif _is_hub_id(base_model):
        fm_lines.append(f"base_model: {base_model}")
    fm_lines.append("tags:" + _yaml_list(tags))
    fm_lines.append("---")
    front_matter = "\n".join(fm_lines)

    # ---- header + provenance ----
    step_str = f" · step {_fmt_int(step)}" if step is not None else ""
    title = f"# Gepard TTS — {stage_pretty} checkpoint{step_str}"

    lead = (
        "**Gepard** is an autoregressive, decoder-only text-to-speech model built on a "
        "stock Qwen3.5 backbone. Speech is tokenised with a GroupFSQ neural codec "
        "(NVIDIA NeMo **NanoCodec**, 21.5 Hz, ~1.89 kbps) whose 8 packed codebooks are "
        "unfolded into 32 orthogonal channels and predicted in parallel by 32 tiny heads "
        "— so the model decodes on stock inference engines (e.g. vLLM) with a clean, "
        "single-pass KV-cache loop and no custom CUDA kernels."
    )

    prov_rows = [
        ("Stage", f"`{stage}` — {stage_pretty}"),
        ("Base checkpoint", _hub_link(base_model)),
        ("Fine-tuned from", _hub_link(finetuned_from) if finetuned_from else None),
        ("Training step", _fmt_int(step) if step is not None else None),
        ("Epoch", f"{epoch:g}" if epoch is not None else None),
        ("Saved", saved_at),
        ("Uploaded", _fmt_dt(uploaded_at) if uploaded_at else None),
        ("Precision", _get(cfg, "model", "dtype") or gcfg.get("model_dtype")),
    ]
    provenance = (
        "> [!NOTE]\n> **Provenance** — this card is generated automatically from the "
        "training metadata frozen into the checkpoint at save time.\n\n"
        + _kv_table(prov_rows)
    )

    # ---- architecture ----
    bb = gcfg.get("backbone_config") or {}
    codec = _get(cfg, "codec") or gcfg.get("codec") or {}
    vc = _get(cfg, "voice_cloning") or gcfg.get("voice_cloning") or {}
    tl = _get(cfg, "text_layout") or gcfg.get("text_repetition") or {}
    audio_heads = gcfg.get("audio_heads") or {}

    arch_rows = [
        ("Backbone", _hub_link(base_model)),
        ("Hidden size", bb.get("hidden_size")),
        ("Layers", bb.get("num_hidden_layers")),
        ("Attention heads", bb.get("num_attention_heads")),
        ("KV heads", bb.get("num_key_value_heads")),
        ("Text vocab", _fmt_int(bb.get("vocab_size")) if bb.get("vocab_size") else None),
        (
            "Codec",
            f"{codec.get('num_layers')} layers · FSQ {codec.get('fsq_levels')} · "
            f"{codec.get('frame_rate_hz')} Hz" if codec else None,
        ),
        ("Audio heads", len(audio_heads) or None),
        ("Voice cloning", "enabled" if vc.get("enabled") else "disabled"),
        (
            "SupCon regulariser",
            "enabled" if _get(vc, "training", "supcon", "enabled") else "disabled",
        ),
        (
            "Text repetition",
            (
                f"enabled (target {tl.get('target_text_tokens')} tokens, "
                f"applied below {tl.get('apply_below')})"
                if tl.get("enabled") else "disabled"
            ) if tl else None,
        ),
    ]

    # ---- training recipe (from the frozen resolved cfg) ----
    tr = _get(cfg, "trainer") or {}
    data = _get(cfg, "data") or {}
    supcon = _get(vc, "training", "supcon") or {}
    sources = data.get("hf_datasets") or []
    src_names = [str(s.get("reponame")) for s in sources if isinstance(s, dict)]

    recipe_rows = [
        ("Objective", "multihead CE (32 codec channels) + weighted BCE stop head"),
        ("Epochs", tr.get("num_train_epochs")),
        (
            "Batch (per device × accum)",
            f"{tr.get('per_device_train_batch_size')} × {tr.get('gradient_accumulation_steps')} = "
            f"{(tr.get('per_device_train_batch_size') or 0) * (tr.get('gradient_accumulation_steps') or 0)}/device-step"
            if tr.get("per_device_train_batch_size") else None,
        ),
        ("Learning rate", f"{tr.get('learning_rate'):g}" if tr.get("learning_rate") else None),
        (
            "Schedule",
            f"{tr.get('lr_scheduler_type')} · warmup {tr.get('warmup_steps')} · "
            f"grad-clip {tr.get('max_grad_norm')}" if tr.get("lr_scheduler_type") else None,
        ),
        ("Optimizer", tr.get("optim")),
        ("Weight decay", tr.get("weight_decay")),
        (
            "LR multipliers",
            f"audio ×{tr.get('audio_lr_multiplier')} · embed ×{tr.get('embed_lr_multiplier')}"
            if tr.get("audio_lr_multiplier") is not None else None,
        ),
        ("Seed", tr.get("seed")),
        (
            "SupCon batch (P·K + M)",
            f"{supcon.get('P')}·{supcon.get('K')} + {supcon.get('M')}"
            if supcon.get("enabled") else None,
        ),
        ("Max audio duration", f"{data.get('max_duration_sec')} s" if data.get("max_duration_sec") else None),
        ("Singleton policy", data.get("singleton_policy")),
        ("Training sources", _fmt_int(len(src_names)) if src_names else None),
    ]
    sources_block = ""
    if src_names:
        listed = "\n".join(f"- `{n}`" for n in src_names)
        sources_block = (
            "\n<details>\n<summary>Training data sources "
            f"({len(src_names)})</summary>\n\n{listed}\n\n</details>\n"
        )

    # ---- usage ----
    repo_ref = repo_id or "<repo_id>"
    usage = f"""```python
from gepard.inference import GepardRunner

# Loads model + tokenizer + text-layout policy from the checkpoint's gepard_config.json
runner = GepardRunner.from_checkpoint("{repo_ref}")

# Returns FSQ codec codes [num_heads, T]; decode to waveform with NanoCodec.
tokens = runner.generate("Hello from Gepard.", temperature=0.8, top_k=50)
```

Zero-shot voice cloning: pass unfolded reference codec codes via `ref_codes` /
`ref_mask`. See the inference notebook and `gepard/inference/runner.py` in the
[source repository](https://github.com/nineninesix-ai) for the full
audio-decode path and CFG options."""

    # ---- reproducibility ----
    env = meta.get("environment") or {}
    wandb_meta = meta.get("wandb") or {}
    repro_rows = [
        ("gepard commit", f"`{env.get('gepard_git_commit')}`" if env.get("gepard_git_commit") else None),
        ("torch", env.get("torch")),
        ("transformers", env.get("transformers")),
        (
            "Weights & Biases",
            f"{wandb_meta.get('project')}/{wandb_meta.get('name')}"
            + (f" (`{wandb_meta.get('run_id')}`)" if wandb_meta.get("run_id") else "")
            if wandb_meta.get("project") else None,
        ),
    ]
    full_cfg = json.dumps(cfg, indent=2, ensure_ascii=False) if cfg else "{}"
    repro_block = (
        _kv_table(repro_rows)
        + "\n\nThe complete resolved training configuration is frozen in "
        f"[`{TRAINING_METADATA_NAME}`]({TRAINING_METADATA_NAME}) in this repository "
        "(and mirrored below), so this run is reproducible from the checkpoint alone.\n\n"
        f"<details>\n<summary>Full resolved training config</summary>\n\n"
        f"```json\n{full_cfg}\n```\n\n</details>"
    )

    # ---- assemble ----
    parts = [
        front_matter,
        "",
        title,
        "",
        lead,
        "",
        provenance,
        "",
        "## Architecture",
        "",
        _kv_table(arch_rows),
        "",
        "## Training",
        "",
        _kv_table(recipe_rows),
        sources_block,
        "## Usage",
        "",
        usage,
        "",
        "## Reproducibility",
        "",
        repro_block,
        "",
        "## Intended use & limitations",
        "",
        "This is a research text-to-speech checkpoint. It generates neural-codec tokens "
        "conditioned on text (and, optionally, a reference voice for zero-shot cloning). "
        "No safety, watermarking, or speaker-consent mechanism is bundled — do not use it "
        "to impersonate real individuals without consent, and disclose synthetic audio "
        "where required. Quality varies with language and speaker coverage in the training "
        "corpus above; this card ships no per-checkpoint evaluation metrics.",
        "",
        "## License & attribution",
        "",
        "Model weights: **Apache-2.0**. Gepard loads the NVIDIA NeMo **NanoCodec** "
        "(`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`) at inference time; that codec is "
        "**not** covered by Apache-2.0 but by the "
        "[NVIDIA Open Model License](https://developer.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf). "
        "You must comply with it to synthesise audio.",
        "",
        "---",
        "",
        "<sub>🐆 Model card generated automatically by "
        "`gepard.logging.model_card` from the checkpoint's frozen training metadata.</sub>",
        "",
    ]
    return "\n".join(parts)


def write_model_card(checkpoint_dir, *, repo_id: Optional[str] = None) -> str:
    """Render + write ``README.md`` into ``checkpoint_dir``; returns its path."""
    md = render_model_card(checkpoint_dir, repo_id=repo_id, uploaded_at=_utc_now_iso())
    path = Path(checkpoint_dir) / MODEL_CARD_NAME
    path.write_text(md, encoding="utf-8")
    return str(path)
