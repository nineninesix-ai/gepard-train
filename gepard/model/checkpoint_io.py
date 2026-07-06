"""Unified checkpoint file resolution (ROADMAP Stage 2, §B2.3).

One place for the "local file / local dir / HF Hub repo → local file path"
logic that used to be copy-pasted across the inference runner, the SFT
checkpoint loader and the gepard-config reader.
"""

from pathlib import Path
from typing import Optional

WEIGHTS_NAME = "model.safetensors"


def resolve_checkpoint_file(
    checkpoint: str,
    filename: str = WEIGHTS_NAME,
    required: bool = True,
) -> Optional[str]:
    """Resolve `filename` inside a checkpoint reference to a local path.

    `checkpoint` may be:
      - a direct file path (returned as-is when it is / stands for `filename`),
      - a local checkpoint directory (looks for `filename` inside),
      - an HF Hub repo id (downloads `filename` to the cache).

    With `required=False` an absent file returns None instead of raising —
    used for optional sidecars like `gepard_config.json` on legacy checkpoints.
    """
    path = Path(str(checkpoint))

    if path.is_file():
        # A direct file path stands for the weights file; a differently-named
        # optional sidecar is absent by definition.
        if path.name == filename or (required and filename == WEIGHTS_NAME):
            return str(path)
        if required:
            raise FileNotFoundError(f"{checkpoint!r} is a file, not {filename}")
        return None

    if path.is_dir():
        local = path / filename
        if local.exists():
            return str(local)
        if required:
            raise FileNotFoundError(f"No {filename} in {path}")
        return None

    # Not on the local filesystem → treat as an HF Hub repo id.
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        if required:
            raise
        return None
    try:
        return hf_hub_download(repo_id=str(checkpoint), filename=filename)
    except EntryNotFoundError:
        if required:
            raise
        return None


def resolve_safetensors(checkpoint: str) -> str:
    """Local path to the checkpoint's `model.safetensors` (downloading if needed)."""
    return resolve_checkpoint_file(checkpoint, WEIGHTS_NAME, required=True)


def normalize_scalar_shapes(state_dict, model) -> list:
    """Reconcile numel-1 parameter shapes between a checkpoint and the live model.

    Pre-FSDP2 checkpoints (all published ones included) store
    `ref_compressor.output_scale` as a 0-dim tensor; the model now declares it
    shape [1] because FSDP2's `fully_shard` rejects 0-dim parameters. The two
    broadcast identically, but `load_state_dict` hard-errors on any shape
    mismatch (even with strict=False) — so every load seam calls this first.

    Reshapes in place (both directions, guarded to numel==1 on both sides so a
    real shape bug still fails loudly). Returns the reshaped keys.
    """
    fixed = []
    params = dict(model.named_parameters())
    for key, tensor in state_dict.items():
        target = params.get(key)
        if target is None:
            continue
        if tensor.shape != target.shape and tensor.numel() == 1 and target.numel() == 1:
            state_dict[key] = tensor.reshape(target.shape)
            fixed.append(key)
    return fixed
