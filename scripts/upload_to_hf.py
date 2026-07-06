#!/usr/bin/env python3
"""
Upload trained model checkpoint to Hugging Face Hub.

Usage:
    python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --private
    python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --public
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer


# Trainer resume state (optimizer moments ≈ 2x model size, RNG, schedules) —
# useless on an inference repo, excluded from upload unless --with-training-state.
TRAINING_STATE_PATTERNS = [
    "optimizer*", "scheduler*", "rng_state*", "scaler.pt",
    "trainer_state.json", "training_args.bin", "pytorch_model_fsdp*",
]


def _has_unmerged_lora(checkpoint_path: Path) -> bool:
    """True when model.safetensors still carries raw lora_A/lora_B tensors
    (a periodic checkpoint from a LoRA run — merge before publishing).
    Reads only the safetensors header, not the weights."""
    st = checkpoint_path / "model.safetensors"
    if not st.exists():
        return False
    from safetensors import safe_open

    with safe_open(str(st), framework="pt") as f:
        return any(k.endswith(".lora_A") for k in f.keys())


_DTYPE_MAP = {"bfloat16": "bfloat16", "bf16": "bfloat16",
              "float16": "float16", "fp16": "float16",
              "float32": "float32", "fp32": "float32"}


def maybe_convert_dtype(checkpoint_path: Path, dtype_arg: str = "auto") -> Path:
    """Return the directory to upload: the original, or a temp copy with the
    float tensors of model.safetensors cast to the publish dtype.

    FSDP2 training keeps fp32 master weights, so exported checkpoints carry
    F32 tensors while gepard_config.json says model_dtype=bfloat16 — twice the
    download for zero inference benefit. Local files are never touched (fp32
    stays best for resume/finetune); only the uploaded copy is cast.

    dtype_arg: "auto" (target = gepard_config's model_dtype; no-op when absent
    or already matching), "keep" (never convert), or an explicit dtype name.
    """
    import json

    if dtype_arg == "keep":
        return checkpoint_path
    if dtype_arg == "auto":
        gcfg = checkpoint_path / "gepard_config.json"
        if not gcfg.exists():
            print("ℹ️  No gepard_config.json — dtype left as-is (use --dtype to force)")
            return checkpoint_path
        target_name = json.load(open(gcfg)).get("model_dtype", "keep")
        if target_name not in _DTYPE_MAP:
            return checkpoint_path
    else:
        target_name = dtype_arg
    st = checkpoint_path / "model.safetensors"
    if not st.exists():
        return checkpoint_path

    import shutil
    import tempfile

    import torch
    from safetensors.torch import load_file, save_file

    target = getattr(torch, _DTYPE_MAP[target_name])
    sd = load_file(str(st), device="cpu")
    to_cast = {k for k, v in sd.items()
               if v.dtype.is_floating_point and v.dtype != target}
    if not to_cast:
        print(f"ℹ️  Weights already {target_name} — no dtype conversion needed")
        return checkpoint_path

    from fnmatch import fnmatch

    out_dir = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    for f in checkpoint_path.iterdir():
        # dtype conversion only runs for the inference artifact, so training
        # resume state never belongs in the copy (it can be GBs of optimizer).
        if f.name == "model.safetensors" or any(
            fnmatch(f.name, p) for p in TRAINING_STATE_PATTERNS
        ):
            continue
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    sd = {k: (v.to(target) if k in to_cast else v) for k, v in sd.items()}
    save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"🔄 Cast {len(to_cast)}/{len(sd)} float tensors → {target_name} "
          f"(publish copy only; local checkpoint untouched)")
    return out_dir


def upload_checkpoint(repo_id: str, checkpoint_path: str, private: bool = True,
                      with_training_state: bool = False, allow_unmerged: bool = False,
                      dtype: str = "auto"):
    """
    Upload a model checkpoint with tokenizer to Hugging Face Hub.

    Args:
        repo_id: Repository ID (e.g., "username/model-name")
        checkpoint_path: Path to checkpoint directory
        private: Whether to make the repository private
        with_training_state: Also upload optimizer/scheduler/RNG resume state
        allow_unmerged: Upload even if the weights carry raw LoRA adapters
        dtype: publish dtype — "auto" (gepard_config's model_dtype), "keep",
            or an explicit name; local files are never modified
    """
    checkpoint_path = Path(checkpoint_path)

    # Validate checkpoint path
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    if not checkpoint_path.is_dir():
        print(f"❌ Error: Checkpoint path is not a directory: {checkpoint_path}")
        sys.exit(1)

    # Check for required model files
    required_files = ["config.json", "model.safetensors"]
    missing_files = []
    for file in required_files:
        if not (checkpoint_path / file).exists():
            # Try .bin format
            if file == "model.safetensors" and (checkpoint_path / "pytorch_model.bin").exists():
                continue
            missing_files.append(file)

    if missing_files:
        print(f"⚠️  Warning: Missing model files in checkpoint: {missing_files}")
        print("Continuing anyway...")

    # Refuse un-merged LoRA weights: the lora_A/B + base.weight layout does not
    # load into the inference runner. Periodic checkpoint-N dirs of a LoRA run
    # look like this — fold the adapters first.
    if not allow_unmerged and _has_unmerged_lora(checkpoint_path):
        print("❌ Error: model.safetensors contains un-merged LoRA adapters (lora_A/lora_B).")
        print("   Merge first:  make merge CHECKPOINT=" + str(checkpoint_path))
        print("   (or pass --allow-unmerged to upload the raw adapter checkpoint anyway)")
        sys.exit(1)

    ignore_patterns = None
    if not with_training_state:
        present = sorted(
            {f.name for pat in TRAINING_STATE_PATTERNS for f in checkpoint_path.glob(pat)}
        )
        if present:
            ignore_patterns = TRAINING_STATE_PATTERNS
            print(f"ℹ️  Skipping training resume state ({', '.join(present)})")
            print("   — pass --with-training-state to include it.")

    # Check for tokenizer files
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
    has_tokenizer = any((checkpoint_path / f).exists() for f in tokenizer_files)

    if not has_tokenizer:
        print("\n⚠️  No tokenizer found in checkpoint directory")
        print("🔧 Attempting to load tokenizer from the configured backbone...")

        try:
            from gepard.config import load_train

            base_model_id = load_train([]).model.backbone_id
            print(f"📝 Loading tokenizer from base model: {base_model_id}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)

            # Save tokenizer to checkpoint directory
            print(f"💾 Saving tokenizer to checkpoint directory...")
            tokenizer.save_pretrained(checkpoint_path)
            print("✅ Tokenizer saved to checkpoint")
        except Exception as e:
            print(f"⚠️  Failed to load and save tokenizer: {e}")
            print("Continuing without tokenizer...")

    # Model card (README.md) — rendered from the checkpoint's own
    # training_metadata.json + gepard_config.json, so it states exactly what
    # this checkpoint is and how it was trained (never re-reads conf/, which
    # could have drifted). Written into the checkpoint dir before the dtype
    # copy below, so it is carried into the uploaded folder.
    try:
        from gepard.logging import write_model_card

        card_path = write_model_card(checkpoint_path, repo_id=repo_id)
        print(f"📝 Model card written: {card_path}")
    except Exception as e:
        print(f"⚠️  Model card generation skipped ({type(e).__name__}: {e})")

    # Publish-dtype conversion (a temp copy when needed; local files untouched).
    # After the tokenizer step so a fetched tokenizer lands in the copy too.
    # A resume snapshot must stay byte-faithful: AdamW moments belong to the
    # fp32 master weights, so uploading training state forces dtype=keep.
    if with_training_state and dtype != "keep":
        print("⚠️  Training state requested → weights uploaded as-is (fp32 master); "
              f"--dtype {dtype} is IGNORED for a resume snapshot.")
        dtype = "keep"
    upload_dir = maybe_convert_dtype(checkpoint_path, dtype)

    print(f"\n📤 Uploading checkpoint to Hugging Face Hub")
    print(f"   Repository: {repo_id}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Private: {private}")
    print()

    # Create repository if it doesn't exist
    try:
        api = HfApi()
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        sys.exit(1)

    # Upload all files in checkpoint directory
    try:
        print("\n📁 Uploading files...")

        # List files being uploaded
        files_to_upload = list(upload_dir.glob("*"))
        print(f"   Files to upload ({len(files_to_upload)}):")
        for f in files_to_upload:
            if f.is_file():
                print(f"     - {f.name}")

        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
        )
        print(f"\n✅ Upload completed!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ Error uploading files: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload model checkpoint to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload private checkpoint
  python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --private

  # Upload public checkpoint
  python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --public
        """
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/model-name)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )

    privacy_group = parser.add_mutually_exclusive_group()
    privacy_group.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private (default)"
    )
    privacy_group.add_argument(
        "--public",
        action="store_true",
        help="Make repository public"
    )

    parser.add_argument(
        "--with-training-state",
        action="store_true",
        help="Also upload optimizer/scheduler/RNG resume state (skipped by default)"
    )
    parser.add_argument(
        "--allow-unmerged",
        action="store_true",
        help="Upload even if model.safetensors carries raw un-merged LoRA adapters"
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "keep", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Publish dtype for float tensors. 'auto' casts to gepard_config's "
             "model_dtype (FSDP2 exports carry fp32 master weights — 2x download "
             "for zero inference benefit); 'keep' uploads as-is. Local files are "
             "never modified."
    )

    args = parser.parse_args()

    # Determine privacy setting
    private = not args.public

    upload_checkpoint(
        repo_id=args.repo,
        checkpoint_path=args.checkpoint,
        private=private,
        with_training_state=args.with_training_state,
        allow_unmerged=args.allow_unmerged,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
