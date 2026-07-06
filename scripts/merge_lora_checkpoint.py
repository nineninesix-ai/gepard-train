"""Merge LoRA adapters into the base weights of a saved fine-tune checkpoint.

CPU-only, pure state_dict surgery — does NOT build the model (avoids GPU / flash
attention), so it is safe to run while a training job holds the GPU.

For every LoRA-wrapped Linear the checkpoint stores three tensors under a shared
prefix P:
    P.base.weight   (frozen base, bf16)
    P.lora_A        (rank x in,  fp32)
    P.lora_B        (out x rank, fp32)
This folds them per LoRALinear.merge_into_base():
    W = base.weight + (lora_B @ lora_A) * (alpha / rank)
and renames P.base.weight -> P.weight (matching the un-wrapped nn.Linear layout
that inference rebuilds). All other tensors pass through unchanged.

Usage:
    python merge_lora_checkpoint.py CKPT_DIR [--alpha 32 --rank 16 --suffix merged]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="checkpoint dir (contains model.safetensors)")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--suffix", type=str, default="merged")
    ap.add_argument("--base-model", type=str, default="nineninesix/qwen3_5-full-attn-only-14",
                    help="FALLBACK stock-backbone id for config.json — used only when the "
                         "fine-tune base checkpoint (training_metadata.finetuned_from) has none")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    src = ckpt / "model.safetensors"
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = ckpt.parent / f"{ckpt.name}-{args.suffix}"
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing {out_dir}")

    scaling = args.alpha / args.rank
    print(f"[merge] reading {src}")
    sd = load_file(str(src), device="cpu")
    print(f"[merge] {len(sd)} tensors loaded; scaling = alpha/rank = {args.alpha}/{args.rank} = {scaling}")

    # Group keys by LoRA prefix.
    prefixes = sorted(k[: -len(".lora_A")] for k in sd if k.endswith(".lora_A"))
    print(f"[merge] {len(prefixes)} LoRA modules to fold")

    merged = {}
    consumed = set()
    for p in prefixes:
        a, b, base_w = f"{p}.lora_A", f"{p}.lora_B", f"{p}.base.weight"
        for req in (a, b, base_w):
            if req not in sd:
                raise KeyError(f"LoRA group {p}: missing {req}")
        delta = (sd[b].float() @ sd[a].float()) * scaling          # [out, in], fp32
        w = sd[base_w]
        if delta.shape != w.shape:
            raise ValueError(f"{p}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        merged[f"{p}.weight"] = (w.float() + delta).to(w.dtype)     # back to base dtype (bf16)
        consumed.update({a, b, base_w})
        # carry a base bias through if the wrapped Linear had one
        base_b = f"{p}.base.bias"
        if base_b in sd:
            merged[f"{p}.bias"] = sd[base_b]
            consumed.add(base_b)

    # Everything not part of a LoRA group passes through verbatim.
    for k, v in sd.items():
        if k in consumed:
            continue
        if k in merged:
            raise KeyError(f"name collision on {k}")
        merged[k] = v

    n_lora = len(consumed)
    print(f"[merge] folded {len(prefixes)} adapters; {len(sd)} -> {len(merged)} tensors "
          f"(dropped {n_lora}, added {len(prefixes)} merged weights)")

    out_dir.mkdir(parents=True)
    out_sf = out_dir / "model.safetensors"
    save_file(merged, str(out_sf), metadata={"format": "pt"})
    print(f"[merge] wrote {out_sf}")

    # ── config.json + gepard_config.json ─────────────────────────────────────
    # A periodic LoRA checkpoint carries NEITHER (they are stamped only once the
    # adapters are merged). Rebuild them from the FINE-TUNE BASE checkpoint — NOT
    # the stock backbone: pretrain may have changed `partial_rotary_factor`, so
    # the stock config.json would misdescribe these weights. The base id lives in
    # the checkpoint's own training_metadata.json (`finetuned_from`).
    meta_path = ckpt / "training_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    base_ckpt = meta.get("finetuned_from") or args.base_model
    ft_text_rep = (meta.get("config") or {}).get("text_layout")

    from gepard.model.checkpoint_io import resolve_checkpoint_file

    # config.json — copy the fine-tune base's verbatim (its partial_rotary_factor
    # is exactly what these weights were trained with). Generate from the stock
    # backbone only as a fallback when the base has none (legacy / non-finetune).
    base_config = resolve_checkpoint_file(base_ckpt, "config.json", required=False)
    if base_config is not None:
        shutil.copy2(base_config, out_dir / "config.json")
        print(f"[merge] config.json from fine-tune base {base_ckpt}")
    else:
        from transformers import AutoConfig
        from gepard.model.configuration import patch_config_json_rotary
        if base_ckpt != args.base_model:
            print(f"[merge] ⚠️  could not resolve config.json from fine-tune base "
                  f"{base_ckpt!r}; falling back to stock backbone {args.base_model} — "
                  f"partial_rotary_factor may be WRONG if pretrain changed it.")
        cfg = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
        cfg.save_pretrained(str(out_dir))
        patch_config_json_rotary(str(out_dir / "config.json"))
        print(f"[merge] config.json generated from stock backbone {args.base_model}")

    # gepard_config.json — take the base's (full, correct architecture) and
    # override `text_repetition` with the fine-tune's layout, so inference replays
    # the SAME repetition the fine-tune trained on (base=pretrain has it OFF; the
    # SFT corpus baked it ON).
    base_gcfg = resolve_checkpoint_file(base_ckpt, "gepard_config.json", required=False)
    if base_gcfg is not None:
        gdict = json.loads(Path(base_gcfg).read_text())
        if ft_text_rep is not None:
            gdict["text_repetition"] = ft_text_rep
            print(f"[merge] gepard_config.text_repetition ← fine-tune "
                  f"(enabled={ft_text_rep.get('enabled')})")
        (out_dir / "gepard_config.json").write_text(
            json.dumps(gdict, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"[merge] gepard_config.json from fine-tune base {base_ckpt}")
    elif (ckpt / "gepard_config.json").exists():
        shutil.copy2(ckpt / "gepard_config.json", out_dir / "gepard_config.json")
        print("[merge] gepard_config.json copied from source checkpoint")

    # Carry THIS checkpoint's provenance forward so `make upload` renders a real
    # model card (stage/step/recipe) instead of the degraded "unknown" one.
    if meta_path.exists():
        shutil.copy2(meta_path, out_dir / "training_metadata.json")
        print("[merge] copied training_metadata.json")

    # tokenizer + chat template — so the merged dir loads standalone.
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        s = ckpt / fname
        if s.exists():
            shutil.copy2(s, out_dir / fname)
            print(f"[merge] copied {fname}")

    print(f"[merge] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
