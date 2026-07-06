#!/usr/bin/env python3
"""
DPO stage 4 — training (run inside the TRAINING venv; no codec/Whisper needed).

Trainable: LoRA adapters on the last N backbone layers + stop_head (fp32),
everything else frozen bf16. Reference logprobs come precomputed from
pairs.jsonl, so only one model lives in memory.

Loss: length-normalized DPO over full trajectory likelihoods INCLUDING the
Bernoulli stop terms (gepard/model/losses/dpo.py) — without them the gradient never reaches
the stop decision, the exact component behind the runaway defect (MODEL_GUIDE §7).

Checkpoints:
    dpo_checkpoints/<run_name>/adapter-step-N.pt   — LoRA + stop_head only (small)
    dpo_checkpoints/<run_name>/merged/             — full inference checkpoint
                                                     (TTSRunner.from_checkpoint-ready)

Usage:
    python -m gepard.training.dpo [overrides...]
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch

from ..config import load_dpo
from ..config.schema import DPOConfig
from ..inference.runner import TTSRunner
from ..model.losses.dpo import dpo_loss, encode_text_ids, trajectory_logprobs
from ..model.lora import lora_state_dict, merge_all_lora
from .base import inject_backbone_lora


class PairsDataset:
    """pairs.jsonl + token files → training items (tokens loaded lazily, LRU=1 group)."""

    def __init__(self, cfg: DPOConfig, device):
        self.cfg = cfg
        self.device = device
        with open(cfg.pairs_path) as f:
            self.pairs = [json.loads(line) for line in f]
        self.prefixes = torch.load(cfg.prefixes_path, map_location=device, weights_only=True)
        self._cache_gid, self._cache = None, None

    def __len__(self):
        return len(self.pairs)

    def _tokens(self, gid, idx):
        if gid != self._cache_gid:
            self._cache = torch.load(
                Path(self.cfg.tokens_dir) / f"{gid}.pt", map_location="cpu", weights_only=True
            )
            self._cache_gid = gid
        return self._cache["tokens"][idx].long()

    def get(self, i, tokenizer, repeater=None):
        p = self.pairs[i]
        prefix = self.prefixes[p["speaker"]]
        text_ids = encode_text_ids(tokenizer, p["text"], self.device, repeater)
        norm = None  # combined ref logp is computed in collate (needs training cfg)
        return {
            "text_ids": text_ids,
            "prefix": prefix,
            "chosen": self._tokens(p["group_id"], p["chosen_idx"]),
            "rejected": self._tokens(p["group_id"], p["rejected_idx"]),
            "chosen_truncated": p["chosen_truncated"],
            "rejected_truncated": p["rejected_truncated"],
            "ref_chosen": p["ref_chosen"],
            "ref_rejected": p["ref_rejected"],
            "reward_margin": p["reward_chosen"] - p["reward_rejected"],
        }


def ref_combined(ref: dict, stop_w: float, length_norm: bool) -> float:
    total = ref["logp_tokens"] + stop_w * ref["logp_stop"]
    return total / max(1, ref["n_frames"]) if length_norm else total


def weight_from_margin(margins, t, device):
    """Per-pair reward-magnitude weights, normalised to mean=1 (MODEL_GUIDE §7.2).

    mode "none" → None (plain-mean DPO, round 1/2). "linear" → 1+margin/scale;
    "clip" → clip(margin/scale, [1/max, max]). Normalised so the batch loss scale
    is unchanged — weighting only redistributes emphasis toward high-margin pairs.
    """
    mode = t.reward_weight_mode
    if mode == "none":
        return None
    m = torch.tensor(margins, dtype=torch.float32, device=device) / max(t.reward_weight_scale, 1e-8)
    if mode == "linear":
        w = (1.0 + m).clamp(min=1.0 / t.reward_weight_max, max=t.reward_weight_max)
    elif mode == "clip":
        w = m.clamp(min=1.0 / t.reward_weight_max, max=t.reward_weight_max)
    else:
        raise ValueError(f"unknown reward_weight_mode: {mode!r}")
    return w / w.mean().clamp(min=1e-8)   # mean(w)=1 → loss scale unchanged


def main():
    ap = argparse.ArgumentParser(description="DPO trainer")
    args, overrides = ap.parse_known_args()

    cfg = load_dpo(overrides)
    t = cfg.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(t.seed)
    random.seed(t.seed)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)

    # ── model: start from the SAMPLING policy (the trajectories are on-policy for it)
    print(f"[dpo_train] run={cfg.run_name}  policy checkpoint={cfg.sampling.checkpoint}")
    runner = TTSRunner.from_checkpoint(cfg.sampling.checkpoint, fallback=cfg)
    model, tokenizer = runner.model, runner.tokenizer
    inject_backbone_lora(model, t.lora, tag="dpo_train")

    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                     "lr": t.learning_rate}]
    if t.train_stop_head:
        model.stop_head.float().requires_grad_(True)
        param_groups.append({"params": list(model.stop_head.parameters()), "lr": t.stop_head_lr})
    if t.train_audio_heads:
        for head in model.codebook_heads:
            head.float().requires_grad_(True)
        param_groups.append({
            "params": [p for h in model.codebook_heads for p in h.parameters()],
            "lr": t.stop_head_lr,
        })
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"[dpo_train] trainable params: {n_train / 1e6:.2f}M")

    dataset = PairsDataset(cfg, device)
    print(f"[dpo_train] {len(dataset)} pairs")
    if len(dataset) == 0:
        sys.exit("[dpo_train] no pairs — run dpo-sample/score/pairs first")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(dataset) / (t.batch_pairs * t.grad_accum))
    total_steps = steps_per_epoch * t.num_epochs

    def lr_lambda(step):
        if step < t.warmup_steps:
            return step / max(1, t.warmup_steps)
        prog = (step - t.warmup_steps) / max(1, total_steps - t.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_wandb = t.report_to == "wandb"
    if use_wandb:
        import wandb
        wandb_kwargs = {"project": t.wandb_project, "name": t.wandb_name or cfg.run_name,
                        "config": {"dpo": True, "beta": t.beta, "pairs": len(dataset)}}
        if t.entity:
            wandb_kwargs["entity"] = t.entity
        wandb.init(**wandb_kwargs)

    model.train()
    global_step, accum, running = 0, 0, {}
    order = list(range(len(dataset)))

    for epoch in range(t.num_epochs):
        random.shuffle(order)
        for batch_start in range(0, len(order), t.batch_pairs):
            idxs = order[batch_start:batch_start + t.batch_pairs]
            items = [dataset.get(i, tokenizer, runner.repeater) for i in idxs]

            def side_logprobs(side, trunc_key):
                return trajectory_logprobs(
                    model,
                    text_ids_list=[it["text_ids"] for it in items],
                    prefix_list=[it["prefix"] for it in items],
                    tokens_list=[it[side] for it in items],
                    truncated_list=[it[trunc_key] for it in items],
                    p_floor=t.p_floor,
                    requires_grad=True,
                )

            pol_c = side_logprobs("chosen", "chosen_truncated")
            pol_r = side_logprobs("rejected", "rejected_truncated")
            ref_c = torch.tensor(
                [ref_combined(it["ref_chosen"], t.stop_term_weight, t.length_normalize)
                 for it in items], device=device)
            ref_r = torch.tensor(
                [ref_combined(it["ref_rejected"], t.stop_term_weight, t.length_normalize)
                 for it in items], device=device)

            weights = weight_from_margin(
                [it["reward_margin"] for it in items], t, device)
            loss, metrics = dpo_loss(
                pol_c, pol_r, ref_c, ref_r,
                beta=t.beta, stop_term_weight=t.stop_term_weight,
                length_normalize=t.length_normalize,
                weights=weights,
            )
            (loss / t.grad_accum).backward()
            accum += 1
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

            if accum == t.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], t.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                if global_step % t.log_steps == 0:
                    n = t.log_steps * t.grad_accum
                    line = {k: round(v / n, 4) for k, v in running.items()}
                    line["lr"] = scheduler.get_last_lr()[0]
                    print(f"[step {global_step}/{total_steps} ep {epoch}] {line}")
                    if use_wandb:
                        wandb.log(line, step=global_step)
                    running = {}

                if global_step % t.save_steps == 0:
                    path = os.path.join(cfg.checkpoints_dir, f"adapter-step-{global_step}.pt")
                    torch.save({"lora": lora_state_dict(model),
                                "stop_head": model.stop_head.state_dict(),
                                "step": global_step}, path)
                    print(f"[dpo_train] adapter → {path}")

    # ── final save: adapter + (optionally) merged inference checkpoint ──
    final_adapter = os.path.join(cfg.checkpoints_dir, "adapter-final.pt")
    torch.save({"lora": lora_state_dict(model),
                "stop_head": model.stop_head.state_dict(),
                "step": global_step}, final_adapter)
    print(f"[dpo_train] final adapter → {final_adapter}")

    if t.save_merged:
        from safetensors.torch import save_file
        from huggingface_hub import snapshot_download

        n_merged = merge_all_lora(model)
        model.stop_head.to(dtype=next(model.model.parameters()).dtype)
        if t.train_audio_heads:
            for head in model.codebook_heads:
                head.to(dtype=next(model.model.parameters()).dtype)
        merged_dir = os.path.join(cfg.checkpoints_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state, os.path.join(merged_dir, "model.safetensors"))
        # config.json + tokenizer files so TTSRunner.from_checkpoint(merged_dir) just works
        src = cfg.sampling.checkpoint
        if not os.path.isdir(src):
            src = snapshot_download(repo_id=src, allow_patterns=["*.json", "*.txt"])
        for f in os.listdir(src):
            if f.endswith((".json", ".txt")) and f != "model.safetensors.index.json":
                shutil.copy(os.path.join(src, f), merged_dir)
        # The copied config.json may predate the vLLM duplication requirement —
        # force the configured partial_rotary_factor into both the flat key and
        # rope_parameters (vLLM reads flat, HF reads nested).
        merged_config = os.path.join(merged_dir, "config.json")
        if os.path.exists(merged_config):
            from ..model.configuration import patch_config_json_rotary

            patch_config_json_rotary(merged_config, value=cfg.model.partial_rotary_factor)
        # Provenance record so the merged checkpoint is self-describing on upload:
        # stage=dpo, step, DPO recipe, and the SFT policy it was tuned from. Without
        # it the model card can't label the stage or the real parent (MODEL_GUIDE §12).
        from ..logging.model_card import write_dpo_training_metadata

        write_dpo_training_metadata(cfg, merged_dir, global_step=global_step)
        print(f"[dpo_train] merged {n_merged} LoRA modules → {merged_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
