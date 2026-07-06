#!/usr/bin/env python3
"""
DPO stage 3 — preference pairs + frozen-reference logprobs (venv or venv_dpo;
needs neither the codec nor Whisper).

Pair policy per group (N scored rollouts of one prompt):
  chosen   — highest reward, must be genuinely good (wer/dur/truncation gates),
  rejected — lowest reward, margin R_w - R_l >= min_reward_margin,
  up to max_pairs_per_group (best vs worst, then 2nd best vs 2nd worst).

Reference logprobs are computed here ONCE with the frozen reference checkpoint
(teacher-forced, batched) and stored in pairs.jsonl — training then never needs
a second model in memory. Raw component sums are stored (tokens / stop / T), so
the trainer can apply any normalization scheme without recomputing.

Usage:
    python -m gepard.data.dpo.pairs [--skip-ref-logprobs] [overrides...]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


from gepard.config import load_dpo
from gepard.config.schema import DPOConfig
from gepard.inference.runner import TTSRunner
from gepard.model.losses.dpo import encode_text_ids, trajectory_logprobs


def build_pairs(cfg: DPOConfig, scores):
    groups = defaultdict(list)
    for rec in scores:
        groups[rec["group_id"]].append(rec)

    p = cfg.pairs
    pairs, n_no_chosen, n_no_margin = [], 0, 0
    for gid, recs in groups.items():
        recs = sorted(recs, key=lambda r: r["reward"], reverse=True)

        def chosen_ok(r):
            if r["wer"] > p.chosen_max_wer or r["empty_asr"]:
                return False
            if p.chosen_not_truncated and r["truncated"]:
                return False
            if p.chosen_dur_in_bounds and not (r["dur_min"] <= r["duration_sec"] <= r["dur_max"]):
                return False
            return True

        good = [r for r in recs if chosen_ok(r)]
        if not good:
            n_no_chosen += 1
            continue

        used = set()
        for k in range(p.max_pairs_per_group):
            if k >= len(good):
                break
            chosen = good[k]
            # k-th worst not already used and not itself an acceptable chosen
            rejected = next(
                (r for r in reversed(recs)
                 if r["sample_idx"] not in used and r["sample_idx"] != chosen["sample_idx"]
                 and not chosen_ok(r)),
                None,
            )
            if rejected is None or chosen["reward"] - rejected["reward"] < p.min_reward_margin:
                if rejected is not None:
                    n_no_margin += 1
                break
            used.update({chosen["sample_idx"], rejected["sample_idx"]})
            pairs.append({
                "group_id": gid, "uuid": chosen["uuid"], "text": chosen["text"],
                "category": chosen.get("category"), "speaker": chosen["speaker"],
                "chosen_idx": chosen["sample_idx"], "rejected_idx": rejected["sample_idx"],
                "chosen_n_frames": chosen["n_frames"], "rejected_n_frames": rejected["n_frames"],
                "chosen_truncated": chosen["truncated"], "rejected_truncated": rejected["truncated"],
                "reward_chosen": chosen["reward"], "reward_rejected": rejected["reward"],
            })

    print(f"[dpo_pairs] groups={len(groups)}  pairs={len(pairs)}  "
          f"skipped: no_chosen={n_no_chosen} no_margin={n_no_margin}")
    return pairs


@torch.no_grad()
def add_ref_logprobs(cfg: DPOConfig, pairs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dpo_pairs] reference checkpoint: {cfg.ref_checkpoint}")
    runner = TTSRunner.from_checkpoint(cfg.ref_checkpoint, fallback=cfg)
    model, tokenizer = runner.model.eval(), runner.tokenizer
    prefixes = torch.load(cfg.prefixes_path, map_location=device, weights_only=True)

    tokens_cache = {}

    def get_tokens(gid, idx):
        if gid not in tokens_cache:
            tokens_cache.clear()
            tokens_cache[gid] = torch.load(
                Path(cfg.tokens_dir) / f"{gid}.pt", map_location="cpu", weights_only=True
            )
        return tokens_cache[gid]["tokens"][idx].long()

    # Flatten: each pair contributes its chosen and rejected trajectory.
    seqs = []
    for pi, p in enumerate(pairs):
        for side, idx, trunc in (
            ("chosen", p["chosen_idx"], p["chosen_truncated"]),
            ("rejected", p["rejected_idx"], p["rejected_truncated"]),
        ):
            seqs.append((pi, side, p["group_id"], idx, trunc, p["text"], p["speaker"]))
    seqs.sort(key=lambda s: get_tokens(s[2], s[3]).shape[1])  # length-bucket batching

    bs = cfg.pairs.ref_logp_batch
    for start in tqdm(range(0, len(seqs), bs), desc="ref logprobs"):
        batch = seqs[start:start + bs]
        lp = trajectory_logprobs(
            model,
            text_ids_list=[encode_text_ids(tokenizer, s[5], device, runner.repeater) for s in batch],
            prefix_list=[prefixes[s[6]] for s in batch],
            tokens_list=[get_tokens(s[2], s[3]) for s in batch],
            truncated_list=[s[4] for s in batch],
            p_floor=cfg.pairs.p_floor,
            requires_grad=False,
        )
        for j, (pi, side, *_rest) in enumerate(batch):
            pairs[pi][f"ref_{side}"] = {
                "logp_tokens": round(float(lp.logp_tokens[j]), 4),
                "logp_stop": round(float(lp.logp_stop[j]), 4),
                "n_frames": int(lp.n_frames[j]),
            }
    return pairs


def main():
    ap = argparse.ArgumentParser(description="DPO pair builder")
    ap.add_argument("--skip-ref-logprobs", action="store_true",
                    help="only build pairs (debug)")
    args, overrides = ap.parse_known_args()

    cfg = load_dpo(overrides)
    with open(cfg.scores_path) as f:
        scores = [json.loads(line) for line in f]

    pairs = build_pairs(cfg, scores)
    if not args.skip_ref_logprobs and pairs:
        pairs = add_ref_logprobs(cfg, pairs)

    with open(cfg.pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[dpo_pairs] {len(pairs)} pairs → {cfg.pairs_path}")

    by_cat = defaultdict(int)
    for p in pairs:
        by_cat[p.get("category") or "?"] += 1
    print(f"[dpo_pairs] by category: {dict(by_cat)}")


if __name__ == "__main__":
    main()
