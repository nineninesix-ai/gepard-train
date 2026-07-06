#!/usr/bin/env python3
"""
DPO stage 1 — rollout sampling (run inside venv_dpo).

For every (text, speaker) group, generates N samples in ONE batched
autoregressive pass (the speaker prefix and text prefill are shared across the
group, diversity comes from sampling). Tokens are saved — unlike
benchmark/short_ex_search.py which only kept wavs — because DPO training needs
the exact trajectories. Runaway generations are not killed by a timeout but
truncated at an adaptive frame cap, so they enter the dataset as negatives.

Resumable: groups whose token file already exists are skipped.

Usage:
    python -m gepard.data.dpo.sample [--limit 8] [overrides...]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F


from gepard.config import load_dpo
from gepard.config.schema import DPOConfig
from gepard.inference.runner import TTSRunner, FullAttnCache
from gepard.logging import (
    get_logger,
    print_dpo_sample_banner,
    setup_train_logging,
    shutdown_train_logging,
)
from gepard.model.losses.dpo import adaptive_max_frames, compute_speaker_prefix, encode_text_ids


NULL_SPEAKER = "__null__"
log = get_logger("dpo.sample")


def speaker_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _encode_wave(player, wave, sr_in: int, device, sample_rate: int,
                 fsq_levels) -> torch.Tensor:
    """resample to the codec sample rate if needed → unfolded codes [1, T_ref, C]."""
    import librosa
    from gepard.model.codec_ops import unfold_tokens

    if sr_in != sample_rate:
        wave = librosa.resample(wave, orig_sr=sr_in, target_sr=sample_rate)
    wave_t = torch.from_numpy(wave).float().unsqueeze(0).to(device)
    wave_len = torch.tensor([wave_t.shape[-1]]).to(device)
    with torch.inference_mode():
        tokens, _ = player.encode(audio=wave_t, audio_len=wave_len)
    return unfold_tokens(tokens.cpu(), num_levels=list(fsq_levels)).permute(0, 2, 1).to(device)


def encode_ref_audios(cfg: DPOConfig, device) -> Dict[str, torch.Tensor]:
    """Build speaker → ref_codes [1, T_ref, C]. Prefers `speaker_pool` (HF dataset
    with an `audio` column, speakers named pool_NN by row index) over `ref_audios`.
    Holdout speakers are excluded — they exist only for unseen-voice evaluation."""
    import librosa
    import numpy as np
    from gepard.inference.codec_wrapper import UnfoldedCodecModel

    s = cfg.sampling
    sr, fsq = cfg.codec.sample_rate, cfg.codec.fsq_levels
    holdout = set(s.holdout_speakers)
    player = UnfoldedCodecModel.from_pretrained(cfg.codec.codec_id)
    out = {}

    if s.speaker_pool:
        from datasets import load_dataset
        ds = load_dataset(s.speaker_pool, split="train")
        for i in range(len(ds)):
            name = f"pool_{i:02d}"
            if name in holdout:
                continue
            a = ds[i]["audio"]
            # datasets Audio: dict {array, sampling_rate} OR a torchcodec decoder
            if isinstance(a, dict):
                wave, sr_in = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
            else:
                samples = a.get_all_samples()
                wave = samples.data.mean(0).cpu().numpy().astype(np.float32)  # mono
                sr_in = samples.sample_rate
            out[name] = _encode_wave(player, wave, sr_in, device, sr, fsq)
    else:
        for path in s.ref_audios:
            name = speaker_name(path)
            if name in holdout:
                continue
            wave, _ = librosa.load(path, sr=sr)
            out[name] = _encode_wave(player, wave, sr, device, sr, fsq)

    del player
    torch.cuda.empty_cache()
    log.info("Encoded %d training speakers%s", len(out),
             f" ({len(holdout)} held out)" if holdout else "")
    return out


def _uncond_text_ids(runner: TTSRunner, device, mode: str) -> torch.LongTensor:
    """Text-free uncond branch for CFG (same prefix, text removed). Mirrors
    GepardRunner: empty_text keeps the special-token frame, audio_only is bare.
    Ids come from the runner (checkpoint / config tree), matching the cond
    branch — not from hardcoded constants."""
    if mode == "empty_text":
        ids = [runner.BOS_TEXT, runner.EOT, runner.BOS_AUDIO]
    elif mode == "audio_only":
        ids = [runner.BOS_AUDIO]
    else:
        raise ValueError(f"unknown cfg_uncond_mode={mode!r}")
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def generate_group(
    runner: TTSRunner,
    text: str,
    prefix: torch.Tensor,          # [K, d]
    n_samples: int,
    max_frames: int,
    temperature: float,
    top_k: int,
    stop_threshold: float,
    cfg_scale: float = 1.0,
    cfg_frames: int = 0,
    cfg_uncond_mode: str = "empty_text",
) -> List[dict]:
    """N rollouts of one prompt in a single batched AR loop.

    With cfg_scale != 1.0, runs text-CFG (MODEL_GUIDE §7.4): a second uncond
    branch (same prefix, no text) is decoded in parallel and per-head logits are
    guided  logit = logit_u + cfg_scale*(logit_c - logit_u)  before sampling.
    cfg_frames > 0 → onset-only (guide first N frames, then drop the uncond pass).
    """
    model, device = runner.model, runner.device
    B = n_samples
    K, d = prefix.shape
    cfg_on = cfg_scale != 1.0
    guide_window = None if cfg_frames <= 0 else cfg_frames
    prefix_b = prefix.unsqueeze(0).to(model.model.embed_tokens.weight.dtype)

    def prefill(text_ids_1d):
        text_ids = text_ids_1d.unsqueeze(0)                            # [1, T_text]
        T = text_ids.shape[1]
        text_emb = model.model.embed_tokens(text_ids)                 # [1, T, d]
        inp = torch.cat([prefix_b, text_emb], dim=1).expand(B, -1, -1).contiguous()
        attn = torch.ones(B, K + T, dtype=torch.long, device=device)
        cache = FullAttnCache(model.config)
        out = model.model(inputs_embeds=inp, attention_mask=attn,
                          use_cache=True, past_key_values=cache)
        return out.last_hidden_state[:, -1, :], out.past_key_values, T

    # Match production inference: adaptive text repetition via runner.repeater
    # (deterministic target_R), same layout the deployed TTSRunner uses (MODEL_GUIDE §5.2/§7.4).
    cond_ids = encode_text_ids(runner.tokenizer, text, device, runner.repeater)  # [T_text]
    cond_h, cond_cache, T_cond = prefill(cond_ids)
    if cfg_on:
        unc_ids = _uncond_text_ids(runner, device, cfg_uncond_mode)[0]
        unc_h, unc_cache, T_unc = prefill(unc_ids)
    else:
        unc_h = unc_cache = None
        T_unc = 0

    frames = torch.zeros(B, model.num_codebook_heads, max_frames,
                         dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    n_frames = torch.full((B,), max_frames, dtype=torch.long, device=device)

    def sample_frame(h_c: torch.Tensor, h_u, scale: float, temp: float) -> torch.Tensor:
        cols = []
        do_cfg = h_u is not None and scale != 1.0
        for head, vocab in zip(model.codebook_heads, model.vocab_sizes):
            logits = head(h_c).float()
            if do_cfg:
                lu = head(h_u).float()
                logits = lu + scale * (logits - lu)
            if temp != 1.0:
                logits = logits / temp
            if top_k > 0:
                k = min(top_k, vocab)
                thr = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < thr, float("-inf"))
            cols.append(torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1))
        return torch.stack(cols, dim=1)                                # [B, C]

    guide_now = cfg_on and (guide_window is None or 0 < guide_window)
    frames[:, :, 0] = sample_frame(cond_h, unc_h if guide_now else None,
                                   cfg_scale if guide_now else 1.0, temperature)

    for step in range(1, max_frames):
        prev = frames[:, :, step - 1]                                  # [B, C]
        channel_tokens = [prev[:, c] for c in range(model.num_codebook_heads)]
        emb = model._embed_audio(channel_tokens).unsqueeze(1)          # [B, 1, d]

        attn = torch.ones(B, K + T_cond + step, dtype=torch.long, device=device)
        out = model.model(inputs_embeds=emb, attention_mask=attn,
                          use_cache=True, past_key_values=cond_cache)
        cond_h = out.last_hidden_state[:, -1, :]
        cond_cache = out.past_key_values

        guide_now = cfg_on and (guide_window is None or step < guide_window)
        if guide_now:
            attn_u = torch.ones(B, K + T_unc + step, dtype=torch.long, device=device)
            out_u = model.model(inputs_embeds=emb, attention_mask=attn_u,
                                use_cache=True, past_key_values=unc_cache)
            unc_h = out_u.last_hidden_state[:, -1, :]
            unc_cache = out_u.past_key_values

        stop_p = torch.sigmoid(model.stop_head(cond_h).squeeze(-1).float())
        newly_done = (~done) & (stop_p > stop_threshold)
        n_frames[newly_done] = step          # frames 0..step-1 were generated
        done |= newly_done
        if bool(done.all()):
            break

        frames[:, :, step] = sample_frame(cond_h, unc_h if guide_now else None,
                                          cfg_scale if guide_now else 1.0, temperature)
        # done rows keep "generating" into ignored positions — cheap for N=8

    return [
        {
            "tokens": frames[i, :, : int(n_frames[i])].to(torch.int16).cpu(),
            "n_frames": int(n_frames[i]),
            "truncated": bool(n_frames[i] >= max_frames),
        }
        for i in range(B)
    ]


def main():
    ap = argparse.ArgumentParser(description="DPO rollout sampler")
    ap.add_argument("--checkpoint", default=None, help="override sampling.checkpoint")
    ap.add_argument("--limit", type=int, default=0, help="only first N groups (smoke test)")
    ap.add_argument("--shard", default="0/1",
                    help="i/N — process only groups with index %% N == i; run N processes "
                         "in parallel on a big GPU (see make dpo-sample-sharded)")
    args, overrides = ap.parse_known_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    assert 0 <= shard_i < shard_n, f"bad --shard {args.shard}"

    cfg = load_dpo(overrides)
    s = cfg.sampling
    if args.checkpoint:
        s.checkpoint = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.tokens_dir, exist_ok=True)

    # Structured logging. Each shard is an independent process → its own writer,
    # so (unlike torchrun rank-0 gating) every shard sets up file logging with a
    # distinct filename inside one shared run dir. Only our `gepard.*` lines are
    # filed; NeMo / TTSRunner / wandb output is left on stdout, untouched.
    log_dir = Path("logs") / f"dpo_sample_{cfg.run_name}"
    filename = f"shard{shard_i}.log" if shard_n > 1 else "sample.log"
    log_state = setup_train_logging(scope="dpo.sample", log_dir=log_dir, filename=filename)

    try:
        if shard_i == 0:
            print_dpo_sample_banner(cfg, log_dir, shard_i, shard_n,
                                    console=log_state["console"])
        cfg_desc = (f"cfg={s.cfg_scale}@{'all' if s.cfg_frames <= 0 else s.cfg_frames}f/{s.cfg_uncond_mode}"
                    if s.cfg_scale != 1.0 else "cfg=off")
        log.info("=== DPO rollout sampling started (shard %d/%d) ===", shard_i, shard_n)
        log.info("run=%s  policy checkpoint=%s", cfg.run_name, s.checkpoint)
        log.info("N=%d  speakers_per_text=%d  temp=%s  %s",
                 s.num_samples, s.speakers_per_text, s.temperature, cfg_desc)

        ref_codes = encode_ref_audios(cfg, device)
        speakers = sorted(ref_codes.keys())

        runner = TTSRunner.from_checkpoint(s.checkpoint, fallback=cfg)
        runner.model.eval()

        # Frozen Q-Former → speaker prefixes are constants: compute once, persist for
        # the pairs/train stages (they then need neither the codec nor the ref wavs).
        # Only shard 0 writes the file (identical content; avoids a write race).
        prefixes = {spk: compute_speaker_prefix(runner.model, rc) for spk, rc in ref_codes.items()}
        # null_prefix is a learnable [K, d] vector (CFG unconditional); store it as a
        # pseudo-speaker so pairs/train consume it through the same path.
        if s.null_prefix_prob > 0 and getattr(runner.model, "null_prefix", None) is not None:
            prefixes[NULL_SPEAKER] = runner.model.null_prefix.detach()
        if shard_i == 0:
            torch.save({k: v.float().cpu() for k, v in prefixes.items()}, cfg.prefixes_path)
            log.info("%d prefixes (null=%s) → %s", len(prefixes),
                     "yes" if NULL_SPEAKER in prefixes else "no", cfg.prefixes_path)

        with open(s.texts_file) as f:
            texts = [json.loads(line) for line in f]

        import random
        groups = []
        for rec in texts:
            for k in range(s.speakers_per_text):
                # deterministic per (text, slot): null with prob null_prefix_prob, else
                # a pool speaker (offset by id+slot to spread voices across the corpus).
                rng = random.Random(s.seed * 7919 + rec["id"] * 31 + k)
                if NULL_SPEAKER in prefixes and rng.random() < s.null_prefix_prob:
                    spk = NULL_SPEAKER
                else:
                    spk = speakers[(rec["id"] + k) % len(speakers)]
                groups.append((len(groups), rec, spk, k))   # global index → seed & sharding
        groups = [g for g in groups if g[0] % shard_n == shard_i]
        if args.limit:
            groups = groups[: args.limit]
        log.info("shard %d/%d: %d texts → %d groups → %d generations",
                 shard_i, shard_n, len(texts), len(groups), len(groups) * s.num_samples)

        # Per-shard manifest so concurrent processes never interleave writes;
        # dpo_score globs manifest*.jsonl.
        manifest_path = (cfg.manifest_path if shard_n == 1
                         else f"{cfg.manifest_path}.shard{shard_i}")
        manifest_f = open(manifest_path, "a", encoding="utf-8")
        t_start, n_done, n_skipped = time.time(), 0, 0

        for gi, rec, spk, k in groups:
            group_id = f"{rec['uuid'][:8]}__{spk}__s{k}"
            token_file = os.path.join(cfg.tokens_dir, f"{group_id}.pt")
            if os.path.exists(token_file):
                n_skipped += 1
                continue

            torch.manual_seed(s.seed * 1_000_003 + gi)   # reproducible, distinct per group
            max_frames = adaptive_max_frames(rec["text"], cfg.reward, s, cfg.codec.frame_rate_hz)
            samples = generate_group(
                runner, rec["text"], prefixes[spk],
                n_samples=s.num_samples, max_frames=max_frames,
                temperature=s.temperature, top_k=s.top_k,
                stop_threshold=s.stop_threshold,
                cfg_scale=s.cfg_scale, cfg_frames=s.cfg_frames,
                cfg_uncond_mode=s.cfg_uncond_mode,
            )

            torch.save(
                {"tokens": [smp["tokens"] for smp in samples],
                 "text": rec["text"], "speaker": spk, "uuid": rec["uuid"]},
                token_file,
            )
            for si, smp in enumerate(samples):
                manifest_f.write(json.dumps({
                    "group_id": group_id, "uuid": rec["uuid"], "text": rec["text"],
                    "category": rec.get("category"), "speaker": spk, "sample_idx": si,
                    "n_frames": smp["n_frames"],
                    "duration_sec": round(smp["n_frames"] / cfg.codec.frame_rate_hz, 3),
                    "truncated": smp["truncated"], "max_frames": max_frames,
                }, ensure_ascii=False) + "\n")
            manifest_f.flush()
            n_done += 1

            if n_done % 25 == 0:
                rate = n_done / (time.time() - t_start)
                eta_h = (len(groups) - n_skipped - n_done) / max(rate, 1e-9) / 3600
                trunc = sum(smp["truncated"] for smp in samples)
                log.info("[%d/%d] %.2f groups/s  ETA %.1fh  last: %r (%s) trunc %d/%d",
                         n_done + n_skipped, len(groups), rate, eta_h,
                         rec["text"][:32], spk, trunc, s.num_samples)

        manifest_f.close()
        log.info("=== shard %d/%d done: %d generated, %d skipped (resume) → %s ===",
                 shard_i, shard_n, n_done, n_skipped, manifest_path)
    except Exception:
        log.exception("DPO sampling (shard %d/%d) failed", shard_i, shard_n)
        raise
    finally:
        shutdown_train_logging(log_state)


if __name__ == "__main__":
    main()
