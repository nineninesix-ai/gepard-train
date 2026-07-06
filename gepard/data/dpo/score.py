#!/usr/bin/env python3
"""
DPO stage 2 — scoring (run inside venv_dpo).

Decodes saved rollout tokens to audio (NeMo codec, batched), transcribes with
Whisper, computes WER/CER and the composite reward:

    R = -w_wer*WER - w_over*max(0, dur - dur_max) - w_short*1[dur < dur_min]
        - w_empty*1[empty_asr]

The duration penalty is TWO-SIDED on purpose: the probe (MODEL_GUIDE §7.3)
found both runaway babble AND premature stops (0.2s for a full sentence);
without the short-side penalty DPO would reward-hack into instant stops, and
silence (empty ASR ⇒ WER=1.0) would beat an honest attempt with WER 2.0.

Rerunnable: scoring is decoupled from sampling, so reward weights can be
changed and this stage re-executed without regenerating rollouts.

Usage:
    python -m gepard.data.dpo.score [overrides...]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


from gepard.config import load_dpo
from gepard.config.schema import DPOConfig



def load_manifest(path: str):
    """Read manifest.jsonl plus any manifest.jsonl.shard* from sharded sampling."""
    import glob
    recs = []
    for p in sorted(glob.glob(path) + glob.glob(path + ".shard*")):
        with open(p) as f:
            recs.extend(json.loads(line) for line in f)
    if not recs:
        raise FileNotFoundError(f"no manifest found at {path}[.shard*] — run dpo-sample first")
    return recs


def decode_all(cfg: DPOConfig, manifest, device):
    """Batched codec decode: group samples by length bucket to limit padding waste."""
    from gepard.inference.codec_wrapper import UnfoldedCodecModel

    player = UnfoldedCodecModel.from_pretrained(cfg.codec.codec_id)
    tokens_cache = {}

    def get_tokens(rec):
        gid = rec["group_id"]
        if gid not in tokens_cache:
            tokens_cache.clear()  # keep at most one group file in RAM
            tokens_cache[gid] = torch.load(
                Path(cfg.tokens_dir) / f"{gid}.pt", map_location="cpu", weights_only=True
            )
        return tokens_cache[gid]["tokens"][rec["sample_idx"]]

    wavs = [None] * len(manifest)
    order = sorted(range(len(manifest)), key=lambda i: manifest[i]["n_frames"])
    bs = cfg.reward.decode_batch_size

    for chunk_start in tqdm(range(0, len(order), bs), desc="codec decode"):
        idxs = order[chunk_start:chunk_start + bs]
        toks = [get_tokens(manifest[i]).long() for i in idxs]
        T_max = max(t.shape[1] for t in toks)
        batch = torch.zeros(len(toks), toks[0].shape[0], T_max, dtype=torch.long)
        lens = torch.tensor([t.shape[1] for t in toks])
        for j, t in enumerate(toks):
            batch[j, :, : t.shape[1]] = t
        with torch.inference_mode():
            audio, audio_len = player.decode_from_codes(batch.to(device), lens.to(device))
        for j, i in enumerate(idxs):
            wavs[i] = audio[j, : int(audio_len[j])].float().cpu().numpy()

    del player
    torch.cuda.empty_cache()
    return wavs


def transcribe_all(cfg: DPOConfig, wavs, device):
    import librosa
    import numpy as np
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # Direct processor + model instead of pipeline(): transformers 5.x's ASR
    # pipeline hard-imports torchcodec in preprocess, which fails to load on this
    # CUDA stack (missing libnppicc.so.13 under torch cu130) even for array input.
    # Feeding arrays through the feature extractor bypasses torchcodec entirely.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(cfg.reward.whisper_model)
    model = (
        WhisperForConditionalGeneration.from_pretrained(cfg.reward.whisper_model, dtype=dtype)
        .to(device).eval()
    )
    gen_kwargs = (
        {"language": cfg.reward.whisper_language} if cfg.reward.whisper_language else {}
    )
    bs = cfg.reward.whisper_batch_size
    sr = cfg.codec.sample_rate

    texts = []
    for i in tqdm(range(0, len(wavs), bs), desc="whisper"):
        batch = []
        for w in wavs[i:i + bs]:
            a = np.asarray(w, dtype=np.float32)[: sr * 30]
            # Whisper feature extractor expects 16 kHz; codec output is sr.
            if sr != 16000:
                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            batch.append(a)
        feats = processor(
            batch, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype)
        with torch.no_grad():
            pred_ids = model.generate(feats, **gen_kwargs)
        texts.extend(t.strip() for t in processor.batch_decode(pred_ids, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return texts


def score_sim(cfg: DPOConfig, wavs, manifest, device):
    """WavLM-SV speaker similarity (MODEL_GUIDE §7.3). Returns per-sample cosine to the
    speaker's reference embedding, or None where SIM is undefined (null prefix,
    a speaker with no ref wav, or an audio clip too short to embed)."""
    import os
    import librosa
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, WavLMForXVector

    rw = cfg.reward
    sr = cfg.codec.sample_rate
    min_samp = int(0.2 * sr)   # WavLM conv stack needs a little audio
    extractor = AutoFeatureExtractor.from_pretrained(rw.sim_model)
    sv = WavLMForXVector.from_pretrained(rw.sim_model).to(device).eval()

    @torch.inference_mode()
    def embed(wav_list):
        chunk = [librosa.resample(w.astype("float32"), orig_sr=sr, target_sr=rw.sim_sr)
                 for w in wav_list]
        inp = extractor(chunk, sampling_rate=rw.sim_sr, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        return F.normalize(sv(**inp).embeddings, dim=-1).cpu()

    # Reference embedding per speaker (from the same ref wavs used for the prefix).
    ref_emb = {}
    for path in cfg.sampling.ref_audios:
        name = os.path.splitext(os.path.basename(path))[0]
        wav, _ = librosa.load(path, sr=sr)
        ref_emb[name] = embed([wav])[0]

    sims = [None] * len(manifest)
    valid = [i for i, w in enumerate(wavs)
             if ref_emb.get(manifest[i]["speaker"]) is not None and len(w) >= min_samp]
    bs = 16
    for s in tqdm(range(0, len(valid), bs), desc="wavlm-sim"):
        idxs = valid[s:s + bs]
        gen = embed([wavs[i] for i in idxs])
        for j, i in enumerate(idxs):
            sims[i] = float(torch.dot(gen[j], ref_emb[manifest[i]["speaker"]]))

    del sv
    torch.cuda.empty_cache()
    return sims


def main():
    ap = argparse.ArgumentParser(description="DPO rollout scorer")
    args, overrides = ap.parse_known_args()

    import jiwer
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    cfg = load_dpo(overrides)
    rw = cfg.reward
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = load_manifest(cfg.manifest_path)
    print(f"[dpo_score] run={cfg.run_name}  {len(manifest)} samples")

    wavs = decode_all(cfg, manifest, device)
    transcripts = transcribe_all(cfg, wavs, device)
    sims = score_sim(cfg, wavs, manifest, device) if rw.sim_enabled else [None] * len(manifest)

    normalize = BasicTextNormalizer()
    out_f = open(cfg.scores_path, "w", encoding="utf-8")
    n_clean = 0

    for rec, hyp_raw, sim in zip(manifest, transcripts, sims):
        ref, hyp = normalize(rec["text"]), normalize(hyp_raw)
        empty = not hyp
        wer = 1.0 if empty else float(jiwer.wer(ref, hyp))
        cer = 1.0 if empty else float(jiwer.cer(ref, hyp))

        dur = rec["duration_sec"]
        dur_min, dur_max = rw.dur_bounds(rec["text"])
        r_wer = -rw.w_wer * wer
        r_over = -rw.w_over * max(0.0, dur - dur_max)
        r_short = -rw.w_short * (1.0 if dur < dur_min else 0.0)
        r_empty = -rw.w_empty * (1.0 if empty else 0.0)
        # SIM anchor: penalize voice drift. Undefined (null prefix / no ref) → 0.
        r_sim = -rw.w_sim * (1.0 - sim) if sim is not None else 0.0
        reward = r_wer + r_over + r_short + r_empty + r_sim

        clean = (wer <= cfg.pairs.chosen_max_wer and dur_min <= dur <= dur_max
                 and not rec["truncated"] and not empty)
        n_clean += clean

        out_f.write(json.dumps({
            **rec,
            "transcript": hyp_raw, "wer": round(wer, 4), "cer": round(cer, 4),
            "empty_asr": empty, "dur_min": round(dur_min, 2), "dur_max": round(dur_max, 2),
            "sim": round(sim, 4) if sim is not None else None,
            "reward": round(reward, 4), "clean": clean,
        }, ensure_ascii=False) + "\n")
    out_f.close()

    print(f"[dpo_score] clean rate: {n_clean / len(manifest):.1%} "
          f"({n_clean}/{len(manifest)})")
    print(f"[dpo_score] scores → {cfg.scores_path}")


if __name__ == "__main__":
    main()
