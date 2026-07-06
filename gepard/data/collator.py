"""Batch collation for the Gepard multihead architecture."""

from typing import Any, Dict, List, Optional

import torch

from ..model.losses.supcon import NULL_SPEAKER_INT

__all__ = ["DataCollator", "NULL_SPEAKER_INT", "NULL_SPEAKER_SENTINEL"]


# Speaker-id value used to mark rows that have no usable cross-recording reference
# (low-count speakers under singleton_policy=null_prefix, or rows from datasets that
# don't declare a speaker_id column). Read by ReferenceSamplingDataset to route the
# row through the null_prefix branch at training time. The DataCollator emits a
# `force_null` boolean per sample; the model swaps the prefix with `null_prefix`
# for those samples (in addition to the stochastic cfg_dropout swap).
NULL_SPEAKER_SENTINEL = "__null_ref__"


class DataCollator:
    """
    Data collator for multihead TTS.

    Pads text_ids, all audio channel tensors, labels, and attention_mask to batch
    max lengths. Text and audio are padded independently; the model concatenates embeddings.

    Audio channels are identified by name, derived from audio_heads config
    (same keys as dataset columns: level_audio_0..N).

    When `vc_enabled=True`, the collator also pads per-sample ref tensors emitted
    by ReferenceSamplingDataset:
      - ref_codes:    (B, max_T_ref, C_ref) long, zero-padded
      - ref_mask:     (B, max_T_ref)        bool, True over real frames
      - force_null:   (B,)                  bool, True for samples whose speaker
                      is the null sentinel — the model swaps prefix with
                      `null_prefix` for these (in addition to stochastic CFG).
                      Always emitted alongside ref_codes; defaults to all-False
                      if the upstream dataset doesn't set the per-sample flag
                      (legacy datasets without ReferenceSamplingDataset wrapper).
      - speaker_ints: (B,)                  long, integer speaker labels for
                      SupCon-style losses. Null-ref samples get
                      `NULL_SPEAKER_INT` (-1). Always emitted alongside
                      ref_codes; defaults to all-NULL_SPEAKER_INT for legacy
                      paths without per-sample labels.
    When vc_enabled=False (default), these keys are not emitted — the model takes
    the legacy path (no prefix).
    """

    def __init__(
        self,
        pad_token_id: int,
        audio_heads: Dict[str, int],
        vc_enabled: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.channel_names: List[str] = list(audio_heads.keys())
        self.label_names: List[str] = [f"labels_{name}" for name in self.channel_names]

        self.vc_enabled = bool(vc_enabled)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text_lens = [len(f["text_ids"]) for f in features]
        audio_lens = [len(f[self.channel_names[0]]) for f in features]
        max_text = max(text_lens)
        max_audio = max(audio_lens)

        # Model sequence = [text(max_text) | audio(max_audio)] after embedding concat.
        # Labels and attention_mask must be aligned to this layout.
        max_seq = max_text + max_audio

        batch_size = len(features)

        # Left-pad text_ids: [B, max_text]
        # PAD tokens on the left so SOS is always at position max_text-1,
        # immediately before frame[0] at max_text. Eliminates the gap between
        # text and audio that right-padding creates for shorter texts.
        text_ids = torch.full((batch_size, max_text), self.pad_token_id, dtype=torch.long)
        for i, f in enumerate(features):
            t = f["text_ids"]
            text_ids[i, max_text - len(t):] = torch.tensor(t, dtype=torch.long)

        # Pad audio channel tensors: [B, max_audio] each, pad with 0
        audio_tensors = {}
        for name in self.channel_names:
            tensor = torch.zeros(batch_size, max_audio, dtype=torch.long)
            for i, f in enumerate(features):
                vals = f[name]
                tensor[i, :len(vals)] = torch.tensor(vals, dtype=torch.long)
            audio_tensors[name] = tensor

        # Pad label tensors: [B, max_seq], pad with -100.
        # Dataset labels are laid out as [-100]*text_len + [audio_labels].
        # We must place audio_labels at position max_text (not text_len) to match
        # the model's concatenated sequence [text(max_text) | audio(max_audio)].
        label_tensors = {}
        for key in self.label_names + ["labels_stop"]:
            tensor = torch.full((batch_size, max_seq), -100, dtype=torch.long)
            for i, f in enumerate(features):
                vals = f[key]
                tl = text_lens[i]
                audio_labels = vals[tl:]  # strip the per-sample text padding (-100s)
                tensor[i, max_text:max_text + len(audio_labels)] = torch.tensor(audio_labels, dtype=torch.long)
            label_tensors[key] = tensor

        # Attention mask: [B, max_seq], 1 for real text and real audio.
        # Text region: left-padded, so real tokens are at max_text-tl..max_text-1.
        # Audio region: positions max_text..max_seq-1, only first audio_len are real.
        attention_mask = torch.zeros(batch_size, max_seq, dtype=torch.long)
        for i, (tl, al) in enumerate(zip(text_lens, audio_lens)):
            attention_mask[i, max_text - tl:max_text] = 1     # real text (right-aligned)
            attention_mask[i, max_text:max_text + al] = 1     # real audio

        result: Dict[str, torch.Tensor] = {
            "text_ids": text_ids,
            **audio_tensors,
            "attention_mask": attention_mask,
            **label_tensors,
        }

        # Voice cloning: pad per-sample ref tensors.
        if self.vc_enabled:
            ref_lens = [int(f["ref_len"]) for f in features]
            max_ref = max(ref_lens) if ref_lens else 0
            # Determine C_ref from first non-empty sample (all samples agree by dataset config).
            c_ref = features[0]["ref_codes"].shape[1] if max_ref > 0 else 0
            ref_codes = torch.zeros(batch_size, max_ref, c_ref, dtype=torch.long)
            ref_mask = torch.zeros(batch_size, max_ref, dtype=torch.bool)
            for i, f in enumerate(features):
                rl = int(f["ref_len"])
                if rl > 0:
                    ref_codes[i, :rl] = torch.from_numpy(f["ref_codes"]).long()
                    ref_mask[i, :rl] = True
            result["ref_codes"] = ref_codes
            result["ref_mask"] = ref_mask
            # force_null: per-sample flag set by ReferenceSamplingDataset when
            # the row's speaker_id is the null sentinel. Older datasets won't
            # have this key — default to False so behaviour is unchanged.
            force_null = torch.tensor(
                [bool(f.get("force_null", 0)) for f in features], dtype=torch.bool,
            )
            result["force_null"] = force_null
            # speaker_ints: integer speaker labels for SupCon. Null-ref rows
            # carry NULL_SPEAKER_INT (-1). Default to all-null when the dataset
            # doesn't tag the field (e.g. tests / legacy paths).
            speaker_ints = torch.tensor(
                [int(f.get("speaker_int", NULL_SPEAKER_INT)) for f in features],
                dtype=torch.long,
            )
            result["speaker_ints"] = speaker_ints

        return result
