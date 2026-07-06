import numpy as np
import torch
from omegaconf import open_dict
from nemo.collections.tts.models import AudioCodecModel


class UnfoldedCodecModel(AudioCodecModel):
    """Extends AudioCodecModel with direct decoding from per-dimension
    discrete FSQ codes, bypassing mixed-radix composition/decomposition.

    Works with any GroupFiniteScalarQuantizer configuration — the number
    of groups, dimensions per group, and FSQ levels are read from the
    model's vector_quantizer at runtime.
    """

    def __init__(self, cfg, trainer=None):
        # SLMDiscriminator downloads microsoft/wavlm-base-plus (~360MB) and is
        # only used during training — strip it from the config before init.
        with open_dict(cfg):
            disc = cfg.get("discriminator", None)
            if disc is not None and "discriminators" in disc:
                disc.discriminators = [
                    d for d in disc.discriminators if "SLM" not in d._target_
                ]
        super().__init__(cfg, trainer)

    def decode_from_codes(
        self, codes: torch.Tensor, codes_len: torch.Tensor
    ):
        """Decode audio from unfolded per-dimension discrete codes.

        Args:
            codes: (B, D, T) — per-dimension discrete values, where
                   D = num_groups * dims_per_group (e.g. 16 for 4 groups × 4 dims).
                   Each group of values must follow the FSQ levels defined in the model.
            codes_len: (B,) — valid frame count per batch element.

        Returns:
            audio: (B, T_audio) — decoded waveform
            audio_len: (B,) — valid audio lengths in samples
        """
        num_levels = self.vector_quantizer.fsqs[0].num_levels.squeeze()
        scale = (num_levels // 2).float().to(codes.device)

        groups = codes.chunk(self.vector_quantizer.num_groups, dim=1)
        dequantized = torch.cat(
            [(g - scale[None, :, None]) / scale[None, :, None] for g in groups],
            dim=1,
        )

        return self.decode_audio(inputs=dequantized, input_len=codes_len)

    def unfold_tokens_np(self, encoded_tokens: np.ndarray, num_levels=(9, 8, 8, 7)) -> np.ndarray:
        """Mixed-radix decomposition of packed token indices into per-dimension discrete codes.

        Args:
            encoded_tokens: (C, T) — packed token indices, where C is the number of codebooks.
            num_levels: FSQ levels per dimension within each codebook.

        Returns:
            (C * len(num_levels), T) — per-dimension discrete codes.
        """
        levels = np.array(num_levels, dtype=np.int32)
        dim_base = np.cumprod(np.array([1] + list(num_levels[:-1]), dtype=np.int32))

        parts = []
        for cb in range(encoded_tokens.shape[0]):
            idx = encoded_tokens[cb:cb+1, :]  # (1, T)
            codes = (idx // dim_base[:, None]) % levels[:, None]
            parts.append(codes)

        return np.concatenate(parts, axis=0)


    def unfold_tokens(self, encoded_tokens: torch.Tensor, num_levels=(9, 8, 8, 7)) -> torch.Tensor:
        """Mixed-radix decomposition of packed token indices into per-dimension discrete codes.

        Args:
            encoded_tokens: (B, C, T) — packed token indices, where C is the number of codebooks.
            num_levels: FSQ levels per dimension within each codebook.
                        Codebook size equals the product of all levels.

        Returns:
            (B, C * len(num_levels), T) — per-dimension discrete codes.
            Each group of len(num_levels) values corresponds to one codebook.
        """
        num_levels = torch.tensor(num_levels, dtype=torch.int32, device=encoded_tokens.device)
        dim_base = torch.cumprod(torch.tensor([1] + list(num_levels[:-1]), device=encoded_tokens.device), dim=0)

        parts = []
        for cb in range(encoded_tokens.shape[1]):
            idx = encoded_tokens[:, cb:cb+1, :]
            codes = (idx // dim_base[None, :, None]) % num_levels[None, :, None]  # (B, 4, T)
            parts.append(codes)

        return torch.cat(parts, dim=1)
