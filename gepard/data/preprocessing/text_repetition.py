"""Adaptive text-repetition for short-utterance conditioning.

Single source of truth for the "repeat the text until it reaches a minimum
text-token budget" technique (MODEL_GUIDE §5.2, calibrated in §5.3). The exact
same layout logic is used in two places that MUST agree byte-for-byte, or the
model sees a train/inference mismatch and WER collapses (VoiceStar lesson,
MODEL_GUIDE §5.2):

  * training data prep  — `gepard.data.preprocessing.processor::create_input_ids`
  * inference            — `gepard.inference.runner::TTSRunner.generate`

Both import `TextRepeater` and call `build_input_ids`, so the repeated layout
is defined once, here.

The defect
----------
On short inputs the K=8 speaker prefix dominates the hidden state and the
1–2 text tokens drown (MODEL_GUIDE §5). The model never "latches" onto the
speech manifold → runaway / never-stop. The data (MODEL_GUIDE §5.1) shows the failure is a
function of *text-token count*: a cliff below ~6 tokens, a plateau from ~13.

The fix
-------
Repeat the text block `R-1` extra times **before** the canonical copy so the
text region carries ~`target_text_tokens` tokens of mass instead of 1–2:

    [ (SOT text EOT) x (R-1) | SOT text EOT SOS | audio ... ]
      └──── context copies ──┘ └─── canonical ──┘
            (no SOS)            (only this one has SOS)

Why it does not double-speak — three load-bearing properties:
  1. SOS-gating: `start_of_speech` (SOS) is the learned "audio starts now"
     trigger. Only the *canonical* (last) copy carries it, so only that copy
     starts the render. Context copies are read but never voiced.
  2. Invariant target: the audio target is one canonical utterance regardless
     of R; in training the whole text region is already `-100` in the labels
     (processor masks text), so context copies receive zero supervision.
  3. Mixed: a fraction of eligible-short rows are kept at R=1 during training
     (`mixed_keep_prob`) so the model learns repetition is *optional* and does
     not corrupt WER when it does not happen (VoiceStar-style; MODEL_GUIDE §5.2).

R is chosen from the text-token count only — never from sequence length. The
defect is text-content weakness, not sequence length (padding/filler does not
help; MODEL_GUIDE §5.2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TextRepetitionConfig:
    """Config for adaptive text repetition. Disabled by default.

    Mirrors the `text_layout` config group (`conf/text_layout/`).

    Attributes
    ----------
    enabled : bool
        Master switch. False → `target_R` always returns 1 and
        `build_input_ids` reproduces the legacy single-copy layout exactly.
    target_text_tokens : int
        Repeat a short text until its text region holds ~this many text tokens.
        MODEL_GUIDE §5.3 calibration: 16 ≈ p25 of the pretrain length distribution and the
        plateau where short-utterance fail-rate drops to ≤2.5% even on a hard
        external voice.
    apply_below : int
        Only texts with `n_text_tokens < apply_below` are repeated; longer
        texts keep R=1. MODEL_GUIDE §5.3: the plateau begins at ~13 tokens, so texts at or
        above this are already stable and must not be disturbed (repeating the
        working long register is harmful, MODEL_GUIDE §5.2–5.3).
    max_repeats : int
        Hard cap on R. Bounds the prefill blow-up on 1-token inputs
        (e.g. target 16 / 1 token would be R=16; cap keeps it sane).
    mixed_keep_prob : float
        TRAINING ONLY. Probability that an eligible-short row is kept at R=1
        anyway, so the model also sees short texts without repetition and does
        not come to *require* it. Ignored at inference (`target_R` is
        deterministic). 0.0 disables mixing (every eligible row repeated).
    seed : int
        Base RNG seed for the mixed-keep coin flip. Per-shard offset is added
        by the processor so shards are independent yet reproducible.
    """

    enabled: bool = False
    target_text_tokens: int = 16
    apply_below: int = 13
    max_repeats: int = 8
    mixed_keep_prob: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.target_text_tokens < 1:
            raise ValueError(f"target_text_tokens must be >= 1, got {self.target_text_tokens}")
        if self.apply_below < 1:
            raise ValueError(f"apply_below must be >= 1, got {self.apply_below}")
        if self.max_repeats < 1:
            raise ValueError(f"max_repeats must be >= 1, got {self.max_repeats}")
        if not (0.0 <= self.mixed_keep_prob <= 1.0):
            raise ValueError(f"mixed_keep_prob must be in [0, 1], got {self.mixed_keep_prob}")

    @classmethod
    def from_config(cls, node) -> "TextRepetitionConfig":
        """Build from an OmegaConf node / plain dict / None.

        A missing or empty node yields a disabled config — repetition is opt-in.
        """
        if node is None:
            return cls()
        # OmegaConf DictConfig supports .get; plain dict too.
        get = node.get
        return cls(
            enabled=bool(get("enabled", False)),
            target_text_tokens=int(get("target_text_tokens", 16)),
            apply_below=int(get("apply_below", 13)),
            max_repeats=int(get("max_repeats", 8)),
            mixed_keep_prob=float(get("mixed_keep_prob", 0.25)),
            seed=int(get("seed", 0)),
        )


class TextRepeater:
    """Builds the (possibly repeated) text-id layout. Stateless w.r.t. rows.

    Holds the config and the three special token ids that frame the text block.
    The special ids are passed explicitly so the same class serves both the
    training token map (`start_of_text` / `end_of_text` / `start_of_speech`
    from the `tokens` group) and the inference runner's per-instance
    `BOS_TEXT` / `EOT` / `BOS_AUDIO` — which are the same numbers, by
    construction.
    """

    def __init__(
        self,
        config: TextRepetitionConfig,
        start_of_text: int,
        end_of_text: int,
        start_of_speech: int,
    ) -> None:
        self.config = config
        self.sot = int(start_of_text)
        self.eot = int(end_of_text)
        self.sos = int(start_of_speech)

    # ------------------------------------------------------------------
    # R selection
    # ------------------------------------------------------------------

    def target_R(self, n_text_tokens: int) -> int:
        """Deterministic repeat count from the text-token count.

        This is the inference-time policy and the training-time base (before
        the mixed coin flip). Returns 1 when disabled, when the text is already
        long enough (`>= apply_below`), or when it already meets the budget.
        """
        cfg = self.config
        if not cfg.enabled:
            return 1
        if n_text_tokens <= 0 or n_text_tokens >= cfg.apply_below:
            return 1
        if n_text_tokens >= cfg.target_text_tokens:
            return 1
        R = math.ceil(cfg.target_text_tokens / n_text_tokens)
        return max(1, min(R, cfg.max_repeats))

    def sample_R(self, n_text_tokens: int, rng: Optional[random.Random] = None) -> int:
        """Training-time repeat count: `target_R` with the mixed-keep coin flip.

        With probability `mixed_keep_prob`, an otherwise-eligible short row is
        kept at R=1 so the model learns repetition is optional (MODEL_GUIDE §5.2, mixed keep).
        `rng` makes the flip reproducible; if None, a module-global RNG is used.
        """
        R = self.target_R(n_text_tokens)
        if R <= 1:
            return R
        p = self.config.mixed_keep_prob
        if p > 0.0:
            draw = (rng or random).random()
            if draw < p:
                return 1
        return R

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def build_input_ids(self, text_token_ids: List[int], R: int) -> List[int]:
        """Assemble the text-id sequence for a given R.

            R == 1 → [ SOT, *text, EOT, SOS ]                       (legacy)
            R >  1 → [ SOT, *text, EOT ] * (R-1) + [ SOT, *text, EOT, SOS ]

        Only the final (canonical) copy carries SOS. The audio frames are
        appended by the caller; this returns exactly the text region.
        """
        if R < 1:
            raise ValueError(f"R must be >= 1, got {R}")
        text = list(text_token_ids)
        block = [self.sot] + text + [self.eot]
        canonical = block + [self.sos]
        if R == 1:
            return canonical
        return block * (R - 1) + canonical

    def expand(
        self,
        text_token_ids: List[int],
        rng: Optional[random.Random] = None,
        training: bool = False,
    ) -> List[int]:
        """Convenience: pick R then build the layout in one call.

        `training=True` applies the mixed-keep coin flip (`sample_R`);
        otherwise the deterministic `target_R` is used (inference).
        """
        n = len(text_token_ids)
        R = self.sample_R(n, rng) if training else self.target_R(n)
        return self.build_input_ids(text_token_ids, R)
