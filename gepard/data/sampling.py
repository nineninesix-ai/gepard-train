"""Reference-voice sampling: dataset wrapper, SupCon batch sampler, dataset loading."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from ..model.losses.supcon import NULL_SPEAKER_INT
from .collator import NULL_SPEAKER_SENTINEL

if TYPE_CHECKING:
    from ..config.schema import VoiceCloningConfig  # noqa: F401


class ReferenceSamplingDataset(TorchDataset):
    """Training-side dataset wrapper that adds reference-voice sampling.

    Wraps a HuggingFace Dataset (already shuffled before construction — see
    `prepare_dataset`) and on each `__getitem__`:
      1. Returns the target row as-is (all pre-computed columns: text_ids,
         level_audio_*, labels_*, attention_mask, speaker_id, row_id, encoded_len).
      2. Picks a reference index from the same speaker (cross-recording when
         possible, same-audio fallback for singletons) and attaches:
           - `ref_codes`: np.ndarray [T_ref, C_ref] int, stacked codec channels
           - `ref_len`:   int, length of ref_codes before batch padding
           - `force_null`: int (0/1). 1 means the row's speaker_id is the
             null sentinel — the model will swap the compressor prefix with
             `null_prefix` for this sample (in addition to stochastic CFG
             dropout). The ref_codes returned for these rows are a 1-frame
             zero placeholder so the compressor has at least one valid key
             to attend to (avoids NaN softmax); the result is discarded.
      3. Applies a stochastic slice L ∈ [L_min, L_max] frames (or full ref if
         shorter) so the model is robust to variable-length prompts at inference.

    The shape of `ref_codes` depends on `audio_codec.do_unfold`:
      - do_unfold=true  → C_ref = num_layers * len(fsq_Levels)  (e.g. 32 for 8×[8,7,6,6])
      - do_unfold=false → C_ref = num_layers                    (packed per-layer ids)
    Both are supported; the compressor decides at init whether to unfold in-forward.

    Behavior under `dataset_config.singleton_policy`:
      - "remove":      singletons were already dropped in preprocessing.
                       Remaining same-speaker fallback handles edge cases where
                       the only ref candidate is the target itself.
      - "null_prefix": rows whose speaker_id is `NULL_SPEAKER_SENTINEL` are
                       routed through the null branch (force_null=1). The
                       K-query bottleneck still prevents copy-paste leakage on
                       any non-null same-audio fallback.
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        codec_frame_rate_hz: float,
        vc_config: VoiceCloningConfig,
        audio_heads: Dict[str, int],
        speaker_col: str = "speaker_id",
        rng_seed: Optional[int] = None,
    ):
        assert vc_config.enabled, "ReferenceSamplingDataset must not be constructed when VC is disabled"
        self.dataset = hf_dataset
        self.speaker_col = speaker_col
        self.channel_names: List[str] = list(audio_heads.keys())

        # Slim view used only for random ref reads: drops text_ids / labels_* /
        # attention_mask so each ref pickup deserialises just the codec channels.
        # select_columns is metadata-only on Arrow (no data copy) and preserves
        # row order, so ref_idx works identically on both views.
        self.ref_dataset = hf_dataset.select_columns(self.channel_names)

        self.codec_frame_rate_hz = float(codec_frame_rate_hz)
        rs = vc_config.reference_sampling
        self.l_min_seconds = float(rs.l_min_seconds)
        self.l_max_seconds = float(rs.l_max_seconds)
        self.min_ref_duration_seconds = float(rs.min_ref_duration_seconds)
        self.singleton_min_target_for_slice = float(rs.singleton_min_target_for_slice)
        self.use_self_reference = bool(rs.use_self_reference)
        self.l_min_frames = int(self.l_min_seconds * self.codec_frame_rate_hz)
        self.l_max_frames = int(self.l_max_seconds * self.codec_frame_rate_hz)

        # Per-worker RNG: DataLoader forks processes; each worker calls
        # torch.utils.data.get_worker_info() at first use. We keep a module-level
        # `random` call — torch seeds worker RNGs via worker_init_fn in practice,
        # but our policy is fully stochastic so reproducibility is not a goal.
        self._rng = random.Random(rng_seed) if rng_seed is not None else random

        self._build_speaker_indices()

    def _build_speaker_indices(self) -> None:
        """One-time scan: speaker → row indices, durations, ref candidates.

        Rows tagged with `NULL_SPEAKER_SENTINEL` (low-count speakers or
        speaker-less sources under `singleton_policy=null_prefix`) are recorded
        in `null_ref_indices` and excluded from `speaker_to_indices` /
        `speaker_to_ref_candidates`. They take the dummy-ref + force_null path
        in `__getitem__`.

        Also builds:
          - `speaker_str_to_int`: deterministic string→int mapping over the
            non-null speakers (0..N-1). Null rows map to NULL_SPEAKER_INT.
            Consumed by SupCon (collator passes the per-row int through to the
            trainer as `speaker_ints`).
          - `row_to_speaker_int`: per-row precomputed int label (length n).
            Avoids hashing strings on the data-loader hot path.
        """
        n = len(self.dataset)
        print(f"[ReferenceSamplingDataset] Building speaker index over {n} rows...")

        encoded_lens = np.asarray(self.dataset["encoded_len"], dtype=np.int64)
        self.durations = encoded_lens.astype(np.float64) / self.codec_frame_rate_hz

        self.speaker_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.null_ref_indices: set = set()
        sid_column = self.dataset[self.speaker_col]
        for i, sid in enumerate(sid_column):
            if sid == NULL_SPEAKER_SENTINEL or sid is None:
                self.null_ref_indices.add(i)
                continue
            self.speaker_to_indices[sid].append(i)

        self.speaker_to_ref_candidates: Dict[str, List[int]] = {}
        n_single = 0
        for sid, idxs in self.speaker_to_indices.items():
            long_enough = [i for i in idxs if self.durations[i] >= self.min_ref_duration_seconds]
            self.speaker_to_ref_candidates[sid] = long_enough if long_enough else idxs
            if len(idxs) == 1:
                n_single += 1

        # Deterministic speaker_id → int (sorted for cross-rank consistency
        # under DataLoader workers; the same dataset always yields the same
        # mapping). Null speakers are -1 (NULL_SPEAKER_INT).
        self.speaker_str_to_int: Dict[str, int] = {
            sid: i for i, sid in enumerate(sorted(self.speaker_to_indices.keys()))
        }
        # Per-row int label (length n) — cheap lookup at __getitem__ time.
        self.row_to_speaker_int: np.ndarray = np.full(n, NULL_SPEAKER_INT, dtype=np.int64)
        for i, sid in enumerate(sid_column):
            if sid in self.speaker_str_to_int:
                self.row_to_speaker_int[i] = self.speaker_str_to_int[sid]

        n_speakers = len(self.speaker_to_indices)
        n_null = len(self.null_ref_indices)
        print(
            f"[ReferenceSamplingDataset] {n_speakers} speakers, {n_single} singletons "
            f"({100 * n_single / max(n_speakers, 1):.1f}% of speakers); "
            f"{n_null} null-ref rows ({100 * n_null / max(n, 1):.1f}% of rows). "
            f"L_min={self.l_min_frames}f / L_max={self.l_max_frames}f at {self.codec_frame_rate_hz} Hz."
        )

        # Fix C_ref from the first sample: the dummy ref returned for null-ref
        # rows must match the channel dimension other batch items produce.
        if n > 0:
            sample = self.ref_dataset[0]
            self._c_ref = self._stack_codec_layers(sample).shape[1]
        else:
            self._c_ref = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def _stack_codec_layers(self, row: Dict[str, Any]) -> np.ndarray:
        """Stack all `level_audio_*` columns into [T, C] int array."""
        layers = [np.asarray(row[name], dtype=np.int64) for name in self.channel_names]
        # Truncate each layer to the shortest in case of off-by-one; should not
        # trigger in practice since all layers share encoded_len + 1.
        t = min(layer.shape[0] for layer in layers)
        return np.stack([layer[:t] for layer in layers], axis=1)  # [T, C]

    def _stochastic_slice(self, audio: np.ndarray) -> np.ndarray:
        """Random [L_min, L_max] slice; if audio shorter than L_min, return as-is."""
        t = audio.shape[0]
        l_max_f = min(self.l_max_frames, t)
        if l_max_f <= self.l_min_frames:
            return audio
        length = self._rng.randint(self.l_min_frames, l_max_f)
        start = self._rng.randint(0, t - length)
        return audio[start:start + length]

    def _select_reference(self, target_idx: int, target_row: Dict[str, Any]) -> np.ndarray:
        """Apply the reference-selection policy; returns [T_ref, C_ref] int array."""
        sid = target_row[self.speaker_col]
        target_duration_s = float(target_row["encoded_len"]) / self.codec_frame_rate_hz

        # Finetune fast-path: always use the target's own (already-loaded) audio
        # as the reference. Skips the random cross-recording Arrow read. The
        # frozen compressor still emits a real prefix; the K-query bottleneck
        # limits copy-paste leakage just as in the singleton fallback below.
        if self.use_self_reference:
            target_audio = self._stack_codec_layers(target_row)
            if target_duration_s >= self.singleton_min_target_for_slice:
                return self._stochastic_slice(target_audio)
            return target_audio

        candidates = self.speaker_to_ref_candidates.get(sid, [target_idx])
        candidates_excl = [i for i in candidates if i != target_idx]

        # 1. Multi-sample speaker — cross-recording ref.
        if candidates_excl:
            ref_idx = self._rng.choice(candidates_excl)
            ref_row = self.ref_dataset[ref_idx]
            ref_audio = self._stack_codec_layers(ref_row)
            return self._stochastic_slice(ref_audio)

        # 2. Singleton fallback — same audio as target.
        target_audio = self._stack_codec_layers(target_row)
        if target_duration_s >= self.singleton_min_target_for_slice:
            return self._stochastic_slice(target_audio)

        # 3. Very short singleton — use in full. K-query bottleneck prevents
        # copy-paste leakage.
        return target_audio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        target_row = self.dataset[idx]
        if idx in self.null_ref_indices:
            # 1-frame zero placeholder so the compressor still has a valid key
            # to attend over (avoids NaN softmax). The result is replaced by
            # `null_prefix` in the model's forward via force_null.
            ref_codes = np.zeros((1, self._c_ref), dtype=np.int64)
            force_null = 1
        else:
            ref_codes = self._select_reference(idx, target_row)
            force_null = 0
        # np arrays default to int64 here; collator casts to torch.long.
        target_row["ref_codes"] = ref_codes
        target_row["ref_len"] = int(ref_codes.shape[0])
        target_row["force_null"] = force_null
        target_row["speaker_int"] = int(self.row_to_speaker_int[idx])
        return target_row


class SpeakerBucketBatchSampler(Sampler[List[int]]):
    """Speaker-bucket batch sampler for SupCon-style voice cloning training.

    Each batch has shape `P * K + M`:
      - `P` unique speakers × `K` clips per speaker — the SupCon-eligible region.
        The loss treats indices from the same speaker as positives and from
        different speakers as negatives.
      - `M` extra positions sampled from `null_ref_indices` (or random rows if
        the null pool is exhausted) — included so CE / stop heads keep seeing
        the unconditional path, masked out of SupCon loss via `force_null`.

    Batch size = `P*K + M`. Should match `per_device_train_batch_size`.

    Distributed
    -----------
    Each rank seeds its RNG with `base_seed + epoch * 1009 + rank * 7`. Ranks
    produce statistically disjoint batches but no strict partition is enforced
    — with `N_speakers >> P * world_size` overlap is negligible. Total
    optimiser steps per epoch = `num_batches` * gradient_accumulation_steps
    (same across ranks).

    The caller must invoke `set_epoch(epoch)` at each epoch boundary so the
    RNG is re-seeded; the wrapping `MultiheadTTSTrainer` does this in
    `get_train_dataloader`.

    Parameters
    ----------
    speaker_to_indices : Dict[str, List[int]]
        From `ReferenceSamplingDataset.speaker_to_indices`. Only speakers
        with `len(indices) >= K` are kept in the eligible pool — under
        normal config they all satisfy this because `min_clips_per_speaker`
        was set to K at dataset prep time.
    null_ref_indices : Iterable[int]
        Row indices flagged with `force_null=1`. Used to fill the M null
        positions in each batch.
    P, K, M : int
        Speakers / clips-per-speaker / null-row count per batch.
    num_batches : int, optional
        Batches per rank per epoch. When omitted, computed as
        `len(eligible_rows) // (P * K * world_size)`.
    base_seed : int
        Base RNG seed; combined with epoch and rank.
    rank, world_size : int
        Distributed rank / world. Defaults to `torch.distributed` state if
        initialised, else 0 / 1.
    """

    def __init__(
        self,
        speaker_to_indices: Dict[str, List[int]],
        null_ref_indices,
        P: int,
        K: int,
        M: int,
        num_batches: Optional[int] = None,
        base_seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        if P < 2:
            raise ValueError(f"P must be >= 2 (need negatives in batch); got P={P}")
        if K < 2:
            raise ValueError(f"K must be >= 2 (need positives per anchor); got K={K}")
        if M < 0:
            raise ValueError(f"M must be >= 0; got M={M}")

        self.P = int(P)
        self.K = int(K)
        self.M = int(M)
        self.batch_size = self.P * self.K + self.M

        if rank is None or world_size is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.rank = torch.distributed.get_rank()
                self.world_size = torch.distributed.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
        else:
            self.rank = int(rank)
            self.world_size = int(world_size)

        # Keep only speakers that can supply K clips. Under min_clips_per_speaker=K
        # at data prep, this is everyone in the map; the filter is defensive.
        self._eligible_speakers: List[str] = [
            s for s, idx in speaker_to_indices.items() if len(idx) >= self.K
        ]
        if not self._eligible_speakers:
            raise RuntimeError(
                f"SpeakerBucketBatchSampler: no speakers with >={self.K} clips; "
                f"check dataset prep (min_clips_per_speaker)."
            )
        # Store as numpy arrays for fast slicing during sampling.
        self._speaker_indices: Dict[str, np.ndarray] = {
            s: np.asarray(speaker_to_indices[s], dtype=np.int64)
            for s in self._eligible_speakers
        }
        self._null_indices: np.ndarray = np.asarray(sorted(null_ref_indices), dtype=np.int64)

        # Fallback when null pool is empty: sample fillers from the eligible
        # pool (won't be flagged force_null=True, but at least they're valid
        # rows; SupCon loss will treat them as additional anchors).
        self._null_fallback: np.ndarray = np.concatenate(list(self._speaker_indices.values()))

        eligible_rows = sum(len(v) for v in self._speaker_indices.values())
        if num_batches is None:
            denom = max(1, self.P * self.K * self.world_size)
            num_batches = max(1, eligible_rows // denom)
        self._num_batches = int(num_batches)

        self.base_seed = int(base_seed)
        self._epoch = 0

        print(
            f"[SpeakerBucketBatchSampler] P={self.P} K={self.K} M={self.M} "
            f"batch_size={self.batch_size} | eligible_speakers={len(self._eligible_speakers):,} "
            f"eligible_rows={eligible_rows:,} null_rows={len(self._null_indices):,} | "
            f"rank={self.rank}/{self.world_size} num_batches/rank/epoch={self._num_batches:,}"
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the per-epoch RNG. Called by the trainer at epoch start."""
        self._epoch = int(epoch)

    def _make_rng(self) -> np.random.Generator:
        seed = (self.base_seed
                + self._epoch * 1009
                + self.rank * 7)
        return np.random.default_rng(seed)

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._make_rng()
        speakers = np.array(self._eligible_speakers, dtype=object)
        n_speakers = len(speakers)

        for _ in range(self._num_batches):
            # 1. Pick P unique speakers without replacement.
            chosen = rng.choice(n_speakers, size=self.P, replace=False)
            batch_indices: List[int] = []
            for spk_idx in chosen:
                sid = speakers[spk_idx]
                pool = self._speaker_indices[sid]
                # Pick K indices without replacement (pool >= K by construction).
                picks = rng.choice(len(pool), size=self.K, replace=False)
                batch_indices.extend(int(pool[p]) for p in picks)

            # 2. Pad with M null rows (or fallback from eligible pool).
            if self.M > 0:
                src = self._null_indices if self._null_indices.size > 0 else self._null_fallback
                replace = src.size < self.M
                fillers = rng.choice(src.size, size=self.M, replace=replace)
                batch_indices.extend(int(src[f]) for f in fillers)

            assert len(batch_indices) == self.batch_size
            yield batch_indices

    def __len__(self) -> int:
        return self._num_batches


def prepare_dataset(
    dataset_path: str,
    codec_frame_rate_hz: Optional[float] = None,
    vc_config: Optional[VoiceCloningConfig] = None,
    audio_heads: Optional[Dict[str, int]] = None,
    keep_index_path: Optional[str] = None,
):
    """
    Load and prepare the training dataset.

    When `vc_config.enabled` is True, wraps the shuffled HF dataset in a
    ReferenceSamplingDataset so each `__getitem__` emits a per-sample ref slice.
    Otherwise returns the shuffled HF dataset directly (legacy path).

    Args:
        dataset_path: Path to the processed dataset directory
        codec_frame_rate_hz: codec frame rate (required when vc_config.enabled=True;
            drives the ref-slice length math)
        vc_config: VoiceCloningConfig; None or enabled=False → legacy path
        audio_heads: {channel_name: vocab_size} — required when VC is enabled
            so ReferenceSamplingDataset knows which columns to stack for ref_codes.
        keep_index_path: optional .npy of row indices to keep (short-utterance
            reweighting, MODEL_GUIDE §5). Applied before the shuffle. A per-run
            training choice, passed through from trainer.keep_index_path.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Please run 'make dataset' or 'python -m gepard.cli.prepare' first."
        )

    print(f"Loading dataset from {path}")
    hf_dataset = load_from_disk(str(path))

    # Optional row-subset selection (short-utterance reweighting, MODEL_GUIDE §5).
    # MUST run before the shuffle below: the saved index refers to on-disk row
    # order (load_from_disk order == data-NNNNN shard order). Selecting after the
    # shuffle would point the index at permuted rows. None → full dataset.
    if keep_index_path:
        keep_idx = np.load(keep_index_path)
        n_before = len(hf_dataset)
        hf_dataset = hf_dataset.select(keep_idx)
        print(
            f"[prepare_dataset] keep_index {keep_index_path}: "
            f"{n_before:,} -> {len(hf_dataset):,} rows "
            f"({100 * len(hf_dataset) / max(n_before, 1):.1f}% kept)"
        )

    # Fixed seed so all distributed ranks see the same shuffle permutation.
    # Without it, `hf_dataset.shuffle()` draws from system RNG per process,
    # giving each rank a different permutation and making
    # `dataset[ref_idx]` return physically different rows on different ranks
    # (the per-rank speaker_to_indices map stays internally consistent, but
    # cross-rank comparisons/debug become unpredictable). DistributedSampler
    # will partition the same permutation cleanly across ranks.
    hf_dataset = hf_dataset.shuffle(seed=44)

    if len(hf_dataset) == 0:
        raise ValueError(f"Dataset at {path} is empty!")

    print(f"Loaded {len(hf_dataset)} samples")

    if vc_config is None or not vc_config.enabled:
        return hf_dataset

    if codec_frame_rate_hz is None or audio_heads is None:
        raise ValueError(
            "prepare_dataset: codec_frame_rate_hz and audio_heads are required "
            "when voice_cloning.enabled=True"
        )

    return ReferenceSamplingDataset(
        hf_dataset=hf_dataset,
        codec_frame_rate_hz=codec_frame_rate_hz,
        vc_config=vc_config,
        audio_heads=audio_heads,
    )
