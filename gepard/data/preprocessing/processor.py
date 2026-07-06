"""
Dataset preprocessing pipeline for Gepard multihead training.

This module is the **mandatory preparation step** that must be run before
training. It turns raw HuggingFace codec datasets (containing per-layer
packed FSQ indices + transcripts) into a flat on-disk dataset whose schema
exactly matches what `gepard.data.DataCollator` and
`gepard.model.GepardModel` expect.

How to run
----------
Invoke via the CLI wrapper:

    python -m gepard.cli.prepare [--output PATH] [--n-shards N]

Configuration is composed from the Hydra tree (`conf/prepare.yaml`):
    - `hf_datasets`: list of source datasets (HF hub repo or local path) with
      the column names in each source that hold per-layer codec indices,
      transcript and encoded length.
    - `audio_codec.num_layers` and `audio_codec.fsq_Levels`: describe the codec.
      With `do_unfold=True`, each packed codebook index is decomposed into
      `len(fsq_Levels)` per-dimension discrete codes, yielding
      `num_layers * len(fsq_Levels)` channels (32 for the default 8×[8,7,6,6]).
    - `token_map`: special-token ids (SOT, EOT, SOS, PAD) that frame the input.
    - `max_duration_sec`: optional upper bound; longer samples are filtered out.
    - `add_speaker_id`: if True, each dataset item must declare `speaker_id_col_name`;
      that column is renamed to `speaker_id` and preserved in the output. To
      avoid speaker-label collisions across sources (the same string can mean
      different real speakers in two datasets), each value is prefixed with a
      4-char tag unique to its source dataset — generated from `uuid.uuid4()`
      at run start, or pinned via per-item `speaker_prefix:` for reproducibility.
    - `add_row_id`: if True, a `row_id` column (0-based sequential index) is appended
      to the final concatenated dataset after all sources are merged.
    - `singleton_policy`: how to treat rows whose speaker_id is unusable for
      cross-recording reference (low-count speakers or rows from datasets that
      don't declare `speaker_id_col_name`). "remove" drops them (legacy);
      "null_prefix" keeps them, rewriting `speaker_id` to `NULL_SPEAKER_SENTINEL`
      so the ReferenceSamplingDataset emits `force_null=True` and the model
      swaps the compressor prefix with the learnable `null_prefix` for those
      samples (the same branch cfg_dropout uses). Low-count counting is one
      columnar pass per source done in the main process **before sharding**
      (so cross-shard counts stay correct); the rewrite (or drop) bundles into
      the existing per-shard duration filter — zero extra Arrow rewrites.
      Useful because low-count speakers carry no learnable speaker prior for
      voice-cloning, but their text+audio is still valuable for the
      unconditional path.
    - `min_clips_per_speaker`: integer threshold. Speakers with strictly fewer
      than this many clips in a source are classified as low-count and routed
      via `singleton_policy`. 1 → only true singletons. K>1 → enables
      SupCon-style training where the batch sampler needs ≥K clips/speaker
      to form positives; speakers below that go unconditional.
    - `speaker_statistics`: when true, the pipeline dumps
      `speaker_statistics.json` next to the train dataset directory after
      `save_to_disk`. Contains per-source + global clip-count distributions
      and breakdown of rows routed to null vs. SupCon-eligible.
    - `train_dataset_path`: output directory for `save_to_disk`.

High-level flow
---------------
`DatasetProcessor`  (module-level orchestrator)
    └── builds a unique 4-char `speaker_prefix` per source (when add_speaker_id).
    └── for each entry in `hf_datasets`:
        `ItemDataset`  (single-source loader + column renamer)
            └── (optional) one columnar pass over `speaker_id` to find low-count
                speakers (those with strictly fewer than `min_clips_per_speaker`
                clips in this source); the resulting set is forwarded to every
                worker for the filter.
            └── shards the source dataset and runs, in parallel worker processes,
                `TrainDataPreProcessor`  (the actual per-sample transform)
                    - filters in one pass: duration ≤ max_duration_sec AND
                      first codec layer non-empty AND (under
                      `singleton_policy=remove`) speaker_id not in low-count set,
                    - `transform_row`: a single row-level visit composing
                        * `add_codes` — unfolds packed codec indices into N channels,
                        * `create_input_ids` — builds `text_ids`, per-channel
                          audio inputs/labels, `labels_stop`, `attention_mask`,
                        * speaker_id prefixing — `"<prefix>_<original_id>"`,
                        * (under `singleton_policy=null_prefix`) low-count rows
                          are NOT prefixed and their speaker_id is rewritten to
                          NULL_SPEAKER_SENTINEL so the ref sampler routes them
                          through the null_prefix branch.
            └── records per-source clip-count stats (when speaker_statistics=True).
    └── concatenates all processed datasets.
    └── (optional) appends `row_id` column to the final dataset.

Output schema (per row)
-----------------------
Columns the collator reads (all 1-D python lists of equal length within a row):
    - `text_ids`                : [SOT, text_tokens..., EOT, SOS]
    - `level_audio_{i}`         : per-channel discrete inputs for i in [0, N),
                                  shape == audio_len_for_row + 1 (last frame
                                  duplicated so SOS+frames fill the stop slot)
    - `labels_level_audio_{i}`  : [-100]*n_text + codes + [-100], length == n_text + audio_len + 1
    - `labels_stop`             : [-100]*n_text + [0]*audio_len + [1]
    - `attention_mask`          : 1s over the full n_text + audio_len + 1 positions
    - `encoded_len`             : passthrough from the source dataset
    - `speaker_id`              : (optional) speaker identifier, present when
                                  `add_speaker_id: true` in dataset_config.yaml.
                                  Stored as `"<4char_prefix>_<original_id>"` so
                                  ids are globally unique across source datasets.
    - `row_id`                  : (optional) 0-based global row index across all
                                  concatenated datasets, present when `add_row_id: true`

The mandatory trailing `-100` in audio labels and the duplicated last input
frame together implement the stop-prediction trick: after the causal shift in
the model, the last real frame predicts `stop=1` while contributing no audio
cross-entropy at the stop position.
"""

from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import dataclasses
import locale
import os
import random
import uuid
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import numpy as np
import json

from gepard.model.codec_ops import unfold_tokens_np
from gepard.data.collator import NULL_SPEAKER_SENTINEL
from gepard.data.preprocessing.text_repetition import TextRepetitionConfig, TextRepeater
from gepard.logging import get_logger, init_worker_logging, log_event


# Valid values for `data.singleton_policy`.
SINGLETON_POLICIES = {"remove", "null_prefix"}


class TrainDataPreProcessor:
    """Per-shard transform that turns raw codec rows into multihead training rows.

    This class is the core per-sample logic of the pipeline. One instance is
    constructed **per worker process** inside `ItemDataset.__call__` (see
    `process_shard`) and is stateful only w.r.t. configuration — it is safe to
    pickle across processes because the HuggingFace tokenizer is reloaded in
    `__init__` from its name, not passed as a live object.

    Responsibilities
    ----------------
    The shard is processed in exactly two passes: one combined `filter` and
    one combined `map(transform_row)`. Per-row, `transform_row` performs:

    1. Filter (combined, before the map) by audio duration, non-emptiness, and
       optional singleton-speaker drop. Rows with
       `encoded_len / frame_rate_hz > max_dur`, OR with the first codec layer
       null/empty, OR (when `low_count_speakers` is supplied) with a
       `speaker_id` that appears only once in the source dataset, are dropped.
       The frame rate comes from `audio_codec.frame_rate_hz` in the dataset
       config so it matches whichever codec produced the source data.
    2. Unfold each packed codebook index into per-dimension discrete codes
       (`add_codes`). For FSQ with levels `[L0, L1, ...]`, a packed index `k`
       is decomposed as `k // base_d % L_d` for each dimension `d`
       (mixed-radix, little-endian — see `unfold_tokens_np`). With
       `num_layers=8` and `fsq_Levels=[8,7,6,6]` this gives 32 channels.
       Can be disabled via `audio_codec.do_unfold=False`, in which case each
       codec layer becomes one channel and `fsq_Levels` is ignored.
    3. Tokenise the transcript with the configured HF tokenizer and wrap it
       with the TTS special-token frame (`create_input_ids`):
           `[SOT, <text tokens>, EOT, SOS]`
       Optionally prefixed with a language tag (`"{lang}: {text}"`) when
       `language_tag` is set on the item config.
    4. Build per-channel labels and the binary stop labels such that, after
       the model's causal shift (`logits[:-1]` vs `labels[1:]`):
           - SOS predicts frame_0 and stop=0,
           - the last real frame predicts stop=1 and no codebook (label -100),
           - the appended duplicated last frame provides input at the stop
             position but contributes no label.
    5. Prefix `speaker_id` with the per-source 4-char tag when both
       `add_speaker_id` and `speaker_prefix` are set, ensuring the resulting
       string ids are globally unique across source datasets.

    Sequence layout per row (see also `create_input_ids` docstring)
    ---------------------------------------------------------------
        input  : [SOT, txt.., EOT, SOS, frame_0, ..., frame_m, frame_m(dup)]
        labels : [-100]*n_text,          ch_0,    ..., ch_m,   -100          (per channel)
        stop   : [-100]*n_text,           0,      ...,   0,      1

    Parameters
    ----------
    tokenizer_name : str
        HF Hub id (or local path) of the tokenizer used for text. Must match
        the tokenizer the main model was / will be trained on, otherwise
        `text_ids` will be out of vocabulary.
    max_dur : int | None
        Maximum audio duration in seconds. Rows longer than this are filtered
        out in `__call__`. Pass falsy (0 / None) to disable filtering.
    tokens : gepard.config.schema.TokensConfig
        `tokeniser_length`, `start_of_text`, `end_of_text`, `start_of_speech`,
        `tts_pad` — the special-token vocabulary ids.
    codec : gepard.config.schema.CodecConfig
        `num_layers` (int), `fsq_levels` (list[int]) and the `do_unfold` flag.
    language_tag : str, optional
        When provided, prepended to every transcript as `"{tag.lower()}: "`.
    add_speaker_id : bool, optional
        When True, the `speaker_id` column (already renamed by `ItemDataset`) is
        preserved in the output instead of being dropped with other source columns.
    speaker_prefix : str, optional
        4-char tag (or any short string) prepended as `"{speaker_prefix}_"` to
        every `speaker_id` value when `add_speaker_id` is also True. Generated
        upstream by `DatasetProcessor` so each source dataset gets a unique
        prefix and string ids never collide across sources.
    low_count_speakers : set[str], optional
        Set of `speaker_id` values that appear in strictly fewer than the
        configured `min_clips_per_speaker` threshold in this source dataset.
        Computed in the main process before sharding (since shard-local
        counts would be wrong) and pickled into each worker. Action depends
        on `singleton_policy`: under "remove" the row-level filter drops
        `speaker_id in low_count_speakers` as part of the same
        `dataset.filter` pass; under "null_prefix" `transform_row` rewrites
        `speaker_id` to `NULL_SPEAKER_SENTINEL` instead (no row drop). Both
        paths cost zero extra Arrow rewrites.
    singleton_policy : str, optional
        "remove" (drop low-count rows) or "null_prefix" (mark them with a
        sentinel and route through the unconditional path at training time).

    Derived attributes
    ------------------
    channel_columns : list[str]
        Canonical names of the unfolded audio-input columns the model reads
        (`level_audio_0`, `level_audio_1`, …). Consumed directly by
        the derived `model.audio_heads` map.
    label_columns : list[str]
        `labels_level_audio_0`, `labels_level_audio_1`, … — per-channel
        codebook targets aligned with the concatenated text+audio sequence.
    """

    def __init__(self, tokenizer_name: str, max_dur: int, tokens,
                 codec, language_tag: str = None, add_speaker_id: bool = False,
                 speaker_prefix: str = None, low_count_speakers: set = None,
                 singleton_policy: str = "remove", shard_idx: int = 0,
                 repetition: TextRepetitionConfig = None) -> None:
        self.shard_idx = shard_idx
        self.log = get_logger(f"dataset.shard.{shard_idx}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_dur = max_dur
        self.language_tag = language_tag
        locale.getpreferredencoding = lambda: "UTF-8"

        self.tokeniser_length = tokens.tokeniser_length
        self.start_of_text = tokens.start_of_text
        self.end_of_text = tokens.end_of_text
        self.start_of_speech = tokens.start_of_speech
        self.pad_token = tokens.tts_pad

        # Adaptive text repetition (MODEL_GUIDE §5.2–5.3). Disabled config →
        # build_input_ids reproduces the legacy single-copy layout exactly.
        # Per-shard RNG offset keeps shards independent yet reproducible for the
        # training-only mixed-keep coin flip.
        self.repetition_cfg = repetition or TextRepetitionConfig()
        self.repeater = TextRepeater(
            self.repetition_cfg, self.start_of_text, self.end_of_text, self.start_of_speech,
        )
        self._rep_rng = random.Random(self.repetition_cfg.seed + shard_idx)
        if self.repetition_cfg.enabled:
            self.log.info(
                f"text repetition ON: target={self.repetition_cfg.target_text_tokens} "
                f"apply_below={self.repetition_cfg.apply_below} "
                f"max_R={self.repetition_cfg.max_repeats} "
                f"mixed_keep_prob={self.repetition_cfg.mixed_keep_prob}"
            )

        self.num_layers = codec.num_layers
        self.fsq_levels = list(codec.fsq_levels)
        self.do_unfold = codec.do_unfold
        self.frame_rate_hz = float(codec.frame_rate_hz)

        if self.do_unfold:
            self.num_channels = self.num_layers * len(self.fsq_levels)
        else:
            self.num_channels = self.num_layers
            self.log.warning(
                f"do_unfold=False — fsq_levels {self.fsq_levels} ignored, "
                f"using {self.num_channels} channels (one per codec layer)"
            )

        self.add_speaker_id = add_speaker_id
        self.speaker_prefix = speaker_prefix
        self.low_count_speakers = low_count_speakers
        if singleton_policy not in SINGLETON_POLICIES:
            raise ValueError(
                f"singleton_policy={singleton_policy!r} not in {sorted(SINGLETON_POLICIES)}"
            )
        self.singleton_policy = singleton_policy
        self.layer_columns = [f'nano_layer_{i+1}' for i in range(self.num_layers)]
        self.channel_columns = [f'level_audio_{i}' for i in range(self.num_channels)]
        self.label_columns = [f'labels_level_audio_{i}' for i in range(self.num_channels)]

    def add_codes(self, example):
        """Extract codec layers, optionally unfold FSQ into per-dimension channels."""
        codes = np.array([example[col] for col in self.layer_columns]).T  # [frames, num_layers]

        if self.do_unfold:
            # (num_layers, frames) -> (num_channels, frames)
            unfolded = unfold_tokens_np(codes.T, self.fsq_levels)
        else:
            unfolded = codes.T  # (num_layers, frames) as-is

        for i, col in enumerate(self.channel_columns):
            example[col] = unfolded[i].tolist()
        return example


    def create_input_ids(self, example):
        """Build text_ids and labels for multihead architecture.

        Sequence layout:
            input:  [SOT, txt.., EOT, SOS, frame_0, ..., frame_m, frame_m(dup)]
            labels: [-100 * n_text,          ch_0, ..., ch_m,     -100        ]  (per channel)
            stop:   [-100 * n_text,            0,  ...,   0,        1         ]

        The duplicated last frame serves as input at the stop position.
        After causal shift (logits[:-1] vs labels[1:]):
          - SOS predicts frame_0 + stop=0
          - frame_m predicts stop=1, ch=-100
          - dup frame (last pos) is dropped by shift

        When adaptive text repetition is enabled (MODEL_GUIDE §5.2), the text region
        grows to `[SOT,txt,EOT]*(R-1) + [SOT,txt,EOT,SOS]` for short prompts.
        `n_text` absorbs the extra copies, so the label/stop/mask construction
        below — all driven off `n_text` — is unchanged: the whole (repeated)
        text region stays masked (-100) and only the canonical SOS triggers
        audio. Long prompts get R=1 → identical to legacy.
        """
        if self.language_tag is not None:
            text_prompt = f"{self.language_tag.lower()}: {example['text']}"
        else:
            text_prompt = example["text"]

        text_tokens = self.text_tokenizer.encode(text_prompt, add_special_tokens=False)
        R = self.repeater.sample_R(len(text_tokens), self._rep_rng)
        text_ids = self.repeater.build_input_ids(text_tokens, R)
        n_text = len(text_ids)
        n_audio = len(example[self.channel_columns[0]])

        # Duplicate last frame as input for the stop position
        for ch in self.channel_columns:
            example[ch] = example[ch] + [example[ch][-1]]

        # Labels per channel: -100 for text, actual codes for audio, -100 for stop position
        for ch, lbl in zip(self.channel_columns, self.label_columns):
            codes = example[ch][:-1]  # original n_audio codes (without the dup)
            example[lbl] = [-100] * n_text + codes + [-100]

        # Stop labels: -100 for text, 0 for audio frames, 1 at stop position
        example["labels_stop"] = [-100] * n_text + [0] * n_audio + [1]

        example["text_ids"] = text_ids
        total_len = n_text + n_audio + 1  # +1 for dup frame
        example["attention_mask"] = [1] * total_len
        return example

    def transform_row(self, example):
        """Single-pass transform: unfold codes, build input_ids/labels, prefix speaker_id.

        Composes `add_codes` + `create_input_ids` so we touch each row once
        instead of round-tripping the unfolded `level_audio_*` columns through
        Arrow between the two passes. The optional speaker_prefix is applied
        here so it costs nothing on top — same row-level visit.

        Under `singleton_policy=null_prefix`, rows whose speaker is low-count
        (count < min_clips_per_speaker, or whose speaker_id was already
        synthesized as the sentinel by ItemDataset for speaker-less sources)
        keep `speaker_id = NULL_SPEAKER_SENTINEL` — no per-source prefix is
        applied, so all such rows collapse onto the same sentinel value across
        datasets and the ref sampler can index them as one bucket.
        """
        example = self.add_codes(example)
        example = self.create_input_ids(example)
        if self.add_speaker_id:
            sid = example["speaker_id"]
            is_low_count = self.low_count_speakers is not None and sid in self.low_count_speakers
            is_sentinel = sid == NULL_SPEAKER_SENTINEL
            if self.singleton_policy == "null_prefix" and (is_low_count or is_sentinel):
                example["speaker_id"] = NULL_SPEAKER_SENTINEL
            elif self.speaker_prefix:
                example["speaker_id"] = f"{self.speaker_prefix}_{sid}"
        return example

    def __call__(self, dataset: Dataset) -> Dataset:
        # The duration / non-empty filter already ran in `ItemDataset.__init__`
        # (main process, before sharding and before speaker counting). The only
        # row-drop left at the shard stage is low-count speaker removal under
        # `singleton_policy=remove`; under `null_prefix` no rows are dropped
        # here — `transform_row` rewrites low-count speakers to the sentinel.
        rows_in = len(dataset)
        drop_low_count = (
            self.low_count_speakers if self.singleton_policy == "remove" else None
        )
        if drop_low_count:
            log_event(
                self.log,
                f"low-count filter start ({rows_in:,} rows; policy=remove)",
                type="shard", shard=self.shard_idx, stage="filtering", rows_in=rows_in,
            )
            dataset = dataset.filter(
                lambda x: x['speaker_id'] not in drop_low_count,
                desc='Filter rows (drop low-count speakers)',
            )
            log_event(
                self.log,
                f"low-count filter done {rows_in:,} -> {len(dataset):,}",
                type="shard", shard=self.shard_idx, stage="filtered",
                rows_in=rows_in, rows_out=len(dataset),
            )

        log_event(
            self.log,
            "transform start",
            type="shard", shard=self.shard_idx, stage="mapping",
        )
        dataset = dataset.map(
            self.transform_row,
            remove_columns=self.layer_columns + ["text"],
            desc='Transform (unfold + input_ids + prefix): ',
        )

        columns_to_keep = (
            ["text_ids"] + self.channel_columns + self.label_columns +
            ["labels_stop", "attention_mask", "encoded_len"]
        )
        if self.add_speaker_id:
            columns_to_keep.append("speaker_id")
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        rows_out = len(dataset)
        log_event(
            self.log,
            f"transform done ({rows_out:,} rows)",
            type="shard", shard=self.shard_idx, stage="mapped", rows_out=rows_out,
        )
        return dataset


def process_shard(shard_idx, shard_data, tokenizer_name, max_dur, tokens, codec,
                  language_tag, add_speaker_id=False, speaker_prefix=None,
                  low_count_speakers=None, singleton_policy="remove", log_queue=None,
                  repetition=None):
    """Worker entry point used by `ItemDataset.__call__` via ProcessPoolExecutor.

    Instantiates a fresh `TrainDataPreProcessor` inside the worker process
    (the tokenizer and config values are re-materialised here, avoiding
    pickling issues with live HF tokenizer objects) and runs it on one shard
    of the source dataset. Returns the transformed HF `Dataset` shard.
    `low_count_speakers` is a `set` computed globally in the main process and
    pickled into each worker — so the low-count drop happens inside the
    existing per-shard filter pass (no extra Arrow rewrite).

    `log_queue`, when supplied, is the parent's `mp.Queue` used by
    `gepard.logging.init_worker_logging` to route every worker log record back
    into the parent's QueueListener (no stdout interleave, structured files).
    """
    if log_queue is not None:
        init_worker_logging(log_queue)
    log = get_logger(f"dataset.shard.{shard_idx}")
    rows_in = len(shard_data)
    log_event(
        log, f"worker started ({rows_in:,} rows)",
        type="shard", shard=shard_idx, stage="started", rows_in=rows_in,
    )
    try:
        processor = TrainDataPreProcessor(
            tokenizer_name, max_dur, tokens, codec, language_tag,
            add_speaker_id, speaker_prefix, low_count_speakers,
            singleton_policy=singleton_policy, shard_idx=shard_idx,
            repetition=repetition,
        )
        processed_shard = processor(shard_data)
    except Exception as exc:
        log_event(
            log, f"worker failed: {exc}",
            type="shard", shard=shard_idx, stage="error", error=str(exc),
        )
        raise
    log_event(
        log, f"worker done ({len(processed_shard):,} rows)",
        type="shard", shard=shard_idx, stage="done", rows_out=len(processed_shard),
    )
    return processed_shard


class ItemDataset:
    """Loader + parallel processor for a **single** source dataset.

    One `ItemDataset` instance corresponds to one entry under `hf_datasets`
    in `dataset_config.yaml`. It is responsible for:

    1. Loading the source dataset, either from the HuggingFace Hub
       (`local=False`, default) or from a local `save_to_disk` directory
       (`local=True`, useful for re-running on preprocessed inputs or
       offline hosts).
    2. Validating that the source declares exactly `audio_codec.num_layers`
       `nano_layer_*` columns — a mismatch is a configuration bug and raises
       immediately rather than producing silently-incorrect data. When
       `add_speaker_id=True`, also validates that `speaker_id_col_name` is
       present and non-null in the item config.
    3. Renaming source-specific columns to the canonical names the rest of
       the pipeline expects:
           - `<text_col_name>`         → `text`
           - `<encoded_len>`           → `encoded_len`
           - `<nano_layer_i source>`   → `nano_layer_i` for i in [1, num_layers]
           - `<speaker_id_col_name>`   → `speaker_id`  (only when `add_speaker_id=True`)
       The mapping is read from the item config keys
       `text_col_name`, `encoded_len`, `nano_layer_1`, …, `nano_layer_{N}`,
       and optionally `speaker_id_col_name`.
    3a. (When `add_speaker_id=True` and the source declares
        `speaker_id_col_name`) one columnar pass over the renamed `speaker_id`
        column with `Counter` to build the low-count-speaker set (speakers
        with strictly fewer than `min_clips_per_speaker` clips). The set is
        forwarded to every worker; the per-shard filter then either drops
        those rows (`singleton_policy=remove`) or `transform_row` rewrites
        them to `NULL_SPEAKER_SENTINEL` (`singleton_policy=null_prefix`).
        Either path bundles into the existing filter/map, so zero extra
        Arrow rewrites are paid for it. For sources WITHOUT a speaker_id
        column, an `add_column` pass synthesizes a sentinel-valued column
        in one shot (only allowed under `singleton_policy=null_prefix`).
        Per-source clip-count statistics are stored on `self.item_stats`
        and surfaced by `DatasetProcessor` when speaker_statistics=True.
    4. Splitting the (possibly huge) dataset into `n_shards` contiguous shards
       and dispatching each shard to a worker process via `ProcessPoolExecutor`,
       running `process_shard` in each. The per-worker transform is
       `TrainDataPreProcessor` — see its docstring for the row-level semantics.
    5. Concatenating processed shards back in original order and optionally
       subsampling via `item_cfg.max_len` (shuffled with a fixed seed for
       reproducibility).

    Parallelism
    -----------
    `n_shards` defaults to `min(cpu_count(), 8)` when not explicitly set; the
    CLI (`python -m data.prepare --n-shards N`) plumbs through to this. Each
    shard becomes one child process. Tokenizer init happens once per worker,
    so heavy dataset loading on the main process is not blocked.

    Parameters
    ----------
    item_cfg : OmegaConf
        One entry of the `hf_datasets` list. Expected keys:
            `reponame` (str): HF repo id or local dir.
            `local` (bool, default False): if True, use `load_from_disk`.
            `name` (str | None): HF dataset config name.
            `split` (str): HF split, e.g. "train" or "train[:1000]".
            `text_col_name` (str): source column holding the transcript.
            `encoded_len` (str): source column with frame-count per sample.
            `nano_layer_1..N` (str): source column names per codec layer.
            `language_tag` (str, optional): passed through to
                `TrainDataPreProcessor` to prefix text with a language marker.
            `max_len` (int, optional): cap on number of rows retained after
                processing (useful for dev / debug).
            `speaker_id_col_name` (str): source column holding the speaker
                identifier. Required (non-null) when `add_speaker_id=True`.
            `speaker_prefix` (str, optional): pin the per-source 4-char tag
                used to namespace `speaker_id` (e.g. `"jjrw"`). When omitted,
                `DatasetProcessor` generates a uuid4-based one at run start.
                Set this if you need stable speaker ids across re-runs.
    tokenizer_name : str
        Tokenizer id forwarded to each worker.
    max_dur : int | None
        Max-duration filter forwarded to each worker.
    tokens, codec : gepard.config.schema.{TokensConfig, CodecConfig}
        Global special-token + codec configuration forwarded to each worker.
    n_shards : int, optional
        Number of parallel shards; falls back to `min(cpu_count(), 8)`.
    add_speaker_id : bool, optional
        When True, validates `speaker_id_col_name`, renames it to `speaker_id`,
        and forwards the flag to each worker so the column is kept in output.
    speaker_prefix : str, optional
        Per-source 4-char tag (resolved by `DatasetProcessor`) prepended to each
        `speaker_id` value as `"{speaker_prefix}_{original_id}"`. Forwarded to
        every worker so the rename happens in the same `transform_row` map pass.
    singleton_policy : str, optional
        "remove" (legacy: drop low-count rows; speaker-less datasets are
        rejected) or "null_prefix" (keep low-count rows + synthesize sentinel
        speaker_id for speaker-less sources, route through null_prefix at
        train time). When `add_speaker_id=True`, this triggers one columnar
        pass over `speaker_id` after rename to build a `set` of speakers
        with strictly fewer than `min_clips_per_speaker` clips; the set is
        forwarded to every worker, and the rewrite (or drop) is bundled into
        the existing per-shard filter.
    min_clips_per_speaker : int, optional
        Threshold for the low-count classifier. A speaker with fewer than
        this many clips in the source is routed through `singleton_policy`.
        Default 1 reproduces the legacy "singleton" semantics.
    """

    def __init__(self, item_cfg: OmegaConf, tokenizer_name: str, max_dur: int,
                 tokens=None, codec=None, n_shards: int = None,
                 load_dataset_num_proc: int = 10, filter_num_proc: int = 20,
                 add_speaker_id: bool = False, speaker_prefix: str = None,
                 singleton_policy: str = "remove", min_clips_per_speaker: int = 1,
                 log_queue=None, item_idx: int = 1, item_total: int = 1,
                 repetition: TextRepetitionConfig = None):
        self.log = get_logger("dataset.item")
        self.item_cfg = item_cfg
        self.tokenizer_name = tokenizer_name
        self.max_dur = max_dur
        self.tokens = tokens
        self.codec = codec
        self.add_speaker_id = add_speaker_id
        self.speaker_prefix = speaker_prefix
        self.repetition = repetition or TextRepetitionConfig()
        self.load_dataset_num_proc = load_dataset_num_proc
        self.filter_num_proc = filter_num_proc
        if singleton_policy not in SINGLETON_POLICIES:
            raise ValueError(
                f"singleton_policy={singleton_policy!r} not in {sorted(SINGLETON_POLICIES)}"
            )
        self.singleton_policy = singleton_policy
        self.min_clips_per_speaker = max(1, int(min_clips_per_speaker))
        self.log_queue = log_queue
        self.item_idx = item_idx
        self.item_total = item_total
        self.language_tag = self.item_cfg.get('language_tag')
        self.max_len = self.item_cfg.get('max_len')
        is_local = bool(self.item_cfg.get('local', False))
        # Filled by _compute_low_count_speakers / __init__ tail; consumed by
        # DatasetProcessor for the speaker_statistics dump.
        self.speaker_counts: Optional[Counter] = None
        self.item_stats: Dict[str, Any] = {}

        if n_shards is None:
            self.n_shards = min(mp.cpu_count(), 8)
        else:
            self.n_shards = n_shards

        if is_local:
            self.log.info(f"loading local dataset: {item_cfg.reponame}")
            self.dataset = load_from_disk(self.item_cfg.reponame)
        else:
            self.log.info(f"loading {item_cfg.reponame} (name={item_cfg.name}, split={item_cfg.split})")
            self.dataset = load_dataset(
                self.item_cfg.reponame,
                self.item_cfg.name,
                split=self.item_cfg.split,
                num_proc=self.load_dataset_num_proc,
            )

        self.log.info(f"loaded {len(self.dataset):,} rows; will use {self.n_shards} shards")

        num_layers = codec.num_layers
        self._validate_layer_columns(num_layers)
        # Whether this source actually carries speaker labels. Under
        # singleton_policy=remove this is required (legacy hard error). Under
        # singleton_policy=null_prefix a missing column is allowed: every row
        # gets a synthesized `speaker_id == NULL_SPEAKER_SENTINEL`, which the
        # ref sampler will route through the null_prefix path.
        self._has_speaker_col = bool(self.item_cfg.get('speaker_id_col_name'))
        if add_speaker_id and not self._has_speaker_col:
            if singleton_policy == "remove":
                self._validate_speaker_id_column()  # raises with the legacy message
            else:
                self.log.info(
                    f'no speaker_id_col_name for "{item_cfg.reponame}" — '
                    f'all rows will be marked as null-ref under singleton_policy=null_prefix'
                )

        rename_dict = {self.item_cfg.text_col_name: 'text', self.item_cfg.encoded_len: 'encoded_len'}
        for i in range(1, num_layers + 1):
            src_col = self.item_cfg[f'nano_layer_{i}']
            rename_dict[src_col] = f'nano_layer_{i}'
        if add_speaker_id and self._has_speaker_col:
            rename_dict[self.item_cfg.speaker_id_col_name] = 'speaker_id'
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.log.debug(f"renamed columns: {rename_dict}")

        # Synthesize a sentinel-valued speaker_id column for sources that don't
        # provide one. Done after rename so the rest of the pipeline can treat
        # the column as always present when add_speaker_id=True.
        if add_speaker_id and not self._has_speaker_col:
            self.dataset = self.dataset.add_column(
                'speaker_id', [NULL_SPEAKER_SENTINEL] * len(self.dataset)
            )
            self.log.info(
                f'synthesized speaker_id column with sentinel for {len(self.dataset):,} rows'
            )

        # Duration / non-empty filter — applied HERE, in the main process,
        # before sharding and before speaker counting. Previously it ran
        # per-shard inside the workers (AFTER counting), so a speaker counted
        # with >=K clips pre-filter could end up with <K post-filter: a real
        # (non-sentinel) speaker_id that the SupCon sampler then rejects as
        # ineligible, leaving its rows in no batch at all. Counting on the
        # already-filtered dataset closes that gap. Running before sharding
        # also keeps `add_codes` protected from None/empty codec inputs.
        # Parallelism is configurable via `processing.filter_num_proc`.
        self._n_rows_loaded = len(self.dataset)
        fr = codec.frame_rate_hz
        max_dur = self.max_dur
        self.dataset = self.dataset.filter(
            lambda x: (x['nano_layer_1'] is not None and len(x['nano_layer_1']) > 0)
                      and (not max_dur or x['encoded_len'] / fr <= max_dur),
            num_proc=self.filter_num_proc,
            desc='Filter rows (duration + non-empty)',
        )
        self.log.info(
            f"duration/non-empty filter: {self._n_rows_loaded:,} -> "
            f"{len(self.dataset):,} rows"
        )

        # Low-count speakers (those whose count in this source is strictly less
        # than `min_clips_per_speaker`) are useless for cross-recording
        # voice-cloning AND insufficient as a positive-bucket for a SupCon-style
        # batch sampler that wants K samples per speaker. Counting must happen
        # here, before sharding, since shard-local counts would be wrong (a
        # speaker could be a low-count in one shard but not globally). What
        # happens to those rows depends on `singleton_policy`:
        #   - remove:      drop them in the per-shard filter pass.
        #   - null_prefix: keep them; transform_row rewrites speaker_id to
        #                  NULL_SPEAKER_SENTINEL. No row drop.
        # Counting is skipped when we already know every row is sentinel
        # (speaker-less source under null_prefix policy) — every speaker would
        # be the same sentinel and the low-count set would be the singleton
        # `{NULL_SPEAKER_SENTINEL}` only when count is 0 which never happens.
        needs_count = add_speaker_id and self._has_speaker_col
        if needs_count:
            self.low_count_speakers, self.speaker_counts = (
                self._compute_low_count_speakers(self.min_clips_per_speaker)
            )
        else:
            self.low_count_speakers = None
            # Speaker-less source under null_prefix: synthesize a Counter so
            # the statistics dump remains uniform across sources.
            if add_speaker_id:
                self.speaker_counts = Counter({NULL_SPEAKER_SENTINEL: len(self.dataset)})
            else:
                self.speaker_counts = None

        # Record raw per-source stats. Finalised in `__call__` once the post-
        # filter row count is known.
        self.item_stats = self._build_initial_stats()

    def _compute_low_count_speakers(self, min_count: int):
        """Single columnar pass over `speaker_id` to find low-count speakers.

        Returns a tuple `(set[str], Counter)`:
          - set: speaker ids whose count is strictly less than `min_count`
          - Counter: full speaker_id → clip-count mapping (kept for stats)

        Iteration is batched (`Dataset.iter`) so we never materialise the full
        column as one giant Python list, which matters for multi-million-row
        sources.
        """
        self.log.info(
            f"counting speakers across {len(self.dataset):,} rows "
            f"(low-count threshold: <{min_count} clips)"
        )
        counts: Counter = Counter()
        for batch in self.dataset.select_columns(['speaker_id']).iter(batch_size=50_000):
            counts.update(batch['speaker_id'])
        low_count = {sp for sp, c in counts.items() if c < min_count}
        rows_low_count = sum(c for sp, c in counts.items() if sp in low_count)
        action = "dropped" if self.singleton_policy == "remove" else "routed to null_prefix"
        self.log.info(
            f"{len(counts):,} unique speakers; {len(low_count):,} have <{min_count} clips "
            f"({rows_low_count:,} rows) — will be {action} at the filter stage"
        )
        return low_count, counts

    @staticmethod
    def _clip_count_histogram(counts: Optional[Counter]) -> Dict[str, int]:
        """Bucket clip-counts into a fixed histogram for the statistics dump."""
        if not counts:
            return {}
        buckets = {"1": 0, "2": 0, "3": 0, "4-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
        for c in counts.values():
            if c == 1:        buckets["1"] += 1
            elif c == 2:      buckets["2"] += 1
            elif c == 3:      buckets["3"] += 1
            elif c <= 5:      buckets["4-5"] += 1
            elif c <= 10:     buckets["6-10"] += 1
            elif c <= 20:     buckets["11-20"] += 1
            else:             buckets["21+"] += 1
        return buckets

    def _build_initial_stats(self) -> Dict[str, Any]:
        """Compute the per-source clip-count stats from `self.speaker_counts`.

        Speaker counts here are already post duration/non-empty filter (that
        filter runs in `__init__`, before counting). `n_rows_initial` is the
        raw loaded count; `n_rows_final` (post low-count drop + transform) is
        patched in by `__call__`. Safe to call even when speaker_statistics is
        disabled — it is a few Python comprehensions.
        """
        cnt = self.speaker_counts or Counter()
        threshold = self.min_clips_per_speaker
        low_count = sum(1 for c in cnt.values() if c < threshold)
        rows_low = sum(c for c in cnt.values() if c < threshold)
        rows_high = sum(c for c in cnt.values() if c >= threshold)
        return {
            "name": self.item_cfg.get("name") or self.item_cfg.reponame,
            "speaker_prefix": self.speaker_prefix,
            "has_speaker_col": self._has_speaker_col,
            "min_clips_per_speaker": threshold,
            "n_rows_initial": self._n_rows_loaded,
            "n_rows_final": None,  # filled after filter pass in __call__
            "n_speakers": len(cnt),
            "n_low_count_speakers": low_count,
            "n_low_count_rows": rows_low,
            "n_supcon_eligible_speakers": max(0, len(cnt) - low_count),
            "n_supcon_eligible_rows": rows_high,
            "clip_count_histogram": self._clip_count_histogram(cnt),
        }

    def _validate_layer_columns(self, num_layers: int):
        """Check that item_cfg declares exactly num_layers nano_layer_* fields."""
        cfg_layer_keys = [k for k in self.item_cfg if k.startswith('nano_layer_')]
        if len(cfg_layer_keys) != num_layers:
            raise ValueError(
                f'codec.num_layers={num_layers} but dataset "{self.item_cfg.reponame}" '
                f'declares {len(cfg_layer_keys)} nano_layer_* columns: {cfg_layer_keys}'
            )

    def _validate_speaker_id_column(self):
        """Check that item_cfg declares a non-null speaker_id_col_name when add_speaker_id is True."""
        speaker_col = self.item_cfg.get('speaker_id_col_name')
        if not speaker_col:
            raise ValueError(
                f'add_speaker_id=True but dataset "{self.item_cfg.reponame}" '
                f'does not declare speaker_id_col_name (or it is null)'
            )

    def __call__(self):
        # Tell the dashboard a new source is starting so it resets the
        # per-shard table to a clean slate. The dashboard ignores this event
        # if no live UI is attached.
        item_name = self.item_cfg.get("name") or self.item_cfg.reponame
        log_event(
            self.log, f"start [{self.item_idx}/{self.item_total}] {self.item_cfg.reponame}",
            type="item_start", name=item_name,
            idx=self.item_idx, total=self.item_total, n_shards=self.n_shards,
        )

        shards = []
        for i in range(self.n_shards):
            shard = self.dataset.shard(num_shards=self.n_shards, index=i)
            shards.append((shard, i))
        self.log.debug(f"sharded into {self.n_shards} pieces")

        processed_shards = []

        with ProcessPoolExecutor(max_workers=self.n_shards) as executor:
            future_to_shard = {
                executor.submit(
                    process_shard, shard_idx, shard, self.tokenizer_name, self.max_dur,
                    self.tokens, self.codec, self.language_tag,
                    self.add_speaker_id, self.speaker_prefix, self.low_count_speakers,
                    self.singleton_policy, self.log_queue, self.repetition,
                ): shard_idx
                for shard, shard_idx in shards
            }

            for future in as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                try:
                    processed_shard = future.result()
                    processed_shards.append((shard_idx, processed_shard))
                except Exception as exc:
                    self.log.error(f"shard {shard_idx} failed: {exc}")
                    raise

        processed_shards.sort(key=lambda x: x[0])
        final_shards = [shard for _, shard in processed_shards]

        self.log.info(f"concatenating {len(final_shards)} shards")
        final_dataset = concatenate_datasets(final_shards)
        if self.max_len is not None:
            final_dataset = final_dataset.shuffle(seed=42).select(range(int(self.max_len)))
            self.log.info(f"capped at max_len={self.max_len:,}")

        # Patch the final row count into stats. `self.speaker_counts` is
        # already post duration/non-empty filter (counted in __init__); this
        # records the count after the optional low-count drop + transform.
        self.item_stats["n_rows_final"] = len(final_dataset)

        log_event(
            self.log, f"done [{self.item_idx}/{self.item_total}] {item_name}: {len(final_dataset):,} rows",
            type="item_done", name=item_name, rows=len(final_dataset),
        )
        return final_dataset


class DatasetProcessor:
    """Top-level orchestrator that builds the final training dataset.

    This is the class invoked by `gepard.cli.prepare`. It:

    1. Takes the composed `PrepareConfig` via
       `load_config()`. All downstream behaviour is driven from that file.
    2. When `add_speaker_id: true`, generates a unique 4-char `speaker_prefix`
       per source dataset (`uuid.uuid4().hex[:4]`, deduped within the run, or
       a value pinned via per-item `speaker_prefix:` for re-run reproducibility).
       The chosen prefixes are logged at startup. Each source's `speaker_id`
       column is rewritten as `"{prefix}_{original_id}"` inside the worker so
       string ids are globally unique across sources — no collisions across
       datasets that happen to use the same speaker labels.
    2a. The `singleton_policy` field (one of "remove" / "null_prefix") and
        the `min_clips_per_speaker` threshold are forwarded to each
        `ItemDataset`. When `add_speaker_id: true`, that triggers a one-pass
        `Counter` over `speaker_id` before sharding; depending on the policy,
        low-count rows (count < threshold) are either dropped in the per-shard
        filter or rewritten to a null sentinel in `transform_row`.
    3. For each entry under `hf_datasets`, constructs an `ItemDataset` and
       runs it (load → rename → parallel-transform → concat → optional cap).
    4. Concatenates the per-source processed datasets into a single HF
       `Dataset` ready for `save_to_disk`. The caller (prepare.py) handles
       the actual write and the final output path resolution.
    5. Optionally appends a `row_id` column (0-based sequential index) to the
       final concatenated dataset when `add_row_id: true` in the config.
    6. Optionally collects per-source + global clip-count statistics across
       *all* processed sources when `speaker_statistics: true` in the config.
       Use `save_speaker_statistics(path)` after `__call__` to dump a JSON
       summarising the speaker → clip-count distribution, the partition
       between low-count (routed via `singleton_policy`) and SupCon-eligible
       rows, and a clip-count histogram. Useful for calibrating
       `min_clips_per_speaker` and the trainer-side SupCon batch sampler.

    Usage
    -----
    Typical call site (see prepare.py):

        processor = DatasetProcessor(n_shards_per_dataset=args.n_shards)
        ds = processor()
        ds.save_to_disk(processor.cfg.train_dataset_path)
        if processor.cfg.get('speaker_statistics', False):
            processor.save_speaker_statistics(output_path / "speaker_statistics.json")

    Parameters
    ----------
    n_shards_per_dataset : int, optional
        Number of parallel worker shards to use *per source dataset*.
        Forwarded verbatim to every `ItemDataset`. When None, each
        `ItemDataset` picks its own default (see its docstring).

    Attributes
    ----------
    cfg : OmegaConf
        The loaded dataset configuration.
    speaker_prefixes : list[str | None]
        One entry per source dataset (same order as `cfg.hf_datasets`). Each
        is a 4-char tag forwarded to the corresponding `ItemDataset`/worker,
        or `None` when `cfg.add_speaker_id` is False.
    singleton_policy : str
        Mirror of `cfg.singleton_policy` (default "remove"). Forwarded to
        every `ItemDataset`. Validated against `SINGLETON_POLICIES` at init.
        Setting this to anything other than "remove" without
        `add_speaker_id=True` raises at construction time (no speaker_id
        column to count or rewrite without it).
    min_clips_per_speaker : int
        Mirror of `cfg.min_clips_per_speaker` (default 1). Threshold used by
        each `ItemDataset` to classify speakers as low-count.
    all_item_stats : list[dict]
        Per-source clip-count statistics collected after every `ItemDataset`
        runs. Used by `save_speaker_statistics`. Populated regardless of
        whether `cfg.speaker_statistics` is True (cost is negligible);
        only the on-disk JSON dump is gated on the flag.
    """

    def __init__(self, cfg=None, n_shards_per_dataset: int = None, log_queue=None):
        self.log = get_logger("dataset.processor")
        if cfg is None:
            from gepard.config import load_prepare
            cfg = load_prepare([])
        self.cfg = cfg
        self.tokens = cfg.tokens
        self.codec = cfg.codec
        self.tokenizer_name = cfg.tokens.tokenizer_name
        data = cfg.data
        self.add_speaker_id = bool(data.add_speaker_id)
        self.add_row_id = bool(data.add_row_id)
        self.singleton_policy = str(data.singleton_policy)
        if self.singleton_policy not in SINGLETON_POLICIES:
            raise ValueError(
                f'data.singleton_policy={self.singleton_policy!r} '
                f'must be one of {sorted(SINGLETON_POLICIES)}'
            )
        if self.singleton_policy != "remove" and not self.add_speaker_id:
            raise ValueError(
                f'data.singleton_policy={self.singleton_policy!r} requires '
                'data.add_speaker_id=true (no speaker_id column to count or rewrite without it)'
            )
        self.min_clips_per_speaker = max(1, int(data.min_clips_per_speaker))
        # Adaptive text repetition (MODEL_GUIDE §5). Opt-in via the
        # `text_layout` group; disabled → legacy single-copy layout.
        self.repetition = TextRepetitionConfig.from_config(
            dataclasses.asdict(cfg.text_layout)
        )
        if self.repetition.enabled:
            self.log.info(
                f"text_repetition enabled: target={self.repetition.target_text_tokens}, "
                f"apply_below={self.repetition.apply_below}, "
                f"max_repeats={self.repetition.max_repeats}, "
                f"mixed_keep_prob={self.repetition.mixed_keep_prob}"
            )
        if self.min_clips_per_speaker > 1 and self.singleton_policy == "remove":
            self.log.warning(
                f"min_clips_per_speaker={self.min_clips_per_speaker} with "
                f"singleton_policy='remove' will DROP all rows whose speaker has "
                f"<{self.min_clips_per_speaker} clips. Use 'null_prefix' to keep "
                "them as unconditional training data."
            )
        # Data-prep parallelism — `data.processing`. A `--n-shards` CLI arg,
        # when given, overrides processing.num_shards for that run.
        proc_cfg = data.processing
        self.n_shards_per_dataset = (
            n_shards_per_dataset if n_shards_per_dataset is not None
            else int(proc_cfg.num_shards)
        )
        self.load_dataset_num_proc = int(proc_cfg.load_dataset_num_proc)
        self.filter_num_proc = int(proc_cfg.filter_num_proc)
        self.log_queue = log_queue
        # Source entries as OmegaConf nodes: item configs are open dicts
        # (per-source column names vary), consumed via attribute access.
        self.hf_datasets = [OmegaConf.create(item) for item in data.hf_datasets]
        # Per-source stats collected after each ItemDataset finishes.
        self.all_item_stats: List[Dict[str, Any]] = []

        # Build a unique 4-char prefix per source dataset so that speaker_id
        # collisions across sources are impossible (the same string label can
        # mean different real speakers in two datasets). If item_cfg sets
        # `speaker_prefix:` explicitly, use it; otherwise generate a uuid4-based
        # 4-char hex tag and dedupe within the run.
        if self.add_speaker_id:
            self.speaker_prefixes = self._build_speaker_prefixes()
            prefix_str = ", ".join(
                f"{p}→{item_cfg.get('name') or item_cfg.reponame}"
                for item_cfg, p in zip(self.hf_datasets, self.speaker_prefixes)
            )
            self.log.info(f"speaker prefixes: {prefix_str}")
        else:
            self.speaker_prefixes = [None] * len(self.hf_datasets)

        self.log.info(
            f"ready: {len(self.hf_datasets)} sources, tokenizer={self.tokenizer_name}, "
            f"shards={self.n_shards_per_dataset}, "
            f"load_num_proc={self.load_dataset_num_proc}, filter_num_proc={self.filter_num_proc}"
        )

    def _build_speaker_prefixes(self):
        """Generate one unique 4-char prefix per dataset.

        Resolution order per dataset:
          1. `item_cfg.speaker_prefix` if set (lets the user pin a stable tag
             across re-runs so trained speaker embeddings stay valid).
          2. `uuid.uuid4().hex[:4]` — random 4 hex chars, regenerated until
             unique within this run.
        """
        prefixes = []
        used = set()
        for item_cfg in self.hf_datasets:
            explicit = item_cfg.get('speaker_prefix')
            if explicit:
                if explicit in used:
                    raise ValueError(
                        f'duplicate speaker_prefix "{explicit}" in data.hf_datasets '
                        f'(dataset {item_cfg.reponame})'
                    )
                prefixes.append(explicit)
                used.add(explicit)
                continue
            for _ in range(1000):
                candidate = uuid.uuid4().hex[:4]
                if candidate not in used:
                    used.add(candidate)
                    prefixes.append(candidate)
                    break
            else:
                raise RuntimeError('failed to generate a unique 4-char speaker prefix')
        return prefixes


    def __call__(self):
        self.log.info(
            f"starting master pipeline: {len(self.hf_datasets)} sources, "
            f"singleton_policy={self.singleton_policy}, "
            f"min_clips_per_speaker={self.min_clips_per_speaker}"
        )
        datasets = []
        total = len(self.hf_datasets)

        for i, item_cfg in enumerate(self.hf_datasets, 1):
            item_ds_maker = ItemDataset(
                item_cfg=item_cfg,
                tokenizer_name=self.tokenizer_name,
                max_dur=self.cfg.data.max_duration_sec,
                tokens=self.tokens,
                codec=self.codec,
                n_shards=self.n_shards_per_dataset,
                load_dataset_num_proc=self.load_dataset_num_proc,
                filter_num_proc=self.filter_num_proc,
                add_speaker_id=self.add_speaker_id,
                speaker_prefix=self.speaker_prefixes[i - 1],
                singleton_policy=self.singleton_policy,
                min_clips_per_speaker=self.min_clips_per_speaker,
                log_queue=self.log_queue,
                item_idx=i,
                item_total=total,
                repetition=self.repetition,
            )
            processed_dataset = item_ds_maker()
            datasets.append(processed_dataset)
            # item_stats is finalised by ItemDataset.__call__ (post-filter row
            # count patched in); store it for save_speaker_statistics.
            self.all_item_stats.append(item_ds_maker.item_stats)

        self.log.info(f"concatenating {len(datasets)} processed sources")
        final_dataset = concatenate_datasets(datasets)
        self.log.info(f"final dataset: {len(final_dataset):,} rows")

        if self.add_row_id:
            final_dataset = final_dataset.add_column('row_id', list(range(len(final_dataset))))
            self.log.info(f"row_id column added (0..{len(final_dataset) - 1:,})")

        return final_dataset

    def save_speaker_statistics(self, output_path: str):
        """Aggregate per-source clip-count statistics and dump them to JSON.

        The dump captures, for each source and globally:
          - total speaker count and total row count,
          - number of low-count speakers (count < min_clips_per_speaker) and
            the number of rows they account for (routed via singleton_policy),
          - number of SupCon-eligible speakers (count >= threshold) and their
            row contribution,
          - a clip-count histogram (1, 2, 3, 4-5, 6-10, 11-20, 21+).

        Used to calibrate `min_clips_per_speaker` and the trainer-side
        SupCon batch sampler (e.g. picking P×K so the eligible bucket can
        sustain the sampler's draw rate).

        Args:
            output_path: Path where to save the statistics JSON file.
        """
        if not self.all_item_stats:
            self.log.warning("no per-source stats collected; skipping dump")
            return

        # Per-source rows kept as-is. Build a global aggregate too.
        def _zero_hist() -> Dict[str, int]:
            return {"1": 0, "2": 0, "3": 0, "4-5": 0, "6-10": 0, "11-20": 0, "21+": 0}

        global_hist = _zero_hist()
        total_speakers = 0
        total_rows_initial = 0
        total_rows_final = 0
        total_low_speakers = 0
        total_low_rows = 0
        total_supcon_speakers = 0
        total_supcon_rows = 0

        for s in self.all_item_stats:
            total_speakers += int(s.get("n_speakers", 0))
            total_rows_initial += int(s.get("n_rows_initial", 0))
            total_rows_final += int(s.get("n_rows_final", 0) or 0)
            total_low_speakers += int(s.get("n_low_count_speakers", 0))
            total_low_rows += int(s.get("n_low_count_rows", 0))
            total_supcon_speakers += int(s.get("n_supcon_eligible_speakers", 0))
            total_supcon_rows += int(s.get("n_supcon_eligible_rows", 0))
            for k, v in (s.get("clip_count_histogram") or {}).items():
                if k in global_hist:
                    global_hist[k] += int(v)

        # Effective null exposure from the data side alone (before stochastic
        # cfg_dropout). Speakers below threshold → null_prefix path.
        denom = max(1, total_low_rows + total_supcon_rows)
        pct_structural_null = 100.0 * total_low_rows / denom

        statistics = {
            "config": {
                "singleton_policy": self.singleton_policy,
                "min_clips_per_speaker": self.min_clips_per_speaker,
            },
            "global": {
                "n_sources": len(self.all_item_stats),
                "total_speakers": total_speakers,
                "total_rows_initial": total_rows_initial,
                "total_rows_final": total_rows_final,
                "low_count": {
                    "n_speakers": total_low_speakers,
                    "n_rows": total_low_rows,
                },
                "supcon_eligible": {
                    "n_speakers": total_supcon_speakers,
                    "n_rows": total_supcon_rows,
                },
                "pct_structural_null_exposure": round(pct_structural_null, 2),
                "clip_count_histogram": global_hist,
            },
            "per_source": self.all_item_stats,
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        self.log.info(f"speaker stats saved to {output_path}")
        self.log.info(
            f"speakers total={total_speakers:,}  "
            f"low_count={total_low_speakers:,} ({total_low_rows:,} rows)  "
            f"supcon_eligible={total_supcon_speakers:,} ({total_supcon_rows:,} rows)  "
            f"structural_null={pct_structural_null:.1f}%"
        )



