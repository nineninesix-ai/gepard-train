"""Structured-config schema for the Gepard pipeline.

These dataclasses are the single schema behind three surfaces:
  1. Hydra ConfigStore — type + required-field checks at compose time.
  2. `gepard.config.load_*` — merge YAML → typed object (`OmegaConf.to_object`).
  3. the checkpoint `gepard_config.json` — the MODEL-BUILD subset marked [B]
     below is what `save_model` serializes so inference stops reading YAML.

Field roles: [B]=model-build (→config.json), [T]=train-only, [D]=data/prep,
[R]=derived. Values live in the `conf/` YAML tree, NOT here — the defaults below
are only fallbacks; mandatory identity/architecture fields are `MISSING` so a bad
composition fails loudly. Design rationale for the field *meanings* lives in
docs/MODEL_GUIDE.md; this file is deliberately comment-light.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


# ─────────────────────────── global invariants (→ config.json) ──────────────
@dataclass
class TokensConfig:
    """Special-token map + tokenizer id. Single source of truth for the token
    ids used across prep/train/inference. MODEL_GUIDE §2.3, §5.2. [B]"""
    tokenizer_name: str = MISSING
    tokeniser_length: int = MISSING
    start_of_text: int = MISSING
    end_of_text: int = MISSING
    start_of_speech: int = MISSING
    end_of_speech: int = MISSING
    tts_pad: int = MISSING

    def token_map(self) -> Dict[str, int]:
        """The special-token id map in the checkpoint `gepard_config.json`
        convention (token ids only, no tokenizer id)."""
        d = dataclasses.asdict(self)
        d.pop("tokenizer_name")
        return d


@dataclass
class CodecConfig:
    """GroupFSQ codec geometry. Drives the 32 audio heads and the compressor
    input dim. MODEL_GUIDE §2.2–2.3. [B]"""
    num_layers: int = MISSING
    fsq_levels: List[int] = MISSING
    do_unfold: bool = True
    frame_rate_hz: float = MISSING
    codec_id: str = MISSING
    sample_rate: int = MISSING


@dataclass
class TextLayoutConfig:
    """Adaptive text-repetition layout. Must match train↔inference byte-for-byte.
    MODEL_GUIDE §5.2–5.3. build-relevant fields [B]; `mixed_keep_prob`/`seed` [T]."""
    enabled: bool = MISSING
    target_text_tokens: int = 16
    apply_below: int = 13
    max_repeats: int = 8
    mixed_keep_prob: float = 0.25
    seed: int = 0


# ─────────────────────────────── model architecture ────────────────────────
@dataclass
class ModelConfig:
    """Backbone + multihead architecture. MODEL_GUIDE §2–3, §6. [B]
    `audio_heads` is [R] derived from `codec` via the `gepard.audio_heads`
    resolver (conf/model/gepard.yaml); `stop_*_weight` are [T] loss weights."""
    backbone_id: str = MISSING
    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"
    audio_embed_dim: int = 32
    partial_rotary_factor: float = 1.0       # backbone RoPE coverage; reconciled on load
    stop_loss_weight: float = 1.0
    stop_pos_weight: float = 1.0
    audio_heads: Dict[str, int] = field(default_factory=dict)


# ─────────────────────────────── voice cloning ─────────────────────────────
@dataclass
class ReferenceSamplingConfig:
    """Reference-slice selection policy. MODEL_GUIDE §4. [T/D]"""
    l_min_seconds: float = 3.0
    l_max_seconds: float = 15.0
    min_ref_duration_seconds: float = 3.0
    singleton_min_target_for_slice: float = 6.0
    use_self_reference: bool = False


@dataclass
class CompressorConfig:
    """RefCompressor (Q-Former) architecture. MODEL_GUIDE §4.1. [B]
    Architecture is fixed qformer/RMSNorm/SwiGLU/sinusoidal — only the
    dimensions below are configurable."""
    num_queries: int = 8
    num_layers: int = 2
    num_heads: int = 8
    d_model: Optional[int] = None            # null → inherit backbone hidden_size
    ffn_hidden_size_multiplier: int = 4
    dropout: float = 0.1
    queries_init_std: float = 0.02
    lr_multiplier: float = 1.0               # [T]


@dataclass
class DiversityLossConfig:
    """Hinge-variance query regularizer. MODEL_GUIDE §4.3. [T]"""
    enabled: bool = False
    gamma: float = 0.5
    weight: float = 1.0
    warmup_start: int = 2000
    ramp_steps: int = 3000


@dataclass
class SupConConfig:
    """Supervised-contrastive loss + P·K+M batch composition. MODEL_GUIDE §4.3. [T]"""
    enabled: bool = False
    P: int = 4
    K: int = 3
    M: int = 4
    weight: float = 0.3
    warmup_start: int = 2000
    ramp_steps: int = 1000
    temperature: float = 0.1
    use_projection: bool = True
    projection_hidden_dim: int = 128
    projection_dim: int = 128
    gather_across_ranks: bool = True


@dataclass
class VCTrainingConfig:
    """VC training-time knobs (CFG dropout + aux losses). MODEL_GUIDE §4.2–4.3."""
    cfg_dropout_prob: float = 0.15           # [T]
    null_prefix_init_std: float = 0.02       # [B] (init)
    diversity_loss: DiversityLossConfig = field(default_factory=DiversityLossConfig)
    supcon: SupConConfig = field(default_factory=SupConConfig)


@dataclass
class VoiceCloningConfig:
    """Reference-voice feature. `enabled: false` = legacy no-prefix path.
    MODEL_GUIDE §4. `enabled`+`compressor` are [B]; the rest [T]."""
    enabled: bool = False
    reference_sampling: ReferenceSamplingConfig = field(default_factory=ReferenceSamplingConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    training: VCTrainingConfig = field(default_factory=VCTrainingConfig)


# ─────────────────────────────── data / prep ───────────────────────────────
@dataclass
class ProcessingConfig:
    """Data-prep parallelism (worker/process counts). [D]"""
    num_shards: int = 20
    load_dataset_num_proc: int = 10
    filter_num_proc: int = 20


@dataclass
class DataConfig:
    """Dataset sourcing + prep policy. `train_dataset_path` is the SINGLE source
    of the prepared-dataset location: prepare WRITES it, and pretrain/SFT READ it
    (no separate `dataset_path` — training checks the dir is built at start).
    MODEL_GUIDE §4.2 (singleton / null-prefix policy). [D]"""
    train_dataset_path: str = MISSING
    max_duration_sec: Optional[float] = None
    add_row_id: bool = False
    add_speaker_id: bool = False
    singleton_policy: str = "remove"
    min_clips_per_speaker: int = 1
    speaker_statistics: bool = False
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    hf_datasets: List[Any] = field(default_factory=list)


# ─────────────────────────────── trainer ───────────────────────────────────
@dataclass
class WandbConfig:
    project: str = "gepard"
    name: str = "run"
    entity: Optional[str] = None


@dataclass
class TrainerConfig:
    """Optimizer / schedule / dataloader / checkpoint / logging. All [T].
    MODEL_GUIDE §3 (per-group LR multipliers). `seed`/`weight_decay` are now
    explicit (were relying on HF TrainingArguments defaults 42/0.0)."""
    output_dir: str = "./checkpoints"
    save_steps: int = 5000
    save_total_limit: int = 10
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 9e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    max_grad_norm: float = 3.0
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 1
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    dataloader_num_workers: int = 0
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    average_tokens_across_devices: bool = False
    # Dataset location is NOT here — it is the single source `data.train_dataset_path`.
    keep_index_path: Optional[str] = None    # subset reweight on top of that dataset
    expensive_metrics_every: int = 1000
    audio_lr_multiplier: float = 1.0
    embed_lr_multiplier: float = 1.0
    gradient_checkpointing: bool = True
    seed: int = 42
    weight_decay: float = 0.0
    wandb: WandbConfig = field(default_factory=WandbConfig)


# ─────────────────────────────── finetune / LoRA ───────────────────────────
@dataclass
class LoraConfig:
    """LoRA adapter spec (merged into base weights at export). MODEL_GUIDE §7.4. [T]"""
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    last_n_layers: int = 16


@dataclass
class FinetuneConfig:
    """Checkpoint to load + freeze flags + LoRA. `freeze_*` are no-ops when
    `lora.enabled` (the whole model is frozen before injection). [T]"""
    checkpoint_path: str = ""
    freeze_backbone: bool = False
    freeze_ref_compressor: bool = False
    freeze_supcon_head: bool = False
    freeze_null_prefix: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class RunConfig:
    resume_from: Optional[str] = None
    # Training-phase label frozen into the checkpoint's training_metadata.json
    # (and the rendered model card). Free-form, but keep to pretrain/sft/dpo so
    # the provenance reads consistently. Override per phase, e.g. an SFT
    # experiment sets `run.stage=sft`.
    stage: str = "pretrain"


# ─────────────────────────────── DPO (Phase 3) ─────────────────────────────
@dataclass
class DPOSamplingConfig:
    """Stage 1 rollouts. MODEL_GUIDE §7.4 (offline CFG distillation)."""
    checkpoint: str = MISSING
    texts_file: str = MISSING
    ref_audios: List[str] = field(default_factory=list)
    speaker_pool: str = ""
    holdout_speakers: List[str] = field(default_factory=list)
    null_prefix_prob: float = 0.0
    speakers_per_text: int = 2
    num_samples: int = 8
    temperature: float = 0.4
    top_k: int = 0
    stop_threshold: float = 0.5
    cfg_scale: float = 1.0
    cfg_frames: int = 0
    cfg_uncond_mode: str = "empty_text"
    cap_expected_multiple: float = 4.0
    cap_min_frames: int = 80
    cap_max_frames: int = 300
    seed: int = 17


@dataclass
class DPORewardConfig:
    """Stage 2 programmatic reward. MODEL_GUIDE §7.3."""
    whisper_model: str = "distil-whisper/distil-large-v3"
    whisper_language: Optional[str] = "en"
    whisper_batch_size: int = 16
    decode_batch_size: int = 32
    sec_per_word: float = 0.4
    sec_base: float = 0.7
    dur_max_ratio: float = 2.0
    dur_min_ratio: float = 0.3
    w_wer: float = 1.0
    w_over: float = 0.5
    w_short: float = 2.0
    w_empty: float = 2.0
    sim_enabled: bool = False
    w_sim: float = 0.5
    sim_model: str = "microsoft/wavlm-base-plus-sv"
    sim_sr: int = 16000

    def expected_sec(self, text: str) -> float:
        return self.sec_base + self.sec_per_word * max(1, len(text.split()))

    def dur_bounds(self, text: str) -> tuple:
        exp = self.expected_sec(text)
        return exp * self.dur_min_ratio, exp * self.dur_max_ratio


@dataclass
class DPOPairsConfig:
    """Stage 3 pair construction + frozen-ref logprobs. MODEL_GUIDE §7.1.
    `p_floor` is kept equal to the pipeline-level `dpo.p_floor` (validated)."""
    chosen_max_wer: float = 0.2
    chosen_dur_in_bounds: bool = True
    chosen_not_truncated: bool = True
    min_reward_margin: float = 0.5
    max_pairs_per_group: int = 2
    ref_checkpoint: str = ""
    ref_logp_batch: int = 8
    p_floor: float = 1e-4


@dataclass
class DPOLoraConfig:
    """DPO LoRA — legitimately distinct from FinetuneConfig.lora."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    last_n_layers: int = 16


@dataclass
class DPOTrainingConfig:
    """Stage 4 DPO training. MODEL_GUIDE §7.2.
    `p_floor` is kept equal to the pipeline-level `dpo.p_floor` (validated)."""
    out_dir: str = "dpo_checkpoints"
    beta: float = 2.0
    length_normalize: bool = True
    stop_term_weight: float = 1.0
    p_floor: float = 1e-4
    reward_weight_mode: str = "none"
    reward_weight_scale: float = 1.0
    reward_weight_max: float = 4.0
    lora: DPOLoraConfig = field(default_factory=DPOLoraConfig)
    train_stop_head: bool = True
    train_audio_heads: bool = False
    learning_rate: float = 1e-5
    stop_head_lr: float = 1e-4
    num_epochs: int = 2
    batch_pairs: int = 4
    grad_accum: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500
    save_merged: bool = True
    log_steps: int = 10
    seed: int = 17
    wandb_project: str = "gepard-dpo"
    wandb_name: str = ""
    entity: Optional[str] = None     # wandb team/org; None → personal default
    report_to: str = "none"


# ─────────────────────────────── top-level entry schemas ───────────────────
@dataclass
class TrainConfig:
    """`conf/train.yaml` — pretrain / SFT."""
    codec: CodecConfig = field(default_factory=CodecConfig)
    tokens: TokensConfig = field(default_factory=TokensConfig)
    text_layout: TextLayoutConfig = field(default_factory=TextLayoutConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    voice_cloning: VoiceCloningConfig = field(default_factory=VoiceCloningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    run: RunConfig = field(default_factory=RunConfig)


@dataclass
class PrepareConfig:
    """`conf/prepare.yaml` — dataset build."""
    codec: CodecConfig = field(default_factory=CodecConfig)
    tokens: TokensConfig = field(default_factory=TokensConfig)
    text_layout: TextLayoutConfig = field(default_factory=TextLayoutConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class DPOConfig:
    """`conf/dpo.yaml` — the 4-stage DPO pipeline. Also composes the model groups
    so the runner rebuilds the model from the same source."""
    run_name: str = "round1"
    p_floor: float = 1e-4
    sampling: DPOSamplingConfig = field(default_factory=DPOSamplingConfig)
    reward: DPORewardConfig = field(default_factory=DPORewardConfig)
    pairs: DPOPairsConfig = field(default_factory=DPOPairsConfig)
    training: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    # model groups for the runner (TTSRunner rebuild)
    codec: CodecConfig = field(default_factory=CodecConfig)
    tokens: TokensConfig = field(default_factory=TokensConfig)
    text_layout: TextLayoutConfig = field(default_factory=TextLayoutConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    voice_cloning: VoiceCloningConfig = field(default_factory=VoiceCloningConfig)

    # ── derived paths (everything for a round lives under one directory) ──
    # Rooted at `dpo_data/` — the legacy loader wrote outputs to a `dpo_dataset/`
    # dir that nothing else used while inputs lived in `dpo_data/` (ROADMAP §G4);
    # the migration unifies on `dpo_data/`.
    @property
    def run_dir(self) -> str:
        return os.path.join("dpo_data", self.run_name)

    @property
    def tokens_dir(self) -> str:
        return os.path.join(self.run_dir, "tokens")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.run_dir, "manifest.jsonl")

    @property
    def prefixes_path(self) -> str:
        return os.path.join(self.run_dir, "prefixes.pt")

    @property
    def scores_path(self) -> str:
        return os.path.join(self.run_dir, "scores.jsonl")

    @property
    def pairs_path(self) -> str:
        return os.path.join(self.run_dir, "pairs.jsonl")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.training.out_dir, self.run_name)

    @property
    def ref_checkpoint(self) -> str:
        return self.pairs.ref_checkpoint or self.sampling.checkpoint
