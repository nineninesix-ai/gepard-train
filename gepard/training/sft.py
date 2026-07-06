"""
Trainer class for Gepard multihead architecture (SFT: pretrain + finetune).
"""

from __future__ import annotations

import dataclasses

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from .callbacks import (
    _is_main_process,
    DiagnosticsCallback,
    EagerOptimizerStateCallback,
    GepardConfigSaveCallback,
    MultiheadLossLogCallback,
    TokenizerSaveCallback,
    TrainingMetadataCallback,
)
from ..config.schema import VoiceCloningConfig
from ..model.configuration import GepardConfig, save_gepard_config
from ..model.modeling import GepardModel, config_from_model, resolve_dtype
from ..data import (
    DataCollator,
    ReferenceSamplingDataset,
    SpeakerBucketBatchSampler,
    prepare_dataset,
)
from ..logging import get_logger

# Routed to logs/train_*/train_main.log (+ console) when the CLI wires logging;
# a bare logger otherwise, so importing this module never forces log setup.
log = get_logger("train.sft")

if TYPE_CHECKING:
    from ..config.schema import TrainConfig  # noqa: F401


class MultiheadTTSTrainer(Trainer):
    """
    HF Trainer subclass that supports per-group learning rates.

    Non-backbone parameters (audio_embeddings, codebook_heads, stop_head) get
    learning_rate * audio_lr_multiplier; the pretrained text embedding table
    (model.embed_tokens) gets learning_rate * embed_lr_multiplier; the rest of
    the backbone keeps the base learning_rate. All groups follow the same
    lr_scheduler curve (cosine + warmup), just at different absolute scales.

    Set all multipliers to 1.0 to fall back to standard single-LR behavior.
    """

    def __init__(
        self,
        *args,
        audio_lr_multiplier: float = 1.0,
        ref_lr_multiplier: float = 1.0,
        embed_lr_multiplier: float = 1.0,
        vc_config: Optional[VoiceCloningConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.audio_lr_multiplier = audio_lr_multiplier
        self.ref_lr_multiplier = ref_lr_multiplier
        self.embed_lr_multiplier = embed_lr_multiplier
        self.vc_config = vc_config
        self._supcon_batch_sampler: Optional[SpeakerBucketBatchSampler] = None
        self._last_supcon_epoch: int = -1

    def _supcon_enabled(self) -> bool:
        return (
            self.vc_config is not None
            and self.vc_config.enabled
            and self.vc_config.training.supcon.enabled
        )

    def training_step(self, model, inputs, *args, **kwargs):
        # Sync global_step into the model so the diversity/SupCon warmup-ramp
        # curricula see the real step.
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if hasattr(unwrapped, "_global_step"):
            unwrapped._global_step = self.state.global_step

        # SpeakerBucketBatchSampler needs `set_epoch` for per-epoch reshuffling.
        # HF Trainer's built-in epoch hook only handles `dataloader.sampler`,
        # not `batch_sampler`. Detect the transition manually.
        if self._supcon_batch_sampler is not None:
            cur_epoch = int(self.state.epoch or 0)
            if cur_epoch != self._last_supcon_epoch:
                self._supcon_batch_sampler.set_epoch(cur_epoch)
                self._last_supcon_epoch = cur_epoch

        loss = super().training_step(model, inputs, *args, **kwargs)

        # Safety net against NaN/Inf divergence (MODEL_GUIDE §6.2): super().training_step
        # has already run backward, so non-finite grads (e.g. a bf16 logit
        # overflow surviving the CE guard) are now in `.grad`. Neutralise them
        # before the optimizer step so a single bad microbatch can't poison the
        # fp32 weights forever. Cheap relative to a TTS step; no-op when clean.
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        return loss

    def get_train_dataloader(self) -> DataLoader:
        """Override to use `SpeakerBucketBatchSampler` when SupCon is enabled.

        Bypasses HF Trainer's `_get_train_sampler` machinery (which assumes a
        plain index Sampler combined with a BatchSampler chosen by HF) and
        wires our custom batch_sampler directly. The sampler itself handles
        distributed partitioning via per-rank RNG seeding.
        """
        if not self._supcon_enabled():
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not isinstance(self.train_dataset, ReferenceSamplingDataset):
            raise RuntimeError(
                "SupCon requires train_dataset to be a ReferenceSamplingDataset "
                f"but got {type(self.train_dataset).__name__}. "
                "Check the voice_cloning config group."
            )

        sup_cfg = self.vc_config.training.supcon
        expected_bs = sup_cfg.P * sup_cfg.K + sup_cfg.M
        if expected_bs != self.args.per_device_train_batch_size:
            raise ValueError(
                f"SupCon: P*K + M = {sup_cfg.P}*{sup_cfg.K} + {sup_cfg.M} = {expected_bs} "
                f"must equal per_device_train_batch_size="
                f"{self.args.per_device_train_batch_size}. "
                "Fix the config: either change P/K/M in the voice_cloning group or "
                "trainer.per_device_train_batch_size."
            )

        batch_sampler = SpeakerBucketBatchSampler(
            speaker_to_indices=self.train_dataset.speaker_to_indices,
            null_ref_indices=self.train_dataset.null_ref_indices,
            P=sup_cfg.P,
            K=sup_cfg.K,
            M=sup_cfg.M,
            base_seed=int(self.args.seed) if self.args.seed is not None else 42,
        )
        self._supcon_batch_sampler = batch_sampler

        params = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            params["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(self.train_dataset, **params)
        return self.accelerator.prepare(dataloader)

    def create_optimizer(self):
        if (
            self.audio_lr_multiplier == 1.0
            and self.ref_lr_multiplier == 1.0
            and self.embed_lr_multiplier == 1.0
        ):
            return super().create_optimizer()

        # Unwrap FSDP/DDP to iterate named parameters
        model = self.model
        unwrapped = model
        while hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module

        # Audio param names: TTS heads/embeddings that live outside the backbone.
        audio_prefixes = (
            "audio_embeddings.",
            "audio_embed_proj.",
            "codebook_heads.",
            "stop_head.",
        )
        # Voice-cloning params (RefCompressor + learnable null prefix + SupCon
        # projection head). `null_prefix` is a top-level nn.Parameter (no dot
        # suffix). The SupCon head is also random-init and benefits from the
        # same ref_lr_multiplier as the compressor.
        ref_prefixes = (
            "ref_compressor.",
            "null_prefix",
            "supcon_head.",
        )
        # Pretrained text embedding table — its own (lower) LR group so it
        # adapts gently instead of over-rotating away from the pretrained init.
        embed_prefixes = ("model.embed_tokens.",)

        # Exclude weight-decay params the same way HF Trainer does
        decay_params = get_parameter_names(unwrapped, [torch.nn.LayerNorm])
        decay_params = [n for n in decay_params if "bias" not in n]

        backbone_decay, backbone_nodecay = [], []
        audio_decay, audio_nodecay = [], []
        ref_decay, ref_nodecay = [], []
        embed_decay, embed_nodecay = [], []

        for name, param in unwrapped.named_parameters():
            if not param.requires_grad:
                continue
            is_ref = any(name == p or name.startswith(p) for p in ref_prefixes)
            is_audio = (not is_ref) and any(name.startswith(p) for p in audio_prefixes)
            is_embed = (
                (not is_ref) and (not is_audio)
                and any(name.startswith(p) for p in embed_prefixes)
            )
            is_decay = name in decay_params

            if is_ref:
                (ref_decay if is_decay else ref_nodecay).append(param)
            elif is_audio:
                (audio_decay if is_decay else audio_nodecay).append(param)
            elif is_embed:
                (embed_decay if is_decay else embed_nodecay).append(param)
            else:
                (backbone_decay if is_decay else backbone_nodecay).append(param)

        base_lr = self.args.learning_rate
        audio_lr = base_lr * self.audio_lr_multiplier
        ref_lr = base_lr * self.ref_lr_multiplier
        embed_lr = base_lr * self.embed_lr_multiplier
        wd = self.args.weight_decay

        param_groups = [
            {"params": backbone_decay,   "lr": base_lr,  "weight_decay": wd},
            {"params": backbone_nodecay, "lr": base_lr,  "weight_decay": 0.0},
            {"params": audio_decay,      "lr": audio_lr, "weight_decay": wd},
            {"params": audio_nodecay,    "lr": audio_lr, "weight_decay": 0.0},
            {"params": ref_decay,        "lr": ref_lr,   "weight_decay": wd},
            {"params": ref_nodecay,      "lr": ref_lr,   "weight_decay": 0.0},
            {"params": embed_decay,      "lr": embed_lr, "weight_decay": wd},
            {"params": embed_nodecay,    "lr": embed_lr, "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        # HF injects lr into optimizer_kwargs — remove it so per-group lr takes effect
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        if _is_main_process():
            n_backbone = sum(p.numel() for p in backbone_decay + backbone_nodecay)
            n_audio = sum(p.numel() for p in audio_decay + audio_nodecay)
            n_ref = sum(p.numel() for p in ref_decay + ref_nodecay)
            n_embed = sum(p.numel() for p in embed_decay + embed_nodecay)
            msg = (
                f"[LR groups] backbone {n_backbone/1e6:.1f}M @ lr={base_lr:.2e} | "
                f"audio {n_audio/1e6:.1f}M @ lr={audio_lr:.2e} (×{self.audio_lr_multiplier})"
            )
            if n_embed:
                msg += (
                    f" | embed {n_embed/1e6:.1f}M @ lr={embed_lr:.2e} "
                    f"(×{self.embed_lr_multiplier})"
                )
            if n_ref:
                msg += f" | ref {n_ref/1e6:.1f}M @ lr={ref_lr:.2e} (×{self.ref_lr_multiplier})"
            log.info(msg)
        return self.optimizer


class GepardTrainer:
    """
    High-level trainer for Gepard multihead pretraining and fine-tuning.

    Consumes one composed `gepard.config.schema.TrainConfig` (Hydra) and handles
    model init, tokenizer, dataset, and HF Trainer setup. When
    `cfg.finetune.checkpoint_path` is set the model is loaded from a TTS
    checkpoint (instead of a fresh Qwen3.5 init); `cfg.finetune.lora.enabled`
    switches to frozen-model + backbone-adapter training.
    """

    def __init__(self, cfg: "TrainConfig"):
        self.cfg = cfg
        self.vc_config = cfg.voice_cloning

        self.model: Optional[GepardModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self.dataset = None
        self._lora_enabled = False

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.cfg.model.dtype)

    def setup(self):
        """Setup all components for training."""
        self._init_wandb()
        self._load_tokenizer()
        self._load_model()
        ft = self.cfg.finetune
        if ft.checkpoint_path:
            self._load_tts_checkpoint()
        if ft.lora.enabled:
            self._inject_lora()
        else:
            self._apply_freezing()
        self._load_dataset()
        self._create_trainer()

    def _init_wandb(self):
        tr = self.cfg.trainer
        if "wandb" in tr.report_to and _is_main_process():
            wandb_kwargs = {"project": tr.wandb.project, "name": tr.wandb.name}
            if tr.wandb.entity:
                wandb_kwargs["entity"] = tr.wandb.entity
            wandb.init(**wandb_kwargs)
            log.info("wandb: %s/%s", tr.wandb.project, tr.wandb.name)

    def _load_tokenizer(self):
        log.info("Loading tokenizer from %s", self.cfg.model.backbone_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.backbone_id)
        log.info("Tokenizer loaded (vocab_size=%d)", len(self.tokenizer))

    def _load_model(self):
        m = self.cfg.model
        log.info("Loading model from %s", m.backbone_id)
        self.model = GepardModel.from_pretrained(
            m.backbone_id,
            audio_heads=m.audio_heads,
            stop_loss_weight=m.stop_loss_weight,
            stop_pos_weight=m.stop_pos_weight,
            vc_config=self.vc_config,
            codec=self.cfg.codec,
            audio_embed_dim=m.audio_embed_dim,
            partial_rotary_factor=m.partial_rotary_factor,
            attn_implementation=m.attn_implementation,
            dtype=self.torch_dtype,
        )
        self.model = self.model.to(self.torch_dtype)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("Model loaded: %.1fM params (%.1fM trainable)",
                 total_params / 1e6, trainable / 1e6)
        log.info("  Audio heads: %d channels, vocab sizes: %s",
                 len(m.audio_heads), list(m.audio_heads.values()))
        log.info("  Stop loss weight: %s", m.stop_loss_weight)
        log.info("  Attention: %s", m.attn_implementation)

    def _load_tts_checkpoint(self):
        """Overwrite model weights from the pretrained TTS checkpoint
        (`cfg.finetune.checkpoint_path`: HF repo id or local dir/file)."""
        from .base import load_tts_checkpoint

        load_tts_checkpoint(self.model, self.cfg.finetune.checkpoint_path, tag="finetune")

    def _apply_freezing(self):
        """Set requires_grad=False on components flagged in cfg.finetune.

        No-op for a plain pretrain composition (all freeze_* flags false)."""
        cfg = self.cfg.finetune
        if not (cfg.freeze_backbone or cfg.freeze_ref_compressor
                or cfg.freeze_supcon_head or cfg.freeze_null_prefix):
            return
        frozen_params = 0
        for name, param in self.model.named_parameters():
            should_freeze = (
                (cfg.freeze_backbone and name.startswith("model."))
                or (cfg.freeze_ref_compressor and name.startswith("ref_compressor."))
                or (cfg.freeze_supcon_head and name.startswith("supcon_head."))
                or (cfg.freeze_null_prefix and name == "null_prefix")
            )
            if should_freeze:
                param.requires_grad_(False)
                frozen_params += param.numel()

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            "[finetune] Frozen %.1fM params | Trainable %.1fM / %.1fM total",
            frozen_params / 1e6, trainable / 1e6, total / 1e6,
        )

    def _inject_lora(self):
        """Backbone-only LoRA: freeze the whole model, then inject adapters
        (shared lifecycle in `gepard.training.base`)."""
        from .base import inject_backbone_lora

        inject_backbone_lora(self.model, self.cfg.finetune.lora)
        self._lora_enabled = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            "[lora] Trainable %.3fM / %.1fM total (%.3f%%) — "
            "audio heads / stop_head / ref_compressor frozen",
            trainable / 1e6, total / 1e6, 100 * trainable / total,
        )

    def _load_dataset(self):
        self.dataset = prepare_dataset(
            self.cfg.data.train_dataset_path,
            codec_frame_rate_hz=self.cfg.codec.frame_rate_hz,
            vc_config=self.vc_config,
            audio_heads=self.cfg.model.audio_heads,
            keep_index_path=self.cfg.trainer.keep_index_path,
        )

    def _create_trainer(self):
        tr = self.cfg.trainer
        training_args = TrainingArguments(
            output_dir=tr.output_dir,
            num_train_epochs=tr.num_train_epochs,
            per_device_train_batch_size=tr.per_device_train_batch_size,
            gradient_accumulation_steps=tr.gradient_accumulation_steps,
            learning_rate=tr.learning_rate,
            lr_scheduler_type=tr.lr_scheduler_type,
            warmup_steps=tr.warmup_steps,
            max_grad_norm=tr.max_grad_norm,
            optim=tr.optim,
            bf16=tr.bf16,
            fp16=tr.fp16,
            logging_steps=tr.logging_steps,
            save_steps=tr.save_steps,
            save_total_limit=tr.save_total_limit,
            report_to=tr.report_to,
            dataloader_num_workers=tr.dataloader_num_workers,
            dataloader_persistent_workers=tr.dataloader_persistent_workers,
            dataloader_prefetch_factor=(
                tr.dataloader_prefetch_factor
                if tr.dataloader_num_workers > 0 else None
            ),
            dataloader_pin_memory=tr.dataloader_pin_memory,
            remove_unused_columns=tr.remove_unused_columns,
            average_tokens_across_devices=tr.average_tokens_across_devices,
            gradient_checkpointing=tr.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # Explicit in conf/trainer/* since the Hydra migration; the values
            # 42 / 0.0 reproduce the former silent HF defaults (ROADMAP §G2).
            seed=tr.seed,
            weight_decay=tr.weight_decay,
        )

        data_collator = DataCollator(
            pad_token_id=self.cfg.tokens.tts_pad,
            audio_heads=self.cfg.model.audio_heads,
            vc_enabled=self.vc_config.enabled,
        )

        callbacks = [
            EagerOptimizerStateCallback(),
            TokenizerSaveCallback(tokenizer=self.tokenizer),
            GepardConfigSaveCallback(config_provider=self.build_gepard_config),
            TrainingMetadataCallback(self.cfg),
            MultiheadLossLogCallback(),
            DiagnosticsCallback(expensive_every=tr.expensive_metrics_every),
        ]

        ref_lr_multiplier = (
            float(self.vc_config.compressor.lr_multiplier)
            if self.vc_config.enabled
            else 1.0
        )

        self.trainer = MultiheadTTSTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            audio_lr_multiplier=tr.audio_lr_multiplier,
            ref_lr_multiplier=ref_lr_multiplier,
            embed_lr_multiplier=tr.embed_lr_multiplier,
            vc_config=self.vc_config,
        )

        # The banner (gepard.cli.train) already shows these on the console; keep
        # a compact copy at DEBUG so the log file is self-contained.
        log.debug(
            "Trainer built: epochs=%s batch=%s grad_accum=%s lr=%s warmup=%s save_steps=%s",
            tr.num_train_epochs, tr.per_device_train_batch_size,
            tr.gradient_accumulation_steps, tr.learning_rate, tr.warmup_steps,
            tr.save_steps,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        log.info("Starting training (%d samples)", len(self.dataset))
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        log.info("Training loop completed")

    def build_gepard_config(self) -> GepardConfig:
        """Assemble the self-describing checkpoint config (MODEL_GUIDE §10.9).

        Shape drivers are introspected from the live model; the token map, codec
        geometry and text layout — which the model does not hold — come from the
        dataset config so inference stops depending on the training YAMLs.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        unwrapped = self.model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # Un-merged LoRA adapters change the state_dict layout (lora_A/B +
        # base.weight); a vanilla config would falsely describe such a
        # checkpoint as self-contained. Return None so the save callback skips
        # stamping — the final export merges adapters first and IS stamped.
        from ..model.lora import LoRALinear

        if any(isinstance(m, LoRALinear) for m in unwrapped.modules()):
            return None

        return config_from_model(
            unwrapped,
            special_tokens=self.cfg.tokens.token_map(),
            text_repetition=dataclasses.asdict(self.cfg.text_layout),
            codec=dataclasses.asdict(self.cfg.codec),
            model_dtype=self.cfg.model.dtype,
        )

    def save_model(self, output_dir: str):
        from safetensors.torch import save_file

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        # Distributed-safe export. The CLI calls this on EVERY rank; under FSDP
        # the parameters are sharded in place, so a raw state_dict() would hold
        # local shards. accelerator.get_state_dict gathers the full weights — a
        # collective, so all ranks must call it — and only the main process
        # writes files; the rest just participate and wait at the barrier.
        accelerator = getattr(self.trainer, "accelerator", None)
        is_main = (
            accelerator.is_main_process if accelerator is not None else _is_main_process()
        )

        output_path = Path(output_dir)
        if is_main:
            output_path.mkdir(parents=True, exist_ok=True)

        # LoRA run: fold adapters into their base weights so the exported model
        # has the stock key layout and loads directly into the inference runner /
        # GepardModel without any LoRA awareness. merge_all_lora swaps
        # LoRALinear back to nn.Linear in place on every rank — fine here since
        # this is the final export after training, and the gather below must see
        # the merged weights. The separate adapter is also dumped for
        # reuse/inspection.
        if self._lora_enabled:
            from ..model.lora import lora_state_dict, merge_all_lora

            adapter = {k: v.detach().cpu() for k, v in lora_state_dict(self.model).items()}
            if is_main:
                torch.save(adapter, output_path / "lora_adapter.pt")
            n = merge_all_lora(self.model)
            if is_main:
                log.info("[lora] merged %d adapters into base weights for export", n)

        if accelerator is not None:
            state_dict = accelerator.get_state_dict(self.trainer.model)
        else:
            state_dict = self.model.state_dict()

        if is_main:
            # Real safetensors in both branches (the non-LoRA path used to
            # torch.save a pickle under the .safetensors name — unloadable by
            # every reader).
            sd = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
            save_file(sd, str(output_path / "model.safetensors"))

            gepard_cfg = self.build_gepard_config()
            if gepard_cfg is not None:   # None only with un-merged LoRA adapters
                save_gepard_config(gepard_cfg, output_dir)

            # Freeze the resolved training recipe next to the final weights, so
            # the exported checkpoint self-documents its provenance (the model
            # card is rendered from this at upload time).
            from ..logging import write_training_metadata

            tstate = getattr(self.trainer, "state", None)
            write_training_metadata(
                self.cfg,
                output_dir,
                stage=getattr(self.cfg.run, "stage", "pretrain"),
                global_step=getattr(tstate, "global_step", None),
                epoch=getattr(tstate, "epoch", None),
            )

            # Backbone config.json — serving engines (vLLM) and AutoConfig read
            # this, and upload_to_hf.py requires it. patch_config_json_rotary
            # guarantees the vLLM-required duplication: partial_rotary_factor
            # flat top-level AND inside rope_parameters, both at the value the
            # model was trained with.
            from ..model.configuration import (
                effective_partial_rotary_factor,
                patch_config_json_rotary,
            )

            unwrapped = self.model
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            unwrapped.config.save_pretrained(output_dir)
            patch_config_json_rotary(
                str(output_path / "config.json"),
                value=effective_partial_rotary_factor(unwrapped.config),
            )

            self.tokenizer.save_pretrained(output_dir)
            log.info("Model, config.json, gepard_config.json and tokenizer saved to %s",
                     output_dir)

        if accelerator is not None:
            accelerator.wait_for_everyone()
