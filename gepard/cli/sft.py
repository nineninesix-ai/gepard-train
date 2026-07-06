#!/usr/bin/env python3
"""SFT / LoRA short-phrase fine-tune entry point (composes `conf/sft.yaml`).

Thin wrapper over `gepard.cli.train.main` with `config_name="sft"` — same
trainer, model init and TrainConfig schema as pretrain, different phase
composition. Launched by `make finetune` (single-GPU accelerate profile):

    accelerate launch --config_file accelerate/finetune_single.yaml -m gepard.cli.sft
    python -m gepard.cli.sft run.resume_from=checkpoints/checkpoint-2000
"""
import os

# Must be set before the first CUDA allocation (i.e. before torch loads a model).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from gepard.cli.train import main

if __name__ == "__main__":
    main(config_name="sft")
