#!/usr/bin/env python3
"""Training entry point (pretrain + SFT/LoRA) on the Hydra config tree.

Usage:
    # pretrain (conf/train.yaml defaults)
    accelerate launch --config_file accelerate/pretrain_fsdp.yaml -m gepard.cli.train

    # LoRA short-phrase fine-tune (conf/sft.yaml — the `make finetune` path)
    accelerate launch --config_file accelerate/finetune_single.yaml -m gepard.cli.sft

    # ad-hoc overrides / resume
    python -m gepard.cli.train trainer.learning_rate=3e-4 voice_cloning=disabled
    python -m gepard.cli.train run.resume_from=checkpoints/checkpoint-5000
"""
import os

# Must be set before the first CUDA allocation (i.e. before torch loads a model).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None, config_name: str = "train"):
    """Run a training phase composed from `conf/{config_name}.yaml`.

    `config_name="train"` → pretrain; `"sft"` → the LoRA fine-tune entry
    (`gepard.cli.sft`). Both share this driver, the `GepardTrainer`, and the
    TrainConfig schema — only the composed default groups differ.
    """
    from gepard.config import load_sft, load_train, require_dataset_built
    from gepard.training import GepardTrainer
    from gepard.training.callbacks import _is_main_process
    from gepard.logging import (
        get_logger,
        print_train_banner,
        setup_train_logging,
        shutdown_train_logging,
    )

    load = {"train": load_train, "sft": load_sft}[config_name]

    overrides = list(sys.argv[1:] if argv is None else argv)
    is_main = _is_main_process()

    # File + console logging under gepard.* (rank 0 writes the file; other ranks
    # stay quiet). wandb/transformers logging is deliberately left untouched.
    log_state = setup_train_logging(scope=config_name, is_main=is_main)
    log = get_logger(config_name)

    try:
        log.info("Composing configuration (conf/%s.yaml)%s", config_name,
                 f" — overrides: {' '.join(overrides)}" if overrides else "")
        cfg = load(overrides)
        require_dataset_built(cfg)
        log.info("Configuration composed — backbone=%s dataset=%s voice_cloning=%s",
                 cfg.model.backbone_id, cfg.data.train_dataset_path,
                 "on" if cfg.voice_cloning.enabled else "off")

        if is_main:
            print_train_banner(cfg, log_state["log_dir"], console=log_state["console"])
            # Full field-by-field config → log file only (DEBUG < console INFO),
            # so the run is reproducible from the log without cluttering stdout.
            try:
                from omegaconf import OmegaConf
                log.debug("Full resolved config:\n%s",
                          OmegaConf.to_yaml(OmegaConf.structured(cfg)))
            except Exception as e:  # dump is best-effort, never fatal
                log.debug("Config YAML dump unavailable: %s", e)

        log.info("=== Gepard training run started ===")
        trainer = GepardTrainer(cfg)
        trainer.setup()
        trainer.train(resume_from_checkpoint=cfg.run.resume_from)

        final_output_dir = Path(cfg.trainer.output_dir) / "final"
        trainer.save_model(str(final_output_dir))
        log.info("Final model exported to %s", final_output_dir)
        log.info("=== Training pipeline completed successfully ===")
    except Exception:
        log.exception("Training run failed")
        raise
    finally:
        shutdown_train_logging(log_state)


if __name__ == "__main__":
    main()
