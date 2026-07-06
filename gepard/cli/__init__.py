"""Thin CLI entry points (Hydra-composed configs).

Each module exposes a `main(argv)` that accepts Hydra-style overrides
(`key=value`, `group=option`) and hands the composed, validated config to the
corresponding engine:

    python -m gepard.cli.train    trainer.learning_rate=3e-4   # pretrain (conf/train.yaml)
    python -m gepard.cli.sft      run.resume_from=...          # LoRA fine-tune (conf/sft.yaml)
    python -m gepard.cli.prepare  data.processing.num_shards=8
"""
