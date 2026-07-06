#!/usr/bin/env python3
"""
Dataset preparation entry point.

Loads the source datasets from HuggingFace, processes them, and saves the
training dataset to disk. Configuration is composed from the Hydra tree
(`conf/prepare.yaml`): the `tokens` / `codec` / `text_layout` global groups
plus the `data` sourcing group.

Usage:
    python -m gepard.cli.prepare
    python -m gepard.cli.prepare --n-shards 8
    python -m gepard.cli.prepare data.max_duration_sec=15   # Hydra override
"""

import argparse
from pathlib import Path

from gepard.config import load_prepare
from gepard.logging import (
    LiveDashboard,
    enable_hf_progress,
    get_logger,
    print_dataset_banner,
    setup_main_logging,
    shutdown_logging,
)

from .processor import DatasetProcessor


def parse_args(argv=None):
    """Parse command line arguments; unknown args are Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Gepard training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for processed dataset (overrides data.train_dataset_path)"
    )

    parser.add_argument(
        "--n-shards",
        type=int,
        default=None,
        help="Override data.processing.num_shards for this run"
    )

    return parser.parse_known_args(argv)


def main(argv=None):
    """Main dataset preparation function."""
    args, overrides = parse_args(argv)

    log_state = setup_main_logging(scope="dataset")
    log = get_logger("dataset.prepare")
    try:
        # Compose + validate the config first so a bad composition fails
        # before any worker spawns, and the banner renders resolved values.
        cfg = load_prepare(overrides)
        processor = DatasetProcessor(
            cfg=cfg,
            n_shards_per_dataset=args.n_shards,
            log_queue=log_state["queue"],
        )

        print_dataset_banner(
            cfg, args, log_state["log_dir"], console=log_state["console"]
        )

        # The dashboard owns the rich.live region; it consumes shard/item
        # events emitted by the processor and workers via the root logger
        # and renders the per-shard table.
        with LiveDashboard(log_state):
            dataset = processor()

        if args.output is not None:
            output_path = Path(args.output)
        else:
            output_path = Path(cfg.data.train_dataset_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # save_to_disk lives outside the dashboard so HF's native tqdm shard
        # progress can render to stderr without fighting the rich Live region.
        log.info(f"saving {len(dataset):,} rows to {output_path}")
        enable_hf_progress()
        dataset.save_to_disk(str(output_path))
        log.info(f"saved to {output_path}")

        if cfg.data.speaker_statistics:
            statistics_path = output_path / "speaker_statistics.json"
            processor.save_speaker_statistics(str(statistics_path))

        log.info(
            f"done. samples={len(dataset):,}  output={output_path}  logs={log_state['log_dir']}"
        )
        log.info("next: make train  (accelerate launch -m gepard.cli.train)")
    finally:
        shutdown_logging(log_state)


if __name__ == "__main__":
    main()
