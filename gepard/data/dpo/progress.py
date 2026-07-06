#!/usr/bin/env python3
"""Live progress monitor for DPO rollout sampling (``make dpo-progress``).

Stage 1 (``dpo-sample`` / ``dpo-sample-sharded``) runs N independent shard
processes that each only write to their own ``/tmp/dpo_sample_shard_i.log`` — so
the launching terminal shows nothing until ``wait`` returns. This monitor is a
**read-only** aggregator: run it in a second terminal and it renders one rich
progress view over all shards by polling on-disk state (token files + per-shard
manifests). It never touches the running shards, so it is safe to start, stop,
and restart at any time during (or after) a run.

Usage:
    python -m gepard.data.dpo.progress [--shards N] [--interval S] [overrides...]
    make dpo-progress SHARDS=4              # matches make dpo-sample-sharded

`--shards` must match the sampling run so per-shard totals are exact; the
overall bar is shard-agnostic (one token file == one finished group).
"""

import argparse
import os
import time
from pathlib import Path

from gepard.config import load_dpo


def _count_tokens(tokens_dir: str) -> int:
    """Finished groups == token .pt files (one per group; resume-safe)."""
    try:
        return sum(1 for e in os.scandir(tokens_dir) if e.name.endswith(".pt"))
    except FileNotFoundError:
        return 0


def _count_lines(path: str) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _shard_done(manifest_path: str, shard_i: int, shard_n: int, num_samples: int) -> int:
    """Groups a shard has written == its manifest lines / num_samples.

    Single-shard runs write the bare ``manifest_path``; sharded runs write
    ``manifest_path.shard{i}`` (num_samples lines per group, flushed per group).
    """
    path = manifest_path if shard_n == 1 else f"{manifest_path}.shard{shard_i}"
    return _count_lines(path) // max(num_samples, 1)


def main():
    ap = argparse.ArgumentParser(description="DPO sampling progress monitor (read-only)")
    ap.add_argument("--shards", type=int, default=4, help="N used for dpo-sample-sharded")
    ap.add_argument("--interval", type=float, default=2.0, help="poll seconds")
    args, overrides = ap.parse_known_args()

    cfg = load_dpo(overrides)
    s = cfg.sampling
    N = max(1, args.shards)

    n_texts = _count_lines(s.texts_file)
    total = n_texts * s.speakers_per_text                     # groups over the whole run
    per_shard = [total // N + (1 if i < total % N else 0) for i in range(N)]

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    console = Console()
    console.print(
        f"[bold cyan]DPO sampling[/] — run [bold]{cfg.run_name}[/]  "
        f"checkpoint [dim]{s.checkpoint}[/]\n"
        f"[dim]{n_texts} texts x {s.speakers_per_text} speakers = {total:,} groups "
        f"x {s.num_samples} samples · {N} shard(s) · tokens → {cfg.tokens_dir}[/]"
    )
    if total == 0:
        console.print("[yellow]No groups to track (empty texts file?). Exiting.[/]")
        return

    progress = Progress(
        TextColumn("[bold]{task.description:<9}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )

    with progress:
        overall = progress.add_task("overall", total=total)
        shard_tasks = [
            progress.add_task(f"shard {i}", total=per_shard[i])
            for i in range(N) if per_shard[i] > 0
        ]
        try:
            while True:
                done = _count_tokens(cfg.tokens_dir)
                progress.update(overall, completed=min(done, total))
                for i, task in enumerate(shard_tasks):
                    di = _shard_done(cfg.manifest_path, i, N, s.num_samples)
                    progress.update(task, completed=min(di, per_shard[i]))
                if done >= total:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            console.print("\n[dim]monitor stopped (sampling continues in the background)[/]")
            return

    console.print(f"[bold green]✓ all {total:,} groups sampled[/] → {cfg.tokens_dir}")


if __name__ == "__main__":
    main()
