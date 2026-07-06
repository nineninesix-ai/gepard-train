#!/usr/bin/env python3
"""Build a row-keep index that reweights the short-utterance regime by dataset
composition (MODEL_GUIDE §5.3).

The training dataset has text-repetition baked into `text_ids`
(`[SOT t EOT]x(R-1) [SOT t EOT] SOS`, MODEL_GUIDE §5.2). So the *stored*
`text_ids` length is inflated for short rows. The TRUE original text-token
count is recovered per row as:

    R    = count(SOT) in text_ids                    # number of repeated copies
    orig = (len(text_ids) - 1) // R - 2              # -1 drops trailing SOS; -2 drops SOT,EOT/copy

A row is "short" iff `orig < SHORT_T`. We keep ALL short rows, drop long rows
whose `orig > TAIL_CAP` (OOM/slow tail), and randomly subsample the remaining
longs so the kept set is ~`TARGET_FRACTION` of the corpus — which lands the
short share in the desired band (e.g. SHORT_T=13, TARGET=0.5 -> ~27% short).
`SHORT_OVERSAMPLE > 1` additionally tiles the short indices to push the share
higher without touching long count.

Outputs `keep_idx.npy` (int64 row indices into on-disk order — apply in
prepare_dataset BEFORE the shuffle) and `orig_len.npy` (per-KEPT-row original
length, aligned to keep_idx, for an optional length-aware sampler).

Usage:
    python -m data.build_short_keep_index \
        --dataset /opt/dlami/nvme/finetun_dataset \
        --short-t 13 --target-fraction 0.5 --tail-cap 150 \
        --out-dir /opt/dlami/nvme/finetun_dataset
    # add --write to also materialise ds.select(keep_idx).save_to_disk(out/_materialized)
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pyarrow as pa

SOT, EOT, SOS = 248073, 248074, 248070


def compute_orig_len(dataset_dir: str) -> np.ndarray:
    """Full scan of all shards -> per-row original text-token length (int32)."""
    shards = sorted(glob.glob(os.path.join(dataset_dir, "data-*.arrow")))
    if not shards:
        raise FileNotFoundError(f"No data-*.arrow shards in {dataset_dir}")
    out = []
    t0 = time.time()
    for f in shards:
        with pa.memory_map(f, "r") as src:
            col = pa.ipc.open_stream(src).read_all().column("text_ids").combine_chunks()
        offs = col.offsets.to_numpy()
        vals = col.values.to_numpy()
        lens = np.diff(offs).astype(np.int64)
        nsot = np.add.reduceat((vals == SOT).astype(np.int64), offs[:-1])
        nsot = np.maximum(nsot, 1)
        out.append((((lens - 1) // nsot) - 2).astype(np.int32))
    orig = np.concatenate(out)
    print(f"[scan] {len(shards)} shards, {len(orig):,} rows in {time.time()-t0:.1f}s")
    return orig


def build(orig: np.ndarray, short_t: int, target_fraction: float,
          tail_cap: int, short_oversample: int, seed: int):
    N = len(orig)
    rng = np.random.default_rng(seed)

    short_idx = np.where(orig < short_t)[0]
    long_idx = np.where(orig >= short_t)[0]
    # Tail cap applies to the long pool only (shorts are always short).
    tail_drop = long_idx[orig[long_idx] > tail_cap]
    long_eligible = long_idx[orig[long_idx] <= tail_cap]

    n_short_eff = len(short_idx) * short_oversample
    n_keep_total = round(N * target_fraction)
    n_long_keep = n_keep_total - n_short_eff
    if n_long_keep < 0:
        print(f"[warn] shorts (x{short_oversample}) = {n_short_eff:,} already exceed "
              f"target total {n_keep_total:,}; keeping 0 longs.")
        n_long_keep = 0
    n_long_keep = min(n_long_keep, len(long_eligible))

    pick_long = rng.choice(long_eligible, size=n_long_keep, replace=False)
    short_keep = np.tile(short_idx, short_oversample) if short_oversample > 1 else short_idx

    keep = np.concatenate([short_keep, pick_long])
    keep.sort(kind="stable")  # sorted -> faster Arrow select; duplicates (oversample) preserved
    orig_kept = orig[keep]

    total = len(keep)
    short_share = n_short_eff / max(total, 1)
    print("\n=== resulting mix ===")
    print(f"  corpus N            : {N:,}")
    print(f"  short (orig<{short_t})     : {len(short_idx):,} ({len(short_idx)/N*100:.2f}%) "
          f"-> kept x{short_oversample} = {n_short_eff:,}")
    print(f"  long eligible       : {len(long_eligible):,}  (tail orig>{tail_cap} dropped: {len(tail_drop):,})")
    print(f"  longs kept          : {n_long_keep:,} ({n_long_keep/max(len(long_eligible),1)*100:.1f}% of eligible)")
    print(f"  KEEP total          : {total:,} ({total/N*100:.1f}% of corpus)")
    print(f"  >>> short share     : {short_share*100:.1f}%")
    return keep.astype(np.int64), orig_kept.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--short-t", type=int, default=13)
    ap.add_argument("--target-fraction", type=float, default=0.5)
    ap.add_argument("--tail-cap", type=int, default=150)
    ap.add_argument("--short-oversample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out-dir", default=None, help="default: --dataset dir")
    ap.add_argument("--write", action="store_true", help="also materialise the subset to disk")
    args = ap.parse_args()

    out_dir = args.out_dir or args.dataset
    os.makedirs(out_dir, exist_ok=True)

    orig = compute_orig_len(args.dataset)
    keep, orig_kept = build(orig, args.short_t, args.target_fraction,
                            args.tail_cap, args.short_oversample, args.seed)

    tag = f"short{args.short_t}_t{int(args.target_fraction*100)}"
    keep_path = os.path.join(out_dir, f"keep_idx_{tag}.npy")
    orig_path = os.path.join(out_dir, f"orig_len_{tag}.npy")
    np.save(keep_path, keep)
    np.save(orig_path, orig_kept)
    print(f"\n[saved] {keep_path}  ({keep.nbytes/1e6:.1f} MB)")
    print(f"[saved] {orig_path}")
    print(f"\nWire it up:  dataset_config.keep_index_path = '{keep_path}'")

    if args.write:
        from datasets import load_from_disk
        mat = os.path.join(out_dir, f"_materialized_{tag}")
        print(f"\n[write] materialising subset -> {mat} (this copies data, may be large)...")
        ds = load_from_disk(args.dataset).select(keep)
        ds.save_to_disk(mat)
        print(f"[write] done: {len(ds):,} rows at {mat}")


if __name__ == "__main__":
    main()
