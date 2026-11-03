"""Compare BLOCK sizes and N; repeat in randomized order with CUDA graphs.

Effective BW = 3 * N * 4 / elapsed_seconds, for float32 A+B->C.
This is *logical* traffic, not profiler-measured DRAM bytes.
"""
import argparse
import csv
import random
import statistics
from pathlib import Path
import sys

import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel import launch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 65536, 1048576, 16777216])
    p.add_argument("--blocks", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--rep", type=int, default=50, help="CUDA graph benchmark duration in ms")
    p.add_argument("--output", type=Path, default=Path("outputs/block.csv"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    if any(n <= 0 for n in args.sizes) or args.trials <= 0 or args.rep <= 0:
        p.error("sizes, trials and rep must be positive")
    rows = []
    for n in args.sizes:
        a = torch.rand(n, device="cuda", dtype=torch.float32)
        b = torch.rand_like(a)
        c = torch.empty_like(a)
        for block in args.blocks:
            launch(a, b, c, block)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, a + b, rtol=0, atol=0)
        samples = {block: [] for block in args.blocks}
        for trial in range(args.trials):
            order = args.blocks[:]
            random.Random(42 + trial).shuffle(order)
            for block in order:
                ms = triton.testing.do_bench_cudagraph(
                    lambda block=block: launch(a, b, c, block),
                    rep=args.rep, return_mode="median")
                samples[block].append(ms)
        for block in args.blocks:
            ms = statistics.median(samples[block])
            bw = (3 * n * 4) / (ms * 1e-3) / 1e9
            rows.append(dict(n=n, block=block, warps=4, median_ms=f"{ms:.8f}", effective_gbps=f"{bw:.3f}", trials=args.trials))
            print(f"N={n:,} BLOCK={block:4d} time={ms * 1000:.3f} us BW={bw:.2f} GB/s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
