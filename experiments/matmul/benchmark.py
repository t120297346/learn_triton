"""Compare Triton tiled FP16 MatMul to torch.mm with preallocated outputs.

CUDA Graph replay timing; TFLOPS counts 2*M*N*K mathematical operations.
Same buffers are reused, so cache effects may influence measurements.
"""
import argparse
import csv
import random
import statistics
from pathlib import Path

import torch
import triton.testing
from kernel import launch


CONFIGS = [
    (32, 64, 32, 4),
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (64, 128, 64, 4),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048],
                   help="Square M=N=K dimensions")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--rep", type=int, default=50, help="CUDA Graph benchmark duration in ms")
    p.add_argument("--output", type=Path, default=Path("outputs/matmul/benchmark.csv"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    if args.trials <= 0 or args.rep <= 0 or any(x <= 0 for x in args.sizes):
        p.error("sizes/trials/rep must be positive")
    rows = []
    torch.manual_seed(0)
    for n in args.sizes:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        out = torch.empty((n, n), device="cuda", dtype=torch.float16)
        ref_out = torch.empty_like(out)
        torch.mm(a, b, out=ref_out)
        for bm, bn, bk, warps in CONFIGS:
            launch(a, b, out, bm, bn, bk, warps)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, ref_out, atol=0.125, rtol=0.02)

        all_configs = [("triton", *cfg) for cfg in CONFIGS] + [("pytorch", 0, 0, 0, 0)]
        samples = {cfg: [] for cfg in all_configs}
        for trial in range(args.trials):
            order = all_configs[:]
            random.Random(42 + trial).shuffle(order)
            for cfg in order:
                label, bm, bn, bk, warps = cfg
                if label == "pytorch":
                    fn = lambda: torch.mm(a, b, out=ref_out)
                else:
                    fn = lambda bm=bm, bn=bn, bk=bk, warps=warps: launch(a, b, out, bm, bn, bk, warps)
                ms = triton.testing.do_bench_cudagraph(fn, rep=args.rep, return_mode="median")
                samples[cfg].append(ms)

        for cfg in all_configs:
            label, bm, bn, bk, warps = cfg
            ms = statistics.median(samples[cfg])
            tflops = (2 * n**3) / (ms * 1e-3) / 1e12
            rows.append(dict(m=n, n=n, k=n, backend=label, bm=bm, bn=bn, bk=bk,
                             warps=warps, median_ms=f"{ms:.8f}",
                             effective_tflops=f"{tflops:.3f}", trials=args.trials))
            print(f"N={n:4d} {label:7s} tile=({bm},{bn},{bk}) warps={warps} time={ms*1000:.3f}us TFLOPS={tflops:.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
