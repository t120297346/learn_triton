"""Compare BLOCK and num_warps, record TTGIR, PTX and register metrics."""
import argparse
import csv
import random
import re
import statistics
from pathlib import Path
import sys

import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel import launch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1048576)
    p.add_argument("--blocks", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--output", type=Path, default=Path("outputs/layout.csv"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    if args.n <= 0 or args.trials <= 0 or args.rep <= 0:
        p.error("n, trials and rep must be positive")
    a = torch.rand(args.n, device="cuda")
    b = torch.rand_like(a)
    c = torch.empty_like(a)
    configs = [(block, warps) for block in args.blocks for warps in args.warps]
    metadata = {}
    for block, warps in configs:
        kernel = launch(a, b, c, block, warps)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a + b, rtol=0, atol=0)
        layout = next((line.strip() for line in kernel.asm["ttgir"].splitlines() if "#ttg.blocked<" in line), "")
        ptx = kernel.asm["ptx"]
        loads = re.findall(r"\bld\.global(?:\.[\w]+)*", ptx)
        stores = re.findall(r"\bst\.global(?:\.[\w]+)*", ptx)
        metadata[(block, warps)] = (layout, kernel.n_regs, kernel.metadata.shared, ";".join(loads), ";".join(stores))
    samples = {cfg: [] for cfg in configs}
    for trial in range(args.trials):
        order = configs[:]
        random.Random(42 + trial).shuffle(order)
        for block, warps in order:
            ms = triton.testing.do_bench_cudagraph(
                lambda block=block, warps=warps: launch(a, b, c, block, warps),
                rep=args.rep, return_mode="median")
            samples[(block, warps)].append(ms)
    rows = []
    for block, warps in configs:
        ms = statistics.median(samples[(block, warps)])
        layout, regs, shared, loads, stores = metadata[(block, warps)]
        row = dict(n=args.n, block=block, warps=warps, median_us=f"{ms*1000:.4f}",
                   registers_per_thread=regs, shared_bytes=shared, layout=layout,
                   ptx_loads=loads, ptx_stores=stores, trials=args.trials)
        rows.append(row)
        print(f"BLOCK={block} WARPS={warps} time={ms*1000:.3f}us registers={regs} shared={shared}")
        print(f"  {layout}")
        print(f"  loads={loads} stores={stores}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
