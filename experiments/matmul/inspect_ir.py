"""Compile a selected MatMul and export human-readable IR and PTX.

Example:
 python experiments/matmul/inspect_ir.py --m 256 --n 256 --k 256 --bm 64 --bn 64 --bk 32
"""
import argparse
from pathlib import Path

import torch
from kernel import launch


def main():
    p = argparse.ArgumentParser()
    for name, default in (("m", 256), ("n", 256), ("k", 256),
                          ("bm", 64), ("bn", 64), ("bk", 32), ("warps", 4)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/matmul/ir"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    out = torch.empty((args.m, args.n), device="cuda", dtype=torch.float16)
    compiled = launch(a, b, out, args.bm, args.bn, args.bk, args.warps)
    torch.cuda.synchronize()
    reference = torch.mm(a, b)
    torch.testing.assert_close(out, reference, atol=0.125, rtol=0.02)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU={torch.cuda.get_device_name(0)} shape=({args.m},{args.k})x({args.k},{args.n})")
    print(f"Tile=({args.bm},{args.bn},{args.bk}) WARPS={args.warps}")
    print(f"Correct within FP16 tolerance: True; registers/thread={compiled.n_regs}; shared bytes={compiled.metadata.shared}")
    for name, content in compiled.asm.items():
        if isinstance(content, str):
            path = args.output_dir / f"{name}.txt"
            path.write_text(content, encoding="utf-8")
            print(f"Saved {path}")
    print("Note: search TTGIR for dot layouts and PTX for mma instructions; precise instruction family depends on backend.")


if __name__ == "__main__":
    main()
