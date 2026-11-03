"""Inspect correctness and compiler outputs at selected N/BLOCK/WARPS.

Examples:
 python experiments/vector_add/inspect_ir.py --n 1024 --block 128
 python experiments/vector_add/inspect_ir.py --n 1048576 --block 256 --output-dir outputs/ir256
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel import launch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ir"))
    args = p.parse_args()
    if args.n <= 0:
        p.error("--n must be positive")
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    a = torch.rand(args.n, device="cuda", dtype=torch.float32)
    b = torch.rand_like(a)
    c = torch.empty_like(a)
    compiled = launch(a, b, c, args.block, args.warps)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU={torch.cuda.get_device_name(0)} N={args.n} BLOCK={args.block} WARPS={args.warps} GRID={(args.n + args.block - 1)//args.block}")
    print("Correct: True")
    print(f"Registers/thread={compiled.n_regs}; shared_bytes={compiled.metadata.shared}")
    for name, content in compiled.asm.items():
        if isinstance(content, str):
            (args.output_dir / f"{name}.txt").write_text(content, encoding="utf-8")
            if name == "ttgir":
                for line in content.splitlines():
                    if "#ttg.blocked<" in line:
                        print(line.strip())
                        break
    print(f"Saved text IR/PTX to: {args.output_dir}")


if __name__ == "__main__":
    main()
