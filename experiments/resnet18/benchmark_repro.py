"""Reproducible ResNet18 FP16 latency + numerical consistency experiment.

Measures GPU elapsed time with CUDA events; not CPU-to-CPU end-to-end time.
The existing benchmark_resnet18.py is preserved unchanged as a first run.
"""
import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import torchvision
from torchvision.models import resnet18

from analysis_utils import compare_outputs


def benchmark(fn, x, repeat):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            fn(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def environment(seed):
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "gpu": torch.cuda.get_device_name(0),
        "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else "see nvidia-smi",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "seed": seed,
        "weights": "none (random initialization)",
        "dtype": "float16",
        "input_shape": "[batch,3,224,224]",
        "timing": "CUDA event; excludes initial compile and warmup",
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/resnet18/repro"))
    args = p.parse_args()
    if any(n < 1 for n in args.batches) or min(args.rounds, args.repeat) < 1 or args.warmup < 0:
        p.error("batches, repeat, rounds must be positive; warmup nonnegative")
    if not torch.cuda.is_available():
        p.error("CUDA required")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    eager = resnet18(weights=None).eval().half().cuda()
    models = {
        "Eager": eager,
        "Default": torch.compile(eager, mode="default"),
        "ReduceOverhead": torch.compile(eager, mode="reduce-overhead"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment(args.seed), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    rows = []
    for batch in args.batches:
        x = torch.randn((batch, 3, 224, 224), dtype=torch.float16, device="cuda")
        with torch.inference_mode():
            reference = eager(x).detach().clone()
            errors = {}
            for name, fn in models.items():
                y = fn(x)
                errors[name] = compare_outputs(reference, y, atol=args.atol, rtol=args.rtol)
                if not errors[name]["allclose"]:
                    raise RuntimeError(f"Numerical mismatch: batch={batch} mode={name}")
            for fn in models.values():
                for _ in range(args.warmup):
                    fn(x)
            torch.cuda.synchronize()

        samples = {name: [] for name in models}
        for round_index in range(args.rounds):
            order = list(models)
            random.Random(args.seed + round_index).shuffle(order)
            for name in order:
                samples[name].append(benchmark(models[name], x, args.repeat))

        eager_median = statistics.median(samples["Eager"])
        for name in models:
            times = samples[name]
            median = statistics.median(times)
            row = {
                "batch": batch, "mode": name, "median_ms": round(median, 8),
                "p10_ms": round(quantile(times, .1), 8),
                "p90_ms": round(quantile(times, .9), 8),
                "throughput_img_s": round(batch * 1000 / median, 3),
                "speedup_vs_eager": round(eager_median / median, 4),
                "rounds": args.rounds, "repeat": args.repeat,
                "atol": args.atol, "rtol": args.rtol,
                **errors[name],
            }
            rows.append(row)
            print(f"batch={batch:<2} mode={name:<14} median={median:.4f}ms "
                  f"speedup={eager_median/median:.2f}x allclose={errors[name]['allclose']}")
    csv_file = args.output_dir / "benchmark.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_file}")


if __name__ == "__main__":
    main()
