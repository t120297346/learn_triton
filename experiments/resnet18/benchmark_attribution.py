"""PR #4: matched-mode comparison of CUDA-graph settings and host vs GPU latency.

Examples:
 python experiments/resnet18/benchmark_attribution.py --batch 1 --output-dir outputs/resnet18/attribution
 python experiments/resnet18/benchmark_attribution.py --batch 8 --rounds 3 --output-dir outputs/resnet18/attribution_b8

Do not treat the ON/OFF difference as exclusive causal CUDA Graph contribution:
the compiler may select a different execution path and some graph requests
may fall back. Confirm with an execution trace.
"""
import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import torchvision
from torchvision.models import resnet18

from analysis_utils import compare_outputs


def make_models(model):
    # Explicit flags make the ON/OFF comparison more interpretable than
    # comparing two modes with multiple implicit optimization differences.
    return {
        "eager": model,
        "inductor_graph_off": torch.compile(
            model, mode="default", options={"triton.cudagraphs": False}),
        "inductor_graph_on": torch.compile(
            model, mode="default", options={"triton.cudagraphs": True}),
    }


def gpu_elapsed_ms(fn, x, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeats):
            fn(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def synchronized_wall_ms(fn, x, repeats):
    # Includes host dispatch and GPU wait. The per-invocation synchronize
    # changes the workload's queuing pattern versus GPU-event timing.
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(repeats):
            fn(x)
            torch.cuda.synchronize()
        return (time.perf_counter_ns() - start) / 1e6 / repeats


def environment(batch, seed):
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        smi = "unavailable"
    try:
        modes = torch._inductor.list_mode_options()
    except Exception as exc:
        modes = {"unavailable": type(exc).__name__}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": sha, "gpu": torch.cuda.get_device_name(),
        "nvidia_smi": smi, "python": sys.version.split()[0],
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_inductor_mode_options": modes, "batch": batch, "seed": seed,
        "weights": None, "dtype": "float16", "input": [batch, 3, 224, 224],
        "interpretation": "No confirmation of actual graph capture from config alone",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--gpu-repeats", type=int, default=100)
    parser.add_argument("--wall-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/resnet18/attribution"))
    args = parser.parse_args()
    if args.batch < 1 or args.rounds < 1 or args.warmup < 0 or min(
            args.gpu_repeats, args.wall_repeats) < 1:
        parser.error("invalid batch/round/repeat/warmup")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model = resnet18(weights=None).eval().half().cuda()
    modes = make_models(model)
    x = torch.randn((args.batch, 3, 224, 224),
                    device="cuda", dtype=torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment(args.batch, args.seed), indent=2, default=str) + "\n",
        encoding="utf-8")
    with torch.inference_mode():
        ref = model(x).detach().clone()
        errors = {}
        for name, fn in modes.items():
            y = fn(x).detach().clone()
            errors[name] = compare_outputs(ref, y)
            if not errors[name]["allclose"]:
                raise RuntimeError(f"Output mismatch: {name}")
            for _ in range(args.warmup):
                fn(x)
        torch.cuda.synchronize()

    samples = {name: {"gpu_event_ms": [], "sync_wall_ms": []} for name in modes}
    for iteration in range(args.rounds):
        names = list(modes)
        random.Random(args.seed + iteration).shuffle(names)
        for name in names:
            samples[name]["gpu_event_ms"].append(
                gpu_elapsed_ms(modes[name], x, args.gpu_repeats))
        random.Random(args.seed + 10000 + iteration).shuffle(names)
        for name in names:
            samples[name]["sync_wall_ms"].append(
                synchronized_wall_ms(modes[name], x, args.wall_repeats))

    rows = []
    for name, measurements in samples.items():
        for metric, values in measurements.items():
            rows.append({
                "batch": args.batch, "mode": name, "metric": metric,
                "median_ms": round(statistics.median(values), 8),
                "min_ms": round(min(values), 8),
                "max_ms": round(max(values), 8),
                "rounds": args.rounds,
                "repeats_per_round": (
                    args.gpu_repeats if metric == "gpu_event_ms"
                    else args.wall_repeats),
                **errors[name],
            })
            print(f"batch={args.batch} mode={name:<19} {metric:<13} "
                  f"median={statistics.median(values):.5f}ms "
                  f"range=[{min(values):.5f}, {max(values):.5f}]")
    path = args.output_dir / "timing.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.output_dir / "raw_rounds.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    print("Wrote", path)
    print("NOTE: synchronized wall latency is not equivalent to GPU-event latency.")


if __name__ == "__main__":
    main()
