"""Profile GPU kernel launches and operations for Eager/torch.compile modes.

Event counts reflect the profiler's visible CUDA events. They are not proof of
fusion, and CUDA Graph replay may change event observability. Trace separately
from the unprofiled latency benchmark.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torchvision.models import resnet18


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/resnet18/profile"))
    args = p.parse_args()
    if args.batch < 1 or args.steps < 1 or args.warmup < 0:
        p.error("batch/steps must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    eager = resnet18(weights=None).eval().half().cuda()
    models = {
        "Eager": eager,
        "Default": torch.compile(eager, mode="default"),
        "ReduceOverhead": torch.compile(eager, mode="reduce-overhead"),
    }
    x = torch.randn((args.batch, 3, 224, 224), device="cuda", dtype=torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, fn in models.items():
        print(f"Profiling {name} batch={args.batch}", flush=True)
        # Compile outside the profiler; exclude compile and warmup.
        with torch.inference_mode():
            for _ in range(args.warmup):
                fn(x)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False) as prof:
            with torch.inference_mode():
                for _ in range(args.steps):
                    fn(x)
                torch.cuda.synchronize()
        events = prof.events()
        device_events = [
            e for e in events
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        counts = Counter(e.name for e in device_events)
        label = name.lower()
        trace_path = args.output_dir / f"{label}_batch{args.batch}.json"
        prof.export_chrome_trace(str(trace_path))
        table = prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=30)
        (args.output_dir / f"{label}_batch{args.batch}_top.txt").write_text(
            table + "\n", encoding="utf-8")
        for kernel_name, count in counts.most_common():
            summary.append({"mode": name, "batch": args.batch, "steps": args.steps,
                            "kernel_name": kernel_name, "visible_cuda_events": count})
        print(f"  Visible CUDA events: {len(device_events)} in {args.steps} iterations")
        print(f"  Trace: {trace_path}")
        if not device_events:
            print("  WARNING: profiler exposed zero CUDA events; no fusion claim can be made.")

    csv_path = args.output_dir / f"kernel_counts_batch{args.batch}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["mode", "batch", "steps", "kernel_name", "visible_cuda_events"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "readme.json").write_text(json.dumps({
        "batch": args.batch, "steps": args.steps,
        "interpretation": "Counts are visible CUDA events under profiling, "
                          "not independently measured device launch counters. "
                          "CUDA Graph replay may alter attribution.",
        "note": "Profiler adds overhead; do not compare event trace durations "
                "directly with unprofiled benchmark latency.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
