"""Capture short separate CPU/CUDA traces for explicit graph toggles.

The traces help inspect CUDA graph replay, CPU launch and fused pointwise names;
they cannot isolate the exact causal speedup of fusion without a separate
controlled fusion toggle.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision.models import resnet18

from benchmark_attribution import environment, make_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/resnet18/attribution_trace"))
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("invalid batch/steps/warmup")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    model = resnet18(weights=None).eval().half().cuda()
    modes = make_models(model)
    x = torch.randn((args.batch, 3, 224, 224),
                    device="cuda", dtype=torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment(args.batch, 42), indent=2, default=str) + "\n",
        encoding="utf-8")

    counts = []
    for name, fn in modes.items():
        with torch.inference_mode():
            for _ in range(args.warmup):
                fn(x)
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False,
                     with_stack=False) as prof:
            with torch.inference_mode():
                for _ in range(args.steps):
                    with record_function(f"resnet18_forward_{name}"):
                        fn(x)
                torch.cuda.synchronize()

        trace_file = args.output_dir / f"{name}_batch{args.batch}.json"
        prof.export_chrome_trace(str(trace_file))
        operator_table = prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=40)
        (args.output_dir / f"{name}_top.txt").write_text(
            operator_table + "\n", encoding="utf-8")
        device = [e for e in prof.events()
                  if e.device_type == torch.autograd.DeviceType.CUDA]
        frequencies = Counter(e.name for e in device)
        for kernel, occurrences in frequencies.most_common():
            counts.append({"mode": name, "batch": args.batch, "steps": args.steps,
                           "event_name": kernel, "visible_events": occurrences,
                           "fused_named_event": kernel.startswith("triton_poi_fused")})
        print(f"{name}: {len(device)} profiler-visible CUDA events; "
              f"trace={trace_file} ({trace_file.stat().st_size} bytes)")
    out = args.output_dir / "events.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(counts[0]) if counts else [
            "mode", "batch", "steps", "event_name",
            "visible_events", "fused_named_event"])
        w.writeheader()
        w.writerows(counts)
    print("Wrote", out)
    print("Review JSON traces to confirm actual CUDA graph replay and CPU dispatch; "
          "kernel names are clues, not a causal decomposition.")


if __name__ == "__main__":
    main()
