# Phase 4 — CNN Compiler Optimization Attribution

Phase 3 demonstrated ResNet18 speedups and identified fused Triton pointwise
kernels in PyTorch Profiler. Phase 4 compares explicit CUDA Graph ON/OFF
settings in the same Inductor compile mode and measures CPU-synchronized
wall-clock latency separately from GPU-event latency.

## Steps (run from repository root)

~~~bash
python -m pytest -q tests/test_cnn_attribution.py
python experiments/resnet18/benchmark_attribution.py --batch 1 --output-dir outputs/resnet18/attribution_b1
python experiments/resnet18/benchmark_attribution.py --batch 8 --output-dir outputs/resnet18/attribution_b8
python experiments/resnet18/trace_attribution.py --batch 1 --output-dir outputs/resnet18/attribution_trace
~~~

You may add Batch=32 if resources permit.

Generated files:

- `attribution_b*/timing.csv`: per-mode median/min/max for GPU event and synchronized wall latency.
- `attribution_b*/raw_rounds.json`: full round measurements.
- `attribution_b*/environment.json`: software and GPU metadata.
- `attribution_trace/*_batch1.json`: Chrome/Perfetto timeline.
- `attribution_trace/*_top.txt`: profiler top operator summaries.
- `attribution_trace/events.csv`: visible device event names/counts.

Before sharing logs: trace JSON and environment files may include kernel metadata,
local paths and software details. Inspect for sensitive information. Full traces
may be large; compress them rather than committing them to the public repository.

Do not interpret `triton.cudagraphs=True` as proof of capture; inspect the
trace for graph behavior and limitations. For details, see
`docs/cnn_attribution_design.md`.
