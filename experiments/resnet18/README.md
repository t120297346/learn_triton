# Phase 3 — ResNet18 Inference Benchmark and Profiling

The existing `benchmark_resnet18.py` is preserved as the **original exploratory script**.
The newer `benchmark_repro.py` adds multi-round CUDA-event measurements and CSV output.
`profile_resnet18.py` provides separate PyTorch CPU/CUDA profiling traces.

## Environment

Use your pre-existing CUDA-enabled PyTorch, torchvision, Triton and pytest
environment. Known historical reference: RTX 5080, Triton 3.9.0.

## Run from repository root

~~~bash
python -m pytest -q tests/test_resnet18.py
python experiments/resnet18/benchmark_repro.py --output-dir outputs/resnet18/repro
python experiments/resnet18/profile_resnet18.py --batch 1 --output-dir outputs/resnet18/profile
~~~

The first run of torch.compile may take longer due to compilation. These scripts warm up
compiled functions before measurements and never include initial compile time in CUDA Event latency.

For repeatability, run multiple independent sessions and retain hardware/driver metadata.
Benchmark sessions randomize mode order per round; the error reference is Eager FP16.

## Metrics and interpretability

- Latency: mean GPU-event time per forward invocation in a repeated timing section.
  Reported `median_ms` is the median of per-round means, not P50 of each individual inference.
- Throughput: batch / latency.
- Numerical: max/mean/P99 absolute difference, maximum relative difference using a 1e-3
  denominator floor, reference/test value at that position, and allclose with recorded tolerances.
- `weights=None` means random model weights; **no ImageNet accuracy claim**.
- `torch.compile` speedup is not automatically attributable to Triton. TorchInductor,
  cuDNN algorithm choice, fusion, graph replay and launch overhead may all contribute.
- Profiling can change timing and CUDA Graph replay can change how many CUDA events are visible.
  Trace inspection is needed before asserting kernel fusion. Do not use profiler event
  counts as ground-truth hardware launch counters.
- RTX 5080 runs are **not executed by the PR creation environment**. GPU and numerical
  results recorded under `results/` are transcripts supplied by the user.

## Suggested next analysis

Compare Eager vs Default vs ReduceOverhead event names, total calls, GPU self-time,
and CPU launch/dispatch activity. Decide whether the next PR should target a
specific pointwise fusion pattern, memory layout or convolution bottleneck.
