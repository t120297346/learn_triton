# Phase 3: CNN inference — experiment design and validity rules

## Research questions

1. How much does `torch.compile(mode="default")` and
   `mode="reduce-overhead"` change ResNet18 FP16 inference latency at batch 1, 8 and 32?
2. Does the numerical difference relative to Eager FP16 remain within an
   explicitly defined tolerance?
3. Which kernel/CPU-side execution patterns are visible to PyTorch Profiler?
   Are these consistent with fusion or CUDA Graph replay?

## Design

- Hardware target: NVIDIA RTX 5080; input N×3×224×224, contiguous FP16.
- Model: `torchvision.models.resnet18(weights=None)`, eval mode. No trained weights
  or true-image classification accuracy should be claimed.
- Inputs: deterministic PRNG seed, fresh random input per batch, shared among modes.
- Modes: Eager, torch.compile Default, torch.compile ReduceOverhead.
- Pre-timing: explicit warmup for each shape/mode; compile initialization excluded.
- Timing: CUDA events around 100 forward invocations; median of 5 shuffled rounds;
  report P10/P90 across rounds. A CUDA-event measurement is not CPU-to-CPU
  latency; include a separate host-side timer if launch overhead must be quantified.
- Correctness: compare Eager to each mode at the *same input*, report max/mean/P99
  absolute error, clamped-relative diagnostic and allclose (rtol=0.01, atol=0.01 by default).
  Note the maximum-relative-error's denominator floor of 1e-3; it can exceed 100%
  when values are near zero even if absolute errors are small.

## Profiler

- Capture Eager/Default/ReduceOverhead separately, after full warmup.
- Export Chrome Trace and per-mode top operator table.
- Count profiler-visible CUDA kernel events, record their names and workloads.
- Keep profiling separate from latency benchmark. CUDA Graph replay may make
  visible event count incomplete or different. The profiler alone does not
  identify exact fractions of speedup causally due to fusion vs launch overhead.
- Later: use Nsight Systems for launch timeline, Nsight Compute for per-kernel
  memory/occupancy counters. Do not overclaim from PyTorch summary tables.

## Success criteria for PR #3

- Scripts are syntactically valid, CPU-side metric tests pass, CUDA model test
  and timing work on the user's RTX 5080.
- Three mode × three batch results exported to CSV with environment metadata.
- Numeric outputs pass documented tolerances and outliers are interpreted.
- Profile traces and event summaries are inspected before any claim of exact fusion.
- Source and results are separated; generated large JSON traces remain in
  ignored `outputs/`, not in Git history.

## Scope boundary

No custom Triton convolution, no manual fusion kernel and no TPU/Pallas backend.
Those should be separate PRs after bottleneck analysis.
