# Phase 3 — ResNet18 profiling reproduction, 2026-11-03

## Provenance and verification

Source: user-uploaded `cnn_pr3_results.tar.gz`. SHA256: `b37ce8d9e7a0a557d5c359b7e285a3d810a9c2711a083b00154f50d715c6d30b`.

This document is derived from the supplied `resnet18_pytest.log`, `resnet18_benchmark.log`, `resnet18_profile.log`, `benchmark.csv`, `kernel_counts_batch1.csv`, and three profiler top-operator summaries. The archive omits the large Chrome Trace JSON files and `environment.json`, as specified by the collection command. Accordingly, the GPU/software versions for this individual run are **not independently recorded** in this archive. The broader project used an RTX 5080; that should not be mistaken for a verified environment snapshot here.

**Unit tests:** `4 passed in 3.16s` (including numeric metric tests and CUDA-dependent ResNet18 output comparison in this environment).

**Benchmark:** five timing rounds per configuration, 100 forward invocations per round; the reported latency is the median of per-round mean CUDA-event durations, not an end-to-end CPU wall-clock latency.

## Full measurements

Original-reported numerical columns are preserved at [2026-11-03-resnet18-benchmark.csv](2026-11-03-resnet18-benchmark.csv).

| Batch | Eager ms | Default ms | ReduceOverhead ms | Best speedup | Allclose |
|---:|---:|---:|---:|---:|---|
| 1 | 0.588734 | 0.265969 | 0.235385 | **2.5012×** | True for all modes |
| 8 | 0.862834 | 0.706746 | 0.707475 | **1.2209×** | True for all modes |
| 32 | 2.528049 | 2.065274 | 2.040138 | **1.2392×** | True for all modes |

Compared to the exploratory first-run Batch=1 result of 2.40×, this independent benchmark protocol reports 2.50× with ReduceOverhead. Do not infer statistical significance across sessions from these two experiments.

### Numerical observations

- Default/ReduceOverhead vs Eager: batch 1 max abs error 0.001953125, mean abs 0.000554707, P99 abs 0.001953125.
- Batch 8: max abs 0.00390625, mean abs 0.000583932, P99 abs 0.001953125.
- Batch 32: max abs 0.00390625, mean abs 0.000645018, P99 abs 0.001953125.
- All 9 mode×batch combinations pass `allclose` with `rtol=0.01, atol=0.01`; Eager is its own reference.
- Maximum clamped-relative diagnostics for compiled modes: 0.102676, 0.761270, 1.007080 for batches 1,8,32. The denominator floor is 1e-3. Batch 32's near-zero reference is -0.000366211; candidate -0.001373291; abs difference 0.00100708. This is not a classification-error rate.
- Random weights `weights=None`: no claim of top-1 accuracy or deployment quality.

## Batch=1 profiler: 10 iterations per mode

| Mode | Profiler-visible CUDA events | Per iteration | Named Triton events | Distinct names |
|---|---:|---:|---:|---:|
| Eager | 1660 | 166 | 0 | 19 |
| Default | 580 | 58 | 360 | 31 |
| ReduceOverhead | 590 | 59 | 360 | 32 |

Event reduction: **65.06%** for Eager→Default; **64.46%** for Eager→ReduceOverhead. These counts include CUDA device event types visible to the profiler (potentially copies/memsets), **not** a controlled independent hardware launch count or evidence that latency reduction equals event reduction.

Full categorized totals above are recorded in [2026-11-03-profile-summary.csv](2026-11-03-profile-summary.csv). The archive's full per-kernel `kernel_counts_batch1.csv` was inspected; it has not been copied verbatim due to verbose kernel signatures.

### Positive evidence of fusion

The compiled profiling reports contain names beginning with:

```text
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu
triton_poi_fused__native_batch_norm_legit_no_training_add_relu
```

The names show TorchInductor generated fused Triton **pointwise** kernels involving listed operators. This is direct evidence that fusion occurred, **not** a measurement proving how much of the 2.50× speedup is caused by fusion alone.

The same runs retain cuDNN/CUTLASS-like convolution kernels named `sm80_xmma_fprop_implicit_gemm_...`; therefore the main convolution calculations are **not all** replaced by Triton.

### Profiler timing evidence (not comparable to unprofiled latency)

From the user-provided top-operator tables for 10 iterations:
- Eager `aten::cudnn_convolution` self CUDA time: **3.167 ms**, 200 operator calls (73.92% of profiler-reported self CUDA time).
- Default `aten::cudnn_convolution` self CUDA time: **1.682 ms**, 200 operator calls (75.27% of profiler-reported self CUDA time).
- ReduceOverhead contains a compiled graph call with **2.200 ms** self CUDA time under the profiler; the attribution differs from Eager/Default. Do **not** directly compare these parent/operator rows as if they were equivalent hardware kernels.
- Eager shows 370 CUDA events in an NCHW→NHWC conversion kernel family across 10 iterations and 200 for NHWC→NCHW. Compiled modes have substantially different visible operator families, supporting a layout/dispatch path difference that should be investigated with a full trace.

Self CUDA time and nested device-event rows can refer to the same executed work; do not add top-table operator times to child kernel times indiscriminately.

## Interpretation and research implications

**Established:** The CPU-side and GPU-side numerical tests passed; new CSV measurements reproduce a large Batch=1 improvement; the profiler reports dramatically fewer visible events and explicitly named fused Triton pointwise kernels in compiled runs.

**Plausible but not isolated:** Layout conversion reduction, pointwise fusion, CPU dispatch and CUDA Graph replay likely contribute in different proportions. This evidence cannot apportion the speedup to an individual optimization.

**Unresolved:** Chrome Trace JSON, environment snapshot, CPU launch timelines and Nsight counters. The actual trace files were intentionally excluded from the submitted archive. If attributing causal contributions, compare controlled toggles (e.g. fusion/graph behavior) under equivalent profiling rather than relying only on raw event totals.

## Merge recommendation

The PR's initial goal (reproducible benchmark, correctness, and first profiler summary) has been reached on the user's machine. It can be considered for merge after review, while advanced causal profiling remains a separate future PR. No custom Triton CNN optimization or TPU backend was tested.
