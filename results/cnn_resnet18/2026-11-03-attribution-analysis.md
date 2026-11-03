# Phase 4: CUDA Graph ablation and trace observations (2026-11-03)

## Evidence, provenance and environment

Analysis based on user-uploaded `cnn_pr4_results.tar.gz` SHA256
`8035bf483c29e3f1ff90340b945233c507abe4a379f04363e3bee2dfc414758a`.
The archive includes the 2 benchmark CSVs and raw rounds, 3 Chrome Trace
JSONs, profiler operator tables, event summaries, environment JSON, and logs.
This report was produced by inspecting that archive, **not** by rerunning a GPU.

The environment JSON identifies RTX 5080; NVIDIA driver 595.99.02;
Python 3.12.3; PyTorch 2.15.0+cu132; torchvision 0.29.2+cu132;
Triton 3.9.0; CUDA runtime 13.2. Commit
`f9e509ebef76901f65fbccee04beb82631153171`.
It also confirms the tested branch uses the corrected
`torch.compile(model, options={"triton.cudagraphs": ...})`
API without an explicit `mode`.

**Tests:** `2 passed in 1.24s` (PR #4 unit test cases). Benchmark rows
for batches 1 and 8 all report `allclose=True` against Eager FP16
with tolerance inherited from Phase 3 (`atol=rtol=0.01`).
The scripts use random untrained ResNet18 weights (`weights=None`);
no classification accuracy result is implied.

## Latency results

Five rounds, each with 100 GPU-event invocations or 20 synchronized
wall-clock invocations. Times in milliseconds, **median across rounds**:

| Batch | Mode | GPU event ms | Sync wall ms |
|---:|---|---:|---:|
| 1 | Eager | 0.587833 | 0.628024 |
| 1 | Inductor graph OFF | 0.274076 | 0.316222 |
| 1 | Inductor graph ON | **0.238605** | **0.305273** |
| 8 | Eager | 0.855658 | 0.860031 |
| 8 | Inductor graph OFF | **0.531366** | **0.576081** |
| 8 | Inductor graph ON | 0.534691 | 0.600979 |

Full preserved values: [2026-11-03-attribution-timing.csv](2026-11-03-attribution-timing.csv).

Graph ON vs OFF:
- Batch 1 GPU-event median: **12.94% lower** with Graph ON.
  Synchronized wall median: **3.46% lower** with Graph ON.
- Batch 8 GPU-event median: **0.63% higher** with Graph ON.
  Synchronized wall median: **4.32% higher** with Graph ON.
- This is a *single experiment session* and two differently designed
  timers, **not** strong evidence of statistical significance or the isolated
  cost of CPU launch. The per-request synchronized timer forces a GPU sync
  after each forward; the GPU-event timer does not.

Numeric comparison in both batches:
- max abs error 0.00390625;
- batch 1 mean abs error 0.000558865, P99 abs 0.001953125;
- batch 8 mean abs error 0.000587692, P99 abs 0.001953125.
- Diagnostic maximum clamped relative error was 0.25921 (batch 1)
  and 0.57764 (batch 8), not a classification error rate.

## Full Chrome Trace: graph replay observed

Five forwards recorded per mode in the short CPU/CUDA trace. Categorized
`traceEvents` with phase X, distinguishing GPU device kernels and CPU CUDA
runtime calls:

| Mode | GPU kernel events | `cudaGraphLaunch` CPU API events | `cudaLaunchKernel` | `cudaLaunchKernelExC` |
|---|---:|---:|---:|---:|
| Eager | 825 | 0 | 750 | 75 |
| Inductor graph OFF | 285 | 0 | 5 | 100 |
| Inductor graph ON | 290 | **5** | 5 | 0 |

The ON trace confirms **five `cudaGraphLaunch` calls over five forwards**,
whereas the OFF trace confirms zero. Under this workload this is
direct evidence of graph replay, not merely that the compile option was set.

Additional profiler-visible CUDA device events include memsets, etc.:
Eager **830**, OFF **310**, ON **315** across the five profiled forwards
as recorded by the user script. Full aggregated trace data:
[2026-11-03-attribution-trace-summary.csv](2026-11-03-attribution-trace-summary.csv).

**Key distinction:** Inductor OFF and ON show very similar GPU kernel
counts (285 vs 290 over five forwards, including some graph-dependent
differences), while CPU CUDA API dispatch differs dramatically. Hence
the observed graph-on benefit at batch 1 is consistent with replay
reducing submission overhead, **not** an extra 2x kernel fusion.
This does not quantify a causal latency contribution for one optimization.

Both Inductor traces include fused Triton pointwise kernel names,
including `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu`
and `triton_poi_fused__native_batch_norm_legit_no_training_add_relu`.
The word `convolution` in fused pointwise kernel names does not mean
the entire cuDNN convolution was implemented as a Triton fused kernel.

## What still needs work?

- Run Batch 32 to determine whether the near-tie/Graph OFF benefit at Batch 8
  continues at larger workloads.
- Repeat independent sessions and control clocks/temperature if making a
  statistically robust performance claim.
- Inspect host CUDA API timings with Nsight Systems for a more precise
  launch-path decomposition; avoid directly subtracting GPU-event latency
  from synchronized host-wall latency.
- Measure any actual custom Triton fusion separately in a future PR after
  selecting a bottleneck.

## PR status recommendation

**The original PR #4 experimental gate has been met**: option compatibility
fixed, tests pass, numerical results reported, and complete trace confirms
CUDA Graph replay. The scope can be merged after review. More elaborate
attribution, Batch 32 and Nsight profiling can be separate future phases.
