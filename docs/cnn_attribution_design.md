# Phase 4: Attribution of CNN Compiler Optimizations

## Motivation

The Phase 3 ResNet18 experiment found approximately **2.50x** Batch=1
speedup for `torch.compile(mode="reduce-overhead")` over Eager, but it did
not isolate the sources of improvement. PyTorch Profiler found fewer
visible device events and Triton fused pointwise kernel names, while
cuDNN convolution kernels remained.

The next step is a **controlled comparison**, not a claim of improved
CNN accuracy or a new optimized convolution kernel.

## Comparison strategy

1. Eager ResNet18 baseline.
2. Inductor `mode="default"` with `options={"triton.cudagraphs": False}`.
3. Inductor `mode="default"` with `options={"triton.cudagraphs": True}`.

This pair keeps the requested compile mode fixed and explicitly toggles
one CUDA Graph configuration option. It provides stronger evidence than
comparing only `default` to `reduce-overhead`, but the compiler may
change implementation details or fall back; **a requested graph option
is not proof that capture/replay actually happened**.

Refer to the upstream PyTorch `torch.compile` documentation. The
`triton.cudagraphs` option is known for the target PyTorch version but
may vary across releases. The scripts print/save `list_mode_options()`
when available and should fail visibly if an option is unsupported.

## Two timing measurements

- **GPU event latency**: CUDA events around repeated inference forwards.
  Measures elapsed GPU timeline between events (possibly including gaps),
  but does not count total host/application handling time.
- **Synchronized wall latency**: host `perf_counter_ns`, with a
  `torch.cuda.synchronize()` after each forward. This intentionally
  serializes requests, and so differs from the unsynchronized GPU event
  benchmark. It is a separately labeled estimate of single-request
  end-to-end latency, not a direct measurement of 'launch overhead'.

Warm up models and synchronize before measurements; report medians and
min/max for shuffled rounds. Use identical seeded FP16 random inputs
per mode and verify Eager agreement.

## Trace analysis

Capture short separate CPU/CUDA traces for each mode *after* first compilation,
then inspect the JSON in Perfetto/Chrome Trace Viewer or a trace analysis tool.

Questions:
- Does the `triton.cudagraphs` toggle lead to observed graph replay on the
  actual runtime? Check CPU CUDA API activity and graph replay markers.
- How many device kernels and graph-related activities appear per iteration?
- What fused `triton_poi_fused_...` names persist in both Inductor modes?
- Do convolution and layout-conversion events differ?

Beware that profiler event visibility can differ with CUDA Graph replay.
For detailed CUDA runtime API launches, Nsight Systems is a future step.

## What can and cannot be inferred

- Inductor-off vs Inductor-on is a test of requested graph configuration
  within the same compile mode.
- Eager vs Inductor-off is an **aggregate compiler-path contrast**, not the
  isolated causal contribution of Fusion.
- Finding fused kernel names documents the presence of fused code. It does
  not quantify the speedup from Fusion, as the compiler may change other
  scheduling/layout choices.
- `weights=None` uses random parameters. Accuracy on real data is not tested.
- CUDA events and wall-clock numbers use different timing semantics.
- Do not claim statistics significance with only five rounds. Repeat across
  independent sessions after checking clocks/temperature.

## Run plan

~~~bash
python -m pytest -q tests/test_cnn_attribution.py
python experiments/resnet18/benchmark_attribution.py --batch 1
python experiments/resnet18/benchmark_attribution.py --batch 8 --output-dir outputs/resnet18/attribution_b8
python experiments/resnet18/trace_attribution.py --batch 1
~~~

Keep generated full JSON traces and environment metadata under the ignored
`outputs/` folder. Archive them externally for analysis rather than commit
large machine-generated files directly to Git.

## PR gate

Before merge: confirm script correctness on RTX 5080, numerical consistency,
environment JSON, timing CSVs and a meaningful trace. If graph replay is not
observed, record this explicitly rather than inferring graph benefits.
