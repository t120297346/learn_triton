# Phase 2 — FP16 MatMul benchmark and IR evidence (2026-11-03)

## Source and status

- Source: user-uploaded `matmul_results.tar.gz` containing `outputs/matmul/benchmark.csv` and five textual IR exports: `source.txt`, `ttir.txt`, `ttgir.txt`, `llir.txt`, `ptx.txt`.
- Archive SHA256: `88857468f987654b7e46cf5d22d4c1b8dea06c0ea44fb8ca568db48896318b17`.
- Intended system: RTX 5080 and Triton 3.9.0 (per Phase 1). **This archive does not contain an independent environment log**, so the exact runtime versions for this run were not independently confirmed from it.
- **Correctness**: benchmark script checks Triton vs preallocated `torch.mm` before timing each configuration, and the CSV implies the benchmark ran through its configurations. However **no `pytest` transcript is in the archive**; do not record unit tests as passing without that log.
- This document records externally supplied measurements, not re-execution by GitHub tooling.

## Experiment design

- Research goal: inspect tile-based reuse and tensor/matrix instructions, compare explicit configurations against PyTorch for square FP16 `A[M,K] @ B[K,N]`.
- Variables: `M=N=K ∈ {256,512,1024,2048}`, configurations `BM,BN,BK,WARPS ∈ {(32,64,32,4),(64,64,32,4),(64,128,32,4),(64,128,64,4)}`.
- Controlled: same kernel, contiguous row-major FP16 tensors, FP32 accumulation, repeated same allocations, CUDA Graph benchmark, 5 trial rounds per setting.
- Metrics: reported median CUDA Graph latency and mathematical TFLOPS `2MNK / elapsed_seconds / 1e12`.
- Note: each reported latency is based on medians and rounding in the submitted CSV. No cross-session variance or GPU hardware counters supplied.

## Measurements

Full 20-row data: [2026-11-03-matmul-benchmark.csv](2026-11-03-matmul-benchmark.csv).

| Size (square) | Fastest Triton BM×BN×BK | Triton time (μs) | PyTorch time (μs) | Triton TFLOPS | PyTorch TFLOPS |
|---:|---|---:|---:|---:|---:|
| 256 | 32×64×32 | 2.26533 | 2.55192 | 14.812 | 13.149 |
| 512 | 32×64×32 | 5.05925 | 5.16604 | 53.058 | 51.962 |
| 1024 | 32×64×32 | 24.53884 | 25.89445 | 87.514 | 82.932 |
| 2048 | 64×64×32 | 156.97656 | 159.15426 | 109.442 | 107.945 |

The best observed configuration changes at 2048. These results neither establish a global optimum nor prove robust speedups over `torch.mm` across GPUs and repeated sessions.

## Compiler evidence from supplied 256×256 IR (BM=64, BN=64, BK=32, WARPS=4)

In `ttir.txt`, the compiler preserves the high-level operation as `tt.dot` over `tensor<64x32xf16>` and `tensor<32x64xf16>`, producing `tensor<64x64xf32>`.

In `ttgir.txt`:
```mlir
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0,
                         warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
```
The generated code contains `ttg.local_alloc`, `ttg.async_copy_global_to_local`, `ttg.async_commit_group`, `ttg.async_wait`, and `ttg.local_load` staging operations, along with `tt.dot` and an `#mma` accumulator layout.

In `ptx.txt`, examples of emitted instruction families:
```ptx
cp.async.cg.shared.global
ldmatrix.sync.aligned.m8n8.x4.shared.b16
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

These directly establish a generated PTX path using asynchronous global-to-shared copies, matrix operand loads, and NVIDIA matrix multiply-accumulate instructions. They **do not by themselves** establish dynamic Tensor Core utilization, occupancy, throughput limits or memory traffic; use Nsight Compute for those claims. The recorded `#ttg.nvidia_mma versionMajor=2` denotes the compiler's chosen MMA layout family; it does not mean an RTX 5080 is a second-generation GPU.

## Interpretation and next checks

1. Phase 1 vector addition had zero shared-memory allocation and ordinary scalar/vector global reads; this MatMul compilation includes shared staging and matrix instructions, showing why compute kernels require a different memory and layout model.
2. Larger tiles are not consistently faster; candidate explanations include occupancy, registers, staging costs, launch grid and scheduling, but none is isolated by the existing sweep.
3. Numerical comparisons are tolerant FP16 checks rather than strict equality. Add explicit max/mean absolute error recording in a future study.
4. Attach the missing `pytest -q tests/test_matmul.py` output (and environment snapshot) before merging PR #2. Keep benchmark outputs and direct evidence distinct from hypotheses.
