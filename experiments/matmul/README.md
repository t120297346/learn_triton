# Phase 2 — FP16 tiled Matrix Multiplication

## Topic and learning goal

Vector addition had little data reuse and zero shared-memory allocation. This phase studies a more compute-intensive kernel: matrix multiplication `C = A @ B`. A block computes a `BM × BN` output tile, accumulating products over the reduction axis in chunks of `BK`.

The `tl.dot()` expression allows Triton to lower the matrix multiplication using the target GPU's supported matrix instructions. This is a **baseline**, not a hand-tuned/optimal Tensor Core implementation.

## Terms

- **M, N, K**: output rows, output columns, reduction dimension.
- **BM, BN, BK**: dimensions of the block-level tile.
- **Reduction**: sum over K to form each output element.
- **Data reuse**: a loaded A/B element can contribute to multiple outputs.
- **Tensor Core**: NVIDIA matrix-operation hardware; verify generated PTX and hardware profiling before describing instruction utilization.
- **TFLOPS**: theoretical FLOP count `2*M*N*K` divided by measured runtime in seconds and `10^12`.
- **Shared memory**: on-chip memory that the compiler may use for staging tiles; inspect generated metadata rather than assuming usage.

## Requirements

CUDA-enabled NVIDIA GPU, compatible PyTorch, Triton (historical reference: 3.9.0), NumPy and pytest.

```bash
export TRITON_DEFAULT_FP_FUSION=1
python -m pytest tests/test_matmul.py -q
python experiments/matmul/inspect_ir.py --m 256 --n 256 --k 256 --bm 64 --bn 64 --bk 32
python experiments/matmul/benchmark.py --output outputs/matmul/benchmark.csv
```

Each benchmark compares four explicit Triton tile configurations to preallocated-output `torch.mm`. Measurements are generated locally; no performance results are committed by this PR.

## Caveats

- All inputs and outputs are contiguous FP16 row-major, accumulated internally in FP32.
- Correctness uses `atol=0.125, rtol=0.02` for teaching baseline comparisons; numerical tolerance is not evidence of bitwise equivalence. Later experiments should analyze error distributions and reference precision.
- GPU kernels are asynchronous; correctness synchronizes before comparing.
- Preallocated outputs and CUDA Graph replay reduce allocation/CPU overhead but can influence caching. Real memory traffic, occupancy and Tensor Core utilization require a profiler.
- Not all tile/warp configurations are valid on all GPU architectures. The scripts were not executed on the user's RTX 5080 in this PR environment.
