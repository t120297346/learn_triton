# Experiment design and interpretation

## Goal

Understand how a simple memory-bound Triton vector-add kernel is compiled and scheduled on an RTX 5080. The work connects source code to TTIR, TTGIR blocked layout, LLVM IR, PTX, and CUDA performance measurements.

## Terminology

- **N**: number of valid float32 elements in each tensor.
- **BLOCK**: logical number of indices handled per Triton program instance (tile length).
- **Grid**: number of launched programs, computed as `ceil(N/BLOCK)`.
- **Mask**: protects the last partial tile against out-of-bounds memory operations.
- **Warp**: 32 NVIDIA threads; **CTA**: a CUDA thread block containing selected warps.
- **TTIR / TTGIR**: Triton compiler IR before/after GPU layout assignment.
- **sizePerThread**: blocked-layout parameter; not always the *total* number of tensor elements processed by a thread.
- **PTX**: NVIDIA virtual ISA. Actual native SASS may differ.
- **Registers/thread**: compiled register allocation; insufficient by itself to establish occupancy.
- **Effective bandwidth**: `(read A + read B + write C) / latency`, i.e. `12*N / seconds` for float32. Not a DRAM counter.

## Experiment 1: correctness and IR inspection

**Question:** Does the compiled kernel compute all elements, and how are offsets/masks lowered?

- Independent variables: `--n`, `--block`, optionally `--warps`.
- Hold: float32, same GPU, same kernel.
- Observables: correctness, Grid, TTIR/TTGIR layout, PTX, register allocation.
- Run: `python experiments/vector_add/inspect_ir.py --n 1024 --block 128 --output-dir outputs/a`, repeat for `--n 1048576 --block 256`.
- Important correction: the earliest exploratory script allocated only 8 elements while constructing its grid from 1,048,576; its kernel N was 8. This *only* validated eight additions, not a million. The corrected scripts allocate and launch with the same N.

The two historical configurations changed both N and BLOCK; they cannot isolate which parameter caused the layout change. Compare BLOCK with fixed N for causal interpretation.

## Experiment 2: block size sweep

**Question:** How does BLOCK affect runtime across workload sizes?

- N = 1,024 / 65,536 / 1,048,576 / 16,777,216.
- BLOCK = 128 / 256 / 512 / 1024; hold `num_warps=4`.
- Use CUDA Graph benchmark; five rounds with shuffled configuration order.
- Compute median timing, then effective logical GB/s.
- Same buffers are reused; a value above advertised DRAM bandwidth can arise from caching, and does not imply such DRAM throughput.
- At small N, fixed overheads dominate; at large N, times converge. Hypotheses require profiling and additional independent runs.

## Experiment 3: warp/layout sweep

**Question:** How do warp count and layout choices correlate with runtime and PTX vector width?

- Hold N=1,048,576, float32, identical kernel/GPU.
- BLOCK = 128,256,512,1024; WARPS = 4,8.
- Capture TTGIR blocked layout, register count, shared bytes, PTX load/store mnemonics and CUDA Graph latency.
- Differences in block size, number of CTAs, vector width, threads, and registers can co-vary. Associations are not single-factor causal proof.
- PTX instruction counts in source text are *static code occurrences*, not hardware execution counters.

## Reproducibility checklist

Record: GPU, CUDA driver, PyTorch/Triton versions, exact commands, clocks/thermal state, power mode, benchmark variant, repetitions and statistics. Commit hand-curated observations only; keep raw generated IR/CSV under ignored `outputs/` unless intentionally sharing a curated result. Benchmarking via CUDA Graph replays the same allocations and may show cache residency; validate memory traffic with Nsight Compute before making DRAM claims.

## Next

Study tiled matmul, shared-memory reuse, Tensor Core lowering, and then compare GPU execution models with TPU/Pallas. No TPU backend implementation is claimed here.
