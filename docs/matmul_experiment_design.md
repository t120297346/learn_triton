# Phase 2 experiment design: Triton FP16 MatMul

## Research question

How do block tiling and data reuse affect a Triton FP16 matrix multiply on an RTX 5080, and how does its compiler lowering differ from the Phase 1 scalar/vector-load addition kernel?

The purpose is to **learn the compiler transformation path**, not to claim a new optimization.

## Experiment A — correctness and boundary masks

- Inputs A[M,K], B[K,N], output C[M,N]; FP16 storage and FP32 accumulation.
- Compare with `torch.mm` at a documented numerical tolerance.
- Test square, rectangular, and non-divisible dimension cases to exercise the masks on M/N/K.
- One output program computes an BM×BN tile; its inner loop spans K in BK chunks.
- Capture the exact tensor shapes, configuration, GPU/device and software versions.

## Experiment B — compiler lowering

- Hold M=N=K=256 and tile (BM,BN,BK)=(64,64,32), warps=4 initially.
- Inspect `ttir.txt`, `ttgir.txt`, `llir.txt`, `ptx.txt`, as available.
- Search for `tt.dot`, layout transformations, `mma` instruction families, local/shared-memory operations.
- Record `registers/thread` and `shared bytes`. Not all target layouts will look the same across versions.
- Hypothesis: MatMul lowers to matrix-oriented instructions unlike Phase 1's scalar FP32 add.

## Experiment C — performance and tiling

- Independent variables: square dimension `N ∈ {256,512,1024,2048}`; explicit tiling configurations `(32,64,32,4)`, `(64,64,32,4)`, `(64,128,32,4)`, `(64,128,64,4)` where tuple = BM,BN,BK,WARPS.
- Control: FP16, row-major, random seed, same RTX 5080, same benchmark method, 5 shuffled repetitions per configuration.
- Comparison: `torch.mm` with preallocated output to avoid Python-side allocation for the timed call.
- Dependent variables: CUDA Graph median latency in ms and mathematical effective TFLOPS.
- Save raw per-configuration CSV. Compare trends; do not assume optimal tile selection.

## Important limitations

- A change of tile size affects multiple hardware parameters at once: CTA count, loads, register pressure, scheduling, memory staging and occupancy. Timing correlation is not causal isolation.
- Effective TFLOPS uses mathematical FLOPs; it is not a measured hardware Tensor Core instruction throughput.
- The PyTorch and Triton implementations may use different internal algorithms and precision pathways; correctness tolerance should be reviewed before more precise claims.
- CUDA Graph replay repeatedly uses the same tensors; cache residency can affect timing.
- Nsight Compute is needed for matrix-instruction counts, cache traffic and bottleneck identification.
- Reproduction evidence and CSV results must be added **after** the user runs these scripts on hardware; this PR does not assert tests passed.

## Follow-up

Analyze TTGIR/Tensor Core lowering and memory staging before attempting layout pass modification or a Pallas/TPU comparative kernel. Keep the TPU research as a separate stage.
