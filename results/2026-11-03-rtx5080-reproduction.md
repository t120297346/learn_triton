# Reproduction run — 2026-11-03 (Asia/Taipei)

## Provenance

- **User-supplied** archive: `triton_results.tar.gz`; SHA256: `74a7432312ffa4c50c47a94aaf3905a768a7b2c02dc765875ac3885db007daf3`.
- Local timestamp: 2026-11-03 09:56:53+08:00.
- Code commit tested: `1bb6e823d46abcd96da137a927797112547addd9` (PR #1).
- Hardware: NVIDIA GeForce RTX 5080, 16 GB (16303 MiB reported); driver 595.99.02; NVIDIA-SMI CUDA 13.2.
- Python 3.12.3; PyTorch 2.15.0+cu132; Triton 3.9.0.
- Tests executed **on the user's GPU** and supplied as archived logs; not re-run in the GitHub update environment.

## Test 1: correctness

`python -m pytest -q` → **6 passed in 1.40s**. Tests cover N=8, 1024, 1025, 65536, 1048576 and mismatched input length rejection. This supports basic functionality, not exhaustive numerical validation.

## Test 2: compiler inspection with N fixed at 1,048,576

| BLOCK | WARPS | Grid | Correct | sizePerThread | Registers/thread | Shared bytes | PTX load |
|---:|---:|---:|---|---:|---:|---:|---|
| 128 | 4 | 8192 | True | 1 | 14 | 0 | `ld.global.b32` |
| 256 | 4 | 4096 | True | 2 | 16 | 0 | `ld.global.v2.b32` |

Both layouts use `threadsPerWarp=[32]`, `warpsPerCTA=[4]`, `order=[0]`.
Unlike the earliest exploratory test, the new runs use matching allocation length, kernel N and grid.

## Test 3: block-size sweep, CUDA Graph benchmark

See `2026-11-03-block.csv` for exact medians, effective bandwidth and trial count.

| N | B128 μs | B256 μs | B512 μs | B1024 μs |
|---:|---:|---:|---:|---:|
| 1,024 | 0.804 | 0.812 | 0.864 | 0.914 |
| 65,536 | 0.947 | 0.978 | 1.013 | 1.025 |
| 1,048,576 | 4.487 | 3.055 | 3.024 | 3.624 |
| 16,777,216 | 236.284 | 236.643 | 237.232 | 238.474 |

At 16M, effective logical bandwidth was **844.23–852.06 GB/s**. At 1M, the highest calculated value was **4161.525 GB/s** (BLOCK=512), which **is not measured DRAM throughput**: repeated CUDA Graph replays of the same buffers may benefit from cache residency.

## Test 4: layout sweep with N fixed at 1,048,576

See `2026-11-03-layout.csv` for full records and static PTX instruction mnemonics.

| BLOCK | WARPS | Time μs | sizePerThread | Registers/thread | PTX load width |
|---:|---:|---:|---:|---:|---|
| 128 | 4 | 4.4719 | 1 | 14 | scalar |
| 128 | 8 | 5.9482 | 1 | 14 | scalar |
| 256 | 4 | 3.0469 | 2 | 16 | v2 |
| 256 | 8 | 3.6732 | 1 | 14 | scalar |
| 512 | 4 | 3.0784 | 4 | 22 | v4 |
| 512 | 8 | 3.0606 | 2 | 16 | v2 |
| 1024 | 4 | 3.6161 | 4 | 28 | v4 (two groups) |
| 1024 | 8 | 3.0882 | 4 | 22 | v4 |

All configurations reported `shared_bytes=0`. PTX mnemonic occurrences are static code listings, not dynamic instruction or transaction counters.

## Comparison with 2026-11-02 observations

- Correctness and TTGIR/PTX patterns reproduce: 128/4 maps to scalar, 256/4 to v2; register counts are unchanged.
- Earlier 16M graph sweep ranged around 845.22–848.83 GB/s vs. 844.23–852.06 GB/s on this run: similar scale and trend.
- Earlier 1M, BLOCK=256: 3.04 μs, now 3.055 μs; BLOCK=512: 3.04 μs, now 3.024 μs.
- Earlier layout, BLOCK=256/4: 3.039 μs, now 3.0469 μs; BLOCK=256/8: 3.693 μs, now 3.6732 μs. Differences in a small fraction of a microsecond cannot establish stable ranking without distributions/replications across sessions.

## Conclusions and caveats

**Directly observed:** six tests passed; TTGIR layout, vectorized PTX, register counts and timing depend on compilation configuration; these findings were broadly reproduced.

**Not yet demonstrated:** actual L2 hit rate/DRAM traffic, occupancy or stalls, the causal impact of register count or vectorization in isolation, statistical significance across sessions. Next: Nsight Compute profiling; then tiled matrix multiplication and Tensor Core compilation.

The original archive is not committed as a binary; exact selected CSV snapshots are preserved in this PR and the archive SHA256 enables provenance checking.
