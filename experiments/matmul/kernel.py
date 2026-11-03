"""Baseline FP16 tiled matrix multiplication using tl.dot.

A [M,K] and B [K,N] are contiguous row-major CUDA float16 tensors.
Accumulation uses float32 and results are stored as float16.
"""
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A, B, C,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)

    a_ptrs = A + rows[:, None] * K + ks[None, :]
    b_ptrs = B + ks[:, None] * N + cols[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)

    for ki in range(tl.cdiv(K, BK)):
        k_mask = ki * BK + ks
        aa = tl.load(a_ptrs, mask=(rows[:, None] < M) & (k_mask[None, :] < K), other=0)
        bb = tl.load(b_ptrs, mask=(k_mask[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(aa, bb, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    c_ptrs = C + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def launch(a, b, out, bm=64, bn=64, bk=32, num_warps=4):
    """Validate shapes/dtypes and launch one output tile per Triton program."""
    import torch
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("Expected 2D matrices")
    m, k = a.shape
    kb, n = b.shape
    if k != kb or out.shape != (m, n) or min(m, n, k) <= 0:
        raise ValueError("Incompatible matrix shapes")
    if any(x.device.type != "cuda" for x in (a, b, out)):
        raise ValueError("CUDA tensors required")
    if any(x.dtype != torch.float16 for x in (a, b, out)):
        raise ValueError("FP16 tensors required")
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("Contiguous row-major tensors required")
    if any(x < 16 or x & (x - 1) for x in (bm, bn, bk)):
        raise ValueError("Tile dimensions must be powers of two >= 16")
    if num_warps not in (4, 8):
        raise ValueError("Supported benchmark warp counts: 4 or 8")
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    return matmul_kernel[grid](a, b, out, m, n, k,
                               BM=bm, BN=bn, BK=bk, num_warps=num_warps)
