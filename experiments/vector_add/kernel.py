"""Shared Triton vector-add kernel, used by each experiment."""
import triton
import triton.language as tl


@triton.jit
def add_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)
    tl.store(C + offsets, a + b, mask=mask)


def launch(a, b, c, block: int, num_warps: int = 4):
    """Launch with tensor length consistently used for allocation, N and grid."""
    if a.numel() != b.numel() or a.numel() != c.numel():
        raise ValueError("Input/output lengths must match")
    if block <= 0 or block & (block - 1):
        raise ValueError("block must be a positive power of two")
    n = a.numel()
    grid = (triton.cdiv(n, block),)
    return add_kernel[grid](a, b, c, n, BLOCK=block, num_warps=num_warps)
