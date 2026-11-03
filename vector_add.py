
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(A, B, C, N: tl.constexpr,
               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)

    tl.store(C + offsets, a + b, mask=mask)


N = 1048576
BLOCK_SIZE = 256
grid = (triton.cdiv(N, BLOCK_SIZE),)

# 真正建立 N 個元素
a = torch.rand(N, device="cuda", dtype=torch.float32)
b = torch.rand(N, device="cuda", dtype=torch.float32)
c = torch.empty_like(a)

grid = (triton.cdiv(N, BLOCK_SIZE),)

kernel = add_kernel[grid](
    a, b, c, N, BLOCK=BLOCK_SIZE
)

print("Result:", c)
print("Correct:", torch.allclose(c, a + b))

# 觀察編譯結果
for name, code in kernel.asm.items():
    print(f"\n=== {name} ===")
    print(str(code)[:1500])

