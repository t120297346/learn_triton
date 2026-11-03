"""MatMul correctness tests, including tail masks for non-divisible dimensions."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "matmul"))
from kernel import launch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("m,n,k,bm,bn,bk", [
    (64, 64, 64, 32, 64, 32),
    (128, 128, 128, 64, 64, 32),
    (255, 257, 129, 64, 64, 32),
    (512, 256, 128, 64, 128, 64),
])
def test_fp16_matmul(m, n, k, bm, bn, bk):
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    launch(a, b, out, bm, bn, bk)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.mm(a, b), atol=0.125, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_mismatched_inner_dimension():
    a = torch.empty((16, 32), device="cuda", dtype=torch.float16)
    b = torch.empty((33, 16), device="cuda", dtype=torch.float16)
    out = torch.empty((16, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="Incompatible"):
        launch(a, b, out)
