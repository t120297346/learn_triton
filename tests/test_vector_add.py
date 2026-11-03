"""GPU correctness tests; skip gracefully when CUDA is not available."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "vector_add"))
from kernel import launch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("n,block,warps", [
    (8, 128, 4), (1024, 128, 4), (1025, 256, 4),
    (65536, 512, 4), (1048576, 1024, 8),
])
def test_add_kernel(n, block, warps):
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")
    c = torch.empty_like(a)
    launch(a, b, c, block, warps)
    torch.testing.assert_close(c, a+b, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_length_mismatch():
    a = torch.empty(8, device="cuda")
    b = torch.empty(9, device="cuda")
    c = torch.empty(8, device="cuda")
    with pytest.raises(ValueError, match="lengths"):
        launch(a, b, c, 128)
