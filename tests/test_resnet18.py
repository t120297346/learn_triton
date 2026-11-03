"""CPU-safe numerical metric tests and optional CUDA ResNet18 correctness."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "resnet18"))
from analysis_utils import compare_outputs


def test_identical_outputs():
    x = torch.tensor([[0.0, -1.0, 2.0]])
    m = compare_outputs(x, x)
    assert m["allclose"]
    assert m["max_abs"] == 0
    assert m["p99_abs"] == 0


def test_relative_error_clamp_is_diagnostic():
    ref = torch.tensor([0.0, 2.0])
    candidate = torch.tensor([0.0005, 2.0])
    m = compare_outputs(ref, candidate, atol=1e-2, rtol=1e-2)
    assert m["allclose"]
    assert m["max_rel_clamped"] == pytest.approx(.5)
    assert m["ref_at_max_rel"] == 0.0


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError, match="shapes"):
        compare_outputs(torch.zeros(2), torch.zeros(3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_compiled_resnet18_fp16_consistency():
    from torchvision.models import resnet18
    torch.manual_seed(42)
    model = resnet18(weights=None).eval().half().cuda()
    x = torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.float16)
    compiled = torch.compile(model, mode="default")
    with torch.inference_mode():
        ref = model(x)
        test = compiled(x)
    assert compare_outputs(ref, test)["allclose"]
