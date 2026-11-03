"""CPU-compatible unit tests for the attribution helper functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "resnet18"))
from benchmark_attribution import make_models


def test_make_models_has_explicit_modes(monkeypatch):
    import torch
    captured = []

    def fake_compile(model, **kwargs):
        captured.append(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    marker = object()
    models = make_models(marker)
    assert list(models) == ["eager", "inductor_graph_off", "inductor_graph_on"]
    assert models["eager"] is marker
    assert captured[0]["options"]["triton.cudagraphs"] is False
    assert captured[1]["options"]["triton.cudagraphs"] is True
    assert all(c["mode"] == "default" for c in captured)


def test_invalid_numerical_inputs():
    import torch
    from analysis_utils import compare_outputs
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_outputs(torch.tensor([float("nan")]), torch.tensor([0.0]))
