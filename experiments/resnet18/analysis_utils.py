"""Numerical comparison for FP16 CNN outputs.

The clamped relative error is a diagnostic, not a model accuracy measure.
"""
import torch


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor,
                    atol: float = 1e-2, rtol: float = 1e-2,
                    rel_floor: float = 1e-3):
    if reference.shape != candidate.shape:
        raise ValueError("Output shapes differ")
    if reference.numel() == 0 or rel_floor <= 0 or atol < 0 or rtol < 0:
        raise ValueError("Invalid output or tolerance")
    ref = reference.detach().float().flatten()
    test = candidate.detach().float().flatten()
    if not bool(torch.isfinite(ref).all() and torch.isfinite(test).all()):
        raise ValueError("Nonfinite model output")
    diff = (ref - test).abs()
    rel = diff / ref.abs().clamp(min=rel_floor)
    index = int(rel.argmax().item())
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(diff, 0.99).item()),
        "max_rel_clamped": float(rel[index].item()),
        "ref_at_max_rel": float(ref[index].item()),
        "test_at_max_rel": float(test[index].item()),
        "abs_at_max_rel": float(diff[index].item()),
        "allclose": bool(torch.allclose(ref, test, atol=atol, rtol=rtol)),
    }
