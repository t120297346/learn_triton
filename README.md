# Learning Triton: GPU kernels, compiler IR and benchmarks

Hands-on Triton / MLIR study using an NVIDIA GeForce RTX 5080, Triton 3.9.0.

## Repository layout

- `vector_add.py`: original first experiment, preserved for reference.
- `experiments/vector_add/kernel.py`: shared Triton vector-add kernel.
- `experiments/vector_add/inspect_ir.py`: correctness, grid/mask inspection, TTIR/TTGIR/LLVM IR/PTX export.
- `experiments/vector_add/benchmark_block.py`: launch-size vs. block-size performance study.
- `experiments/vector_add/benchmark_layout.py`: block/warp layout and register/PTX instruction study.
- `tests/test_vector_add.py`: CUDA-gated correctness tests, including non-divisible lengths.
- `docs/experiment_design.md`: hypotheses, controlled variables, caveats, and reproducibility.
- `results/2026-11-02-rtx5080.md`: transcribed historical observations (not newly re-run).

## Install

Python 3.12, CUDA-enabled PyTorch, Triton and NumPy. Existing reference environment: Triton 3.9.0.

```bash
python -m pip install torch triton numpy pytest
export TRITON_DEFAULT_FP_FUSION=1
python experiments/vector_add/inspect_ir.py --n 1024 --block 128 --output-dir outputs/ir128
python experiments/vector_add/inspect_ir.py --n 1048576 --block 256 --output-dir outputs/ir256
python experiments/vector_add/benchmark_block.py --output outputs/block.csv
python experiments/vector_add/benchmark_layout.py --output outputs/layout.csv
python -m pytest -q
```

Use the CUDA-compatible PyTorch installer for your machine. Benchmarks require a supported NVIDIA/AMD backend; the archived timings are from RTX 5080.

The benchmark outputs are ignored by Git. Do not mistake effective logical bandwidth for measured DRAM bandwidth. See `docs/experiment_design.md`.
