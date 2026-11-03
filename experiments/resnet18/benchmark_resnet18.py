
import torch
from torchvision.models import resnet18

# ==========================================
# 1. Environment
# ==========================================

device = "cuda"

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

# ==========================================
# 2. Model
# ==========================================

model = resnet18(weights=None).eval().half().to(device)

compiled_default = torch.compile(
    model,
    mode="default"
)

compiled_reduce_overhead = torch.compile(
    model,
    mode="reduce-overhead"
)

models = {
    "Eager": model,
    "Default": compiled_default,
    "ReduceOverhead": compiled_reduce_overhead,
}

# ==========================================
# 3. Benchmark
# ==========================================

def bench(fn, x, warmup=30, repeat=100):
    with torch.inference_mode():

        # Warmup (includes first-time compilation)
        for _ in range(warmup):
            fn(x)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(repeat):
            fn(x)

        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / repeat


# ==========================================
# 4. Error Analysis
# ==========================================


@torch.inference_mode()
def check_error(reference, test, x):

    y_ref = reference(x).float()
    y_test = test(x).float()

    diff = (y_ref - y_test).abs()
    relative = diff / y_ref.abs().clamp(min=1e-3)

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    p99_abs = torch.quantile(
        diff.flatten(), 0.99
    ).item()

    max_relative = relative.max().item()

    # 找出最大相對誤差對應的輸出
    idx = relative.flatten().argmax()

    ref_at_max_rel = (
        y_ref.flatten()[idx].item()
    )
    test_at_max_rel = (
        y_test.flatten()[idx].item()
    )
    abs_at_max_rel = (
        diff.flatten()[idx].item()
    )

    passed = torch.allclose(
        y_ref, y_test,
        rtol=1e-2,
        atol=1e-2
    )

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "p99_abs": p99_abs,
        "max_relative": max_relative,
        "ref_at_max_rel": ref_at_max_rel,
        "test_at_max_rel": test_at_max_rel,
        "abs_at_max_rel": abs_at_max_rel,
        "allclose": passed,
    }


# ==========================================
# 5. Run Experiments
# ==========================================

for batch in [1, 8, 32]:

    x = torch.randn(
        batch, 3, 224, 224,
        device=device,
        dtype=torch.float16
    )

    print(f"\n{'=' * 55}")
    print(f"Batch Size = {batch}")
    print(f"{'=' * 55}")

    # Eager is the reference
    eager_ms = bench(model, x)

    print(f"Eager: {eager_ms:.4f} ms")

    for name, fn in models.items():

        if name == "Eager":
            continue

        latency = bench(fn, x)

        speedup = eager_ms / latency
        throughput = batch / (latency / 1000)

        error = check_error(model, fn, x)

        print(f"\n{name}")
        print(f"  Latency:           {latency:.4f} ms")
        print(f"  Speedup:           {speedup:.2f}x")
        print(f"  Throughput:        {throughput:.2f} images/s")
        print(f"  Max Abs Error:     {error['max_abs']:.6f}")
        print(f"  Mean Abs Error:    {error['mean_abs']:.6f}")
        print(f"  Max Rel Error:     {error['max_relative']:.6f}")
        print(f"  P99 Abs Error:     {error['p99_abs']:.6f}")
        print(f"  Ref at MaxRel:     {error['ref_at_max_rel']:.6f}")
        print(f"  Test at MaxRel:    {error['test_at_max_rel']:.6f}")
        print(f"  Abs at MaxRel:     {error['abs_at_max_rel']:.6f}")
        print(f"  Allclose:          {error['allclose']}")

print("\nAll experiments completed!")

