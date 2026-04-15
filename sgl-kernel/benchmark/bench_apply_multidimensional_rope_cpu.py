import itertools
import time

import torch

from sglang.utils import is_in_ci

IS_CI = is_in_ci()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope_ref(x, cos, sin):
    ndim = 2
    chunk_size = x.shape[-1] // ndim
    cos_3d = cos.unsqueeze(1)
    sin_3d = sin.unsqueeze(1)
    x_parts = x.split(chunk_size, dim=-1)
    cos_parts = cos_3d.split(chunk_size, dim=-1)
    sin_parts = sin_3d.split(chunk_size, dim=-1)
    y_parts = [
        _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k]) for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


def benchmark(num_tokens, num_heads, head_dim, dtype, num_warmup=5, num_iters=50):
    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cpu")
    cos = torch.randn(num_tokens, head_dim, dtype=dtype, device="cpu")
    sin = torch.randn(num_tokens, head_dim, dtype=dtype, device="cpu")

    # Warmup
    for _ in range(num_warmup):
        x_clone = x.clone()
        torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(x_clone, cos, sin)

    # Measure sgl_kernel
    start = time.perf_counter()
    for _ in range(num_iters):
        x_clone = x.clone()
        torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(x_clone, cos, sin)
    sgl_time = (time.perf_counter() - start) / num_iters * 1e6  # us

    # Warmup ref
    for _ in range(num_warmup):
        _apply_multidimensional_rope_ref(x.clone(), cos, sin)

    # Measure reference
    start = time.perf_counter()
    for _ in range(num_iters):
        _apply_multidimensional_rope_ref(x.clone(), cos, sin)
    ref_time = (time.perf_counter() - start) / num_iters * 1e6  # us

    speedup = ref_time / sgl_time if sgl_time > 0 else float("inf")
    return sgl_time, ref_time, speedup


if __name__ == "__main__":
    dtypes = [torch.bfloat16] if IS_CI else [torch.bfloat16, torch.float16]
    num_tokens_list = [64] if IS_CI else [1, 16, 64, 256, 1024]
    num_heads_list = [8] if IS_CI else [4, 8, 16]
    head_dims = [128] if IS_CI else [64, 128, 256]

    configs = list(itertools.product(dtypes, num_tokens_list, num_heads_list, head_dims))

    print(f"{'dtype':<12} {'tokens':<8} {'heads':<8} {'head_dim':<10} "
          f"{'SGL(us)':<12} {'Ref(us)':<12} {'Speedup':<10}")
    print("-" * 72)

    for dtype, num_tokens, num_heads, head_dim in configs:
        sgl_us, ref_us, speedup = benchmark(num_tokens, num_heads, head_dim, dtype)
        print(f"{str(dtype):<12} {num_tokens:<8} {num_heads:<8} {head_dim:<10} "
              f"{sgl_us:<12.2f} {ref_us:<12.2f} {speedup:<10.2f}x")
