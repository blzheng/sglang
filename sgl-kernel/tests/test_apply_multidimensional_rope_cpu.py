import pytest
import torch

import sgl_kernel  # noqa: F401 – ensure ops are registered


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Reference rotate_half: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference single-chunk rotary."""
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference Python implementation of multidimensional RoPE (ndim=2).

    x:   [num_tokens, num_heads, head_dim]
    cos: [num_tokens, head_dim]  (or broadcastable)
    sin: [num_tokens, head_dim]
    """
    ndim = 2
    chunk_size = x.shape[-1] // ndim
    # Unsqueeze cos/sin to broadcast over heads
    cos_3d = cos.unsqueeze(1)  # [num_tokens, 1, head_dim]
    sin_3d = sin.unsqueeze(1)
    x_parts = x.split(chunk_size, dim=-1)
    cos_parts = cos_3d.split(chunk_size, dim=-1)
    sin_parts = sin_3d.split(chunk_size, dim=-1)
    y_parts = [
        _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k]) for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
@pytest.mark.parametrize("num_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
@pytest.mark.parametrize("cos_sin_dtype", ["same", "float32"])
def test_apply_multidimensional_rope_cpu_correctness(
    dtype, num_tokens, num_heads, head_dim, cos_sin_dtype
):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cpu")
    x_ref = x.clone()

    if cos_sin_dtype == "float32":
        cs_dtype = torch.float32
    else:
        cs_dtype = dtype

    cos = torch.randn(num_tokens, head_dim, dtype=cs_dtype, device="cpu")
    sin = torch.randn(num_tokens, head_dim, dtype=cs_dtype, device="cpu")

    # Reference
    expected = _apply_multidimensional_rope_ref(x_ref.float(), cos.float(), sin.float()).to(dtype)

    # Kernel (in-place)
    result = torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(x, cos, sin)
    assert result.data_ptr() == x.data_ptr(), "should be in-place"

    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


def test_apply_multidimensional_rope_cpu_non_contiguous():
    """Test that non-contiguous cos/sin raises an error."""
    x = torch.randn(4, 8, 64, dtype=torch.bfloat16)
    cos = torch.randn(4, 128, dtype=torch.float32)[:, ::2]  # non-contiguous
    sin = torch.randn(4, 64, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(x, cos, sin)


def test_apply_multidimensional_rope_cpu_odd_head_dim():
    """Test that odd head_dim raises an error."""
    x = torch.randn(4, 8, 63, dtype=torch.bfloat16)
    cos = torch.randn(4, 63, dtype=torch.float32)
    sin = torch.randn(4, 63, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="head_dim must be divisible by 2"):
        torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(x, cos, sin)


def test_apply_multidimensional_rope_cpu_dim_mismatch():
    """Test that mismatched dimensions raise an error."""
    x = torch.randn(4, 8, 64, dtype=torch.bfloat16)
    cos = torch.randn(4, 32, dtype=torch.float32)
    sin = torch.randn(4, 64, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(x, cos, sin)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
