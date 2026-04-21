import sys

import pytest
import torch


def _reference_clamp(input, min_val, max_val):
    return torch.clamp(input, min_val, max_val)


def _has_clamp_cpu():
    try:
        torch.ops.sgl_kernel.clamp_cpu
        return True
    except (AttributeError, RuntimeError):
        return False


requires_sgl_kernel_cpu = pytest.mark.skipif(
    not _has_clamp_cpu(),
    reason="sgl_kernel CPU clamp_cpu op not available",
)


@requires_sgl_kernel_cpu
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (7,),
        (128,),
        (4097,),
        (2, 128),
        (4, 8, 256),
        (16, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
class TestClampCPU:
    def test_basic(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype, device="cpu")
        min_val, max_val = torch.tensor(-0.5), torch.tensor(0.5)
        expected = _reference_clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_no_clamp(self, shape, dtype):
        """When bounds are ±inf, clamp should be a no-op."""
        x = torch.randn(shape, dtype=dtype, device="cpu")
        min_val, max_val = torch.tensor(float("-inf")), torch.tensor(float("inf"))
        expected = _reference_clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_clamp_min_only(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype, device="cpu")
        min_val, max_val = torch.tensor(-0.3), torch.tensor(float("inf"))
        expected = _reference_clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_clamp_max_only(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype, device="cpu")
        min_val, max_val = torch.tensor(float("-inf")), torch.tensor(0.3)
        expected = _reference_clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_tight_bounds(self, shape, dtype):
        """Clamp to a single value."""
        x = torch.randn(shape, dtype=dtype, device="cpu")
        min_val, max_val = torch.tensor(0.0), torch.tensor(0.0)
        expected = _reference_clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_inplace(self, shape, dtype):
        """Verify clamp_cpu modifies the tensor in-place."""
        x = torch.randn(shape, dtype=dtype, device="cpu")
        original_data_ptr = x.data_ptr()
        min_val, max_val = torch.tensor(-0.5), torch.tensor(0.5)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        assert x.data_ptr() == original_data_ptr, "clamp_cpu should modify tensor in-place"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
