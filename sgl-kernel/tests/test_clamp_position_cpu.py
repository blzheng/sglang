import sys

import pytest
import torch


def _reference_clamp_position(seq_lens):
    return torch.clamp(seq_lens - 1, min=0)


def _has_clamp_position_cpu():
    try:
        torch.ops.sgl_kernel.clamp_position_cpu
        return True
    except (AttributeError, RuntimeError):
        return False


requires_sgl_kernel_cpu = pytest.mark.skipif(
    not _has_clamp_position_cpu(),
    reason="sgl_kernel CPU clamp_position_cpu op not available",
)


@requires_sgl_kernel_cpu
@pytest.mark.parametrize("size", [1, 2, 7, 8, 15, 16, 127, 128, 255, 256, 1024, 4097])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
class TestClampPositionCPU:
    def test_normal(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.randint(1, 10000, (size,), dtype=dtype, device="cpu")
        expected = _reference_clamp_position(seq_lens)
        result = torch.ops.sgl_kernel.clamp_position_cpu(seq_lens)
        assert torch.equal(result, expected)

    def test_zeros(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.zeros(size, dtype=dtype, device="cpu")
        expected = _reference_clamp_position(seq_lens)
        result = torch.ops.sgl_kernel.clamp_position_cpu(seq_lens)
        assert torch.equal(result, expected)

    def test_ones(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.ones(size, dtype=dtype, device="cpu")
        expected = _reference_clamp_position(seq_lens)
        result = torch.ops.sgl_kernel.clamp_position_cpu(seq_lens)
        assert torch.equal(result, expected)

    def test_mixed(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.randint(0, 10000, (size,), dtype=dtype, device="cpu")
        expected = _reference_clamp_position(seq_lens)
        result = torch.ops.sgl_kernel.clamp_position_cpu(seq_lens)
        assert torch.equal(result, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
