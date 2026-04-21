import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestClampCPU(CustomTestCase):
    shapes = [
        (1,),
        (7,),
        (128,),
        (4097,),
        (2, 128),
        (4, 8, 256),
        (16, 1024),
    ]
    dtype = [torch.bfloat16, torch.float16]

    def _clamp_basic_test(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype)
        min_val, max_val = torch.tensor(-0.5), torch.tensor(0.5)
        expected = torch.clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def _clamp_no_clamp_test(self, shape, dtype):
        """When bounds are ±inf, clamp should be a no-op."""
        x = torch.randn(shape, dtype=dtype)
        min_val, max_val = torch.tensor(float("-inf")), torch.tensor(float("inf"))
        expected = torch.clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def _clamp_min_only_test(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype)
        min_val, max_val = torch.tensor(-0.3), torch.tensor(float("inf"))
        expected = torch.clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def _clamp_max_only_test(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype)
        min_val, max_val = torch.tensor(float("-inf")), torch.tensor(0.3)
        expected = torch.clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def _clamp_tight_bounds_test(self, shape, dtype):
        """Clamp to a single value."""
        x = torch.randn(shape, dtype=dtype)
        min_val, max_val = torch.tensor(0.0), torch.tensor(0.0)
        expected = torch.clamp(x, min_val, max_val)
        torch.ops.sgl_kernel.clamp_cpu(x, min_val, max_val)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)

    def test_clamp(self):
        for params in itertools.product(self.shapes, self.dtype):
            with self.subTest(shape=params[0], dtype=params[1]):
                self._clamp_basic_test(*params)
                self._clamp_no_clamp_test(*params)
                self._clamp_min_only_test(*params)
                self._clamp_max_only_test(*params)
                self._clamp_tight_bounds_test(*params)


if __name__ == "__main__":
    unittest.main()
