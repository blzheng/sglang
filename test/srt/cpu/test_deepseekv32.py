import unittest

import sgl_kernel
import torch
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestDeepSeekV32(CustomTestCase):
    def _deepseekv32_index_native(
        self,
        q: torch.Tensor,
        weight: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        _, M, _, _ = q.shape
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        k_exp = k_f32.unsqueeze(1).expand(-1, M, -1, -1)
        q_T = q_f32.transpose(-1, -2)
        logits = torch.matmul(k_exp, q_T)
        logits = torch.relu(logits)
        logits = logits * weight.unsqueeze(2)
        out = logits.sum(dim=-1)
        return out

    def test_deepseekv32_index(self):
        B = 1
        M = 1
        H = 64
        D = 128
        N = 14
        query = torch.rand(B, M, H, D, dtype=torch.bfloat16)
        key = torch.rand(B, N, D, dtype=torch.bfloat16)
        weight = torch.rand(B, M, H, dtype=torch.float32)
        out_ref = self._deepseekv32_index_native(query, weight, key)
        out = torch.ops.sgl_kernel.deepseek_index_cpu(query, weight, key)
        atol = rtol = precision[out_ref.dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
