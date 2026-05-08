import itertools
import unittest

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(42)

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# Reference implementation (from index_buf_accessor_v4.py)
def _set_k_and_s_torch(buf, loc, k_nope, k_rope, scale_k_nope, page_size):
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape

    nope_dim = k_nope.shape[1]
    rope_dim = k_rope.shape[1]
    scale_dim = scale_k_nope.shape[1]

    buf_fp8 = buf.view(fp8_dtype).flatten()
    buf_bf16 = buf.view(torch.bfloat16).flatten()
    buf_scale = buf.view(torch.uint8).flatten()

    loc_page_index = loc // page_size
    loc_token_offset_in_page = loc % page_size

    s_offset_nbytes_in_page = page_size * (nope_dim + rope_dim * 2)

    nope_offset = loc_page_index * buf_numel_per_page + loc_token_offset_in_page * (
        nope_dim + rope_dim * 2
    )

    rope_offset = (
        loc_page_index * buf_numel_per_page // 2
        + (loc_token_offset_in_page * (nope_dim + rope_dim * 2) + nope_dim) // 2
    )

    s_offset = (
        loc_page_index * buf_numel_per_page
        + s_offset_nbytes_in_page
        + loc_token_offset_in_page * (scale_dim + 1)
    )

    for i in range(num_tokens_to_write):
        buf_fp8[nope_offset[i] : nope_offset[i] + nope_dim] = k_nope[i]
        buf_bf16[rope_offset[i] : rope_offset[i] + rope_dim] = k_rope[i]
        buf_scale[s_offset[i] : s_offset[i] + scale_dim] = scale_k_nope[i]


def make_test_data(num_pages, page_size, num_tokens, nope_dim=448, rope_dim=64, scale_dim=7):
    """Create test data matching the buffer layout."""
    nope_rope_bytes_per_token = nope_dim + rope_dim * 2
    s_bytes_per_token = scale_dim + 1
    buf_numel_per_page = page_size * nope_rope_bytes_per_token + page_size * s_bytes_per_token

    buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8)

    # Generate random non-overlapping locations
    total_slots = num_pages * page_size
    assert num_tokens <= total_slots
    perm = torch.randperm(total_slots)[:num_tokens]
    loc = perm.to(torch.int64)

    k_nope = torch.randint(0, 256, (num_tokens, nope_dim), dtype=torch.uint8).view(fp8_dtype)
    k_rope = torch.randn(num_tokens, rope_dim, dtype=torch.bfloat16)
    scale_k_nope = torch.randint(0, 256, (num_tokens, scale_dim), dtype=torch.uint8)

    return buf, loc, k_nope, k_rope, scale_k_nope


class TestSetKAndS(CustomTestCase):
    num_pages_list = [4, 16]
    page_size_list = [1, 16]
    num_tokens_list = [1, 7, 32]

    def _test_set_k_and_s(self, num_pages, page_size, num_tokens):
        max_tokens = num_pages * page_size
        if num_tokens > max_tokens:
            num_tokens = max_tokens

        buf, loc, k_nope, k_rope, scale_k_nope = make_test_data(
            num_pages, page_size, num_tokens
        )

        # Reference
        buf_ref = buf.clone()
        _set_k_and_s_torch(buf_ref, loc, k_nope, k_rope, scale_k_nope, page_size)

        # C++ kernel
        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_and_s_cpu(
            buf_test, loc, k_nope, k_rope, scale_k_nope, page_size
        )

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_k_and_s(self):
        for params in itertools.product(
            self.num_pages_list, self.page_size_list, self.num_tokens_list
        ):
            with self.subTest(
                num_pages=params[0], page_size=params[1], num_tokens=params[2]
            ):
                self._test_set_k_and_s(*params)

    def test_set_k_and_s_int32_loc(self):
        """Test with int32 loc tensor."""
        buf, loc, k_nope, k_rope, scale_k_nope = make_test_data(8, 16, 20)
        loc_i32 = loc.to(torch.int32)

        buf_ref = buf.clone()
        _set_k_and_s_torch(buf_ref, loc, k_nope, k_rope, scale_k_nope, 16)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_and_s_cpu(
            buf_test, loc_i32, k_nope, k_rope, scale_k_nope, 16
        )

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_k_and_s_large(self):
        """Larger stress test."""
        num_pages, page_size, num_tokens = 64, 16, 512
        buf, loc, k_nope, k_rope, scale_k_nope = make_test_data(
            num_pages, page_size, num_tokens
        )

        buf_ref = buf.clone()
        _set_k_and_s_torch(buf_ref, loc, k_nope, k_rope, scale_k_nope, page_size)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_and_s_cpu(
            buf_test, loc, k_nope, k_rope, scale_k_nope, page_size
        )

        torch.testing.assert_close(buf_ref, buf_test)


if __name__ == "__main__":
    unittest.main()
