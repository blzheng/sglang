"""Unit tests for compress_decode_cpu and compress_extend_cpu kernels."""

import itertools
import time
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(42)


# ───────────────── Reference Python implementation ─────────────────


def compute_state_len(seq_len, ratio):
    return seq_len % ratio + (ratio == 4) * ratio


def overlap_transform(tensor, fill_value, ratio, head_dim):
    """tensor: [S, ratio, 2*head_dim] -> [S, 2*ratio, head_dim]"""
    s, r, _ = tensor.shape
    d = head_dim
    new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
    new_tensor[:, r:] = tensor[:, :, d:]
    if s > 1:
        new_tensor[1:, :r] = tensor[:-1, :, :d]
    return new_tensor


def overlap_transform_decode(tensor, ratio, head_dim):
    """tensor: [bs, 2*ratio, 2*head_dim] -> [bs, 2*ratio, head_dim]"""
    r, d = ratio, head_dim
    return torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)


def rmsnorm_ref(x, weight, eps):
    """RMS norm reference: x is [head_dim], weight is [head_dim]."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(variance + eps) * weight.float()).float()


def apply_rotary_emb_ref(x, freqs):
    """Interleaved rotary embedding reference. x, freqs are 1D [rope_dim]."""
    x = x.float().clone()
    f = freqs.float()
    for k in range(0, x.size(0), 2):
        xr, xi = x[k].item(), x[k + 1].item()
        cr, ci = f[k].item(), f[k + 1].item()
        x[k] = xr * cr - xi * ci
        x[k + 1] = xr * ci + xi * cr
    return x


def hadamard_transform_ref(x, scale):
    n = x.size(-1)
    out = x.float().clone()
    h = 1
    while h < n:
        for j in range(0, n, 2 * h):
            for k in range(h):
                a, b = out[j + k].item(), out[j + k + h].item()
                out[j + k] = a + b
                out[j + k + h] = a - b
        h <<= 1
    return out * scale


def compress_decode_ref(
    pool_kv, pool_score, kv, score, seq_lens, req_pool_indices,
    ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
    overlap, rotate, norm_eps
):
    """Pure Python reference matching compress_decode_old."""
    bs = kv.size(0)
    coff = 1 + (1 if overlap else 0)
    coff_hd = coff * head_dim

    output = torch.empty(bs, head_dim, dtype=torch.float32)

    for b in range(bs):
        seq_len = seq_lens[b].item()
        req_idx = req_pool_indices[b].item()
        write_pos = (seq_len - 1) % ratio + (ratio if overlap else 0)

        pool_kv[req_idx, write_pos] = kv[b]
        pool_score[req_idx, write_pos] = score[b]

        kv_buf = pool_kv[req_idx].clone()
        score_buf = pool_score[req_idx].clone()

        if overlap and (seq_len % ratio == 0):
            pool_kv[req_idx, :ratio] = kv_buf[ratio:]
            pool_score[req_idx, :ratio] = score_buf[ratio:]

        # Reshape to [coff, ratio, coff_hd] then add APE
        kv_r = kv_buf.view(coff, ratio, coff_hd)
        score_r = score_buf.view(coff, ratio, coff_hd)
        score_r = score_r + ape.unsqueeze(0)

        if overlap:
            kv_r2 = kv_r.view(1, coff * ratio, coff_hd)
            score_r2 = score_r.view(1, coff * ratio, coff_hd)
            kv_r2 = overlap_transform_decode(kv_r2, ratio, head_dim)
            score_r2 = overlap_transform_decode(score_r2, ratio, head_dim)
            kv_final = kv_r2.view(ratio * coff, head_dim)
            score_final = score_r2.view(ratio * coff, head_dim)
        else:
            kv_final = kv_r.view(ratio, head_dim)
            score_final = score_r.view(ratio, head_dim)

        weights = score_final.softmax(dim=0)
        compressed = (kv_final * weights).sum(dim=0)
        normed = rmsnorm_ref(compressed, norm_weight, norm_eps)

        freq_pos = (seq_len - 1) // ratio * ratio
        freq = freqs_cis[freq_pos]
        normed[-rope_head_dim:] = apply_rotary_emb_ref(normed[-rope_head_dim:], freq)

        if rotate:
            normed = hadamard_transform_ref(normed, head_dim ** -0.5)

        output[b] = normed

    return output


def compress_extend_ref(
    pool_kv, pool_score, kv, score, prefix_lens, extend_lens,
    req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
    rope_head_dim, overlap, rotate, norm_eps
):
    """Pure Python reference matching compress_extend_old."""
    total_tokens = kv.size(0)
    coff = 1 + (1 if overlap else 0)
    coff_hd = coff * head_dim
    bs = prefix_lens.size(0)

    output = torch.full((total_tokens, head_dim), 10000.0, dtype=torch.float32)

    pt = 0
    for i in range(bs):
        prefix_len = prefix_lens[i].item()
        extend_len = extend_lens[i].item()
        req_idx = req_pool_indices[i].item()

        if prefix_len == 0:
            pool_kv[req_idx].zero_()
            pool_score[req_idx].fill_(float("-inf"))

        pre_state_len = compute_state_len(prefix_len, ratio)
        valid_kv_len = pre_state_len + extend_len

        buf_kv = torch.empty(valid_kv_len, coff_hd, dtype=torch.float32)
        buf_score = torch.empty(valid_kv_len, coff_hd, dtype=torch.float32)
        if pre_state_len > 0:
            buf_kv[:pre_state_len] = pool_kv[req_idx, :pre_state_len]
            buf_score[:pre_state_len] = pool_score[req_idx, :pre_state_len]
        buf_kv[pre_state_len:] = kv[pt:pt + extend_len]
        buf_score[pre_state_len:] = score[pt:pt + extend_len]

        post_state_len = compute_state_len(valid_kv_len, ratio)
        if post_state_len > 0:
            start = valid_kv_len - post_state_len
            pool_kv[req_idx, :post_state_len] = buf_kv[start:valid_kv_len]
            pool_score[req_idx, :post_state_len] = buf_score[start:valid_kv_len]

        compress_len = (valid_kv_len // ratio) * ratio
        if compress_len == 0:
            pt += extend_len
            continue

        num_groups = compress_len // ratio
        kv_r = buf_kv[:compress_len].view(num_groups, ratio, -1)
        score_r = buf_score[:compress_len].view(num_groups, ratio, -1)
        score_r = score_r + ape.unsqueeze(0)

        if overlap:
            kv_ot = overlap_transform(kv_r, 0, ratio, head_dim)
            score_ot = overlap_transform(score_r, float("-inf"), ratio, head_dim)
            kv_ot = kv_ot[1:]
            score_ot = score_ot[1:]
            if kv_ot.size(0) == 0:
                pt += extend_len
                continue
            out_groups = kv_ot.size(0)
            compress_rows = 2 * ratio
        else:
            kv_ot = kv_r.view(num_groups, ratio, head_dim)
            score_ot = score_r.view(num_groups, ratio, head_dim)
            out_groups = num_groups
            compress_rows = ratio

        beg_idx = (prefix_len // ratio) * ratio
        end_idx = ((prefix_len + extend_len) // ratio) * ratio
        freq_indices = list(range(beg_idx, end_idx, ratio))

        for g in range(out_groups):
            weights = score_ot[g].softmax(dim=0)
            compressed = (kv_ot[g] * weights).sum(dim=0)
            normed = rmsnorm_ref(compressed, norm_weight, norm_eps)

            freq = freqs_cis[freq_indices[g]]
            normed[-rope_head_dim:] = apply_rotary_emb_ref(normed[-rope_head_dim:], freq)

            if rotate:
                normed = hadamard_transform_ref(normed, head_dim ** -0.5)

            start = prefix_len
            start = start + ratio - 1 - start % ratio
            indices_in_seq = list(range(start, prefix_len + extend_len, ratio))
            out_idx = indices_in_seq[g]
            flat_idx = out_idx - prefix_len + pt
            if flat_idx < total_tokens:
                output[flat_idx] = normed

        pt += extend_len

    return output


class TestCompressDecodeKernel(CustomTestCase):

    def _make_inputs(self, bs, ratio, head_dim, rope_head_dim, overlap, max_reqs=4, max_seq=256):
        coff = 1 + (1 if overlap else 0)
        coff_hd = coff * head_dim
        state_len = ratio * coff

        pool_kv = torch.randn(max_reqs, state_len, coff_hd, dtype=torch.float32)
        pool_score = torch.randn(max_reqs, state_len, coff_hd, dtype=torch.float32)
        kv = torch.randn(bs, coff_hd, dtype=torch.float32)
        score = torch.randn(bs, coff_hd, dtype=torch.float32)

        seq_lens = torch.randint(ratio, max_seq, (bs,), dtype=torch.int64)
        req_pool_indices = torch.arange(bs, dtype=torch.int64)

        ape = torch.randn(ratio, coff_hd, dtype=torch.float32)
        norm_weight = torch.randn(head_dim, dtype=torch.float32).abs() + 0.1

        # freqs_cis: [max_seq, rope_head_dim] interleaved cos/sin
        freqs_cis = torch.randn(max_seq, rope_head_dim, dtype=torch.float32)

        return (pool_kv, pool_score, kv, score, seq_lens, req_pool_indices,
                ape, norm_weight, freqs_cis)

    def test_decode_no_overlap(self):
        bs, ratio, head_dim, rope_head_dim = 4, 128, 256, 64
        overlap, rotate = False, False
        norm_eps = 1e-6

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap)
        pool_kv, pool_score, kv, score, seq_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        # Reference
        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        ref = compress_decode_ref(
            pool_kv_ref, pool_score_ref, kv, score, seq_lens, req_pool_indices,
            ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
            overlap, rotate, norm_eps)

        # Kernel
        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        out = torch.ops.sgl_kernel.compress_decode_cpu(
            pool_kv_kern, pool_score_kern, kv, score, seq_lens, req_pool_indices,
            ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
            overlap, rotate, norm_eps)

        torch.testing.assert_close(ref, out, atol=1e-4, rtol=1e-4)

    def test_decode_with_overlap(self):
        bs, ratio, head_dim, rope_head_dim = 4, 4, 256, 64
        overlap, rotate = True, False
        norm_eps = 1e-6

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap)
        pool_kv, pool_score, kv, score, seq_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        ref = compress_decode_ref(
            pool_kv_ref, pool_score_ref, kv, score, seq_lens, req_pool_indices,
            ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
            overlap, rotate, norm_eps)

        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        out = torch.ops.sgl_kernel.compress_decode_cpu(
            pool_kv_kern, pool_score_kern, kv, score, seq_lens, req_pool_indices,
            ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
            overlap, rotate, norm_eps)

        torch.testing.assert_close(ref, out, atol=1e-4, rtol=1e-4)

    def test_decode_with_rotate(self):
        bs, ratio, head_dim, rope_head_dim = 2, 4, 128, 64
        overlap, rotate = True, True
        norm_eps = 1e-6

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap)
        pool_kv, pool_score, kv, score, seq_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        ref = compress_decode_ref(
            pool_kv_ref, pool_score_ref, kv, score, seq_lens, req_pool_indices,
            ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
            overlap, rotate, norm_eps)

        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        out = torch.ops.sgl_kernel.compress_decode_cpu(
            pool_kv_kern, pool_score_kern, kv, score, seq_lens, req_pool_indices,
            ape, norm_weight, freqs_cis, ratio, head_dim, rope_head_dim,
            overlap, rotate, norm_eps)

        torch.testing.assert_close(ref, out, atol=1e-3, rtol=1e-3)


class TestCompressExtendKernel(CustomTestCase):

    def _make_inputs(self, bs, ratio, head_dim, rope_head_dim, overlap,
                     prefix_lens_list, extend_lens_list, max_reqs=8, max_seq=512):
        coff = 1 + (1 if overlap else 0)
        coff_hd = coff * head_dim
        state_len = ratio * coff

        total_tokens = sum(extend_lens_list)

        pool_kv = torch.randn(max_reqs, state_len, coff_hd, dtype=torch.float32)
        pool_score = torch.randn(max_reqs, state_len, coff_hd, dtype=torch.float32)
        kv = torch.randn(total_tokens, coff_hd, dtype=torch.float32)
        score = torch.randn(total_tokens, coff_hd, dtype=torch.float32)

        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int64)
        extend_lens = torch.tensor(extend_lens_list, dtype=torch.int64)
        req_pool_indices = torch.arange(bs, dtype=torch.int64)

        ape = torch.randn(ratio, coff_hd, dtype=torch.float32)
        norm_weight = torch.randn(head_dim, dtype=torch.float32).abs() + 0.1
        freqs_cis = torch.randn(max_seq, rope_head_dim, dtype=torch.float32)

        return (pool_kv, pool_score, kv, score, prefix_lens, extend_lens,
                req_pool_indices, ape, norm_weight, freqs_cis)

    def test_extend_no_overlap(self):
        bs, ratio, head_dim, rope_head_dim = 2, 128, 256, 64
        overlap, rotate = False, False
        norm_eps = 1e-6
        prefix_lens_list = [0, 128]
        extend_lens_list = [256, 128]

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap,
                                   prefix_lens_list, extend_lens_list)
        pool_kv, pool_score, kv, score, prefix_lens, extend_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        ref = compress_extend_ref(
            pool_kv_ref, pool_score_ref, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, rotate, norm_eps)

        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        out = torch.ops.sgl_kernel.compress_extend_cpu(
            pool_kv_kern, pool_score_kern, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, rotate, norm_eps)

        # Only compare non-fill positions
        mask = ref != 10000.0
        if mask.any():
            torch.testing.assert_close(ref[mask], out[mask], atol=1e-4, rtol=1e-4)

    def test_extend_with_overlap(self):
        bs, ratio, head_dim, rope_head_dim = 2, 4, 256, 64
        overlap, rotate = True, False
        norm_eps = 1e-6
        prefix_lens_list = [0, 16]
        extend_lens_list = [32, 16]

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap,
                                   prefix_lens_list, extend_lens_list)
        pool_kv, pool_score, kv, score, prefix_lens, extend_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        ref = compress_extend_ref(
            pool_kv_ref, pool_score_ref, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, rotate, norm_eps)

        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        out = torch.ops.sgl_kernel.compress_extend_cpu(
            pool_kv_kern, pool_score_kern, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, rotate, norm_eps)

        mask = ref != 10000.0
        if mask.any():
            torch.testing.assert_close(ref[mask], out[mask], atol=1e-4, rtol=1e-4)

    def test_extend_with_rotate(self):
        bs, ratio, head_dim, rope_head_dim = 1, 4, 128, 64
        overlap, rotate = True, True
        norm_eps = 1e-6
        prefix_lens_list = [0]
        extend_lens_list = [32]

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap,
                                   prefix_lens_list, extend_lens_list)
        pool_kv, pool_score, kv, score, prefix_lens, extend_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        ref = compress_extend_ref(
            pool_kv_ref, pool_score_ref, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, rotate, norm_eps)

        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        out = torch.ops.sgl_kernel.compress_extend_cpu(
            pool_kv_kern, pool_score_kern, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, rotate, norm_eps)

        mask = ref != 10000.0
        if mask.any():
            torch.testing.assert_close(ref[mask], out[mask], atol=1e-3, rtol=1e-3)

    def test_pool_state_consistency(self):
        """Test that pool state is updated identically by kernel and reference."""
        bs, ratio, head_dim, rope_head_dim = 2, 4, 128, 64
        overlap = True
        norm_eps = 1e-6

        inputs = self._make_inputs(bs, ratio, head_dim, rope_head_dim, overlap,
                                   [0, 8], [16, 8])
        pool_kv, pool_score, kv, score, prefix_lens, extend_lens, req_pool_indices, ape, norm_weight, freqs_cis = inputs

        pool_kv_ref = pool_kv.clone()
        pool_score_ref = pool_score.clone()
        compress_extend_ref(
            pool_kv_ref, pool_score_ref, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, False, norm_eps)

        pool_kv_kern = pool_kv.clone()
        pool_score_kern = pool_score.clone()
        torch.ops.sgl_kernel.compress_extend_cpu(
            pool_kv_kern, pool_score_kern, kv, score, prefix_lens, extend_lens,
            req_pool_indices, ape, norm_weight, freqs_cis, ratio, head_dim,
            rope_head_dim, overlap, False, norm_eps)

        torch.testing.assert_close(pool_kv_ref[:bs], pool_kv_kern[:bs], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pool_score_ref[:bs], pool_score_kern[:bs], atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
