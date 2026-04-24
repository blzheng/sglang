"""Unit tests for Gemma4 audio conformer CPU kernels.

Tests the fused kernels against reference PyTorch implementations:
  1. QKV preprocessing (cast + scale + block/context extraction)
  2. Relative position attention logits (sinusoidal embedding + projection + dot-product with shift)
  3. Softcap attention (softcap + mask + softmax + weighted sum)
"""

import math
import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from utils import parametrize

from sglang.test.test_utils import CustomTestCase

gemma4_audio_qkv_preprocess_cpu = torch.ops.sgl_kernel.gemma4_audio_qkv_preprocess_cpu
gemma4_audio_rel_pos_logits_cpu = torch.ops.sgl_kernel.gemma4_audio_rel_pos_logits_cpu
gemma4_audio_softcap_attn_cpu = torch.ops.sgl_kernel.gemma4_audio_softcap_attn_cpu

torch.manual_seed(42)


# ============================================================================
# Reference implementations — QKV Preprocessing
# ============================================================================


def _pad_dim1(x, dim10_val, dim11_val):
    """Pad dimension 1 of tensor x."""
    padding_tuple = [0] * x.ndim * 2
    dim_idx_from_end = x.ndim - 2
    start_idx_for_dim = 2 * dim_idx_from_end
    padding_tuple[start_idx_for_dim] = dim10_val
    padding_tuple[start_idx_for_dim + 1] = dim11_val
    return F.pad(x, tuple(padding_tuple))


def _convert_to_block(x, chunk_size):
    """Convert tensor to blocked layout."""
    shape = x.shape
    b, t = shape[:2]
    num_blocks = (t + chunk_size - 1) // chunk_size
    padding_len = num_blocks * chunk_size - t
    if padding_len > 0:
        x = _pad_dim1(x, 0, padding_len)
    permute_dims = (b, num_blocks, chunk_size) + shape[2:]
    return x.reshape(permute_dims).contiguous()


def _extract_block_context(x, chunk_size, max_past, max_future, context_size):
    """Extract sliding context windows."""
    pad_left = max_past
    pad_right = max_future + chunk_size - 1
    x = _pad_dim1(x, pad_left, pad_right)
    x_unfolded = x.unfold(dimension=1, size=context_size, step=chunk_size)
    if x.ndim > 2 and x_unfolded.ndim > 3:
        x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)
    return x_unfolded.contiguous()


def reference_qkv_preprocess(
    q,
    k,
    v,
    mask,
    per_dim_scale,
    q_scale,
    k_scale,
    num_heads,
    head_dim,
    chunk_size,
    max_past,
    max_future,
):
    """Reference implementation matching the original Python code."""
    B, T, _ = q.shape
    context_size = chunk_size + max_past + max_future

    qkv_shape = (B, T, num_heads, head_dim)
    query_states = q.float().reshape(qkv_shape).contiguous()
    key_states = k.float().reshape(qkv_shape).contiguous()
    value_states = v.float().reshape(qkv_shape).contiguous()

    per_dim_scale_sp = F.softplus(per_dim_scale)
    broadcast_shape = (1, 1, 1, head_dim)
    query_states = query_states * q_scale * per_dim_scale_sp.view(broadcast_shape)
    key_states = key_states * k_scale

    query_blocks = _convert_to_block(query_states, chunk_size)
    key_blocks = _extract_block_context(
        key_states, chunk_size, max_past, max_future, context_size
    )
    value_blocks = _extract_block_context(
        value_states, chunk_size, max_past, max_future, context_size
    )

    original_valid_mask = ~mask
    validity_mask_blocks = _extract_block_context(
        original_valid_mask, chunk_size, max_past, max_future, context_size
    )

    U = query_blocks.shape[1]
    if (
        validity_mask_blocks.ndim == 4
        and validity_mask_blocks.shape[0] == B
        and validity_mask_blocks.shape[1] == U
        and validity_mask_blocks.shape[2] * validity_mask_blocks.shape[3]
        == context_size
    ):
        validity_mask_blocks = validity_mask_blocks.reshape(B, U, context_size)

    return query_blocks, key_blocks, value_blocks, validity_mask_blocks


# ============================================================================
# Reference implementations — Relative Position Logits
# ============================================================================


def _relative_shift(term_bd, W, C, S):
    """Reference relative shift (skew trick)."""
    B_H = term_bd.shape[0]
    U = term_bd.shape[1]
    pad_amount = (C + 1) - S
    padded = F.pad(term_bd, (0, pad_amount))
    flat = padded.reshape(B_H, U, W * (C + 1))
    sliced = flat[:, :, : W * C]
    shifted = sliced.reshape(B_H, U, W, C)
    return shifted


def reference_generate_sin_emb(
    inv_timescales, pos_proj_weight, max_backward, max_forward, num_heads, head_dim
):
    """Reference sinusoidal embedding + projection."""
    S = max_backward + max_forward + 1
    channels = inv_timescales.shape[0] * 2

    pos_indices = torch.arange(max_backward, -max_forward - 1, -1, dtype=torch.float32)
    scaled = pos_indices.unsqueeze(1) * inv_timescales.unsqueeze(0)
    timing_signal = torch.cat([scaled.sin(), scaled.cos()], dim=-1).unsqueeze(0)

    projected = torch.matmul(
        timing_signal.to(pos_proj_weight.dtype), pos_proj_weight.t()
    )
    sin_emb = projected.squeeze(0).reshape(S, num_heads, head_dim).float()
    return sin_emb


def reference_rel_pos_logits(
    queries,
    keys,
    inv_timescales,
    pos_proj_weight,
    max_backward,
    max_forward,
    num_heads,
    head_dim,
):
    """Full reference implementation matching Gemma4AudioRelativePositionEmbedding.forward()."""
    B, U, W, H, D = queries.shape
    C = keys.shape[2]

    sin_emb = reference_generate_sin_emb(
        inv_timescales, pos_proj_weight, max_backward, max_forward, num_heads, head_dim
    )
    S = sin_emb.shape[0]

    queries_p = queries.permute(0, 3, 1, 2, 4)
    keys_p_t = keys.permute(0, 3, 1, 4, 2)
    term_ac = torch.matmul(queries_p, keys_p_t)

    s_permuted = sin_emb.permute(1, 2, 0)
    q_reshaped = queries_p.reshape(B, H, U * W, D)
    term_bd_unshifted = torch.matmul(q_reshaped, s_permuted)
    term_bd_unshifted = term_bd_unshifted.reshape(B, H, U, W, S)

    term_bd_flat = term_bd_unshifted.reshape(B * H, U, W, S)
    term_bd_shifted = _relative_shift(term_bd_flat, W, C, S)
    term_bd_shifted = term_bd_shifted.reshape(B, H, U, W, C)

    return term_ac + term_bd_shifted


# ============================================================================
# Reference implementations — Softcap Attention
# ============================================================================


def reference_softcap_attn(
    logits,
    validity_mask,
    causal_mask,
    value_blocks,
    softcap,
    invalid_logits_value,
    q_time,
):
    """Reference implementation matching the original Gemma4AudioAttention code."""
    B, H, U, W, C = logits.shape
    D = value_blocks.shape[-1]

    logits = logits / softcap
    logits = torch.tanh(logits)
    logits = logits * softcap

    cond_validity = validity_mask.unsqueeze(1).unsqueeze(-2)
    cond_causal = causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mask = torch.logical_and(cond_validity, cond_causal)

    logits = torch.where(mask, logits, invalid_logits_value)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)

    b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
    h_dim = value_blocks.shape[-1]
    prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
    v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
    result_bmm = torch.bmm(prob_bun, v_bun)
    context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(
        0, 1, 3, 2, 4
    )
    context_vectors = context_vectors.reshape(B, U * W, H, D)
    context_vectors = context_vectors[:, :q_time]
    return context_vectors


# ============================================================================
# Test: QKV Preprocessing
# ============================================================================


class TestGemma4AudioQKVPreprocess(CustomTestCase):

    @parametrize(
        batch_size=[1, 2],
        num_heads=[4, 8],
        head_dim=[64, 128],
        chunk_size=[8, 16],
        max_past=[8, 16],
        max_future=[0, 8],
        seq_len=[32, 64],
        dtype=[torch.bfloat16, torch.float32],
    )
    def test_correctness(
        self,
        batch_size,
        num_heads,
        head_dim,
        chunk_size,
        max_past,
        max_future,
        seq_len,
        dtype,
    ):
        B, H, D, T = batch_size, num_heads, head_dim, seq_len
        HD = H * D

        q = torch.randn(B, T, HD, dtype=dtype)
        k = torch.randn(B, T, HD, dtype=dtype)
        v = torch.randn(B, T, HD, dtype=dtype)
        mask = torch.randint(0, 2, (B, T), dtype=torch.bool)
        per_dim_scale = torch.randn(D, dtype=torch.float32)

        q_scale = (D**-0.5) / math.log(2)
        k_scale = math.log(1 + math.e) / math.log(2)

        ref_qb, ref_kb, ref_vb, ref_vm = reference_qkv_preprocess(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            q_scale,
            k_scale,
            H,
            D,
            chunk_size,
            max_past,
            max_future,
        )
        out_qb, out_kb, out_vb, out_vm = gemma4_audio_qkv_preprocess_cpu(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            q_scale,
            k_scale,
            H,
            D,
            chunk_size,
            max_past,
            max_future,
        )

        rtol = 1e-4 if dtype == torch.float32 else 1e-3
        atol = 1e-4 if dtype == torch.float32 else 1e-3

        torch.testing.assert_close(out_qb, ref_qb, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_kb, ref_kb, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_vb, ref_vb, rtol=rtol, atol=atol)
        self.assertTrue(torch.equal(out_vm, ref_vm))

    @parametrize(
        batch_size=[1],
        num_heads=[4],
        head_dim=[128],
        chunk_size=[8],
        max_past=[8],
        max_future=[0],
    )
    def test_output_shapes(
        self,
        batch_size,
        num_heads,
        head_dim,
        chunk_size,
        max_past,
        max_future,
    ):
        B, H, D = batch_size, num_heads, head_dim
        T = 33  # non-aligned with chunk_size
        HD = H * D
        W = chunk_size
        U = (T + W - 1) // W
        C = W + max_past + max_future

        q = torch.randn(B, T, HD, dtype=torch.bfloat16)
        k = torch.randn(B, T, HD, dtype=torch.bfloat16)
        v = torch.randn(B, T, HD, dtype=torch.bfloat16)
        mask = torch.zeros(B, T, dtype=torch.bool)
        per_dim_scale = torch.zeros(D, dtype=torch.float32)

        out_qb, out_kb, out_vb, out_vm = gemma4_audio_qkv_preprocess_cpu(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            1.0,
            1.0,
            H,
            D,
            chunk_size,
            max_past,
            max_future,
        )

        self.assertEqual(out_qb.shape, (B, U, W, H, D))
        self.assertEqual(out_kb.shape, (B, U, C, H, D))
        self.assertEqual(out_vb.shape, (B, U, C, H, D))
        self.assertEqual(out_vm.shape, (B, U, C))
        self.assertEqual(out_qb.dtype, torch.float32)
        self.assertEqual(out_vm.dtype, torch.bool)

    def test_padding_positions_masked(self):
        """Padding positions should produce zeros for k/v and False for validity."""
        B, H, D, T, W = 1, 4, 64, 16, 8
        max_past, max_future = 4, 0
        C = W + max_past + max_future
        U = (T + W - 1) // W

        q = torch.randn(B, T, H * D, dtype=torch.bfloat16)
        k = torch.randn(B, T, H * D, dtype=torch.bfloat16)
        v = torch.randn(B, T, H * D, dtype=torch.bfloat16)
        mask = torch.ones(B, T, dtype=torch.bool)  # all padding
        per_dim_scale = torch.zeros(D, dtype=torch.float32)

        _, _, _, out_vm = gemma4_audio_qkv_preprocess_cpu(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            1.0,
            1.0,
            H,
            D,
            W,
            max_past,
            max_future,
        )

        ref_vm = reference_qkv_preprocess(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            1.0,
            1.0,
            H,
            D,
            W,
            max_past,
            max_future,
        )[3]
        self.assertTrue(torch.equal(out_vm, ref_vm))

    def test_zero_per_dim_scale(self):
        """With zero per_dim_scale, softplus(0) = ln(2), should still work."""
        B, H, D, T, W = 1, 4, 64, 16, 8
        max_past, max_future = 4, 0

        q = torch.randn(B, T, H * D, dtype=torch.bfloat16)
        k = torch.randn(B, T, H * D, dtype=torch.bfloat16)
        v = torch.randn(B, T, H * D, dtype=torch.bfloat16)
        mask = torch.zeros(B, T, dtype=torch.bool)
        per_dim_scale = torch.zeros(D, dtype=torch.float32)

        q_scale = (D**-0.5) / math.log(2)
        k_scale = math.log(1 + math.e) / math.log(2)

        ref_qb, ref_kb, ref_vb, ref_vm = reference_qkv_preprocess(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            q_scale,
            k_scale,
            H,
            D,
            W,
            max_past,
            max_future,
        )
        out_qb, out_kb, out_vb, out_vm = gemma4_audio_qkv_preprocess_cpu(
            q,
            k,
            v,
            mask,
            per_dim_scale,
            q_scale,
            k_scale,
            H,
            D,
            W,
            max_past,
            max_future,
        )

        torch.testing.assert_close(out_qb, ref_qb, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_kb, ref_kb, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_vb, ref_vb, rtol=1e-3, atol=1e-3)
        self.assertTrue(torch.equal(out_vm, ref_vm))


# ============================================================================
# Test: Relative Position Attention Logits
# ============================================================================


class TestGemma4AudioRelPosLogits(CustomTestCase):

    @parametrize(
        batch_size=[1, 2],
        num_heads=[4, 8],
        head_dim=[64, 128],
        chunk_size=[8, 16],
        channels=[64, 128],
        max_past=[8, 16],
        max_future=[0, 4],
    )
    def test_correctness(
        self,
        batch_size,
        num_heads,
        head_dim,
        chunk_size,
        channels,
        max_past,
        max_future,
    ):
        B, H, D, W = batch_size, num_heads, head_dim, chunk_size
        U = 4
        C = W + max_past + max_future
        HD = H * D

        queries = torch.randn(B, U, W, H, D, dtype=torch.float32)
        keys = torch.randn(B, U, C, H, D, dtype=torch.float32)
        inv_timescales = torch.randn(channels // 2, dtype=torch.float32)
        pos_proj_weight = torch.randn(HD, channels, dtype=torch.float32) * 0.01

        ref = reference_rel_pos_logits(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )
        out = gemma4_audio_rel_pos_logits_cpu(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )

        self.assertEqual(out.shape, (B, H, U, W, C))
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    @parametrize(
        batch_size=[1],
        num_heads=[4],
        head_dim=[64],
        chunk_size=[8],
        channels=[64],
    )
    def test_bf16_weight(self, batch_size, num_heads, head_dim, chunk_size, channels):
        """Kernel should handle bf16 pos_proj_weight via internal conversion."""
        B, H, D, W = batch_size, num_heads, head_dim, chunk_size
        U = 3
        max_past, max_future = 8, 0
        C = W + max_past + max_future
        HD = H * D

        queries = torch.randn(B, U, W, H, D, dtype=torch.float32)
        keys = torch.randn(B, U, C, H, D, dtype=torch.float32)
        inv_timescales = torch.randn(channels // 2, dtype=torch.float32)
        pos_proj_weight = (
            torch.randn(HD, channels, dtype=torch.float32) * 0.01
        ).bfloat16()

        ref = reference_rel_pos_logits(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )
        out = gemma4_audio_rel_pos_logits_cpu(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )

        self.assertEqual(out.shape, (B, H, U, W, C))
        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    @parametrize(
        batch_size=[1],
        num_heads=[4],
        head_dim=[64],
        chunk_size=[8],
        channels=[64],
    )
    def test_zero_weight(self, batch_size, num_heads, head_dim, chunk_size, channels):
        """When pos_proj_weight is zero, output should equal Q @ K^T."""
        B, H, D, W = batch_size, num_heads, head_dim, chunk_size
        U = 3
        max_past, max_future = 8, 0
        C = W + max_past + max_future
        HD = H * D

        queries = torch.randn(B, U, W, H, D, dtype=torch.float32)
        keys = torch.randn(B, U, C, H, D, dtype=torch.float32)
        inv_timescales = torch.randn(channels // 2, dtype=torch.float32)
        pos_proj_weight = torch.zeros(HD, channels, dtype=torch.float32)

        ref = reference_rel_pos_logits(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )
        out = gemma4_audio_rel_pos_logits_cpu(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_output_shape(self):
        """Verify output shape is [B, H, U, W, C]."""
        B, U, W, H, D = 2, 5, 8, 4, 64
        channels = 128
        max_past, max_future = 8, 8
        C = W + max_past + max_future
        HD = H * D

        queries = torch.randn(B, U, W, H, D, dtype=torch.float32)
        keys = torch.randn(B, U, C, H, D, dtype=torch.float32)
        inv_timescales = torch.randn(channels // 2, dtype=torch.float32)
        pos_proj_weight = torch.randn(HD, channels, dtype=torch.float32)

        out = gemma4_audio_rel_pos_logits_cpu(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )
        self.assertEqual(out.shape, (B, H, U, W, C))
        self.assertEqual(out.dtype, torch.float32)

    def test_zero_keys(self):
        """When keys are zero, output should equal the shifted position logits."""
        B, U, W, H, D = 1, 3, 8, 4, 64
        channels = 128
        max_past, max_future = 8, 0
        C = W + max_past + max_future
        HD = H * D

        queries = torch.randn(B, U, W, H, D, dtype=torch.float32)
        keys = torch.zeros(B, U, C, H, D, dtype=torch.float32)
        inv_timescales = torch.randn(channels // 2, dtype=torch.float32)
        pos_proj_weight = torch.randn(HD, channels, dtype=torch.float32) * 0.01

        ref = reference_rel_pos_logits(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )
        out = gemma4_audio_rel_pos_logits_cpu(
            queries, keys, inv_timescales, pos_proj_weight, max_past, max_future, H, D
        )

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# ============================================================================
# Test: Softcap Attention
# ============================================================================


class TestGemma4AudioSoftcapAttn(CustomTestCase):

    @parametrize(
        batch_size=[1, 2],
        num_heads=[4, 8],
        head_dim=[64, 128],
        chunk_size=[8, 16],
        context_size=[24, 32],
        seq_len=[32, 64],
    )
    def test_correctness(
        self,
        batch_size,
        num_heads,
        head_dim,
        chunk_size,
        context_size,
        seq_len,
    ):
        B, H, D, W, C = batch_size, num_heads, head_dim, chunk_size, context_size
        U = (seq_len + W - 1) // W
        q_time = seq_len
        softcap = 50.0
        invalid_val = -2.3819763e38

        logits = torch.randn(B, H, U, W, C, dtype=torch.float32)
        validity_mask = torch.randint(0, 2, (B, U, C), dtype=torch.bool)
        causal_mask = torch.randint(0, 2, (W, C), dtype=torch.bool)
        value_blocks = torch.randn(B, U, C, H, D, dtype=torch.float32)

        ref = reference_softcap_attn(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )
        out = gemma4_audio_softcap_attn_cpu(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    @parametrize(
        batch_size=[1],
        num_heads=[4],
        head_dim=[128],
        chunk_size=[8],
        context_size=[24],
    )
    def test_all_masked(
        self, batch_size, num_heads, head_dim, chunk_size, context_size
    ):
        """When all positions are masked, output should match reference (uniform probs)."""
        B, H, D, W, C = batch_size, num_heads, head_dim, chunk_size, context_size
        U = 4
        q_time = U * W
        softcap = 50.0
        invalid_val = -2.3819763e38

        logits = torch.randn(B, H, U, W, C, dtype=torch.float32)
        validity_mask = torch.zeros(B, U, C, dtype=torch.bool)
        causal_mask = torch.ones(W, C, dtype=torch.bool)
        value_blocks = torch.randn(B, U, C, H, D, dtype=torch.float32)

        ref = reference_softcap_attn(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )
        out = gemma4_audio_softcap_attn_cpu(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    @parametrize(
        batch_size=[1],
        num_heads=[4],
        head_dim=[128],
        chunk_size=[8],
        context_size=[24],
    )
    def test_none_masked(
        self, batch_size, num_heads, head_dim, chunk_size, context_size
    ):
        """When no positions are masked, output should match reference."""
        B, H, D, W, C = batch_size, num_heads, head_dim, chunk_size, context_size
        U = 4
        q_time = U * W
        softcap = 50.0
        invalid_val = -2.3819763e38

        logits = torch.randn(B, H, U, W, C, dtype=torch.float32)
        validity_mask = torch.ones(B, U, C, dtype=torch.bool)
        causal_mask = torch.ones(W, C, dtype=torch.bool)
        value_blocks = torch.randn(B, U, C, H, D, dtype=torch.float32)

        ref = reference_softcap_attn(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )
        out = gemma4_audio_softcap_attn_cpu(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_q_time_truncation(self):
        """q_time < U*W should correctly truncate the output."""
        B, H, D, W, C = 1, 4, 64, 8, 24
        U = 4
        q_time = 25
        softcap = 50.0
        invalid_val = -2.3819763e38

        logits = torch.randn(B, H, U, W, C, dtype=torch.float32)
        validity_mask = torch.ones(B, U, C, dtype=torch.bool)
        causal_mask = torch.ones(W, C, dtype=torch.bool)
        value_blocks = torch.randn(B, U, C, H, D, dtype=torch.float32)

        ref = reference_softcap_attn(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )
        out = gemma4_audio_softcap_attn_cpu(
            logits,
            validity_mask,
            causal_mask,
            value_blocks,
            softcap,
            invalid_val,
            q_time,
        )

        self.assertEqual(out.shape, (B, q_time, H, D))
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_different_softcap_values(self):
        """Verify correctness across different softcap values."""
        B, H, D, W, C = 1, 4, 64, 8, 24
        U = 4
        q_time = U * W
        invalid_val = -2.3819763e38

        logits = torch.randn(B, H, U, W, C, dtype=torch.float32)
        validity_mask = torch.ones(B, U, C, dtype=torch.bool)
        causal_mask = torch.ones(W, C, dtype=torch.bool)
        value_blocks = torch.randn(B, U, C, H, D, dtype=torch.float32)

        for softcap in [10.0, 30.0, 50.0, 100.0]:
            with self.subTest(softcap=softcap):
                ref = reference_softcap_attn(
                    logits,
                    validity_mask,
                    causal_mask,
                    value_blocks,
                    softcap,
                    invalid_val,
                    q_time,
                )
                out = gemma4_audio_softcap_attn_cpu(
                    logits,
                    validity_mask,
                    causal_mask,
                    value_blocks,
                    softcap,
                    invalid_val,
                    q_time,
                )
                torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
