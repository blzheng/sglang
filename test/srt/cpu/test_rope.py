import unittest

import torch
from utils import precision

from sglang.srt.layers.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
    apply_rotary_pos_emb_native,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestROPE(CustomTestCase):
    def test_mrope(self):
        torch.manual_seed(100)
        head_size = 128
        seq_len = 512
        num_heads = 16
        num_kv_heads = 1
        rotary_dim = 128
        max_pos = 262144
        base = 5000000
        is_neox_style = True
        dtype = torch.bfloat16
        mrope_section = [24, 20, 20]
        mrope_interleaved = True
        positions_mrope = torch.randint(0, max_pos, (3, seq_len))
        positions_text = torch.randint(0, max_pos, (seq_len,))
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        test_config = [
            # (dtype, is_neox_stype, mrope_interleaved, positions, mrope_section)
            (torch.bfloat16, False, True, positions_mrope, mrope_section),
            (torch.bfloat16, False, False, positions_mrope, mrope_section),
            (torch.bfloat16, False, False, positions_text, None),
            (torch.bfloat16, True, True, positions_mrope, mrope_section),
            (torch.bfloat16, True, False, positions_mrope, mrope_section),
            (torch.bfloat16, True, False, positions_text, None),
        ]
        for (
            dtype,
            is_neox_style,
            mrope_interleaved,
            positions,
            mrope_section,
        ) in test_config:
            rope = MRotaryEmbedding(
                head_size,
                rotary_dim,
                max_pos,
                base,
                is_neox_style,
                dtype,
                mrope_section,
                mrope_interleaved,
            )
            enable_autocast = True

            with torch.no_grad(), torch.amp.autocast("cpu", enabled=enable_autocast):
                q = torch.randn(seq_len, num_heads * head_size, dtype=dtype)
                q_clone = q.clone()
                k = torch.randn(seq_len, num_kv_heads * head_size, dtype=dtype)
                k_clone = k.clone()

                # ref kernel
                q_ref, k_ref = rope.forward_native(
                    query=q,
                    key=k,
                    positions=positions,
                )
                # fused rope kernel
                q_sgl, k_sgl = torch.ops.sgl_kernel.multimodal_rotary_embedding_cpu(
                    positions,
                    q_clone,
                    k_clone,
                    rope.head_size,
                    rope.cos_sin_cache,
                    rope.mrope_section,
                    rope.mrope_interleaved,
                    is_neox_style,
                )
                atol = rtol = precision[q_ref.dtype]
                torch.testing.assert_close(q_ref, q_sgl, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_ref, k_sgl, atol=atol, rtol=rtol)

    def test_deepseek_v2_rope(self):
        num_head = 16
        seq_len = 1024
        q_head_dim = 192
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        max_pos = 256
        k_dim = 576
        rotary_dim = 64
        is_neox_style = False
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        # Create cos_sin_cache
        freqs = torch.rand(max_pos, qk_rope_head_dim // 2)
        cos = freqs.cos() * 0.7
        sin = freqs.sin() * 0.7
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
        positions = torch.randint(0, max_pos, (seq_len,))

        rope = DeepseekScalingRotaryEmbedding(
            qk_rope_head_dim,
            rotary_dim,
            max_pos,
            16,  # not used since cos_sin_cache is provided
            is_neox_style,
            1.0,
            torch.bfloat16,
            device="cpu",
        )
        rope.register_buffer("cos_sin_cache", cos_sin_cache)

        for dtype in [torch.bfloat16]:
            enable_autocast = True

            with torch.no_grad(), torch.amp.autocast("cpu", enabled=enable_autocast):
                q = torch.randn(seq_len, num_head, q_head_dim, dtype=dtype)
                q_clone = q.clone()
                k = torch.randn(seq_len, 1, k_dim, dtype=dtype)
                k_clone = k.clone()
                _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
                _, q_pe_clone = q_clone.split(
                    [qk_nope_head_dim, qk_rope_head_dim], dim=-1
                )
                k_pe = k[:, :, k_dim - qk_rope_head_dim :]
                k_pe_clone = k_clone[:, :, k_dim - qk_rope_head_dim :]

                # ref kernel
                q_pe, k_pe = rope.forward_native(
                    query=q_pe,
                    key=k_pe,
                    positions=positions,
                )

                # fused rope kernel
                q_pe_clone, k_pe_clone = torch.ops.sgl_kernel.rotary_embedding_cpu(
                    positions,
                    q_pe_clone,
                    k_pe_clone,
                    rope.head_size,
                    cos_sin_cache,
                    False,
                )

                atol = rtol = precision[q_pe.dtype]
                torch.testing.assert_close(q_pe, q_pe_clone, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_pe, k_pe_clone, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_pe, k_pe_clone)

    def test_origin_rope(self):
        def single_test(
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            dims: int,
            is_neox_style: bool,
            dtype: torch.dtype,
            device: str,
            batch_size: int,
            seq_len: int,
            num_q_heads: int,
            num_kv_heads: int,
        ):
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
            torch.manual_seed(100)
            rope_ref = RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position_embeddings,
                base,
                is_neox_style,
                dtype,
            ).to(device)
            pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
            query = torch.randn(
                batch_size * seq_len,
                num_q_heads * head_size,
                dtype=dtype,
                device=device,
            )
            key = torch.randn(
                batch_size * seq_len,
                num_kv_heads * head_size,
                dtype=dtype,
                device=device,
            )
            if dims == 4:
                query = query.view(batch_size, seq_len, num_q_heads, head_size)
                key = key.view(batch_size, seq_len, num_kv_heads, head_size)
            query_ref, key_ref = query.clone(), key.clone()
            query_cpu, key_cpu = query.clone(), key.clone()

            query_ref_out, key_ref_out = rope_ref.forward_native(
                pos_ids, query_ref, key_ref
            )
            query_cpu_out, key_cpu_out = torch.ops.sgl_kernel.rotary_embedding_cpu(
                pos_ids,
                query_cpu,
                key_cpu,
                rope_ref.head_size,
                rope_ref.cos_sin_cache.to(query.dtype),
                rope_ref.is_neox_style,
            )
            torch.testing.assert_close(
                query_ref_out, query_cpu_out, atol=1e-2, rtol=1e-2
            )
            torch.testing.assert_close(key_ref_out, key_cpu_out, atol=1e-2, rtol=1e-2)

        test_config = [
            (64, 64, 32, 8000, True, torch.bfloat16, "cpu", 32, 32, 1, 1),
            (256, 128, 4096, 10000, True, torch.bfloat16, "cpu", 2, 512, 32, 8),
            (512, 128, 311, 10000, True, torch.bfloat16, "cpu", 3, 39, 4, 2),
            (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 32, 8),
            (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 16, 4),
            (512, 128, 311, 10000, False, torch.bfloat16, "cpu", 3, 39, 4, 2),
        ]

        for (
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            device,
            batch_size,
            seq_len,
            num_q_heads,
            num_kv_heads,
        ) in test_config:
            for dim in [2, 4]:
                single_test(
                    head_size,
                    rotary_dim,
                    max_position_embeddings,
                    base,
                    dim,
                    is_neox_style,
                    dtype,
                    device,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                )

    def test_apply_rotary_pos_emb(self):
        num_tokens = 1024
        num_heads = 8
        head_size = 72
        qkv = torch.randn(num_tokens, num_heads * head_size * 3).to(torch.bfloat16)
        query, key, _ = qkv.split(
            [num_heads * head_size, num_heads * head_size, num_heads * head_size],
            dim=-1,
        )
        query = query.view(num_tokens, num_heads, head_size)
        key = key.view(num_tokens, num_heads, head_size)
        for sincos_dtype in [torch.float32, torch.bfloat16]:
            cos = torch.rand(num_tokens, head_size).to(sincos_dtype)
            sin = torch.rand(num_tokens, head_size).to(sincos_dtype)
            q_out_ref, k_out_ref = apply_rotary_pos_emb_native(query, key, cos, sin)
            q_out_sgl, k_out_sgl = torch.ops.sgl_kernel.apply_rotary_pos_emb_cpu(
                query, key, cos, sin
            )
            torch.testing.assert_close(q_out_ref, q_out_sgl, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(k_out_ref, k_out_sgl, atol=1e-2, rtol=1e-2)

    def test_apply_multidimensional_rope(self):
        """Test apply_multidimensional_rope_cpu against the native Python reference."""

        def _rotate_half(x):
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
                _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k])
                for k in range(ndim)
            ]
            return torch.cat(y_parts, dim=-1)

        test_configs = [
            # (num_tokens, num_heads, head_dim, dtype, sincos_dtype)
            (4, 8, 64, torch.bfloat16, torch.bfloat16),
            (32, 16, 128, torch.bfloat16, torch.bfloat16),
            (128, 4, 256, torch.bfloat16, torch.bfloat16),
            (1, 1, 32, torch.bfloat16, torch.float32),
            (32, 16, 128, torch.bfloat16, torch.float32),
        ]

        for num_tokens, num_heads, head_dim, dtype, sincos_dtype in test_configs:
            with self.subTest(
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                sincos_dtype=sincos_dtype,
            ):
                torch.manual_seed(42)
                query = torch.randn(
                    num_tokens, num_heads, head_dim, dtype=dtype, device="cpu"
                )
                key = torch.randn(
                    num_tokens, num_heads, head_dim, dtype=dtype, device="cpu"
                )
                cos = torch.randn(
                    num_tokens, head_dim, dtype=sincos_dtype, device="cpu"
                )
                sin = torch.randn(
                    num_tokens, head_dim, dtype=sincos_dtype, device="cpu"
                )

                q_expected = _apply_multidimensional_rope_ref(
                    query.float(), cos.float(), sin.float()
                ).to(dtype)
                k_expected = _apply_multidimensional_rope_ref(
                    key.float(), cos.float(), sin.float()
                ).to(dtype)

                q_out, k_out = torch.ops.sgl_kernel.apply_multidimensional_rope_cpu(
                    query, key, cos, sin
                )
                atol = rtol = precision[dtype]
                torch.testing.assert_close(q_out, q_expected, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_out, k_expected, atol=atol, rtol=rtol)

    def test_noncontiguous_rope(self):
        """Test rotary_embedding_cpu with non-contiguous query/key tensors.

        This simulates the DeepSeek-V2 pattern where q_pe and k_pe are slices
        of larger tensors, making them non-contiguous with strides that differ
        from the output tensors created by at::empty_like().
        """
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        # Test configs for non-contiguous query/key slicing scenarios
        test_configs = [
            # 2D non-neox: slice q from [seq, num_heads * full_head_dim]
            {
                "head_size": 64,
                "rotary_dim": 64,
                "max_pos": 256,
                "base": 10000,
                "is_neox": False,
                "batch_size": 1,
                "seq_len": 32,
                "num_q_heads": 8,
                "num_kv_heads": 2,
                "full_q_head_dim": 192,
                "nope_dim": 128,
                "rope_dim": 64,
                "dims": 2,
            },
            # 2D neox: slice q from [seq, num_heads * full_head_dim]
            {
                "head_size": 64,
                "rotary_dim": 64,
                "max_pos": 256,
                "base": 10000,
                "is_neox": True,
                "batch_size": 1,
                "seq_len": 32,
                "num_q_heads": 8,
                "num_kv_heads": 2,
                "full_q_head_dim": 192,
                "nope_dim": 128,
                "rope_dim": 64,
                "dims": 2,
            },
            # 4D non-neox: slice from [batch, seq, heads, full_head_dim]
            {
                "head_size": 64,
                "rotary_dim": 64,
                "max_pos": 256,
                "base": 10000,
                "is_neox": False,
                "batch_size": 2,
                "seq_len": 16,
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "full_q_head_dim": 192,
                "nope_dim": 128,
                "rope_dim": 64,
                "dims": 4,
            },
            # 4D neox: slice from [batch, seq, heads, full_head_dim]
            {
                "head_size": 64,
                "rotary_dim": 64,
                "max_pos": 256,
                "base": 10000,
                "is_neox": True,
                "batch_size": 2,
                "seq_len": 16,
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "full_q_head_dim": 192,
                "nope_dim": 128,
                "rope_dim": 64,
                "dims": 4,
            },
            # head_size > rotary_dim (non-neox, 2D)
            {
                "head_size": 128,
                "rotary_dim": 64,
                "max_pos": 256,
                "base": 10000,
                "is_neox": False,
                "batch_size": 1,
                "seq_len": 32,
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "full_q_head_dim": 256,
                "nope_dim": 128,
                "rope_dim": 128,
                "dims": 2,
            },
            # head_size > rotary_dim (neox, 4D)
            {
                "head_size": 128,
                "rotary_dim": 64,
                "max_pos": 256,
                "base": 10000,
                "is_neox": True,
                "batch_size": 2,
                "seq_len": 16,
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "full_q_head_dim": 256,
                "nope_dim": 128,
                "rope_dim": 128,
                "dims": 4,
            },
        ]

        for cfg in test_configs:
            with self.subTest(**cfg):
                torch.manual_seed(42)
                head_size = cfg["head_size"]
                rotary_dim = cfg["rotary_dim"]
                max_pos = cfg["max_pos"]
                base = cfg["base"]
                is_neox = cfg["is_neox"]
                batch_size = cfg["batch_size"]
                seq_len = cfg["seq_len"]
                num_q_heads = cfg["num_q_heads"]
                num_kv_heads = cfg["num_kv_heads"]
                full_q_head_dim = cfg["full_q_head_dim"]
                nope_dim = cfg["nope_dim"]
                rope_dim = cfg["rope_dim"]
                dims = cfg["dims"]
                dtype = torch.bfloat16

                # Build cos_sin_cache
                freqs = torch.rand(max_pos, rotary_dim // 2)
                cos = freqs.cos() * 0.7
                sin = freqs.sin() * 0.7
                cos_sin_cache = torch.cat((cos, sin), dim=-1).to(dtype)

                num_tokens = batch_size * seq_len
                positions = torch.randint(0, max_pos, (num_tokens,))

                if dims == 2:
                    # Create 2D tensors and slice to get non-contiguous
                    q_full = torch.randn(
                        num_tokens,
                        num_q_heads * full_q_head_dim,
                        dtype=dtype,
                    )
                    k_full = torch.randn(
                        num_tokens,
                        num_kv_heads * full_q_head_dim,
                        dtype=dtype,
                    )
                    # Slice off the rope portion (last rope_dim * num_heads elements per head)
                    # For 2D, the slicing is per-head; we need to reshape
                    q_full_3d = q_full.view(
                        num_tokens, num_q_heads, full_q_head_dim
                    )
                    k_full_3d = k_full.view(
                        num_tokens, num_kv_heads, full_q_head_dim
                    )
                    # Slice to get the rope portion (non-contiguous)
                    q_pe_3d = q_full_3d[:, :, nope_dim : nope_dim + rope_dim]
                    k_pe_3d = k_full_3d[:, :, nope_dim : nope_dim + rope_dim]
                    # Reshape back to 2D (still non-contiguous)
                    q_pe = q_pe_3d.reshape(num_tokens, num_q_heads * rope_dim)
                    k_pe = k_pe_3d.reshape(num_tokens, num_kv_heads * rope_dim)
                    # Make contiguous copies for reference
                    q_pe_ref = q_pe.contiguous().clone()
                    k_pe_ref = k_pe.contiguous().clone()
                    q_pe_cpu = q_pe.contiguous().clone()
                    k_pe_cpu = k_pe.contiguous().clone()
                else:  # dims == 4
                    q_full = torch.randn(
                        batch_size,
                        seq_len,
                        num_q_heads,
                        full_q_head_dim,
                        dtype=dtype,
                    )
                    k_full = torch.randn(
                        batch_size,
                        seq_len,
                        num_kv_heads,
                        full_q_head_dim,
                        dtype=dtype,
                    )
                    # Slice to get non-contiguous rope portion
                    q_pe = q_full[
                        :, :, :, nope_dim : nope_dim + rope_dim
                    ]
                    k_pe = k_full[
                        :, :, :, nope_dim : nope_dim + rope_dim
                    ]
                    # Make contiguous copies for reference
                    q_pe_ref = q_pe.contiguous().clone()
                    k_pe_ref = k_pe.contiguous().clone()
                    q_pe_cpu = q_pe.contiguous().clone()
                    k_pe_cpu = k_pe.contiguous().clone()

                # Verify inputs are contiguous (our test targets contiguous copies)
                assert q_pe_ref.is_contiguous()
                assert k_pe_ref.is_contiguous()

                # Reference: apply on contiguous copies
                rope_ref = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_pos,
                    base,
                    is_neox,
                    dtype,
                ).to("cpu")
                rope_ref.register_buffer("cos_sin_cache", cos_sin_cache)

                q_ref_out, k_ref_out = rope_ref.forward_native(
                    positions, q_pe_ref, k_pe_ref
                )

                # Kernel under test: apply on contiguous copies
                q_cpu_out, k_cpu_out = (
                    torch.ops.sgl_kernel.rotary_embedding_cpu(
                        positions,
                        q_pe_cpu,
                        k_pe_cpu,
                        head_size,
                        cos_sin_cache,
                        is_neox,
                    )
                )

                atol = rtol = precision[dtype]
                torch.testing.assert_close(
                    q_ref_out, q_cpu_out, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    k_ref_out, k_cpu_out, atol=atol, rtol=rtol
                )


if __name__ == "__main__":
    unittest.main()
