from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

heteroflow_cpp = load(
    name="heteroflow_cpp",
    sources=[
        f"{_abs_path}/heteroflow_bindings.cpp",
        f"{_abs_path}/task_queue.cpp",
        f"{_abs_path}/backend.cpp",
        f"{_abs_path}/shared_mem_buffer.cpp",
        f"{_abs_path}/../cpu/moe.cpp",
        f"{_abs_path}/../cpu/moe_int8.cpp",
        f"{_abs_path}/../cpu/moe_fp8.cpp",
        f"{_abs_path}/../cpu/gemm_fp8.cpp",
        f"{_abs_path}/../cpu/gemm_int8.cpp",
        f"{_abs_path}/../cpu/gemm.cpp",
    ],
    extra_cflags=["-O3", "-std=c++20", "-I/usr/local/cuda/include", f"-I{_abs_path}/../cpu"],
    extra_ldflags=["-L/usr/local/cuda/lib64", "-lcudart"],
)
import torch
from heteroflow_cpp import CPUInfer, fused_moe_cpu
import math


def scaled_weight(weight, scales):
    E, N, K = weight.shape
    pad_N = (BLOCK_N - (N % BLOCK_N)) % BLOCK_N
    pad_K = (BLOCK_K - (K % BLOCK_K)) % BLOCK_K

    if pad_N > 0 or pad_K > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))

    weight_block = (
        weight.view(E, math.ceil(N / BLOCK_N), BLOCK_N, math.ceil(K / BLOCK_K), BLOCK_K)
        .permute(0, 1, 3, 2, 4)
        .float()
        .contiguous()
    )

    weight_scaled = (
        (
            weight_block
            * scales.view(E, math.ceil(N / BLOCK_N), math.ceil(K / BLOCK_K), 1, 1)
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if pad_N > 0 or pad_K > 0:
        weight_scaled = weight_scaled.view(E, N + pad_N, K + pad_K)
        weight_scaled = weight_scaled[..., :N, :K].contiguous()
    else:
        weight_scaled = weight_scaled.view(E, N, K)
    return weight_scaled
from sgl_kernel_cpu import common_ops

M = 2
N = 512
K = 256
E = 8
topk = 4
dtype = torch.bfloat16
BLOCK_N, BLOCK_K = 64, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400

a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)
w1_fp32 = torch.randn(E, 2 * N, K)
w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
w2_fp32 = torch.randn(E, K, N)
w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
w1s = (
    torch.randn(E, math.ceil(2 * N / BLOCK_N), math.ceil(K / BLOCK_K))
    * factor_for_scale
)
w2s = (
    torch.randn(E, math.ceil(K / BLOCK_N), math.ceil(N / BLOCK_K))
    * factor_for_scale
)

w1_scaled = scaled_weight(w1, w1s)
w2_scaled = scaled_weight(w2, w2s)

score = torch.randn((M, E), dtype=dtype)
score = torch.softmax(score, dim=-1, dtype=torch.float32)
topk_weight, topk_ids = torch.topk(score, topk)

w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)
w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)


cpuinfer = CPUInfer(20)
cpuinfer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, fused_moe_cpu(a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            False,
            True,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            True,))

cpuinfer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)