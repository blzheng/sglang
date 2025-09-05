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
from heteroflow_cpp import CPUInfer, submit_simple_add_one

x = torch.tensor([[1.0, 2.0],[3.0,4.0]]).to('cpu')

cpuinfer = CPUInfer(20)
cpuinfer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, submit_simple_add_one(x))

cpuinfer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)