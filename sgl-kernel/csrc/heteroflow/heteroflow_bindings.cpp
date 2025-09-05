#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "cpuinfer.h"
#include "gemm.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

// at::Tensor fused_experts_cpu(
//     at::Tensor& hidden_states,
//     at::Tensor& w1,
//     at::Tensor& w2,
//     at::Tensor& topk_weights,
//     at::Tensor& topk_ids,
//     bool inplace,
//     bool use_int8_w8a8,
//     bool use_fp8_w8a16,
//     const std::optional<at::Tensor>& w1_scale,
//     const std::optional<at::Tensor>& w2_scale,
//     const std::optional<std::vector<int64_t>> block_size,
//     const std::optional<at::Tensor>& a1_scale,
//     const std::optional<at::Tensor>& a2_scale,
//     bool is_vnni);

// class MOEBindings {
//   public:
//     class ForwardBindings {
//       public:
//         struct Args {
//             CPUInfer *cpuinfer;
//             at::Tensor hidden_states;
//             at::Tensor w1;
//             at::Tensor w2;
//             at::Tensor topk_weights;
//             at::Tensor topk_ids;
//             bool inplace;
//             bool use_int8_w8a8;
//             bool use_fp8_w8a16;
//             std::optional<at::Tensor> w1_scale;
//             std::optional<at::Tensor> w2_scale;
//             std::optional<std::vector<int64_t>> block_size;
//             std::optional<at::Tensor> a1_scale;
//             std::optional<at::Tensor> a2_scale;
//             bool is_vnni;
//             at::Tensor output;
//         };
//         static void inner(void *args) {
//             Args *args_ = (Args *)args;
//             args_->cpuinfer->enqueue(
//                 &fused_experts_cpu, args_->hidden_states, args_->w1, args_->w2,
//                 args_->topk_weights, args_->topk_ids, args_->inplace, args_->use_int8_w8a8,
//                 args_->use_fp8_w8a16, args_->w1_scale, args_->w2_scale, args_->block_size,
//                 args_->a1_scale, args_->a2_scale, args_->is_vnni);
//         }
//         static std::pair<intptr_t, intptr_t>
//         cpuinfer_interface(CPUInfer cpuinfer, at::Tensor hidden_states, at::Tensor w1, at::Tensor w2, at::Tensor topk_weights, at::Tensor topk_ids,
//             bool inplace, bool use_int8_w8a8, bool use_fp8_w8a16, std::optional<at::Tensor> w1_scale,
//             std::optional<at::Tensor> w2_scale, std::optional<std::vector<int64_t>> block_size,
//             std::optional<at::Tensor> a1_scale,
//             std::optional<at::Tensor> a2_scale, bool is_vnni) {
//             at::Tensor output = at::empty_like(hidden_states);
//             Args *args = new Args{nullptr,
//                                   hidden_states,
//                                   w1,
//                                   w2,
//                                   topk_weights,
//                                   topk_ids,
//                                   inplace,
//                                   use_int8_w8a8,
//                                   use_fp8_w8a16,
//                                   w1_scale,
//                                   w2_scale,
//                                   block_size,
//                                   a1_scale,
//                                   a2_scale,
//                                   is_vnni,
//                                   output};
//             return std::make_pair((intptr_t)&inner, (intptr_t)args);
//         }
//     };
// };

// PYBIND11_MODULE(heteroflow_cpp, m) {
//     py::class_<CPUInfer>(m, "CPUInfer")
//         .def(py::init<int>())
//         .def("submit", &CPUInfer::submit)
//         .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
//         .def("sync", &CPUInfer::sync)
//         .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    
// }



at::Tensor fused_experts_cpu2(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
    bool is_vnni) {
        return hidden_states;
    }

class SimpleBindings {
public:
    struct Args {
        CPUInfer* cpuinfer;
        at::Tensor hidden_states;
        at::Tensor w1;
        at::Tensor w2;
        at::Tensor topk_weights;
        at::Tensor topk_ids;
        bool inplace;
        bool use_int8_w8a8;
        bool use_fp8_w8a16;
        std::optional<at::Tensor> w1_scale;
        std::optional<at::Tensor> w2_scale;
        std::optional<std::vector<int64_t>> block_size;
        std::optional<at::Tensor> a1_scale;
        std::optional<at::Tensor> a2_scale;
        bool is_vnni;
        at::Tensor output;
    };

    static void inner(void* vargs) {
        Args* args = (Args*)vargs;
        at::Tensor out = fused_experts_cpu2(
            args->hidden_states,
            args->w1,
            args->w2,
            args->topk_weights,
            args->topk_ids,
            args->inplace,
            args->use_int8_w8a8,
            args->use_fp8_w8a16,
            args->w1_scale,
            args->w2_scale,
            args->block_size,
            args->a1_scale,
            args->a2_scale,
            args->is_vnni
        );
        args->output.copy_(out);
    }

    static std::pair<intptr_t,intptr_t> cpuinfer_interface(at::Tensor& hidden_states, at::Tensor& w1, at::Tensor& w2, at::Tensor& topk_weights, at::Tensor& topk_ids,
        bool inplace, bool use_int8_w8a8, bool use_fp8_w8a16, std::optional<at::Tensor> w1_scale,
        std::optional<at::Tensor> w2_scale, std::optional<std::vector<int64_t>> block_size,
        std::optional<at::Tensor> a1_scale,
        std::optional<at::Tensor> a2_scale, bool is_vnni) {
        at::Tensor output = at::empty_like(hidden_states);
        Args* args = new Args{nullptr, hidden_states, w1, w2, topk_weights, topk_ids, inplace, use_int8_w8a8, use_fp8_w8a16, w1_scale, w2_scale, block_size, a1_scale, a2_scale, is_vnni, output};
        return { (intptr_t)&inner, (intptr_t)args };
    }
};


PYBIND11_MODULE(heteroflow_cpp, m) {
    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    m.def("fused_moe_cpu", &SimpleBindings::cpuinfer_interface);
}