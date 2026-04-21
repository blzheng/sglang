#include "common.h"

namespace {

template <typename scalar_t>
void clamp_position_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ seq_lens,
    int64_t n) {
  at::parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      scalar_t val = seq_lens[i] - static_cast<scalar_t>(1);
      output[i] = val < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) : val;
    }
  });
}

}  // anonymous namespace

// seq_lens : {n}
// output   : {n}
at::Tensor clamp_position_cpu(const at::Tensor& seq_lens) {
  RECORD_FUNCTION("sgl-kernel::clamp_position_cpu", std::vector<c10::IValue>({seq_lens}));
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be a 1D tensor");
  int64_t n = seq_lens.numel();
  at::Tensor output = at::empty_like(seq_lens);

  AT_DISPATCH_INTEGRAL_TYPES(seq_lens.scalar_type(), "clamp_position_cpu", [&] {
    clamp_position_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        seq_lens.data_ptr<scalar_t>(),
        n);
  });
  return output;
}
