#include "common.h"
#include "vec.h"

namespace {

// ============================================================================
// clamp_position: clamp(seq_lens - 1, min=0) for integer types
// ============================================================================

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

// ============================================================================
// clamp: clamp(input, min_val, max_val) for reduced floating-point types
// ============================================================================

template <typename scalar_t>
void clamp_kernel_reduced_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    float min_val,
    float max_val,
    int64_t numel) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  const fVec min_fvec(min_val);
  const fVec max_fvec(max_val);

  at::parallel_for(0, numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    int64_t d = begin;
#pragma GCC unroll 4
    for (; d <= end - kVecSize; d += kVecSize) {
      bVec x_bvec = bVec::loadu(input + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

      x_fvec0 = at::vec::minimum(max_fvec, at::vec::maximum(min_fvec, x_fvec0));
      x_fvec1 = at::vec::minimum(max_fvec, at::vec::maximum(min_fvec, x_fvec1));

      x_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
      x_bvec.store(output + d);
    }
#pragma GCC unroll 4
    for (; d < end; ++d) {
      float val = static_cast<float>(input[d]);
      val = std::max(min_val, std::min(max_val, val));
      output[d] = static_cast<scalar_t>(val);
    }
  });
}

// ============================================================================
// clamp: clamp(input, min_val, max_val) for float32
// ============================================================================

void clamp_kernel_float_impl(
    float* __restrict__ output,
    const float* __restrict__ input,
    float min_val,
    float max_val,
    int64_t numel) {
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = fVec::size();
  const fVec min_fvec(min_val);
  const fVec max_fvec(max_val);

  at::parallel_for(0, numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    int64_t d = begin;
#pragma GCC unroll 4
    for (; d <= end - kVecSize; d += kVecSize) {
      fVec x = fVec::loadu(input + d);
      x = at::vec::minimum(max_fvec, at::vec::maximum(min_fvec, x));
      x.store(output + d);
    }
#pragma GCC unroll 4
    for (; d < end; ++d) {
      output[d] = std::max(min_val, std::min(max_val, input[d]));
    }
  });
}

}  // anonymous namespace

// ============================================================================
// clamp_position_cpu: positions = clamp(seq_lens - 1, min=0)
// ============================================================================

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

// ============================================================================
// clamp_cpu: output = clamp(input, min_val, max_val) for floating types
// ============================================================================

at::Tensor clamp_cpu(const at::Tensor& input, double min_val, double max_val) {
  RECORD_FUNCTION("sgl-kernel::clamp_cpu", std::vector<c10::IValue>({input}));
  at::Tensor output = at::empty_like(input);
  int64_t numel = input.numel();

  if (input.scalar_type() == at::kFloat) {
    clamp_kernel_float_impl(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        numel);
  } else {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "clamp_cpu", [&] {
      clamp_kernel_reduced_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          static_cast<float>(min_val),
          static_cast<float>(max_val),
          numel);
    });
  }
  return output;
}
