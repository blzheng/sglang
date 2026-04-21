#include "common.h"
#include "vec.h"

namespace {

// ============================================================================
// clamp: clamp_(data, min_val, max_val) for reduced floating-point types
// ============================================================================

template <typename scalar_t>
void clamp_kernel_reduced_impl(
    scalar_t* __restrict__ data,
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
      bVec x_bvec = bVec::loadu(data + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

      x_fvec0 = at::vec::minimum(max_fvec, at::vec::maximum(min_fvec, x_fvec0));
      x_fvec1 = at::vec::minimum(max_fvec, at::vec::maximum(min_fvec, x_fvec1));

      x_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
      x_bvec.store(data + d);
    }
#pragma GCC unroll 4
    for (; d < end; ++d) {
      float val = static_cast<float>(data[d]);
      val = std::max(min_val, std::min(max_val, val));
      data[d] = static_cast<scalar_t>(val);
    }
  });
}

// ============================================================================
// clamp: clamp_(data, min_val, max_val) for float32
// ============================================================================

void clamp_kernel_float_impl(
    float* __restrict__ data,
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
      fVec x = fVec::loadu(data + d);
      x = at::vec::minimum(max_fvec, at::vec::maximum(min_fvec, x));
      x.store(data + d);
    }
#pragma GCC unroll 4
    for (; d < end; ++d) {
      data[d] = std::max(min_val, std::min(max_val, data[d]));
    }
  });
}

}  // anonymous namespace

// ============================================================================
// clamp_cpu: input.clamp_(min_val, max_val) in-place for floating types
//   min_val and max_val are scalar Tensors.
// ============================================================================

void clamp_cpu(at::Tensor& input, const at::Tensor& min_val, const at::Tensor& max_val) {
  RECORD_FUNCTION("sgl-kernel::clamp_cpu", std::vector<c10::IValue>({input}));
  int64_t numel = input.numel();
  float fmin = min_val.item<float>();
  float fmax = max_val.item<float>();

  if (input.scalar_type() == at::kFloat) {
    clamp_kernel_float_impl(
        input.data_ptr<float>(),
        fmin,
        fmax,
        numel);
  } else {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "clamp_cpu", [&] {
      clamp_kernel_reduced_impl<scalar_t>(
          input.data_ptr<scalar_t>(),
          fmin,
          fmax,
          numel);
    });
  }
}
