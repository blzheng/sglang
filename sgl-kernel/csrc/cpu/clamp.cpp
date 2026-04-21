#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void clamp_kernel_impl(scalar_t* __restrict__ data, float min_val, float max_val, int64_t numel) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  const fVec min_fvec(min_val);
  const fVec max_fvec(max_val);

  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
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

#if defined(CPU_CAPABILITY_AVX512)
template <>
void clamp_kernel_impl(at::BFloat16* __restrict__ data, float min_val, float max_val, int64_t numel) {
  const __m512 vmin = _mm512_set1_ps(min_val);
  const __m512 vmax = _mm512_set1_ps(max_val);

  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t d = begin;
    // 32 bf16 values = 512 bits
#pragma GCC unroll 4
    for (; d <= end - 32; d += 32) {
      __m512i raw = _mm512_loadu_si512(data + d);
      __m512 f0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(raw, 0));
      __m512 f1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(raw, 1));

      f0 = _mm512_min_ps(vmax, _mm512_max_ps(vmin, f0));
      f1 = _mm512_min_ps(vmax, _mm512_max_ps(vmin, f1));

      _mm512_storeu_si512(data + d, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
    }
#pragma GCC unroll 4
    for (; d < end; ++d) {
      float val = static_cast<float>(data[d]);
      val = std::max(min_val, std::min(max_val, val));
      data[d] = static_cast<at::BFloat16>(val);
    }
  });
}

template <>
void clamp_kernel_impl(at::Half* __restrict__ data, float min_val, float max_val, int64_t numel) {
  const __m512 vmin = _mm512_set1_ps(min_val);
  const __m512 vmax = _mm512_set1_ps(max_val);

  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t d = begin;
    // 16 fp16 values = 256 bits → promoted to 512-bit float
#pragma GCC unroll 4
    for (; d <= end - 16; d += 16) {
      __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + d));
      __m512 f = _mm512_cvtph_ps(raw);

      f = _mm512_min_ps(vmax, _mm512_max_ps(vmin, f));

      _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + d), _mm512_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT));
    }
#pragma GCC unroll 4
    for (; d < end; ++d) {
      float val = static_cast<float>(data[d]);
      val = std::max(min_val, std::min(max_val, val));
      data[d] = static_cast<at::Half>(val);
    }
  });
}
#endif

}  // anonymous namespace

void clamp_cpu(at::Tensor& input, const at::Tensor& min_val, const at::Tensor& max_val) {
  RECORD_FUNCTION("sgl-kernel::clamp_cpu", std::vector<c10::IValue>({input}));
  int64_t numel = input.numel();
  float fmin = min_val.item<float>();
  float fmax = max_val.item<float>();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      input.scalar_type(), "clamp_kernel", [&] { clamp_kernel_impl(input.data_ptr<scalar_t>(), fmin, fmax, numel); });
}
