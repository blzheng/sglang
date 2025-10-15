#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void deepseek_index_kernel_impl(
  const scalar_t* __restrict__ query,
  const scalar_t* __restrict__ weight,
  const scalar_t* __restrict__ key,
  float* __restrict__ out,
  int64_t B,
  int64_t M,
  int64_t H,
  int64_t D,
  int64_t N) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();
  at::parallel_for(0, B * M * N, 0, [&](int64_t begin, int64_t end){
    int64_t bi{0}, mi{0}, ni{0};
    data_index_init(begin, bi, B, mi, M, ni, N);
    for (int64_t i = begin; i < end; ++i) {
      int64_t k_offset = bi * N * D + ni * D;
      int64_t out_offset = bi * M * N + mi * N + ni;
      for(int64_t hi = 0; hi < H; ++hi) {
        int64_t q_offset = bi * M * H * D + mi * H * D + hi * D;
        int64_t weight_offset = bi * M * H + mi * H + hi;
        fVec logit_vec = fVec(float(0));
        int64_t di = 0;
        for(; di + bVecSize <= D; di += bVecSize) {
          bVec q_vec = bVec::loadu(query + q_offset + di);
          bVec k_vec = bVec::loadu(key + k_offset + di);
          fVec q_fvec0, q_fvec1, k_fvec0, k_fvec1;
          std::tie(q_fvec0, q_fvec1) = at::vec::convert_to_float(q_vec);
          std::tie(k_fvec0, k_fvec1) = at::vec::convert_to_float(k_vec);
          logit_vec += q_fvec0 * k_fvec0;
          logit_vec += q_fvec1 * k_fvec1;
        }
        float logit = at::vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, logit_vec);
        for(; di < D; ++di) {
          logit += float(query[q_offset + di]) * float(key[k_offset + di]);
        }
        if (logit > 0) {
          out[out_offset] += logit * weight[weight_offset];
        }
      }
      // move to the next index
      data_index_step(bi, B, mi, M, ni, N);
    }
  });
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
void deepseek_index_kernel_impl(
  const at::BFloat16* __restrict__ query,
  const at::BFloat16* __restrict__ weight,
  const at::BFloat16* __restrict__ key,
  float* __restrict__ out,
  int64_t B,
  int64_t M,
  int64_t H,
  int64_t D,
  int64_t N) {
  constexpr int64_t vec_size = 32;
  at::parallel_for(0, B * M * N, 0, [&](int64_t begin, int64_t end){
    int64_t bi{0}, mi{0}, ni{0};
    data_index_init(begin, bi, B, mi, M, ni, N);
    for (int64_t i = begin; i < end; ++i) {
      int64_t k_offset = bi * N * D + ni * D;
      int64_t out_offset = bi * M * N + mi * N + ni;
      for(int64_t hi = 0; hi < H; ++hi) {
        int64_t q_offset = bi * M * H * D + mi * H * D + hi * D;
        int64_t weight_offset = bi * M * H + mi * H + hi;
        __m512 logit_vec = _mm512_setzero_ps();
        int64_t di = 0;
        for(; di + vec_size <= D; di += vec_size) {
          __m512bh q_vec = (__m512bh)(_mm512_loadu_si512(query + q_offset + di));
          __m512bh k_vec = (__m512bh)(_mm512_loadu_si512(key + k_offset + di));
          logit_vec = _mm512_dpbf16_ps(logit_vec, q_vec, k_vec);
        }
        int64_t count = D -di;
        if (count > 0) {
          __mmask32 mask = (1ULL << count) - 1;
          __m512bh q_vec = (__m512bh)(_mm512_maskz_loadu_epi16(mask, query + q_offset + di));
          __m512bh k_vec = (__m512bh)(_mm512_maskz_loadu_epi16(mask, key + k_offset + di));
          logit_vec = _mm512_dpbf16_ps(logit_vec, q_vec, k_vec);
        }
        float logit = _mm512_reduce_add_ps(logit_vec);
        if (logit > 0) {
          out[out_offset] += logit * weight[weight_offset];
        }
      }
      // move to the next index
      data_index_step(bi, B, mi, M, ni, N);
    }
  });
}
#endif

}  // namespace

// query: [B, M, H, D]
// weight: [B, M, H]
// key: [B, N, D]
at::Tensor deepseek_index_cpu(
    at::Tensor& query,
    at::Tensor& weight,
    at::Tensor& key) {
  RECORD_FUNCTION("sgl-kernel::deepseek_index_cpu", std::vector<c10::IValue>({query, weight, key}));
  CHECK_INPUT(query);
  CHECK_INPUT(weight);
  CHECK_INPUT(key);
  CHECK_DIM(4, query);
  CHECK_DIM(3, weight);
  CHECK_DIM(3, key);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t H = query.size(2);
  int64_t D = query.size(3);
  int64_t N = key.size(1);
  CHECK_EQ(weight.size(0), B);
  CHECK_EQ(weight.size(1), M);
  CHECK_EQ(weight.size(2), H);
  CHECK_EQ(key.size(0), B);
  CHECK_EQ(key.size(2), D);

  at::Tensor out = at::zeros({B, M, N}, query.options().dtype(at::kFloat));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "deepseek_index_kernel", [&] {
    deepseek_index_kernel_impl<scalar_t>(
        query.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        out.data_ptr<float>(),
        B,
        M,
        H,
        D,
        N);
  });

  return out;
}
