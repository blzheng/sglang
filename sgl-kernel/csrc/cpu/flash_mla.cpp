// Intel CPU AMX implementation of FlashMLA's `flash_mla_with_kvcache` for the
// sparse decode path used by DeepSeek-V3.2 / DSv4 reference.
//
// This kernel mirrors the interface of:
//   - flash_mla.flash_mla_with_kvcache (DeepSeek FlashMLA upstream)
//   - flash_mla_with_kvcache_torch (sglang reference @
//     python/sglang/srt/layers/attention/debug_flash_mla_adapter.py)
//
// It follows the same online-softmax + AMX brgemm strategy used by
// `decode_attention_mla_kernel_impl` in decode.cpp, but:
//   * K is gathered by per-(batch, query) absolute token indices
//     (`indices_in_kvcache`) instead of via `req_to_token`.
//   * K is stored quantized (FP8 NoPE + BF16 RoPE + per-tile scales) and
//     dequantized to BF16 before being packed for AMX brgemm.
//   * Optional `attn_sink` (per-head bias added in log-sum-exp space) and
//     `topk_length` (variable-length top-k mask) are supported.
//   * Optional `extra_k_cache` / `extra_indices_in_kvcache` /
//     `extra_topk_length` are concatenated along the topk axis and processed
//     in the same online-softmax pass.
//
// Returned tensors:
//   out: (B, S_q, H_q, D_v), bfloat16
//   lse: (B, H_q, S_q),       float32
//
// Parallelism: [batches, S_q, head_blocks, num_kv_splits], identical to the
// existing CPU MLA decode kernel.

#include "common.h"
#include "gemm.h"
#include "vec.h"

#include <algorithm>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// FP8 layouts (mirrors python/sglang/srt/flashmla_tests/quant.py)
// ---------------------------------------------------------------------------
//
// V32_FP8Sparse:    (d=576, d_nope=512, d_rope=64, tile=128, num_tiles=4)
//   per-token storage = d_nope FP8(e4m3) | num_tiles * 4 (fp32 scales) | d_rope * 2 (bf16 RoPE)
//   = 512 + 16 + 128 = 656 bytes
//   tokens are tightly packed:  [num_blocks, block_size, 656]
//
// MODEL1_FP8Sparse: (d=512, d_nope=448, d_rope=64, tile=64,  num_tiles=7)
//   per block:
//     [block_size * (d_nope + 2*d_rope) bytes  ; FP8 NoPE + bf16 RoPE interleaved per token]
//     [block_size * 8 bytes                    ; 7 e8m0 scales per token + 1 byte pad]
//   block stride is padded up to a multiple of 576 bytes.
//
// We dequantize once into a [num_blocks*block_size, d_qk] BF16 buffer and
// reuse it for both the Q@K (head_size = d_qk) and the S@V (head_size_v
// columns of the same buffer).  Memory footprint for typical decode shapes
// (~few MB) is small relative to the QK^T and softmax compute.

enum FP8KVCacheLayout : int64_t {
  kV32FP8Sparse = 1,    // FP8KVCacheLayout.V32_FP8Sparse
  kModel1FP8Sparse = 2  // FP8KVCacheLayout.MODEL1_FP8Sparse
};

struct FP8LayoutMeta {
  int64_t d;
  int64_t d_nope;
  int64_t d_rope;
  int64_t tile_size;
  int64_t num_tiles;
};

inline FP8LayoutMeta get_fp8_meta(int64_t layout) {
  switch (layout) {
    case kV32FP8Sparse:
      return {576, 512, 64, 128, 4};
    case kModel1FP8Sparse:
      return {512, 448, 64, 64, 7};
    default:
      TORCH_CHECK(false, "flash_mla_with_kvcache_cpu: unsupported FP8 layout ", layout);
  }
}

// Convert a single fp8_e4m3 byte to float using ATen's helper to keep
// behaviour aligned with the rest of the codebase.
inline float fp8_e4m3_to_float(uint8_t v) {
  c10::Float8_e4m3fn x;
  x.x = v;
  return static_cast<float>(x);
}

// Convert one fp8_e8m0 byte (= unsigned 8-bit exponent) to float.
// e8m0 only stores an exponent (bias 127); value = 2^(e - 127), with 0xFF
// reserved for NaN.  Mirrors the cast used in quant.py.
inline float fp8_e8m0_to_float(uint8_t v) {
  if (v == 0xFF) return std::numeric_limits<float>::quiet_NaN();
  // exponent of 0 maps to +-0 according to e8m0fnu spec.
  if (v == 0) return 0.f;
  union {
    uint32_t u;
    float f;
  } u;
  u.u = static_cast<uint32_t>(v) << 23;
  return u.f;
}

// Dequantize the active FP8 KV-cache prefix into a contiguous BF16 tensor of
// shape [active_total_tokens, d_qk].  This is parallel over tokens.
//
// `fp8_storage` is the raw byte view of `k_cache` (last dim = bytes_per_token
// for V32 / variable padded for MODEL1).
template <int64_t LAYOUT>
void dequantize_fp8_kvcache_impl(
    at::BFloat16* __restrict__ out,
    const uint8_t* __restrict__ fp8_storage,
    int64_t block_size,
    int64_t active_total_tokens,
    int64_t storage_block_stride_bytes) {
  static_assert(LAYOUT == kV32FP8Sparse || LAYOUT == kModel1FP8Sparse, "bad layout");
  constexpr FP8LayoutMeta meta = (LAYOUT == kV32FP8Sparse) ? FP8LayoutMeta{576, 512, 64, 128, 4}
                                                            : FP8LayoutMeta{512, 448, 64, 64, 7};

  const int64_t total_tokens = active_total_tokens;

  if constexpr (LAYOUT == kV32FP8Sparse) {
    constexpr int64_t bytes_per_token = meta.d_nope + meta.num_tiles * 4 + meta.d_rope * 2;  // 656

    at::parallel_for(0, total_tokens, /*grain_size*/ 16, [&](int64_t begin, int64_t end) {
      for (int64_t t = begin; t < end; ++t) {
        const int64_t b = t / block_size;
        const int64_t s = t - b * block_size;
        const uint8_t* src = fp8_storage + b * storage_block_stride_bytes + s * bytes_per_token;
        at::BFloat16* dst = out + t * meta.d;

        const uint8_t* nope_ptr = src;
        const float* scale_ptr = reinterpret_cast<const float*>(src + meta.d_nope);
        const at::BFloat16* rope_ptr =
            reinterpret_cast<const at::BFloat16*>(src + meta.d_nope + meta.num_tiles * 4);

        for (int64_t tile = 0; tile < meta.num_tiles; ++tile) {
          const float sc = scale_ptr[tile];
          for (int64_t i = 0; i < meta.tile_size; ++i) {
            const int64_t k = tile * meta.tile_size + i;
            dst[k] = static_cast<at::BFloat16>(fp8_e4m3_to_float(nope_ptr[k]) * sc);
          }
        }
        // copy bf16 RoPE part as-is
        for (int64_t k = 0; k < meta.d_rope; ++k) {
          dst[meta.d_nope + k] = rope_ptr[k];
        }
      }
    });
  } else {
    // MODEL1_FP8Sparse: per block layout is
    //   nope_rope_part : block_size * (d_nope + 2*d_rope) bytes, indexed by token
    //   scale_part     : block_size * 8 bytes, where last byte per token is padding
    constexpr int64_t nope_rope_per_token = meta.d_nope + 2 * meta.d_rope;  // 448 + 128 = 576
    constexpr int64_t scale_stride = 8;

    at::parallel_for(0, total_tokens, /*grain_size*/ 16, [&](int64_t begin, int64_t end) {
      for (int64_t t = begin; t < end; ++t) {
        const int64_t b = t / block_size;
        const int64_t s = t - b * block_size;
        const uint8_t* block_base = fp8_storage + b * storage_block_stride_bytes;
        const uint8_t* nope_rope = block_base + s * nope_rope_per_token;
        const uint8_t* scale_base =
            block_base + block_size * nope_rope_per_token + s * scale_stride;
        at::BFloat16* dst = out + t * meta.d;

        const uint8_t* nope_ptr = nope_rope;
        const at::BFloat16* rope_ptr =
            reinterpret_cast<const at::BFloat16*>(nope_rope + meta.d_nope);

        for (int64_t tile = 0; tile < meta.num_tiles; ++tile) {
          const float sc = fp8_e8m0_to_float(scale_base[tile]);
          for (int64_t i = 0; i < meta.tile_size; ++i) {
            const int64_t k = tile * meta.tile_size + i;
            dst[k] = static_cast<at::BFloat16>(fp8_e4m3_to_float(nope_ptr[k]) * sc);
          }
        }
        for (int64_t k = 0; k < meta.d_rope; ++k) {
          dst[meta.d_nope + k] = rope_ptr[k];
        }
      }
    });
  }
}

at::Tensor dequantize_fp8_kvcache(at::Tensor k_cache, int64_t layout, int64_t active_total_tokens) {
  const FP8LayoutMeta meta = get_fp8_meta(layout);

  TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D [num_blocks, block_size, 1, packed_bytes]");
  TORCH_CHECK(k_cache.size(2) == 1, "h_k must be 1 for FlashMLA sparse FP8 path");
  TORCH_CHECK(
      k_cache.dtype() == at::kFloat8_e4m3fn,
      "flash_mla_with_kvcache_cpu: expect FP8 k_cache to be float8_e4m3fn, got ",
      k_cache.dtype());

  const int64_t num_blocks = k_cache.size(0);
  const int64_t block_size = k_cache.size(1);
  const int64_t capacity = num_blocks * block_size;
  TORCH_CHECK(
      active_total_tokens >= 0 && active_total_tokens <= capacity,
      "flash_mla_with_kvcache_cpu: active_total_tokens (",
      active_total_tokens,
      ") must be within KV-cache capacity (",
      capacity,
      ")");

  // Each block is contiguous in storage but its trailing-bytes count differs
  // per layout.  The outermost stride in *bytes* tells us the true padded
  // block stride (= 576-byte aligned for MODEL1).
  const int64_t block_stride_elems = k_cache.stride(0);
  const int64_t storage_block_stride_bytes = block_stride_elems * k_cache.element_size();

  auto out = at::empty({active_total_tokens, meta.d}, k_cache.options().dtype(at::kBFloat16));
  const uint8_t* fp8_storage = static_cast<const uint8_t*>(k_cache.data_ptr());

  if (layout == kV32FP8Sparse) {
    dequantize_fp8_kvcache_impl<kV32FP8Sparse>(
        out.data_ptr<at::BFloat16>(),
        fp8_storage,
        block_size,
        active_total_tokens,
        storage_block_stride_bytes);
  } else {
    dequantize_fp8_kvcache_impl<kModel1FP8Sparse>(
        out.data_ptr<at::BFloat16>(),
        fp8_storage,
        block_size,
        active_total_tokens,
        storage_block_stride_bytes);
  }
  return out;
}

template <typename index_t>
int64_t infer_active_total_tokens(
    const index_t* __restrict__ indices,
    int64_t batches,
    int64_t s_q,
    int64_t topk,
    int64_t capacity,
    const int32_t* __restrict__ topk_length) {
  if (indices == nullptr || topk == 0 || capacity == 0) {
    return 0;
  }

  const int num_threads = at::get_num_threads();
  std::vector<int64_t> thread_max(num_threads, -1);
  at::parallel_for(0, batches * s_q, 0, [&](int64_t begin, int64_t end) {
    const int tid = at::get_thread_num();
    int64_t local_max = -1;
    for (int64_t i = begin; i < end; ++i) {
      const int64_t b = i / s_q;
      const int64_t limit = topk_length != nullptr
          ? std::max<int64_t>(0, std::min<int64_t>(topk_length[b], topk))
          : topk;
      const index_t* row = indices + i * topk;
      for (int64_t k = 0; k < limit; ++k) {
        const int64_t v = static_cast<int64_t>(row[k]);
        if (v >= 0 && v < capacity) {
          local_max = std::max(local_max, v);
        }
      }
    }
    thread_max[tid] = std::max(thread_max[tid], local_max);
  });

  int64_t max_seen = -1;
  for (int64_t v : thread_max) {
    max_seen = std::max(max_seen, v);
  }
  return max_seen + 1;
}

template <typename index_t>
inline bool is_valid_sparse_index(index_t idx, int64_t pos, int64_t topk_limit, int64_t total_tokens) {
  const int64_t v = static_cast<int64_t>(idx);
  return pos < topk_limit && v >= 0 && v < total_tokens;
}

// ---------------------------------------------------------------------------
// AMX-friendly K/V VNNI packer for sparse decode.
//
// The packing strategy mirrors `pack_vnni` in decode.cpp:
//   * For QK^T:  key  packed from [N, K]   -> [K/2, N, 2]   (tile B for tile A=Q)
//   * For S@V:   value packed from [N, Kv] -> [N/2, Kv, 2]
// where N is the BLOCK_N tile of K rows (gathered by per-token indices) and
// K = head_size, Kv = head_size_v.  `indices < 0` (invalid) load zero rows
// (their attention scores are masked out separately).
// ---------------------------------------------------------------------------

#if defined(CPU_CAPABILITY_AVX512)
template <typename scalar_t, typename index_t>
inline void sparse_pack_vnni_Nx32(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int64_t n_start,
    int64_t topk_limit,
    int64_t total_tokens,
    int ld_src,
    int ld_dst0,
    int ld_dst1,
    bool convert_v) {
  __m512i vinputs[16];
  int n = 0;
  for (; n < N; ++n) {
    index_t idx = ind[n];
    if (!is_valid_sparse_index(idx, n_start + n, topk_limit, total_tokens)) {
      vinputs[n] = _mm512_set1_epi32(0);
    } else {
      vinputs[n] = _mm512_loadu_si512(src + idx * ld_src);
    }
  }
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  if (convert_v) {
    for (int nn = 0; nn < 16; nn += 2) {
      __m512i d0, d1;
      std::tie(d0, d1) = transpose_2x32_16bit(vinputs[nn], vinputs[nn + 1]);
      _mm512_storeu_si512(dst1 + (nn >> 1) * ld_dst1 * 2, d0);
      _mm512_storeu_si512(dst1 + (nn >> 1) * ld_dst1 * 2 + 32, d1);
    }
  }

  transpose_16x16_32bit(vinputs);
  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst0 + k * ld_dst0 * 2, vmask, vinputs[k]);
  }
}
#endif

template <typename scalar_t, typename index_t>
void sparse_pack_vnni(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int Kv,
    int64_t n_start,
    int64_t topk_limit,
    int64_t total_tokens,
    int ld_src,
    int ld_dst0,
    int ld_dst1) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;
  const int KBv = Kv / 32;
  for (int nb = 0; nb < NB; ++nb) {
    for (int kb = 0; kb < KB; ++kb) {
      int nb_size = std::min(N - nb * 16, 16);
      sparse_pack_vnni_Nx32<scalar_t, index_t>(
          /*    dst0 */ dst0 + ((kb * 32) >> 1) * ld_dst0 * 2 + nb * 16 * 2,
          /*    dst1 */ dst1 + ((nb * 16) >> 1) * ld_dst1 * 2 + kb * 32 * 2,
          /*     src */ src + kb * 32,
          /*     ind */ ind + nb * 16,
          /*       N */ nb_size,
          /* n_start */ n_start + nb * 16,
          /* top_lim */ topk_limit,
          /*  n_tokn */ total_tokens,
          /*  ld_src */ ld_src,
          /* ld_dst0 */ ld_dst0,
          /* ld_dst1 */ ld_dst1,
          /*   cvt_v */ kb < KBv);
    }
  }
#else
  // Reference scalar fallback (NO-AVX512 build).
  for (int n = 0; n < N; ++n) {
    index_t idx = ind[n];
    const bool valid = is_valid_sparse_index(idx, n_start + n, topk_limit, total_tokens);
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        scalar_t v = !valid ? scalar_t(0) : src[idx * ld_src + k * 2 + d];
        dst0[k * ld_dst0 * 2 + n * 2 + d] = v;
      }
    }
  }
  for (int n = 0; n < (N >> 1) * 2; n += 2) {
    index_t i0 = ind[n + 0];
    index_t i1 = ind[n + 1];
    const bool valid0 = is_valid_sparse_index(i0, n_start + n + 0, topk_limit, total_tokens);
    const bool valid1 = is_valid_sparse_index(i1, n_start + n + 1, topk_limit, total_tokens);
    for (int k = 0; k < Kv; ++k) {
      dst1[(n >> 1) * ld_dst1 * 2 + k * 2 + 0] = !valid0 ? scalar_t(0) : src[i0 * ld_src + k];
      dst1[(n >> 1) * ld_dst1 * 2 + k * 2 + 1] = !valid1 ? scalar_t(0) : src[i1 * ld_src + k];
    }
  }
  if (N % 2 != 0) {
    index_t idx = ind[N - 1];
    const bool valid = is_valid_sparse_index(idx, n_start + N - 1, topk_limit, total_tokens);
    for (int k = 0; k < Kv; ++k) {
      dst1[(N >> 1) * ld_dst1 * 2 + k * 2 + 0] = !valid ? scalar_t(0) : src[idx * ld_src + k];
      dst1[(N >> 1) * ld_dst1 * 2 + k * 2 + 1] = 0;
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// Helpers shared with decode.cpp (small inline duplicates so we don't have to
// expose them through a header).
// ---------------------------------------------------------------------------

template <typename scalar_t>
inline void fmla_fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int BLOCK_N>
inline void fmla_copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    if constexpr (col % 2 == 0) {
      fVec a0 = fVec::loadu(input + col * 16);
      fVec a1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a0, a1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

template <typename scalar_t>
inline void fmla_finalize_out(
    scalar_t* __restrict__ out, const float* __restrict__ acc, float inv_s, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(inv_s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a0 = fVec::loadu(acc + d) * s_fvec;
    fVec a1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a0, a1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * inv_s);
  }
}

// ---------------------------------------------------------------------------
// Main kernel: sparse MLA decode (AMX BF16 brgemm).
//
// query    : [B, S_q, H_q, D_qk]   bf16
// k_main   : [active_main_tokens, D_qk] bf16 or original bf16 cache flattened
// indices  : [B, S_q, topk_main]        int32/int64
// k_extra  : optional extra KV source with its own indices
// topk_len : [B]                   int32 or null
// attn_sink: [H_q]                 float32 or null
// output   : [B, S_q, H_q, D_v]    bf16
// lse      : [B, H_q, S_q]         float32
// ---------------------------------------------------------------------------

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void sparse_mla_decode_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ lse_out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ k_main,
    const scalar_t* __restrict__ k_extra,
    const index_t* __restrict__ indices,
    const index_t* __restrict__ extra_indices,
    const int32_t* __restrict__ topk_length,
    const int32_t* __restrict__ extra_topk_length,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ buffer,
    int64_t batches,
    int64_t s_q,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t topk_main,
    int64_t topk_extra,
    int64_t total_tokens_main,
    int64_t total_tokens_extra,
    int64_t q_strideB,
    int64_t q_strideS,
    int64_t q_strideH,
    int64_t k_main_strideN,
    int64_t k_extra_strideN,
    int64_t idx_strideB,
    int64_t idx_strideS,
    int64_t extra_idx_strideB,
    int64_t extra_idx_strideS,
    float scaling,
    int64_t buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;

  // partition heads
  constexpr int64_t kBLOCK_H_MAX = 16;
  const int64_t BLOCK_H = (batches * s_q) >= 16 ? kBLOCK_H_MAX : 8;
  const int64_t num_h_blocks = div_up(num_heads, BLOCK_H);

  // parallel on [B, S_q, head_block]
  at::parallel_for(0, batches * s_q * num_h_blocks, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, sq{0}, hb{0};
    data_index_init(begin, bs, batches, sq, s_q, hb, num_h_blocks);

    int tid = at::get_thread_num();
    scalar_t* __restrict__ Btmp0 = buffer + tid * buffer_size_per_thread;        // K  packed
    scalar_t* __restrict__ Btmp1 = Btmp0 + BLOCK_N * head_size;                  // V  packed
    // f32 V accumulator follows the bf16 packing region (reinterpret cast).
    float* __restrict__ v_acc_local =
        reinterpret_cast<float*>(Btmp1 + BLOCK_N * head_size_v);
    fmla_fill_stub(Btmp1, 0.f, BLOCK_N * head_size_v);  // initialize V padding

    alignas(64) float s_i[kBLOCK_H_MAX * BLOCK_N];
    float* __restrict__ s_delta = s_i;
    alignas(64) scalar_t s_delta2[kBLOCK_H_MAX * BLOCK_N];

    alignas(64) float s_prime[kBLOCK_H_MAX];
    alignas(64) float m_prime[kBLOCK_H_MAX];

    for (int64_t i = begin; i < end; ++i) {
      const int64_t h_start = hb * BLOCK_H;
      const int64_t h_end = std::min(h_start + BLOCK_H, num_heads);
      const int64_t h_size = h_end - h_start;

      const scalar_t* __restrict__ q_ptr = query + bs * q_strideB + sq * q_strideS + h_start * q_strideH;
      const index_t* __restrict__ idx_ptr = indices + bs * idx_strideB + sq * idx_strideS;
      const index_t* __restrict__ extra_idx_ptr = extra_indices == nullptr
          ? nullptr
          : extra_indices + bs * extra_idx_strideB + sq * extra_idx_strideS;

      fmla_fill_stub(s_prime, 0.f, BLOCK_H);
      fmla_fill_stub(m_prime, -std::numeric_limits<float>::infinity(), BLOCK_H);
      for (int64_t h = 0; h < h_size; ++h) {
        fmla_fill_stub(v_acc_local + h * head_size_v, 0.f, head_size_v);
      }

      auto process_cache = [&](const scalar_t* __restrict__ k_ptr,
                               const index_t* __restrict__ cur_idx_ptr,
                               const int32_t* __restrict__ cur_topk_length,
                               int64_t topk_count,
                               int64_t total_tokens,
                               int64_t k_strideN) {
        if (k_ptr == nullptr || cur_idx_ptr == nullptr || topk_count == 0) {
          return;
        }
        const int64_t topk_limit = cur_topk_length != nullptr
            ? std::max<int64_t>(0, std::min<int64_t>(cur_topk_length[bs], topk_count))
            : topk_count;

        // loop over one top-k source (main or extra). Processing them as two
        // consecutive streams avoids allocating/copying a unified KV buffer and
        // merged indices while preserving online-softmax semantics.
        for (int64_t n = 0; n < topk_count; n += BLOCK_N) {
          int64_t n_size = std::min<int64_t>(BLOCK_N, topk_count - n);
          const int64_t padded_n_size = div_up(int(n_size), TILE_K) * TILE_K;

          // Pack K (BLOCK_N rows via gather) into Btmp0 (key, vnni) and Btmp1
          // (value, vnni). Invalid entries load zeros and are masked below.
          sparse_pack_vnni<scalar_t, index_t>(
              /*    dst0 */ Btmp0,
              /*    dst1 */ Btmp1,
              /*     src */ k_ptr,
              /*     ind */ cur_idx_ptr + n,
              /*       N */ static_cast<int>(n_size),
              /*       K */ static_cast<int>(head_size),
              /*      Kv */ static_cast<int>(head_size_v),
              /* n_start */ n,
              /* top_lim */ topk_limit,
              /*  n_tokn */ total_tokens,
              /*  ld_src */ static_cast<int>(k_strideN),
              /* ld_dst0 */ static_cast<int>(BLOCK_N),
              /* ld_dst1 */ static_cast<int>(head_size_v));

          // Q @ K
          at::native::cpublas::brgemm(
              /* M     */ h_size,
              /* N     */ n_size,
              /* K     */ head_size,
              /* lda   */ q_strideH,
              /* ldb   */ BLOCK_N,
              /* ldc   */ BLOCK_N,
              /* add_C */ false,
              /* A     */ q_ptr,
              /* B     */ Btmp0,
              /* C     */ s_i);

          const Vec scale_vec = Vec(scaling);
          for (int64_t h = 0; h < h_size; ++h) {
            float* row = s_i + h * BLOCK_N;
            // s_i <- s_i * scale, with masking for invalid indices and tail
            at::vec::map<float>([scale_vec](Vec x) { return x * scale_vec; }, row, row, n_size);

            for (int64_t k = 0; k < n_size; ++k) {
              if (!is_valid_sparse_index(cur_idx_ptr[n + k], n + k, topk_limit, total_tokens)) {
                row[k] = -std::numeric_limits<float>::infinity();
              }
            }

            // online softmax update
            float m_i = at::vec::reduce_all<float>(
                [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, row, n_size);
            m_i = std::max(m_i, m_prime[h]);

            // Guard against the all-masked tile (m_i == -inf): keep state unchanged.
            if (!std::isfinite(m_i)) {
              // Still need to produce zeros for s_delta on this tile.
              fmla_fill_stub(s_delta + h * BLOCK_N, 0.f, padded_n_size);
              fmla_copy_stub<scalar_t, BLOCK_N>(s_delta2 + h * BLOCK_N, s_delta + h * BLOCK_N);
              continue;
            }

            const float m_delta = std::exp(m_prime[h] - m_i);
            at::vec::map<float>(
                [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); },
                s_delta + h * BLOCK_N,
                row,
                n_size);

            s_prime[h] *= m_delta;
            s_prime[h] += at::vec::reduce_all<float>(
                [](Vec& x, Vec& y) { return x + y; }, s_delta + h * BLOCK_N, n_size);

            m_prime[h] = m_i;

            // Rescale the running V accumulator for this head.
            float scale_m = m_delta;
            at::vec::map<float>(
                [scale_m](Vec x) { return x * Vec(scale_m); },
                v_acc_local + h * head_size_v,
                v_acc_local + h * head_size_v,
                head_size_v);

            // Pad s_delta with 0 then convert to bf16 (s_delta2)
            fmla_fill_stub(s_delta + h * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
            fmla_copy_stub<scalar_t, BLOCK_N>(s_delta2 + h * BLOCK_N, s_delta + h * BLOCK_N);
          }

          // V' <- s_delta @ V + V'   (accumulate into v_acc_local at f32)
          at::native::cpublas::brgemm(
              /* M     */ h_size,
              /* N     */ head_size_v,
              /* K     */ padded_n_size,
              /* lda   */ BLOCK_N,
              /* ldb   */ head_size_v,
              /* ldc   */ head_size_v,
              /* add_C */ true,
              /* A     */ s_delta2,
              /* B     */ Btmp1,
              /* C     */ v_acc_local);
        }
      };

      process_cache(k_main, idx_ptr, topk_length, topk_main, total_tokens_main, k_main_strideN);
      process_cache(
          k_extra,
          extra_idx_ptr,
          extra_topk_length,
          topk_extra,
          total_tokens_extra,
          k_extra_strideN);

      // Apply attention sink correction directly on the output and lse.
      //   out *= exp(lse_no_sink) / (exp(lse_no_sink) + exp(attn_sink))
      //        = 1 / (1 + exp(attn_sink - lse_no_sink))
      // where lse_no_sink = m_prime + log(s_prime).  When lse_no_sink == -inf
      // (i.e. no valid k), the output is forced to zero and lse to +inf to
      // match the reference.
      for (int64_t h = 0; h < h_size; ++h) {
        const int64_t hh = h_start + h;
        const bool lonely = !std::isfinite(m_prime[h]) || s_prime[h] == 0.f;
        float lse_val = lonely ? std::numeric_limits<float>::infinity()
                                : (m_prime[h] + std::log(s_prime[h]));
        float inv_s = lonely ? 0.f : (1.f / s_prime[h]);

        if (!lonely && attn_sink != nullptr) {
          const float sink = attn_sink[hh];
          // sink scaling on output:  out *= 1 / (1 + exp(sink - lse))
          // (lse here is the un-sinked lse).
          const float corr = 1.f / (1.f + std::exp(sink - lse_val));
          inv_s *= corr;
        }

        // Write final bf16 output row.
        scalar_t* out_row = output + bs * (s_q * num_heads * head_size_v)
            + sq * (num_heads * head_size_v) + hh * head_size_v;
        if (lonely) {
          fmla_fill_stub(out_row, 0.f, head_size_v);
        } else {
          fmla_finalize_out<scalar_t>(out_row, v_acc_local + h * head_size_v, inv_s, head_size_v);
        }

        // lse layout: (B, H_q, S_q)
        lse_out[bs * num_heads * s_q + hh * s_q + sq] = lse_val;
      }

      data_index_step(bs, batches, sq, s_q, hb, num_h_blocks);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point: flash_mla_with_kvcache_cpu
//
// Mirrors the sparse decode path of FlashMLA's flash_mla_with_kvcache.
// Returns (out, lse).
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor> flash_mla_with_kvcache_cpu(
    at::Tensor& q,
    at::Tensor& k_cache,
    int64_t head_dim_v,
    double softmax_scale,
    at::Tensor& indices,                            // [B, S_q, topk]
    std::optional<at::Tensor> topk_length,          // [B]
    std::optional<at::Tensor> attn_sink,            // [H_q]
    std::optional<at::Tensor> extra_k_cache,
    std::optional<at::Tensor> extra_indices,
    std::optional<at::Tensor> extra_topk_length,
    bool is_fp8_kvcache,
    int64_t fp8_layout) {
  RECORD_FUNCTION(
      "sgl-kernel::flash_mla_with_kvcache_cpu", std::vector<c10::IValue>({q, k_cache, indices}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_DIM(4, q);  // [B, S_q, H_q, D_qk]
  CHECK_DIM(4, k_cache);

  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16,
      "flash_mla_with_kvcache_cpu: only bfloat16 query is supported, got ",
      q.scalar_type());
  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "flash_mla_with_kvcache_cpu: indices must be int32 or int64, got ",
      indices.scalar_type());

  const int64_t B = q.size(0);
  const int64_t S_q = q.size(1);
  const int64_t H_q = q.size(2);
  const int64_t D_qk = q.size(3);
  const int64_t D_v = head_dim_v;

  TORCH_CHECK(D_qk >= D_v, "head_dim must be >= head_dim_v");
  CHECK_EQ(indices.size(0), B);
  CHECK_EQ(indices.size(1), S_q);
  const int64_t topk_main = indices.size(2);

  TORCH_CHECK(
      extra_k_cache.has_value() == extra_indices.has_value(),
      "extra_k_cache and extra_indices must be both provided or both omitted");
  bool has_extra = extra_k_cache.has_value();
  int64_t topk_extra = has_extra ? extra_indices.value().size(2) : 0;
  if (has_extra) {
    CHECK_EQ(extra_indices.value().size(0), B);
    CHECK_EQ(extra_indices.value().size(1), S_q);
    TORCH_CHECK(
        extra_indices.value().scalar_type() == indices.scalar_type(),
        "extra_indices dtype must match indices dtype");
  }

  if (topk_length.has_value()) {
    TORCH_CHECK(topk_length.value().scalar_type() == at::kInt, "topk_length must be int32");
    CHECK_EQ(topk_length.value().size(0), B);
  }
  if (extra_topk_length.has_value()) {
    TORCH_CHECK(extra_topk_length.value().scalar_type() == at::kInt, "extra_topk_length must be int32");
    CHECK_EQ(extra_topk_length.value().size(0), B);
  }
  const int32_t* tl_main_ptr =
      topk_length.has_value() ? topk_length.value().data_ptr<int32_t>() : nullptr;
  const int32_t* tl_extra_ptr =
      extra_topk_length.has_value() ? extra_topk_length.value().data_ptr<int32_t>() : nullptr;

  const int64_t capacity_main = k_cache.size(0) * k_cache.size(1);
  int64_t capacity_extra = 0;
  if (has_extra) {
    capacity_extra = extra_k_cache.value().size(0) * extra_k_cache.value().size(1);
  }

  int64_t total_tokens_main = 0;
  int64_t total_tokens_extra = 0;

  if (indices.scalar_type() == at::kInt) {
    total_tokens_main = infer_active_total_tokens<int32_t>(
        indices.data_ptr<int32_t>(),
        B,
        S_q,
        topk_main,
        capacity_main,
        tl_main_ptr);
    if (has_extra) {
      total_tokens_extra = infer_active_total_tokens<int32_t>(
          extra_indices.value().data_ptr<int32_t>(),
          B,
          S_q,
          topk_extra,
          capacity_extra,
          tl_extra_ptr);
    }
  } else {
    total_tokens_main = infer_active_total_tokens<int64_t>(
        indices.data_ptr<int64_t>(),
        B,
        S_q,
        topk_main,
        capacity_main,
        tl_main_ptr);
    if (has_extra) {
      total_tokens_extra = infer_active_total_tokens<int64_t>(
          extra_indices.value().data_ptr<int64_t>(),
          B,
          S_q,
          topk_extra,
          capacity_extra,
          tl_extra_ptr);
    }
  }

  at::Tensor k_dequant_main;
  at::Tensor k_dequant_extra;
  const at::BFloat16* k_main_ptr = nullptr;
  const at::BFloat16* k_extra_ptr = nullptr;
  int64_t k_main_strideN = D_qk;
  int64_t k_extra_strideN = D_qk;

  if (is_fp8_kvcache) {
    k_dequant_main = dequantize_fp8_kvcache(k_cache, fp8_layout, total_tokens_main);
    TORCH_CHECK(
        k_dequant_main.size(1) == D_qk,
        "k_cache dequantized D_qk (",
        k_dequant_main.size(1),
        ") does not match q's last dim (",
        D_qk,
        ")");
    k_main_ptr = k_dequant_main.data_ptr<at::BFloat16>();
    k_main_strideN = k_dequant_main.stride(0);
    if (has_extra) {
      k_dequant_extra = dequantize_fp8_kvcache(extra_k_cache.value(), fp8_layout, total_tokens_extra);
      TORCH_CHECK(
          k_dequant_extra.size(1) == D_qk,
          "extra_k_cache dequantized D_qk (",
          k_dequant_extra.size(1),
          ") does not match q's last dim (",
          D_qk,
          ")");
      k_extra_ptr = k_dequant_extra.data_ptr<at::BFloat16>();
      k_extra_strideN = k_dequant_extra.stride(0);
    }
  } else {
    TORCH_CHECK(
        k_cache.scalar_type() == at::kBFloat16,
        "flash_mla_with_kvcache_cpu: non-FP8 k_cache must be bfloat16, got ",
        k_cache.scalar_type());
    TORCH_CHECK(k_cache.is_contiguous(), "non-FP8 k_cache must be contiguous");
    CHECK_EQ(k_cache.size(2), 1);
    CHECK_EQ(k_cache.size(3), D_qk);
    k_main_ptr = k_cache.data_ptr<at::BFloat16>();
    k_main_strideN = k_cache.stride(1);

    if (has_extra) {
      TORCH_CHECK(
          extra_k_cache.value().scalar_type() == at::kBFloat16,
          "flash_mla_with_kvcache_cpu: non-FP8 extra_k_cache must be bfloat16, got ",
          extra_k_cache.value().scalar_type());
      TORCH_CHECK(extra_k_cache.value().is_contiguous(), "non-FP8 extra_k_cache must be contiguous");
      CHECK_EQ(extra_k_cache.value().size(2), 1);
      CHECK_EQ(extra_k_cache.value().size(3), D_qk);
      k_extra_ptr = extra_k_cache.value().data_ptr<at::BFloat16>();
      k_extra_strideN = extra_k_cache.value().stride(1);
    }
  }

  // 2) allocate outputs
  auto out = at::empty({B, S_q, H_q, D_v}, q.options());
  auto lse = at::empty({B, H_q, S_q}, q.options().dtype(at::kFloat));

  // 3) per-thread B buffer for K (head_size) + V (head_size_v) packing,
  //    plus an f32 V accumulator of size kBLOCK_H_MAX * head_size_v.
  constexpr int64_t BLOCK_N = 128;  // multiple of 32 for AMX brgemm
  constexpr int64_t kBLOCK_H_MAX = 16;
  TORCH_CHECK(D_qk % 32 == 0, "head_dim_qk must be a multiple of 32");
  TORCH_CHECK(D_v % 32 == 0, "head_dim_v must be a multiple of 32");

  const int num_threads = at::get_num_threads();
  // Layout per thread (in bf16 elements):
  //   [Btmp0 : BLOCK_N * D_qk] [Btmp1 : BLOCK_N * D_v] [v_acc_local : kBLOCK_H_MAX * D_v floats]
  // f32 takes 2 bf16 elements -> multiply by 2.
  const int64_t buffer_size_per_thread = BLOCK_N * D_qk + BLOCK_N * D_v + 2 * kBLOCK_H_MAX * D_v;
  auto buffer = at::empty({num_threads, buffer_size_per_thread}, q.options());

  // 4) strides
  const int64_t q_strideB = q.stride(0);
  const int64_t q_strideS = q.stride(1);
  const int64_t q_strideH = q.stride(2);
  const int64_t idx_strideB = indices.stride(0);
  const int64_t idx_strideS = indices.stride(1);
  const int64_t extra_idx_strideB = has_extra ? extra_indices.value().stride(0) : 0;
  const int64_t extra_idx_strideS = has_extra ? extra_indices.value().stride(1) : 0;

  // 5) dispatch on indices dtype
  if (indices.scalar_type() == at::kInt) {
    sparse_mla_decode_kernel_impl<at::BFloat16, int32_t, BLOCK_N>(
        out.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>(),
        q.data_ptr<at::BFloat16>(),
        k_main_ptr,
        k_extra_ptr,
        indices.data_ptr<int32_t>(),
        has_extra ? extra_indices.value().data_ptr<int32_t>() : nullptr,
        tl_main_ptr,
        tl_extra_ptr,
        attn_sink.has_value() ? attn_sink.value().data_ptr<float>() : nullptr,
        buffer.data_ptr<at::BFloat16>(),
        B,
        S_q,
        H_q,
        D_qk,
        D_v,
        topk_main,
        topk_extra,
        total_tokens_main,
        total_tokens_extra,
        q_strideB,
        q_strideS,
        q_strideH,
        k_main_strideN,
        k_extra_strideN,
        idx_strideB,
        idx_strideS,
        extra_idx_strideB,
        extra_idx_strideS,
        static_cast<float>(softmax_scale),
        buffer_size_per_thread);
  } else {
    sparse_mla_decode_kernel_impl<at::BFloat16, int64_t, BLOCK_N>(
        out.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>(),
        q.data_ptr<at::BFloat16>(),
        k_main_ptr,
        k_extra_ptr,
        indices.data_ptr<int64_t>(),
        has_extra ? extra_indices.value().data_ptr<int64_t>() : nullptr,
        tl_main_ptr,
        tl_extra_ptr,
        attn_sink.has_value() ? attn_sink.value().data_ptr<float>() : nullptr,
        buffer.data_ptr<at::BFloat16>(),
        B,
        S_q,
        H_q,
        D_qk,
        D_v,
        topk_main,
        topk_extra,
        total_tokens_main,
        total_tokens_extra,
        q_strideB,
        q_strideS,
        q_strideH,
        k_main_strideN,
        k_extra_strideN,
        idx_strideB,
        idx_strideS,
        extra_idx_strideB,
        extra_idx_strideS,
        static_cast<float>(softmax_scale),
        buffer_size_per_thread);
  }

  return std::make_tuple(out, lse);
}

// Note: operator registration lives in torch_extension_cpu.cpp like the
// other CPU attention kernels.
