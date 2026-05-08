#include "common.h"
#include <cmath>
#include <cstring>
#include <immintrin.h>


// ============================================================================
// quant_to_nope_fp8_rope_bf16_pack_cpu
//
// Equivalent to Python:
//   k_nope_bf16, k_rope_bf16 = k_bf16.split([448, 64], dim=-1)
//   x = k_nope_bf16.view(-1, 7, 64)
//   scale = x.abs().amax(dim=-1).float() / 448.0
//   scale_pow2 = 2^ceil(log2(max(scale, 1e-4)))
//   k_nope_fp8 = (x.float() / scale_pow2.unsqueeze(-1)).to(fp8_e4m3fn)
//   scale_uint8 = e8m0_encode(scale_pow2) -> (ceil_log2 + 127)
// ============================================================================

namespace {

constexpr int64_t QUANT_DIM_NOPE = 448;
constexpr int64_t QUANT_DIM_ROPE = 64;
constexpr int64_t QUANT_TILE_SIZE = 64;
constexpr int64_t QUANT_NUM_TILES = QUANT_DIM_NOPE / QUANT_TILE_SIZE;  // 7
constexpr float QUANT_FP8_MAX = 448.0f;
constexpr float QUANT_FP8_MIN = -448.0f;
constexpr float QUANT_EPS = 1e-4f;

// Float to fp8_e4m3fn using PyTorch's conversion for correctness
inline uint8_t float_to_fp8_e4m3fn(float val) {
  return at::Float8_e4m3fn(val).x;
}

#if defined(CPU_CAPABILITY_AVX512)

// Process one tile (64 bf16 values): find amax, compute scale, quantize to fp8
inline uint8_t quantize_tile_avx512(
    const at::BFloat16* __restrict__ src,
    uint8_t* __restrict__ dst) {

  // Load 64 bf16 -> 4 x 16 floats
  __m256i bf16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
  __m256i bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 16));
  __m256i bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
  __m256i bf16_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 48));

  __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_0), 16));
  __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_1), 16));
  __m512 f2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_2), 16));
  __m512 f3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_3), 16));

  // Absolute values
  const __m512i abs_mask_i = _mm512_set1_epi32(0x7FFFFFFF);
  __m512 abs0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask_i));
  __m512 abs1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask_i));
  __m512 abs2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask_i));
  __m512 abs3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask_i));

  // Horizontal max
  __m512 max01 = _mm512_max_ps(abs0, abs1);
  __m512 max23 = _mm512_max_ps(abs2, abs3);
  __m512 max0123 = _mm512_max_ps(max01, max23);
  float amax = _mm512_reduce_max_ps(max0123);

  // Scale computation
  float scale = std::max(amax / QUANT_FP8_MAX, QUANT_EPS);
  float ceil_log2 = std::ceil(std::log2(scale));
  float scale_pow2 = std::exp2(ceil_log2);
  float scale_inv = 1.0f / scale_pow2;

  int exponent = static_cast<int>(ceil_log2);
  uint8_t scale_uint8 = static_cast<uint8_t>(exponent + 127);

  // Scale all values
  __m512 vinv = _mm512_set1_ps(scale_inv);
  __m512 s0 = _mm512_mul_ps(f0, vinv);
  __m512 s1 = _mm512_mul_ps(f1, vinv);
  __m512 s2 = _mm512_mul_ps(f2, vinv);
  __m512 s3 = _mm512_mul_ps(f3, vinv);

  // Clamp
  __m512 vmax = _mm512_set1_ps(QUANT_FP8_MAX);
  __m512 vmin = _mm512_set1_ps(QUANT_FP8_MIN);
  s0 = _mm512_max_ps(_mm512_min_ps(s0, vmax), vmin);
  s1 = _mm512_max_ps(_mm512_min_ps(s1, vmax), vmin);
  s2 = _mm512_max_ps(_mm512_min_ps(s2, vmax), vmin);
  s3 = _mm512_max_ps(_mm512_min_ps(s3, vmax), vmin);

  // Convert float -> fp8
  alignas(64) float buf[16];

  _mm512_store_ps(buf, s0);
  for (int j = 0; j < 16; ++j) dst[j] = float_to_fp8_e4m3fn(buf[j]);

  _mm512_store_ps(buf, s1);
  for (int j = 0; j < 16; ++j) dst[16 + j] = float_to_fp8_e4m3fn(buf[j]);

  _mm512_store_ps(buf, s2);
  for (int j = 0; j < 16; ++j) dst[32 + j] = float_to_fp8_e4m3fn(buf[j]);

  _mm512_store_ps(buf, s3);
  for (int j = 0; j < 16; ++j) dst[48 + j] = float_to_fp8_e4m3fn(buf[j]);

  return scale_uint8;
}

#endif  // CPU_CAPABILITY_AVX512

inline uint8_t quantize_tile_scalar(
    const at::BFloat16* __restrict__ src,
    uint8_t* __restrict__ dst) {

  float amax = 0.0f;
  for (int j = 0; j < QUANT_TILE_SIZE; ++j) {
    float v = std::abs(static_cast<float>(src[j]));
    if (v > amax) amax = v;
  }

  float scale = std::max(amax / QUANT_FP8_MAX, QUANT_EPS);
  float ceil_log2 = std::ceil(std::log2(scale));
  float scale_pow2 = std::exp2(ceil_log2);
  float scale_inv = 1.0f / scale_pow2;

  int exponent = static_cast<int>(ceil_log2);
  uint8_t scale_uint8 = static_cast<uint8_t>(exponent + 127);

  for (int j = 0; j < QUANT_TILE_SIZE; ++j) {
    float val = static_cast<float>(src[j]) * scale_inv;
    val = std::max(std::min(val, QUANT_FP8_MAX), QUANT_FP8_MIN);
    dst[j] = float_to_fp8_e4m3fn(val);
  }

  return scale_uint8;
}

}  // anonymous namespace

// set_k_and_s_cpu: scatter-copy k_nope (fp8), k_rope (bf16), and
// scale_k_nope (uint8) into a paged KV-cache buffer.
//
// Buffer layout per page (buf shape: [num_pages, buf_numel_per_page]):
//   Token data region: page_size * (nope_dim + rope_dim*2) bytes
//     Per token: [nope_dim bytes fp8 | rope_dim*2 bytes bf16]
//   Scale region: page_size * (scale_dim + 1) bytes
//     Per token: [scale_dim bytes | 1 byte padding]
//
void set_k_and_s_cpu(
    at::Tensor& buf,         // [num_pages, buf_numel_per_page], uint8
    at::Tensor& loc,         // [num_tokens], int32 or int64
    at::Tensor& k_nope,      // [num_tokens, nope_dim], fp8 (viewed as uint8)
    at::Tensor& k_rope,      // [num_tokens, rope_dim], bf16 (viewed as uint8, rope_dim*2 bytes)
    at::Tensor& scale_k_nope,// [num_tokens, scale_dim], uint8
    int64_t page_size) {
  const int64_t num_tokens = loc.size(0);
  const int64_t buf_numel_per_page = buf.size(1);
  const int64_t nope_dim = k_nope.size(1);       // 448 (bytes, fp8 elements)
  const int64_t rope_dim = k_rope.size(1);        // 64 (bf16 elements, 128 bytes)
  const int64_t scale_dim = scale_k_nope.size(1); // 7

  const int64_t nope_rope_bytes = nope_dim + rope_dim * 2;
  const int64_t s_offset_nbytes_in_page = page_size * nope_rope_bytes;
  const int64_t padded_scale_per_token = scale_dim + 1;

  uint8_t* buf_ptr = buf.data_ptr<uint8_t>();
  const uint8_t* k_nope_ptr = reinterpret_cast<const uint8_t*>(k_nope.data_ptr());
  const uint8_t* k_rope_ptr = reinterpret_cast<const uint8_t*>(k_rope.data_ptr());
  const uint8_t* scale_ptr = scale_k_nope.data_ptr<uint8_t>();

  // We handle both int32 and int64 loc
  const bool loc_is_int64 = (loc.scalar_type() == at::kLong);
  const int32_t* loc_i32 = loc_is_int64 ? nullptr : loc.data_ptr<int32_t>();
  const int64_t* loc_i64 = loc_is_int64 ? loc.data_ptr<int64_t>() : nullptr;

  at::parallel_for(0, num_tokens, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t token_loc = loc_is_int64 ? loc_i64[i] : static_cast<int64_t>(loc_i32[i]);
      const int64_t page_idx = token_loc / page_size;
      const int64_t token_off = token_loc % page_size;

      // --- nope: copy nope_dim bytes (fp8) ---
      const int64_t nope_byte_offset =
          page_idx * buf_numel_per_page + token_off * nope_rope_bytes;
      uint8_t* dst_nope = buf_ptr + nope_byte_offset;
      const uint8_t* src_nope = k_nope_ptr + i * nope_dim;

#if defined(CPU_CAPABILITY_AVX512)
      // nope_dim=448: 7 x 64-byte AVX512 stores
      {
        int64_t d = 0;
        for (; d + 64 <= nope_dim; d += 64) {
          __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src_nope + d));
          _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_nope + d), v);
        }
        // Handle remainder (nope_dim % 64 == 0 for 448, but be safe)
        if (d < nope_dim) {
          std::memcpy(dst_nope + d, src_nope + d, nope_dim - d);
        }
      }
#else
      std::memcpy(dst_nope, src_nope, nope_dim);
#endif

      // --- rope: copy rope_dim bf16 values = rope_dim*2 bytes ---
      const int64_t rope_byte_offset =
          page_idx * buf_numel_per_page + token_off * nope_rope_bytes + nope_dim;
      uint8_t* dst_rope = buf_ptr + rope_byte_offset;
      const uint8_t* src_rope = k_rope_ptr + i * rope_dim * 2;
      const int64_t rope_bytes = rope_dim * 2;

#if defined(CPU_CAPABILITY_AVX512)
      // rope_bytes=128: 2 x 64-byte AVX512 stores
      {
        int64_t d = 0;
        for (; d + 64 <= rope_bytes; d += 64) {
          __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src_rope + d));
          _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_rope + d), v);
        }
        if (d < rope_bytes) {
          std::memcpy(dst_rope + d, src_rope + d, rope_bytes - d);
        }
      }
#else
      std::memcpy(dst_rope, src_rope, rope_bytes);
#endif

      // --- scale: copy scale_dim bytes ---
      const int64_t s_byte_offset =
          page_idx * buf_numel_per_page + s_offset_nbytes_in_page
          + token_off * padded_scale_per_token;
      uint8_t* dst_scale = buf_ptr + s_byte_offset;
      const uint8_t* src_scale = scale_ptr + i * scale_dim;
      std::memcpy(dst_scale, src_scale, scale_dim);
    }
  });
}


std::tuple<at::Tensor, at::Tensor, at::Tensor>
quant_to_nope_fp8_rope_bf16_pack_cpu(at::Tensor& k_bf16) {
  TORCH_CHECK(k_bf16.dtype() == at::kBFloat16,
      "quant_to_nope_fp8_rope_bf16_pack_cpu: expect bf16 input, got ", k_bf16.dtype());
  TORCH_CHECK(k_bf16.dim() == 2 && k_bf16.size(1) == 512,
      "quant_to_nope_fp8_rope_bf16_pack_cpu: expect input shape [N, 512]");
  TORCH_CHECK(k_bf16.is_contiguous(),
      "quant_to_nope_fp8_rope_bf16_pack_cpu: expect contiguous input");

  const int64_t num_tokens = k_bf16.size(0);

  auto k_nope_fp8 = at::empty({num_tokens, QUANT_DIM_NOPE},
      k_bf16.options().dtype(at::kFloat8_e4m3fn));
  auto k_rope_bf16 = at::empty({num_tokens, QUANT_DIM_ROPE},
      k_bf16.options().dtype(at::kBFloat16));
  auto scale_out = at::empty({num_tokens, QUANT_NUM_TILES},
      k_bf16.options().dtype(at::kByte));

  const at::BFloat16* input_ptr = k_bf16.data_ptr<at::BFloat16>();
  uint8_t* nope_ptr = reinterpret_cast<uint8_t*>(k_nope_fp8.data_ptr());
  at::BFloat16* rope_ptr = k_rope_bf16.data_ptr<at::BFloat16>();
  uint8_t* scale_ptr = scale_out.data_ptr<uint8_t>();

  at::parallel_for(0, num_tokens, /*grain_size=*/64, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const at::BFloat16* src_token = input_ptr + t * 512;

      // Copy rope portion (last 64 bf16 values = 128 bytes)
      std::memcpy(rope_ptr + t * QUANT_DIM_ROPE,
                  src_token + QUANT_DIM_NOPE,
                  QUANT_DIM_ROPE * sizeof(at::BFloat16));

      // Quantize each nope tile
      for (int64_t tile = 0; tile < QUANT_NUM_TILES; ++tile) {
        const at::BFloat16* tile_src = src_token + tile * QUANT_TILE_SIZE;
        uint8_t* tile_dst = nope_ptr + t * QUANT_DIM_NOPE + tile * QUANT_TILE_SIZE;

#if defined(CPU_CAPABILITY_AVX512)
        scale_ptr[t * QUANT_NUM_TILES + tile] = quantize_tile_avx512(tile_src, tile_dst);
#else
        scale_ptr[t * QUANT_NUM_TILES + tile] = quantize_tile_scalar(tile_src, tile_dst);
#endif
      }
    }
  });

  return std::make_tuple(k_nope_fp8, k_rope_bf16, scale_out);
}
