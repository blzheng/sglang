#include "common.h"
#include <cstring>
#include <immintrin.h>

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
