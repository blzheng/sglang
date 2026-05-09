/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "common.h"
#include "vec.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace {

using fVec = at::vec::Vectorized<float>;

// ──────────────────────────── helpers ────────────────────────────

static inline int64_t compute_state_len(int64_t seq_len, int64_t ratio) {
  return seq_len % ratio + (ratio == 4 ? ratio : 0);
}

// In-place RMS norm on a single row of float data with weight.
static inline void rmsnorm_row(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int64_t dim,
    float eps) {
  float sum_sq = 0.0f;
  int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
  fVec sum_v(0.0f);
  for (; d <= dim - fVec::size(); d += fVec::size()) {
    fVec x = fVec::loadu(input + d);
    sum_v = sum_v + x * x;
  }
  sum_sq = vec_reduce_sum(sum_v);
#endif
  for (; d < dim; ++d) {
    sum_sq += input[d] * input[d];
  }
  float rsqrt_var = 1.0f / std::sqrt(sum_sq / dim + eps);

  d = 0;
#if defined(CPU_CAPABILITY_AVX512)
  fVec scale_v(rsqrt_var);
  for (; d <= dim - fVec::size(); d += fVec::size()) {
    fVec x = fVec::loadu(input + d);
    fVec w = fVec::loadu(weight + d);
    (x * scale_v * w).store(out + d);
  }
#endif
  for (; d < dim; ++d) {
    out[d] = input[d] * rsqrt_var * weight[d];
  }
}

// In-place interleaved rotary embedding on a single row.
// freqs is [rope_dim] float, x is [rope_dim] float.
static inline void apply_rotary_emb_row_f32(
    float* __restrict__ x,
    const float* __restrict__ freqs,
    int64_t rope_dim,
    bool inverse) {
  int64_t k = 0;
#if defined(CPU_CAPABILITY_AVX512)
  // Build sign mask for complex multiply
  __m512 sign_mask;
  if (inverse) {
    sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
        (int)0x80000000, 0, (int)0x80000000, 0,
        (int)0x80000000, 0, (int)0x80000000, 0,
        (int)0x80000000, 0, (int)0x80000000, 0,
        (int)0x80000000, 0, (int)0x80000000, 0));
  } else {
    sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
        0, (int)0x80000000, 0, (int)0x80000000,
        0, (int)0x80000000, 0, (int)0x80000000,
        0, (int)0x80000000, 0, (int)0x80000000,
        0, (int)0x80000000, 0, (int)0x80000000));
  }
  for (; k <= rope_dim - 16; k += 16) {
    __m512 xv = _mm512_loadu_ps(x + k);
    __m512 fv = _mm512_loadu_ps(freqs + k);
    __m512 out = _mm512_fmadd_ps(
        xv,
        _mm512_permute_ps(fv, 0xA0),
        _mm512_mul_ps(
            _mm512_permute_ps(xv, 0xB1),
            _mm512_xor_ps(_mm512_permute_ps(fv, 0xF5), sign_mask)));
    _mm512_storeu_ps(x + k, out);
  }
#endif
  for (; k < rope_dim; k += 2) {
    float xr = x[k], xi = x[k + 1];
    float cr = freqs[k], ci = freqs[k + 1];
    if (inverse) {
      x[k]     = xr * cr + xi * ci;
      x[k + 1] = xi * cr - xr * ci;
    } else {
      x[k]     = xr * cr - xi * ci;
      x[k + 1] = xr * ci + xi * cr;
    }
  }
}

// Softmax along dim=0 of a column across `n` rows, applied to `score[n]`,
// writes softmax weights to `weights[n]`.
static inline void softmax_col(
    float* __restrict__ weights,
    const float* __restrict__ score,
    int64_t n) {
  float max_val = score[0];
  for (int64_t i = 1; i < n; ++i) {
    max_val = std::max(max_val, score[i]);
  }
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    weights[i] = std::exp(score[i] - max_val);
    sum += weights[i];
  }
  float inv_sum = 1.0f / sum;
  for (int64_t i = 0; i < n; ++i) {
    weights[i] *= inv_sum;
  }
}

// Hadamard transform for rotate_activation.
// Operates on float buffer of power-of-2 size.
static void hadamard_transform_row(float* __restrict__ data, int64_t n, float scale) {
  if (n == 1) {
    data[0] *= scale;
    return;
  }
#if defined(CPU_CAPABILITY_AVX512)
  if (n >= 16) {
    // Phase 1: fused butterflies for h=1,2,4,8
    for (int64_t j = 0; j < n; j += 16) {
      __m512 v = _mm512_loadu_ps(data + j);
      __m512 p, s, d;
      // h=1
      p = _mm512_permute_ps(v, 0b10'11'00'01);
      s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xAAAA, s, d);
      // h=2
      p = _mm512_permute_ps(v, 0b01'00'11'10);
      s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xCCCC, s, d);
      // h=4
      p = _mm512_permutexvar_ps(
          _mm512_set_epi32(11,10,9,8, 15,14,13,12, 3,2,1,0, 7,6,5,4), v);
      s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xF0F0, s, d);
      // h=8
      p = _mm512_permutexvar_ps(
          _mm512_set_epi32(7,6,5,4, 3,2,1,0, 15,14,13,12, 11,10,9,8), v);
      s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xFF00, s, d);
      _mm512_storeu_ps(data + j, v);
    }
    // Phase 2: larger butterflies
    for (int64_t h = 16; h < n; h <<= 1) {
      for (int64_t j = 0; j < n; j += 2 * h) {
        for (int64_t k = 0; k < h; k += 16) {
          __m512 a = _mm512_loadu_ps(data + j + k);
          __m512 b = _mm512_loadu_ps(data + j + k + h);
          _mm512_storeu_ps(data + j + k, _mm512_add_ps(a, b));
          _mm512_storeu_ps(data + j + k + h, _mm512_sub_ps(a, b));
        }
      }
    }
    // Apply scale
    __m512 sv = _mm512_set1_ps(scale);
    for (int64_t j = 0; j < n; j += 16) {
      __m512 v = _mm512_loadu_ps(data + j);
      _mm512_storeu_ps(data + j, _mm512_mul_ps(v, sv));
    }
    return;
  }
#endif
  // Scalar fallback
  int64_t h = 1;
  while (h < n) {
    for (int64_t j = 0; j < n; j += 2 * h) {
      for (int64_t k = 0; k < h; ++k) {
        float a = data[j + k], b = data[j + k + h];
        data[j + k]     = a + b;
        data[j + k + h] = a - b;
      }
    }
    h <<= 1;
  }
  for (int64_t j = 0; j < n; ++j) {
    data[j] *= scale;
  }
}

// overlap_transform_decode: tensor [bs, 2*ratio, 2*head_dim] -> [bs, 2*ratio, head_dim]
// ret[:, :r, :] = tensor[:, :r, :d]
// ret[:, r:, :] = tensor[:, r:, d:]
// where r = ratio, d = head_dim
static void overlap_transform_decode_inplace(
    float* __restrict__ out,  // [n_items, head_dim]
    const float* __restrict__ in,  // [n_items, 2*head_dim]
    int64_t ratio,
    int64_t head_dim) {
  // First half: copy in[:ratio, :head_dim]
  for (int64_t i = 0; i < ratio; ++i) {
    const float* src = in + i * 2 * head_dim;
    float* dst = out + i * head_dim;
    std::memcpy(dst, src, head_dim * sizeof(float));
  }
  // Second half: copy in[ratio:2*ratio, head_dim:2*head_dim]
  for (int64_t i = ratio; i < 2 * ratio; ++i) {
    const float* src = in + i * 2 * head_dim + head_dim;
    float* dst = out + i * head_dim;
    std::memcpy(dst, src, head_dim * sizeof(float));
  }
}

// ──────────────────── compress_decode_cpu ────────────────────

// Equivalent to compress_decode_old in Python.
// pool_kv/pool_score: [max_reqs, state_len, coff*head_dim]
// kv/score: [bs, coff*head_dim]
// seq_lens: [bs] int64
// req_pool_indices: [bs] int64
// ape: [ratio, coff*head_dim] float32
// norm_weight: [head_dim] float32
// freqs_cis: [max_seq, rope_dim] float32 (real-valued interleaved cos/sin)
void compress_decode_cpu_impl(
    float* __restrict__ pool_kv,
    float* __restrict__ pool_score,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ req_pool_indices,
    const float* __restrict__ ape,
    const float* __restrict__ norm_weight,
    const float* __restrict__ freqs_cis,
    float* __restrict__ output,  // [bs, head_dim]
    int64_t bs,
    int64_t pool_state_len,
    int64_t pool_row_stride,  // stride between rows in pool (may be > coff_hd for interleaved storage)
    int64_t kv_row_stride,    // stride between rows in kv/score input
    int64_t ratio,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t coff,
    bool overlap,
    bool rotate,
    float norm_eps,
    int64_t freqs_stride) {

  int64_t coff_hd = coff * head_dim;

  at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
    // Scratch buffers per thread
    std::vector<float> kv_buf(ratio * coff * coff_hd);
    std::vector<float> score_buf(ratio * coff * coff_hd);
    std::vector<float> kv_work(ratio * coff * head_dim);
    std::vector<float> score_work(ratio * coff * head_dim);
    std::vector<float> softmax_weights(ratio * coff);
    std::vector<float> compressed(head_dim);

    for (int64_t b = begin; b < end; ++b) {
      int64_t seq_len = seq_lens[b];
      int64_t req_idx = req_pool_indices[b];
      int64_t write_pos = (seq_len - 1) % ratio + (overlap ? ratio : 0);

      float* pool_kv_req = pool_kv + req_idx * pool_state_len * pool_row_stride;
      float* pool_score_req = pool_score + req_idx * pool_state_len * pool_row_stride;

      // Write new kv and score to pool at write_pos
      std::memcpy(pool_kv_req + write_pos * pool_row_stride, kv + b * kv_row_stride, coff_hd * sizeof(float));
      std::memcpy(pool_score_req + write_pos * pool_row_stride, score + b * kv_row_stride, coff_hd * sizeof(float));

      // Copy out entire pool state for this request (strided -> contiguous)
      int64_t total_state = ratio * coff;
      for (int64_t r = 0; r < total_state; ++r) {
        std::memcpy(kv_buf.data() + r * coff_hd, pool_kv_req + r * pool_row_stride, coff_hd * sizeof(float));
        std::memcpy(score_buf.data() + r * coff_hd, pool_score_req + r * pool_row_stride, coff_hd * sizeof(float));
      }

      // Handle overlap shift
      if (overlap && (seq_len % ratio == 0)) {
        // Shift: pool[:ratio] = pool[ratio:2*ratio]
        for (int64_t r = 0; r < ratio; ++r) {
          std::memcpy(pool_kv_req + r * pool_row_stride,
                      pool_kv_req + (ratio + r) * pool_row_stride,
                      coff_hd * sizeof(float));
          std::memcpy(pool_score_req + r * pool_row_stride,
                      pool_score_req + (ratio + r) * pool_row_stride,
                      coff_hd * sizeof(float));
        }
      }

      // Add APE to score
      // kv_buf/score_buf shape: [coff, ratio, coff_hd] (after reshape from [total_state, coff_hd])
      // APE shape: [ratio, coff_hd]
      for (int64_t c = 0; c < coff; ++c) {
        for (int64_t r = 0; r < ratio; ++r) {
          float* sp = score_buf.data() + (c * ratio + r) * coff_hd;
          const float* ap = ape + r * coff_hd;
          int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
          for (; d <= coff_hd - 16; d += 16) {
            __m512 sv = _mm512_loadu_ps(sp + d);
            __m512 av = _mm512_loadu_ps(ap + d);
            _mm512_storeu_ps(sp + d, _mm512_add_ps(sv, av));
          }
#endif
          for (; d < coff_hd; ++d) {
            sp[d] += ap[d];
          }
        }
      }

      if (overlap) {
        // overlap_transform_decode on kv and score
        // Input: [bs=1, coff*ratio=2*ratio, coff*head_dim=2*head_dim]
        // Output: [bs=1, 2*ratio, head_dim]
        overlap_transform_decode_inplace(kv_work.data(), kv_buf.data(), ratio, head_dim);
        overlap_transform_decode_inplace(score_work.data(), score_buf.data(), ratio, head_dim);
      } else {
        // Just copy, treating as [ratio, head_dim]
        for (int64_t r = 0; r < ratio; ++r) {
          std::memcpy(kv_work.data() + r * head_dim, kv_buf.data() + r * coff_hd, head_dim * sizeof(float));
          std::memcpy(score_work.data() + r * head_dim, score_buf.data() + r * coff_hd, head_dim * sizeof(float));
        }
      }

      // Now: kv_work, score_work are [ratio*coff, head_dim]
      int64_t compress_rows = ratio * coff;

      // Softmax over score per head_dim column, then weighted sum
      // For each dim d: softmax(score[:, d]) then sum(kv[:, d] * softmax_weight)
      // But actually: softmax is over dim=1 (the ratio*coff rows) for each element
      // Wait - looking at the Python code more carefully:
      // kv_and_score_to_compress.score.softmax(dim=1)
      // The shapes are [bs, ratio*coff, head_dim]
      // So softmax is over axis 1 (ratio*coff) for each head_dim independently
      // Then: (kv * softmax_weights).sum(dim=1) -> [bs, head_dim]

      // For each head_dim position:
      for (int64_t d = 0; d < head_dim; ++d) {
        // Gather scores for this dim across all rows
        float max_val = -1e30f;
        for (int64_t r = 0; r < compress_rows; ++r) {
          float s = score_work[r * head_dim + d];
          if (s > max_val) max_val = s;
        }
        float sum_exp = 0.0f;
        for (int64_t r = 0; r < compress_rows; ++r) {
          softmax_weights[r] = std::exp(score_work[r * head_dim + d] - max_val);
          sum_exp += softmax_weights[r];
        }
        float inv_sum = 1.0f / sum_exp;
        float weighted_sum = 0.0f;
        for (int64_t r = 0; r < compress_rows; ++r) {
          weighted_sum += kv_work[r * head_dim + d] * softmax_weights[r] * inv_sum;
        }
        compressed[d] = weighted_sum;
      }

      // RMS norm with weight
      rmsnorm_row(output + b * head_dim, compressed.data(), norm_weight, head_dim, norm_eps);

      // Apply rotary embedding to last rope_head_dim elements
      int64_t freq_pos = (seq_len - 1) / ratio * ratio;
      const float* freq_ptr = freqs_cis + freq_pos * freqs_stride;
      apply_rotary_emb_row_f32(
          output + b * head_dim + (head_dim - rope_head_dim),
          freq_ptr,
          rope_head_dim,
          false);

      // Optional rotate (Hadamard transform)
      if (rotate) {
        float scale = std::pow((float)head_dim, -0.5f);
        hadamard_transform_row(output + b * head_dim, head_dim, scale);
      }
    }
  });
}

// ──────────────────── compress_extend_cpu ────────────────────

void compress_extend_cpu_impl(
    float* __restrict__ pool_kv,
    float* __restrict__ pool_score,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const int64_t* __restrict__ prefix_lens,
    const int64_t* __restrict__ extend_lens,
    const int64_t* __restrict__ req_pool_indices,
    const float* __restrict__ ape,
    const float* __restrict__ norm_weight,
    const float* __restrict__ freqs_cis,
    float* __restrict__ output,  // [total_tokens, head_dim], filled with 10000.0
    int64_t bs,
    int64_t pool_state_len,
    int64_t pool_row_stride,
    int64_t kv_row_stride,
    int64_t ratio,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t coff,
    bool overlap,
    bool rotate,
    float norm_eps,
    int64_t freqs_stride,
    int64_t total_tokens) {

  int64_t coff_hd = coff * head_dim;

  // Sequential over batch (matches Python loop)
  int64_t pt = 0;
  for (int64_t i = 0; i < bs; ++i) {
    int64_t prefix_len = prefix_lens[i];
    int64_t extend_len = extend_lens[i];
    int64_t req_idx = req_pool_indices[i];

    float* state_kv = pool_kv + req_idx * pool_state_len * pool_row_stride;
    float* state_score = pool_score + req_idx * pool_state_len * pool_row_stride;

    if (prefix_len == 0) {
      // Clear state: kv=0, score=-inf (strided)
      for (int64_t r = 0; r < pool_state_len; ++r) {
        for (int64_t d = 0; d < coff_hd; ++d) {
          state_kv[r * pool_row_stride + d] = 0.0f;
          state_score[r * pool_row_stride + d] = -std::numeric_limits<float>::infinity();
        }
      }
    }

    int64_t pre_state_len = compute_state_len(prefix_len, ratio);
    int64_t valid_kv_len = pre_state_len + extend_len;

    // Build buffer: [valid_kv_len, coff_hd] (contiguous)
    std::vector<float> buf_kv(valid_kv_len * coff_hd);
    std::vector<float> buf_score(valid_kv_len * coff_hd);

    // Copy pre-state from pool (strided -> contiguous)
    for (int64_t r = 0; r < pre_state_len; ++r) {
      std::memcpy(buf_kv.data() + r * coff_hd, state_kv + r * pool_row_stride, coff_hd * sizeof(float));
      std::memcpy(buf_score.data() + r * coff_hd, state_score + r * pool_row_stride, coff_hd * sizeof(float));
    }

    // Copy extend data (strided -> contiguous)
    for (int64_t r = 0; r < extend_len; ++r) {
      std::memcpy(buf_kv.data() + (pre_state_len + r) * coff_hd,
                  kv + (pt + r) * kv_row_stride,
                  coff_hd * sizeof(float));
      std::memcpy(buf_score.data() + (pre_state_len + r) * coff_hd,
                  score + (pt + r) * kv_row_stride,
                  coff_hd * sizeof(float));
    }

    // Save post-state back to pool (contiguous -> strided)
    int64_t post_state_len = compute_state_len(valid_kv_len, ratio);
    if (post_state_len > 0) {
      int64_t start = valid_kv_len - post_state_len;
      for (int64_t r = 0; r < post_state_len; ++r) {
        std::memcpy(state_kv + r * pool_row_stride,
                    buf_kv.data() + (start + r) * coff_hd,
                    coff_hd * sizeof(float));
        std::memcpy(state_score + r * pool_row_stride,
                    buf_score.data() + (start + r) * coff_hd,
                    coff_hd * sizeof(float));
      }
    }

    int64_t compress_len = (valid_kv_len / ratio) * ratio;
    if (compress_len == 0) {
      pt += extend_len;
      continue;
    }

    // Reshape to [compress_len/ratio, ratio, coff_hd] and add APE
    int64_t num_groups = compress_len / ratio;
    for (int64_t g = 0; g < num_groups; ++g) {
      for (int64_t r = 0; r < ratio; ++r) {
        float* sp = buf_score.data() + (g * ratio + r) * coff_hd;
        const float* ap = ape + r * coff_hd;
        int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
        for (; d <= coff_hd - 16; d += 16) {
          __m512 sv = _mm512_loadu_ps(sp + d);
          __m512 av = _mm512_loadu_ps(ap + d);
          _mm512_storeu_ps(sp + d, _mm512_add_ps(sv, av));
        }
#endif
        for (; d < coff_hd; ++d) {
          sp[d] += ap[d];
        }
      }
    }

    int64_t out_groups = num_groups;

    if (overlap) {
      // overlap_transform on kv and score
      // For kv, fill_value=0; for score, fill_value=-inf
      // Input: [num_groups, ratio, 2*head_dim]
      // Output: [num_groups, 2*ratio, head_dim]
      std::vector<float> ot_kv(num_groups * 2 * ratio * head_dim, 0.0f);
      std::vector<float> ot_score(num_groups * 2 * ratio * head_dim, -std::numeric_limits<float>::infinity());

      for (int64_t g = 0; g < num_groups; ++g) {
        // new_tensor[:, r:] = tensor[:, :, d:]
        for (int64_t r = 0; r < ratio; ++r) {
          const float* kv_src = buf_kv.data() + (g * ratio + r) * coff_hd + head_dim;
          float* kv_dst = ot_kv.data() + g * 2 * ratio * head_dim + (ratio + r) * head_dim;
          std::memcpy(kv_dst, kv_src, head_dim * sizeof(float));

          const float* sc_src = buf_score.data() + (g * ratio + r) * coff_hd + head_dim;
          float* sc_dst = ot_score.data() + g * 2 * ratio * head_dim + (ratio + r) * head_dim;
          std::memcpy(sc_dst, sc_src, head_dim * sizeof(float));
        }
        // new_tensor[1:, :r] = tensor[:-1, :, :d]
        if (g > 0) {
          for (int64_t r = 0; r < ratio; ++r) {
            const float* kv_src = buf_kv.data() + ((g - 1) * ratio + r) * coff_hd;
            float* kv_dst = ot_kv.data() + g * 2 * ratio * head_dim + r * head_dim;
            std::memcpy(kv_dst, kv_src, head_dim * sizeof(float));

            const float* sc_src = buf_score.data() + ((g - 1) * ratio + r) * coff_hd;
            float* sc_dst = ot_score.data() + g * 2 * ratio * head_dim + r * head_dim;
            std::memcpy(sc_dst, sc_src, head_dim * sizeof(float));
          }
        }
      }

      // Drop leading window: skip group 0
      out_groups = num_groups - 1;
      if (out_groups <= 0) {
        pt += extend_len;
        continue;
      }

      // Compress each remaining group
      std::vector<float> compressed(head_dim);
      std::vector<float> softmax_weights(2 * ratio);

      for (int64_t g = 0; g < out_groups; ++g) {
        int64_t src_g = g + 1;  // skip first group
        int64_t compress_rows = 2 * ratio;
        const float* kv_g = ot_kv.data() + src_g * compress_rows * head_dim;
        const float* sc_g = ot_score.data() + src_g * compress_rows * head_dim;

        for (int64_t d = 0; d < head_dim; ++d) {
          float max_val = -1e30f;
          for (int64_t r = 0; r < compress_rows; ++r) {
            float s = sc_g[r * head_dim + d];
            if (s > max_val) max_val = s;
          }
          float sum_exp = 0.0f;
          for (int64_t r = 0; r < compress_rows; ++r) {
            softmax_weights[r] = std::exp(sc_g[r * head_dim + d] - max_val);
            sum_exp += softmax_weights[r];
          }
          float inv_sum = 1.0f / sum_exp;
          float wsum = 0.0f;
          for (int64_t r = 0; r < compress_rows; ++r) {
            wsum += kv_g[r * head_dim + d] * softmax_weights[r] * inv_sum;
          }
          compressed[d] = wsum;
        }

        // RMS norm
        std::vector<float> normed(head_dim);
        rmsnorm_row(normed.data(), compressed.data(), norm_weight, head_dim, norm_eps);

        // RoPE
        int64_t beg_idx = (prefix_len / ratio) * ratio;
        int64_t freq_idx = beg_idx + g * ratio;
        const float* freq_ptr = freqs_cis + freq_idx * freqs_stride;
        apply_rotary_emb_row_f32(
            normed.data() + (head_dim - rope_head_dim),
            freq_ptr,
            rope_head_dim,
            false);

        if (rotate) {
          float scale = std::pow((float)head_dim, -0.5f);
          hadamard_transform_row(normed.data(), head_dim, scale);
        }

        // Write to output at correct position
        int64_t start = prefix_len;
        start = start + ratio - 1 - start % ratio;
        int64_t out_seq_idx = start + (g + 1) * ratio;
        // Actually: indices_in_seq = torch.arange(start, prefix_lens[i] + extend_lens[i], ratio)
        // and then after overlap drop, kv_compressed has out_groups entries
        // Let me recalculate: before drop, we have num_groups compressed outputs
        // After drop (skip first), we have out_groups = num_groups - 1
        // indices_in_seq has the same count as kv_compressed.size(0) = out_groups
        // indices_in_seq = [start, start+ratio, start+2*ratio, ...]
        // So index g -> start + g*ratio
        int64_t out_idx = start + g * ratio;
        if (out_idx < prefix_len + extend_len && out_idx >= prefix_len) {
          int64_t flat_idx = out_idx - prefix_len + pt;
          if (flat_idx < total_tokens) {
            std::memcpy(output + flat_idx * head_dim, normed.data(), head_dim * sizeof(float));
          }
        }
      }
    } else {
      // No overlap case
      std::vector<float> compressed(head_dim);
      std::vector<float> softmax_weights(ratio);
      std::vector<float> normed(head_dim);

      for (int64_t g = 0; g < num_groups; ++g) {
        int64_t compress_rows = ratio;

        for (int64_t d = 0; d < head_dim; ++d) {
          float max_val = -1e30f;
          for (int64_t r = 0; r < compress_rows; ++r) {
            float s = buf_score[(g * ratio + r) * coff_hd + d];
            if (s > max_val) max_val = s;
          }
          float sum_exp = 0.0f;
          for (int64_t r = 0; r < compress_rows; ++r) {
            softmax_weights[r] = std::exp(buf_score[(g * ratio + r) * coff_hd + d] - max_val);
            sum_exp += softmax_weights[r];
          }
          float inv_sum = 1.0f / sum_exp;
          float wsum = 0.0f;
          for (int64_t r = 0; r < compress_rows; ++r) {
            wsum += buf_kv[(g * ratio + r) * coff_hd + d] * softmax_weights[r] * inv_sum;
          }
          compressed[d] = wsum;
        }

        rmsnorm_row(normed.data(), compressed.data(), norm_weight, head_dim, norm_eps);

        int64_t beg_idx = (prefix_len / ratio) * ratio;
        int64_t freq_idx = beg_idx + g * ratio;
        const float* freq_ptr = freqs_cis + freq_idx * freqs_stride;
        apply_rotary_emb_row_f32(
            normed.data() + (head_dim - rope_head_dim),
            freq_ptr,
            rope_head_dim,
            false);

        if (rotate) {
          float scale = std::pow((float)head_dim, -0.5f);
          hadamard_transform_row(normed.data(), head_dim, scale);
        }

        // Write to output
        int64_t start = prefix_len;
        start = start + ratio - 1 - start % ratio;
        int64_t out_idx = start + g * ratio;
        if (out_idx < prefix_len + extend_len && out_idx >= prefix_len) {
          int64_t flat_idx = out_idx - prefix_len + pt;
          if (flat_idx < total_tokens) {
            std::memcpy(output + flat_idx * head_dim, normed.data(), head_dim * sizeof(float));
          }
        }
      }
    }

    pt += extend_len;
  }
}

}  // anonymous namespace

at::Tensor compress_decode_cpu(
    at::Tensor& pool_kv,
    at::Tensor& pool_score,
    at::Tensor& kv,
    at::Tensor& score,
    at::Tensor& seq_lens,
    at::Tensor& req_pool_indices,
    at::Tensor& ape,
    at::Tensor& norm_weight,
    at::Tensor& freqs_cis,
    int64_t ratio,
    int64_t head_dim,
    int64_t rope_head_dim,
    bool overlap,
    bool rotate,
    double norm_eps) {
  RECORD_FUNCTION("sgl-kernel::compress_decode_cpu", {});

  int64_t bs = kv.size(0);
  int64_t coff = 1 + (overlap ? 1 : 0);
  int64_t pool_state_len = pool_kv.size(1);

  // Ensure float32
  TORCH_CHECK(kv.scalar_type() == at::kFloat, "compress_decode_cpu: kv must be float32");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "compress_decode_cpu: score must be float32");
  TORCH_CHECK(pool_kv.scalar_type() == at::kFloat, "compress_decode_cpu: pool_kv must be float32");

  at::Tensor output = at::empty({bs, head_dim}, kv.options());

  // freqs_cis stride: number of elements per position
  int64_t freqs_stride = freqs_cis.size(-1);
  // Pool row stride (may be > coff_hd if pool is a non-contiguous view)
  int64_t pool_row_stride = pool_kv.stride(-2);
  // kv/score row stride
  int64_t kv_row_stride = kv.stride(0);

  compress_decode_cpu_impl(
      pool_kv.data_ptr<float>(),
      pool_score.data_ptr<float>(),
      kv.data_ptr<float>(),
      score.data_ptr<float>(),
      seq_lens.data_ptr<int64_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      ape.data_ptr<float>(),
      norm_weight.data_ptr<float>(),
      freqs_cis.data_ptr<float>(),
      output.data_ptr<float>(),
      bs,
      pool_state_len,
      pool_row_stride,
      kv_row_stride,
      ratio,
      head_dim,
      rope_head_dim,
      coff,
      overlap,
      rotate,
      static_cast<float>(norm_eps),
      freqs_stride);

  return output;
}

at::Tensor compress_extend_cpu(
    at::Tensor& pool_kv,
    at::Tensor& pool_score,
    at::Tensor& kv,
    at::Tensor& score,
    at::Tensor& prefix_lens,
    at::Tensor& extend_lens,
    at::Tensor& req_pool_indices,
    at::Tensor& ape,
    at::Tensor& norm_weight,
    at::Tensor& freqs_cis,
    int64_t ratio,
    int64_t head_dim,
    int64_t rope_head_dim,
    bool overlap,
    bool rotate,
    double norm_eps) {
  RECORD_FUNCTION("sgl-kernel::compress_extend_cpu", {});

  int64_t total_tokens = kv.size(0);
  int64_t coff = 1 + (overlap ? 1 : 0);
  int64_t pool_state_len = pool_kv.size(1);
  int64_t bs = prefix_lens.size(0);

  TORCH_CHECK(kv.scalar_type() == at::kFloat, "compress_extend_cpu: kv must be float32");

  at::Tensor output = at::full({total_tokens, head_dim}, 10000.0f, kv.options());

  int64_t freqs_stride = freqs_cis.size(-1);
  int64_t pool_row_stride = pool_kv.stride(-2);
  int64_t kv_row_stride = kv.stride(0);

  compress_extend_cpu_impl(
      pool_kv.data_ptr<float>(),
      pool_score.data_ptr<float>(),
      kv.data_ptr<float>(),
      score.data_ptr<float>(),
      prefix_lens.data_ptr<int64_t>(),
      extend_lens.data_ptr<int64_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      ape.data_ptr<float>(),
      norm_weight.data_ptr<float>(),
      freqs_cis.data_ptr<float>(),
      output.data_ptr<float>(),
      bs,
      pool_state_len,
      pool_row_stride,
      kv_row_stride,
      ratio,
      head_dim,
      rope_head_dim,
      coff,
      overlap,
      rotate,
      static_cast<float>(norm_eps),
      freqs_stride,
      total_tokens);

  return output;
}
