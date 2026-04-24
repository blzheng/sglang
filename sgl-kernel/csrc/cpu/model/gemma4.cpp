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

// Forward declaration from gemm.cpp
at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni);

namespace {

// In-place add position embeddings from table lookup into output.
// For non-padding patches:
//   output[b, p, :] += table[0, x_pos, :] + table[1, y_pos, :]
// Padding patches are left unchanged (GEMM result is kept as-is, which
// matches the original code: pos_emb is zero for padding, so add is no-op).
template <typename scalar_t>
void add_position_embeddings_inplace_impl(
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ patch_positions,
    const bool* __restrict__ padding_positions,
    const scalar_t* __restrict__ table,
    int64_t batch_size,
    int64_t num_patches,
    int64_t hidden_size,
    int64_t position_embedding_size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();
  const int64_t table_stride = position_embedding_size * hidden_size;

  at::parallel_for(0, batch_size * num_patches, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      // Skip padding patches — their position_embeddings would be 0
      if (padding_positions[idx]) continue;

      scalar_t* __restrict__ out_ptr = output + idx * hidden_size;
      const int64_t* pos_ptr = patch_positions + idx * 2;
      int64_t x_pos = std::max(pos_ptr[0], static_cast<int64_t>(0));
      int64_t y_pos = std::max(pos_ptr[1], static_cast<int64_t>(0));

      const scalar_t* __restrict__ row_x = table + x_pos * hidden_size;
      const scalar_t* __restrict__ row_y = table + table_stride + y_pos * hidden_size;

      int64_t d = 0;
#pragma GCC unroll 4
      for (; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec o_bvec = bVec::loadu(out_ptr + d);
        bVec x_bvec = bVec::loadu(row_x + d);
        bVec y_bvec = bVec::loadu(row_y + d);

        fVec o0, o1, x0, x1, y0, y1;
        std::tie(o0, o1) = at::vec::convert_to_float(o_bvec);
        std::tie(x0, x1) = at::vec::convert_to_float(x_bvec);
        std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

        o0 = o0 + x0 + y0;
        o1 = o1 + x1 + y1;

        bVec out_bvec = convert_from_float_ext<scalar_t>(o0, o1);
        out_bvec.store(out_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float o = static_cast<float>(out_ptr[d]);
        float x = static_cast<float>(row_x[d]);
        float y = static_cast<float>(row_y[d]);
        out_ptr[d] = static_cast<scalar_t>(o + x + y);
      }
    }
  });
}

}  // anonymous namespace

// Fused Gemma4VisionPatchEmbedder forward:
//   output = linear(2*(pixel_values - 0.5), weight) + position_embeddings(positions, table)
//
// pixel_values:              [batch, num_patches, patch_pixels]
// input_proj_weight:         [hidden_size, patch_pixels] (may be VNNI-packed)
// patch_positions:           [batch, num_patches, 2]  (int64)
// padding_positions:         [batch, num_patches]     (bool)
// position_embedding_table:  [2, position_embedding_size, hidden_size]
// is_vnni:                   whether input_proj_weight is pre-packed
//
// returns: [batch, num_patches, hidden_size]
at::Tensor gemma4_patch_embed_fwd_cpu(
    at::Tensor& pixel_values,
    at::Tensor& input_proj_weight,
    at::Tensor& patch_positions,
    at::Tensor& padding_positions,
    at::Tensor& position_embedding_table,
    bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::gemma4_patch_embed_fwd_cpu", std::vector<c10::IValue>({pixel_values, patch_positions}));

  TORCH_CHECK(pixel_values.dim() == 3, "pixel_values must be [batch, num_patches, patch_pixels]");
  TORCH_CHECK(
      patch_positions.dim() == 3 && patch_positions.size(2) == 2, "patch_positions must be [batch, num_patches, 2]");
  TORCH_CHECK(padding_positions.dim() == 2, "padding_positions must be [batch, num_patches]");
  TORCH_CHECK(
      position_embedding_table.dim() == 3 && position_embedding_table.size(0) == 2,
      "position_embedding_table must be [2, position_embedding_size, hidden_size]");

  int64_t batch_size = pixel_values.size(0);
  int64_t num_patches = pixel_values.size(1);
  int64_t patch_pixels = pixel_values.size(2);
  int64_t hidden_size = position_embedding_table.size(2);
  int64_t pos_emb_size = position_embedding_table.size(1);
  auto dtype = input_proj_weight.scalar_type();

  // Step 1: Fused rescale + cast
  // patches = 2 * (pixel_values - 0.5) = 2 * pixel_values - 1, cast to weight dtype
  auto patches = pixel_values.to(dtype).mul_(2.0).sub_(1.0);
  patches = patches.reshape({batch_size * num_patches, patch_pixels}).contiguous();

  // Step 2: GEMM via weight_packed_linear (handles both VNNI-packed and unpacked)
  auto output = weight_packed_linear(patches, input_proj_weight, std::nullopt, is_vnni);
  output = output.reshape({batch_size, num_patches, hidden_size}).contiguous();

  // Step 3: In-place position embedding addition
  auto patch_pos_c = patch_positions.contiguous();
  auto padding_c = padding_positions.contiguous();
  auto table_c = position_embedding_table.contiguous();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "gemma4_patch_embed_fwd_cpu", [&]() {
    add_position_embeddings_inplace_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        patch_pos_c.data_ptr<int64_t>(),
        padding_c.data_ptr<bool>(),
        table_c.data_ptr<scalar_t>(),
        batch_size,
        num_patches,
        hidden_size,
        pos_emb_size);
  });

  return output;
}

// ============================================================================
// Audio Conformer Kernels
// ============================================================================

// ---- QKV Preprocessing ----

namespace {

template <typename scalar_t>
void gemma4_audio_qkv_preprocess_impl(
    float* __restrict__ query_blocks,
    float* __restrict__ key_blocks,
    float* __restrict__ value_blocks,
    bool* __restrict__ validity_mask,
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const bool* __restrict__ mask_ptr,
    const float* __restrict__ q_dim_scale,
    float k_scale,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t D,
    int64_t U,
    int64_t W,
    int64_t C,
    int64_t max_past) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kFVecSize = fVec::size();
  constexpr int64_t kStep = 2 * kFVecSize;

  const int64_t HD = H * D;
  const fVec k_scale_vec(k_scale);

  at::parallel_for(0, B * U, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t b = idx / U;
      const int64_t u = idx % U;

      float* q_block = query_blocks + (b * U + u) * W * HD;
      for (int64_t w = 0; w < W; ++w) {
        const int64_t t = u * W + w;
        float* q_out = q_block + w * HD;
        if (t < T) {
          const scalar_t* q_in = q_ptr + (b * T + t) * HD;
          for (int64_t hd = 0; hd < HD; hd += D) {
            int64_t d = 0;
#pragma GCC unroll 4
            for (; d <= D - kStep; d += kStep) {
              auto [f0, f1] = load_float_vec2(q_in + hd + d);
              fVec s0 = fVec::loadu(q_dim_scale + d);
              fVec s1 = fVec::loadu(q_dim_scale + d + kFVecSize);
              (f0 * s0).store(q_out + hd + d);
              (f1 * s1).store(q_out + hd + d + kFVecSize);
            }
            for (; d < D; ++d) {
              q_out[hd + d] = static_cast<float>(q_in[hd + d]) * q_dim_scale[d];
            }
          }
        } else {
          std::memset(q_out, 0, HD * sizeof(float));
        }
      }

      float* k_block = key_blocks + (b * U + u) * C * HD;
      float* v_block = value_blocks + (b * U + u) * C * HD;
      bool* vm = validity_mask + (b * U + u) * C;

      for (int64_t c = 0; c < C; ++c) {
        const int64_t t = static_cast<int64_t>(u) * W - max_past + c;
        float* k_out = k_block + c * HD;
        float* v_out = v_block + c * HD;

        if (t >= 0 && t < T) {
          const scalar_t* k_in = k_ptr + (b * T + t) * HD;
          const scalar_t* v_in = v_ptr + (b * T + t) * HD;

          int64_t i = 0;
#pragma GCC unroll 4
          for (; i <= HD - kStep; i += kStep) {
            auto [f0, f1] = load_float_vec2(k_in + i);
            (f0 * k_scale_vec).store(k_out + i);
            (f1 * k_scale_vec).store(k_out + i + kFVecSize);
          }
          for (; i < HD; ++i) {
            k_out[i] = static_cast<float>(k_in[i]) * k_scale;
          }

          int64_t j = 0;
#pragma GCC unroll 4
          for (; j <= HD - kStep; j += kStep) {
            auto [f0, f1] = load_float_vec2(v_in + j);
            f0.store(v_out + j);
            f1.store(v_out + j + kFVecSize);
          }
          for (; j < HD; ++j) {
            v_out[j] = static_cast<float>(v_in[j]);
          }

          vm[c] = !mask_ptr[b * T + t];
        } else {
          std::memset(k_out, 0, HD * sizeof(float));
          std::memset(v_out, 0, HD * sizeof(float));
          vm[c] = false;
        }
      }
    }
  });
}

}  // anonymous namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> gemma4_audio_qkv_preprocess_cpu(
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& mask,
    at::Tensor& per_dim_scale,
    double q_scale,
    double k_scale,
    int64_t num_heads,
    int64_t head_dim,
    int64_t chunk_size,
    int64_t max_past_horizon,
    int64_t max_future_horizon) {
  RECORD_FUNCTION("sgl-kernel::gemma4_audio_qkv_preprocess_cpu", std::vector<c10::IValue>({q}));

  TORCH_CHECK(q.dim() == 3, "q must be [B, T, H*D]");
  TORCH_CHECK(k.dim() == 3, "k must be [B, T, H*D]");
  TORCH_CHECK(v.dim() == 3, "v must be [B, T, H*D]");
  TORCH_CHECK(mask.dim() == 2, "mask must be [B, T]");
  TORCH_CHECK(per_dim_scale.dim() == 1, "per_dim_scale must be [D]");

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = num_heads;
  const int64_t D = head_dim;
  const int64_t W = chunk_size;
  const int64_t U = (T + W - 1) / W;
  const int64_t C = W + max_past_horizon + max_future_horizon;

  TORCH_CHECK(q.size(2) == H * D, "q last dim must equal num_heads * head_dim");
  TORCH_CHECK(per_dim_scale.size(0) == D, "per_dim_scale size must match head_dim");

  auto pds_f32 = per_dim_scale.contiguous().to(at::kFloat);
  auto pds = pds_f32.data_ptr<float>();
  std::vector<float> q_dim_scale(D);
  float q_sc = static_cast<float>(q_scale);
  for (int64_t d = 0; d < D; ++d) {
    q_dim_scale[d] = q_sc * std::log1p(std::exp(pds[d]));
  }

  auto options_f32 = at::TensorOptions().dtype(at::kFloat).device(q.device());
  auto options_bool = at::TensorOptions().dtype(at::kBool).device(q.device());

  auto query_blocks = at::empty({B, U, W, H, D}, options_f32);
  auto key_blocks = at::empty({B, U, C, H, D}, options_f32);
  auto value_blocks = at::empty({B, U, C, H, D}, options_f32);
  auto validity_mask = at::empty({B, U, C}, options_bool);

  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto mask_c = mask.contiguous();

  float k_sc = static_cast<float>(k_scale);

  if (q.scalar_type() == at::kFloat) {
    gemma4_audio_qkv_preprocess_impl<float>(
        query_blocks.data_ptr<float>(),
        key_blocks.data_ptr<float>(),
        value_blocks.data_ptr<float>(),
        validity_mask.data_ptr<bool>(),
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        v_c.data_ptr<float>(),
        mask_c.data_ptr<bool>(),
        q_dim_scale.data(),
        k_sc,
        B,
        T,
        H,
        D,
        U,
        W,
        C,
        max_past_horizon);
  } else {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "gemma4_audio_qkv_preprocess_cpu", [&] {
      gemma4_audio_qkv_preprocess_impl<scalar_t>(
          query_blocks.data_ptr<float>(),
          key_blocks.data_ptr<float>(),
          value_blocks.data_ptr<float>(),
          validity_mask.data_ptr<bool>(),
          q_c.data_ptr<scalar_t>(),
          k_c.data_ptr<scalar_t>(),
          v_c.data_ptr<scalar_t>(),
          mask_c.data_ptr<bool>(),
          q_dim_scale.data(),
          k_sc,
          B,
          T,
          H,
          D,
          U,
          W,
          C,
          max_past_horizon);
    });
  }

  return std::make_tuple(query_blocks, key_blocks, value_blocks, validity_mask);
}

// ---- Relative Position Attention Logits (Fully-Fused) ----

namespace {

at::Tensor generate_projected_sin_emb(
    const at::Tensor& inv_timescales,
    const at::Tensor& weight_f32,
    int64_t S,
    int64_t channels,
    int64_t HD,
    int64_t max_backward) {
  const int64_t half_channels = channels / 2;
  const float* __restrict__ inv_ts_ptr = inv_timescales.data_ptr<float>();

  auto timing = at::empty({S, channels}, inv_timescales.options());
  float* __restrict__ timing_data = timing.data_ptr<float>();

  for (int64_t s = 0; s < S; ++s) {
    float pos = static_cast<float>(max_backward - s);
    float* __restrict__ row = timing_data + s * channels;
    for (int64_t i = 0; i < half_channels; ++i) {
      float scaled = pos * inv_ts_ptr[i];
      row[i] = std::sin(scaled);
      row[i + half_channels] = std::cos(scaled);
    }
  }

  return at::mm(timing, weight_f32.t());
}

void gemma4_audio_rel_pos_logits_impl(
    float* __restrict__ output,
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ sin_emb,
    int64_t B,
    int64_t U,
    int64_t W,
    int64_t H,
    int64_t D,
    int64_t C,
    int64_t S) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = fVec::size();

  const int64_t q_h_stride = D;
  const int64_t q_w_stride = H * D;
  const int64_t q_u_stride = W * q_w_stride;
  const int64_t q_b_stride = U * q_u_stride;

  const int64_t k_h_stride = D;
  const int64_t k_c_stride = H * D;
  const int64_t k_u_stride = C * k_c_stride;
  const int64_t k_b_stride = U * k_u_stride;

  const int64_t se_h_stride = D;
  const int64_t se_s_stride = H * D;

  const int64_t o_w_stride = C;
  const int64_t o_u_stride = W * C;
  const int64_t o_h_stride = U * o_u_stride;
  const int64_t o_b_stride = H * o_h_stride;

  at::parallel_for(0, B * U * H, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t b = idx / (U * H);
      const int64_t uh = idx % (U * H);
      const int64_t u = uh / H;
      const int64_t h = uh % H;

      for (int64_t w = 0; w < W; ++w) {
        const float* __restrict__ q = queries + b * q_b_stride + u * q_u_stride + w * q_w_stride + h * q_h_stride;
        float* __restrict__ out = output + b * o_b_stride + h * o_h_stride + u * o_u_stride + w * o_w_stride;

        for (int64_t c = 0; c < C; ++c) {
          const float* __restrict__ k = keys + b * k_b_stride + u * k_u_stride + c * k_c_stride + h * k_h_stride;
          const int64_t s = c - w;

          fVec sum_vec(0.0f);
          float sum_scalar = 0.0f;

          if (s >= 0 && s < S) {
            const float* __restrict__ se = sin_emb + s * se_s_stride + h * se_h_stride;
            int64_t d = 0;
#pragma GCC unroll 4
            for (; d <= D - kVecSize; d += kVecSize) {
              fVec qv = fVec::loadu(q + d);
              fVec kv = fVec::loadu(k + d);
              fVec sv = fVec::loadu(se + d);
              sum_vec = sum_vec + qv * (kv + sv);
            }
            for (; d < D; ++d) {
              sum_scalar += q[d] * (k[d] + se[d]);
            }
          } else {
            int64_t d = 0;
#pragma GCC unroll 4
            for (; d <= D - kVecSize; d += kVecSize) {
              fVec qv = fVec::loadu(q + d);
              fVec kv = fVec::loadu(k + d);
              sum_vec = sum_vec + qv * kv;
            }
            for (; d < D; ++d) {
              sum_scalar += q[d] * k[d];
            }
          }

          alignas(64) float tmp[kVecSize];
          sum_vec.store(tmp);
          for (int i = 0; i < kVecSize; ++i) {
            sum_scalar += tmp[i];
          }

          out[c] = sum_scalar;
        }
      }
    }
  });
}

}  // anonymous namespace

at::Tensor gemma4_audio_rel_pos_logits_cpu(
    at::Tensor& queries,
    at::Tensor& keys,
    at::Tensor& inv_timescales,
    at::Tensor& pos_proj_weight,
    int64_t max_backward,
    int64_t max_forward,
    int64_t num_heads,
    int64_t head_dim) {
  RECORD_FUNCTION("sgl-kernel::gemma4_audio_rel_pos_logits_cpu", std::vector<c10::IValue>({queries}));

  TORCH_CHECK(queries.dim() == 5, "queries must be [B, U, W, H, D]");
  TORCH_CHECK(keys.dim() == 5, "keys must be [B, U, C, H, D]");
  TORCH_CHECK(queries.scalar_type() == at::kFloat, "queries must be float32");
  TORCH_CHECK(keys.scalar_type() == at::kFloat, "keys must be float32");

  const int64_t B = queries.size(0);
  const int64_t U = queries.size(1);
  const int64_t W = queries.size(2);
  const int64_t H = queries.size(3);
  const int64_t D = queries.size(4);
  const int64_t C = keys.size(2);
  const int64_t S = max_backward + max_forward + 1;

  TORCH_CHECK(H == num_heads, "queries H != num_heads");
  TORCH_CHECK(D == head_dim, "queries D != head_dim");
  TORCH_CHECK(
      keys.size(0) == B && keys.size(1) == U && keys.size(3) == H && keys.size(4) == D,
      "keys shape mismatch with queries");

  auto inv_ts = inv_timescales.contiguous().to(at::kFloat).reshape({-1});
  const int64_t half_channels = inv_ts.size(0);
  const int64_t channels = half_channels * 2;
  const int64_t HD = H * D;

  TORCH_CHECK(
      pos_proj_weight.size(0) == HD && pos_proj_weight.size(1) == channels,
      "pos_proj_weight must be [H*D, channels], got [",
      pos_proj_weight.size(0),
      ", ",
      pos_proj_weight.size(1),
      "]");

  auto weight_f32 = pos_proj_weight.contiguous().to(at::kFloat);

  auto sin_emb = generate_projected_sin_emb(inv_ts, weight_f32, S, channels, HD, max_backward);

  auto output = at::empty({B, H, U, W, C}, queries.options());

  gemma4_audio_rel_pos_logits_impl(
      output.data_ptr<float>(),
      queries.contiguous().data_ptr<float>(),
      keys.contiguous().data_ptr<float>(),
      sin_emb.data_ptr<float>(),
      B,
      U,
      W,
      H,
      D,
      C,
      S);

  return output;
}

// ---- Softcap Attention ----

namespace {

void gemma4_audio_softcap_attn_impl(
    float* __restrict__ output,
    const float* __restrict__ logits,
    const bool* __restrict__ vmask,
    const bool* __restrict__ cmask,
    const float* __restrict__ values,
    int64_t B,
    int64_t H,
    int64_t U,
    int64_t W,
    int64_t C,
    int64_t D,
    int64_t q_time,
    float softcap_val,
    float invalid_val) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = fVec::size();

  const float inv_softcap = 1.0f / softcap_val;

  const int64_t lg_h_stride = U * W * C;
  const int64_t lg_b_stride = H * lg_h_stride;

  const int64_t val_c_stride = H * D;
  const int64_t val_u_stride = C * val_c_stride;
  const int64_t val_b_stride = U * val_u_stride;

  const int64_t out_b_stride = q_time * H * D;

  at::parallel_for(0, B * U * H, 0, [&](int64_t begin, int64_t end) {
    std::vector<float> scapped(C);
    std::vector<float> probs(C);
    std::vector<float> acc(D);

    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t b = idx / (U * H);
      const int64_t uh = idx % (U * H);
      const int64_t u = uh / H;
      const int64_t h = uh % H;

      const bool* vm = vmask + (b * U + u) * C;
      const float* lg_base = logits + b * lg_b_stride + h * lg_h_stride + u * W * C;
      const float* v_base = values + b * val_b_stride + u * val_u_stride + h * D;

      for (int64_t w = 0; w < W; ++w) {
        const int64_t t = u * W + w;
        if (t >= q_time) break;

        const float* lg = lg_base + w * C;
        const bool* cm = cmask + w * C;

        float max_val = -std::numeric_limits<float>::infinity();
        fVec max_vec(-std::numeric_limits<float>::infinity());
        fVec sc_vec(softcap_val);
        fVec inv_sc_vec(inv_softcap);
        fVec inv_vec(invalid_val);

        int64_t c = 0;
        for (; c <= C - kVecSize; c += kVecSize) {
          fVec lv = fVec::loadu(lg + c);
          lv = (lv * inv_sc_vec).tanh() * sc_vec;

          alignas(64) float buf[kVecSize];
          lv.store(buf);
          for (int i = 0; i < kVecSize; ++i) {
            if (!(vm[c + i] && cm[c + i])) {
              buf[i] = invalid_val;
            }
          }
          fVec mv = fVec::loadu(buf);
          mv.store(scapped.data() + c);
          max_vec = at::vec::maximum(max_vec, mv);
        }
        for (; c < C; ++c) {
          float v;
          if (vm[c] && cm[c]) {
            v = std::tanh(lg[c] * inv_softcap) * softcap_val;
          } else {
            v = invalid_val;
          }
          scapped[c] = v;
          max_val = std::max(max_val, v);
        }
        {
          alignas(64) float tmp[kVecSize];
          max_vec.store(tmp);
          for (int i = 0; i < kVecSize; ++i)
            max_val = std::max(max_val, tmp[i]);
        }

        float sum_exp = 0.0f;
        fVec max_bcast(max_val);
        fVec sum_vec(0.0f);
        c = 0;
        for (; c <= C - kVecSize; c += kVecSize) {
          fVec sv = fVec::loadu(scapped.data() + c);
          fVec ev = (sv - max_bcast).exp();
          ev.store(probs.data() + c);
          sum_vec = sum_vec + ev;
        }
        for (; c < C; ++c) {
          float p = std::exp(scapped[c] - max_val);
          probs[c] = p;
          sum_exp += p;
        }
        {
          alignas(64) float tmp[kVecSize];
          sum_vec.store(tmp);
          for (int i = 0; i < kVecSize; ++i)
            sum_exp += tmp[i];
        }
        const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

        std::fill(acc.begin(), acc.end(), 0.0f);

        for (c = 0; c < C; ++c) {
          const float p = probs[c] * inv_sum;
          const float* vp = v_base + c * val_c_stride;
          fVec p_vec(p);

          int64_t d = 0;
#pragma GCC unroll 4
          for (; d <= D - kVecSize; d += kVecSize) {
            fVec a = fVec::loadu(acc.data() + d);
            fVec v = fVec::loadu(vp + d);
            a = a + p_vec * v;
            a.store(acc.data() + d);
          }
          for (; d < D; ++d) {
            acc[d] += p * vp[d];
          }
        }

        float* out = output + b * out_b_stride + t * H * D + h * D;
        int64_t d = 0;
        for (; d <= D - kVecSize; d += kVecSize) {
          fVec fv = fVec::loadu(acc.data() + d);
          fv.store(out + d);
        }
        for (; d < D; ++d) {
          out[d] = acc[d];
        }
      }  // for w
    }  // for idx
  });  // parallel_for
}

}  // anonymous namespace

at::Tensor gemma4_audio_softcap_attn_cpu(
    at::Tensor& logits,
    at::Tensor& validity_mask,
    at::Tensor& causal_mask,
    at::Tensor& value_blocks,
    double softcap,
    double invalid_logits_value,
    int64_t q_time) {
  RECORD_FUNCTION("sgl-kernel::gemma4_audio_softcap_attn_cpu", std::vector<c10::IValue>({logits}));

  TORCH_CHECK(logits.dim() == 5, "logits must be [B, H, U, W, C]");
  TORCH_CHECK(validity_mask.dim() == 3, "validity_mask must be [B, U, C]");
  TORCH_CHECK(causal_mask.dim() == 2, "causal_mask must be [W, C]");
  TORCH_CHECK(value_blocks.dim() == 5, "value_blocks must be [B, U, C, H, D]");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits must be float32");
  TORCH_CHECK(value_blocks.scalar_type() == at::kFloat, "value_blocks must be float32");

  const int64_t B = logits.size(0);
  const int64_t H = logits.size(1);
  const int64_t U = logits.size(2);
  const int64_t W = logits.size(3);
  const int64_t C = logits.size(4);
  const int64_t D = value_blocks.size(4);

  TORCH_CHECK(
      validity_mask.size(0) == B && validity_mask.size(1) == U && validity_mask.size(2) == C,
      "validity_mask shape mismatch");
  TORCH_CHECK(causal_mask.size(0) == W && causal_mask.size(1) == C, "causal_mask shape mismatch");
  TORCH_CHECK(
      value_blocks.size(0) == B && value_blocks.size(1) == U && value_blocks.size(2) == C && value_blocks.size(3) == H,
      "value_blocks shape mismatch");
  TORCH_CHECK(q_time > 0 && q_time <= U * W, "invalid q_time");

  auto output = at::empty({B, q_time, H, D}, logits.options());

  gemma4_audio_softcap_attn_impl(
      output.data_ptr<float>(),
      logits.contiguous().data_ptr<float>(),
      validity_mask.contiguous().data_ptr<bool>(),
      causal_mask.contiguous().data_ptr<bool>(),
      value_blocks.contiguous().data_ptr<float>(),
      B,
      H,
      U,
      W,
      C,
      D,
      q_time,
      static_cast<float>(softcap),
      static_cast<float>(invalid_logits_value));

  return output;
}
