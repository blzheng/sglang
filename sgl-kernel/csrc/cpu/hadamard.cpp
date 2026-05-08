#include "common.h"
#include "vec.h"

namespace {

// ──────────────────────────────────────────────────────────────────────
// Iterative Cooley-Tukey-style Fast Walsh-Hadamard Transform (FWHT)
// along the last dimension.  n must be a power of two.
//
// For each row of length n the butterfly loop is:
//   for h = 1, 2, 4, …, n/2
//     for every pair (a[j], b[j]) separated by stride h
//       (a[j], b[j]) ← (a[j]+b[j], a[j]−b[j])
//
// When h >= 16 (== 512-bit / sizeof(float)) we can do the butterfly
// entirely with AVX-512 loads/stores.  The inner two cases (h < 16)
// use in-register shuffles so we never spill to memory at small
// strides.
// ──────────────────────────────────────────────────────────────────────

// Scalar fallback for a single row of length n
inline void fwht_row_scalar(float* __restrict__ row, int64_t n, float scale) {
  for (int64_t h = 1; h < n; h <<= 1) {
    for (int64_t j = 0; j < n; j += 2 * h) {
      for (int64_t k = 0; k < h; ++k) {
        float a = row[j + k];
        float b = row[j + k + h];
        row[j + k]     = a + b;
        row[j + k + h] = a - b;
      }
    }
  }
  for (int64_t d = 0; d < n; ++d) {
    row[d] *= scale;
  }
}

#if defined(CPU_CAPABILITY_AVX512)

// AVX-512 optimized FWHT for a single row of length n (power of 2, n >= 16)
inline void fwht_row_avx512(float* __restrict__ row, int64_t n, float scale) {
  // For small strides (h < 16), we process 16 floats at a time using
  // in-register shuffles to do butterflies without extra loads/stores.
  // For h >= 16, we do standard load-butterfly-store with AVX-512.

  // Phase 1: h = 1, 2, 4, 8 — in-register butterflies
  // Process the row in chunks of 16 floats.
  // For h=1: butterfly pairs at stride 1 within each 16-float chunk.
  // For h=2: butterfly pairs at stride 2 within each 16-float chunk.
  // etc.

  for (int64_t h = 1; h < 16 && h < n; h <<= 1) {
    for (int64_t j = 0; j < n; j += 16) {
      __m512 v = _mm512_loadu_ps(row + j);

      // Shuffle to get the paired element.
      // For stride h, element i is paired with element i^h.
      // We need to swap elements that differ in bit position log2(h).
      __m512 paired;
      if (h == 1) {
        // swap adjacent: 0↔1, 2↔3, 4↔5, …
        paired = _mm512_permute_ps(v, 0b10'11'00'01);
      } else if (h == 2) {
        // swap pairs: (0,1)↔(2,3), (4,5)↔(6,7), …
        paired = _mm512_permute_ps(v, 0b01'00'11'10);
      } else if (h == 4) {
        // swap quads: (0..3)↔(4..7) within each 256-bit half
        paired = _mm512_permutexvar_ps(
            _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4), v);
      } else {  // h == 8
        // swap octets: (0..7)↔(8..15)
        paired = _mm512_permutexvar_ps(
            _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), v);
      }

      // For the "first" slot (bit log2(h) == 0): v[i]=a, paired[i]=b → sum = a+b
      // For the "second" slot (bit log2(h) == 1): v[i]=b, paired[i]=a → diff = a-b
      __m512 sum  = _mm512_add_ps(v, paired);
      __m512 diff = _mm512_sub_ps(paired, v);  // paired - v = a - b for second slots

      // Blend: for indices where bit log2(h) is 0, take sum; else take diff.
      __mmask16 mask;
      if (h == 1) {
        mask = 0xAAAA;  // bits: 1010 1010 1010 1010  (odd positions)
      } else if (h == 2) {
        mask = 0xCCCC;  // bits: 1100 1100 1100 1100
      } else if (h == 4) {
        mask = 0xF0F0;  // bits: 1111 0000 1111 0000
      } else {  // h == 8
        mask = 0xFF00;  // bits: 1111 1111 0000 0000
      }
      __m512 result = _mm512_mask_blend_ps(mask, sum, diff);
      _mm512_storeu_ps(row + j, result);
    }
  }

  // Phase 2: h >= 16 — standard load-add/sub-store
  for (int64_t h = 16; h < n; h <<= 1) {
    for (int64_t j = 0; j < n; j += 2 * h) {
      for (int64_t k = 0; k < h; k += 16) {
        __m512 a = _mm512_loadu_ps(row + j + k);
        __m512 b = _mm512_loadu_ps(row + j + k + h);
        _mm512_storeu_ps(row + j + k,     _mm512_add_ps(a, b));
        _mm512_storeu_ps(row + j + k + h, _mm512_sub_ps(a, b));
      }
    }
  }

  // Apply scale
  __m512 scale_vec = _mm512_set1_ps(scale);
  int64_t d = 0;
  for (; d <= n - 16; d += 16) {
    __m512 v = _mm512_loadu_ps(row + d);
    _mm512_storeu_ps(row + d, _mm512_mul_ps(v, scale_vec));
  }
  for (; d < n; ++d) {
    row[d] *= scale;
  }
}

#endif  // CPU_CAPABILITY_AVX512

}  // anonymous namespace

// ──────────────────────────────────────────────────────────────────────
// fast_hadamard_transform_cpu
//
// Input:  x      – arbitrary-shape tensor whose last dim is a power of 2
//         scale  – multiplicative scale applied after the transform
// Output: tensor of same shape/dtype with FWHT applied along the last dim
// ──────────────────────────────────────────────────────────────────────
at::Tensor fast_hadamard_transform_cpu(const at::Tensor& x, double scale) {
  const int64_t n = x.size(-1);
  TORCH_CHECK(n > 0 && (n & (n - 1)) == 0,
              "fast_hadamard_transform: last dim must be a power of 2, got ", n);

  // Flatten to 2-D, work in float, then cast back.
  const int64_t rows = x.numel() / n;
  auto x_flat = x.reshape({rows, n}).to(at::kFloat).contiguous();
  auto out = x_flat.clone();  // mutable working copy
  float* out_ptr = out.data_ptr<float>();

  at::parallel_for(0, rows, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      float* row = out_ptr + r * n;
#if defined(CPU_CAPABILITY_AVX512)
      if (n >= 16) {
        fwht_row_avx512(row, n, static_cast<float>(scale));
      } else {
        fwht_row_scalar(row, n, static_cast<float>(scale));
      }
#else
      fwht_row_scalar(row, n, static_cast<float>(scale));
#endif
    }
  });

  // Reshape back and cast to original dtype.
  return out.view(x.sizes()).to(x.scalar_type());
}
