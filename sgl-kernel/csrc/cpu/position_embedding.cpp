#include "common.h"
#include "vec.h"

namespace {

// Fused position embedding kernel:
//   For each (batch, patch), gather position_embedding_table[0, x_pos, :] and
//   position_embedding_table[1, y_pos, :], sum them, and zero out if padding.
//
// This avoids creating one-hot tensors and doing matmul, replacing it with
// direct table lookups (gather) which is much more efficient.
//
// patch_positions:          [batch, num_patches, 2]  (int64, x/y indices, -1 for padding)
// padding_positions:        [batch, num_patches]     (bool, true for padding)
// position_embedding_table: [2, position_embedding_size, hidden_size]
// output:                   [batch, num_patches, hidden_size]
template <typename scalar_t>
void position_embeddings_kernel_impl(
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

  // table layout: [2, position_embedding_size, hidden_size]
  const int64_t table_stride = position_embedding_size * hidden_size;

  at::parallel_for(0, batch_size * num_patches, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t b = idx / num_patches;
      int64_t p = idx % num_patches;

      scalar_t* __restrict__ out_ptr = output + (b * num_patches + p) * hidden_size;

      // Check padding
      if (padding_positions[b * num_patches + p]) {
        // Zero out the output
        int64_t d = 0;
        const bVec zero_bvec(static_cast<scalar_t>(0));
        for (; d <= hidden_size - kVecSize; d += kVecSize) {
          zero_bvec.store(out_ptr + d);
        }
        for (; d < hidden_size; ++d) {
          out_ptr[d] = static_cast<scalar_t>(0);
        }
        continue;
      }

      // Clamp positions to >= 0
      const int64_t* pos_ptr = patch_positions + (b * num_patches + p) * 2;
      int64_t x_pos = std::max(pos_ptr[0], static_cast<int64_t>(0));
      int64_t y_pos = std::max(pos_ptr[1], static_cast<int64_t>(0));

      // Gather from table[0, x_pos, :] and table[1, y_pos, :]
      const scalar_t* __restrict__ row_x = table + x_pos * hidden_size;
      const scalar_t* __restrict__ row_y = table + table_stride + y_pos * hidden_size;

      // Sum the two rows into output
      int64_t d = 0;
#pragma GCC unroll 4
      for (; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(row_x + d);
        bVec y_bvec = bVec::loadu(row_y + d);
        fVec x_fvec0, x_fvec1, y_fvec0, y_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);
        std::tie(y_fvec0, y_fvec1) = at::vec::convert_to_float(y_bvec);

        fVec out_fvec0 = x_fvec0 + y_fvec0;
        fVec out_fvec1 = x_fvec1 + y_fvec1;

        bVec out_bvec = convert_from_float_ext<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(out_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(row_x[d]);
        float y_val = static_cast<float>(row_y[d]);
        out_ptr[d] = static_cast<scalar_t>(x_val + y_val);
      }
    }
  });
}

}  // anonymous namespace

// patch_positions:          [batch, num_patches, 2]  (int64)
// padding_positions:        [batch, num_patches]     (bool)
// position_embedding_table: [2, position_embedding_size, hidden_size]
// returns:                  [batch, num_patches, hidden_size]
at::Tensor position_embeddings_cpu(
    at::Tensor& patch_positions, at::Tensor& padding_positions, at::Tensor& position_embedding_table) {
  RECORD_FUNCTION("sgl-kernel::position_embeddings_cpu", std::vector<c10::IValue>({patch_positions}));

  TORCH_CHECK(
      patch_positions.dim() == 3 && patch_positions.size(2) == 2, "patch_positions must be [batch, num_patches, 2]");
  TORCH_CHECK(padding_positions.dim() == 2, "padding_positions must be [batch, num_patches]");
  TORCH_CHECK(
      position_embedding_table.dim() == 3 && position_embedding_table.size(0) == 2,
      "position_embedding_table must be [2, position_embedding_size, hidden_size]");

  int64_t batch_size = patch_positions.size(0);
  int64_t num_patches = patch_positions.size(1);
  int64_t position_embedding_size = position_embedding_table.size(1);
  int64_t hidden_size = position_embedding_table.size(2);

  auto output = at::empty({batch_size, num_patches, hidden_size}, position_embedding_table.options());

  // Ensure contiguous
  auto patch_positions_c = patch_positions.contiguous();
  auto padding_positions_c = padding_positions.contiguous();
  auto table_c = position_embedding_table.contiguous();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(position_embedding_table.scalar_type(), "position_embeddings_cpu", [&]() {
    position_embeddings_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        patch_positions_c.data_ptr<int64_t>(),
        padding_positions_c.data_ptr<bool>(),
        table_c.data_ptr<scalar_t>(),
        batch_size,
        num_patches,
        hidden_size,
        position_embedding_size);
  });

  return output;
}
