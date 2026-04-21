import itertools
import unittest

import torch
import torch.nn.functional as F
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


def position_embeddings_ref(
    patch_positions, padding_positions, position_embedding_table
):
    """Reference implementation matching Gemma4VisionPatchEmbedder._position_embeddings."""
    position_embedding_size = position_embedding_table.size(1)
    clamped_positions = patch_positions.clamp(min=0)
    one_hot = F.one_hot(clamped_positions, num_classes=position_embedding_size)
    one_hot = one_hot.permute(0, 2, 1, 3).to(position_embedding_table)
    position_embeddings = one_hot @ position_embedding_table
    position_embeddings = position_embeddings.sum(dim=1)
    position_embeddings = torch.where(
        padding_positions.unsqueeze(-1), 0.0, position_embeddings
    )
    return position_embeddings


class TestPositionEmbeddings(CustomTestCase):
    batch_sizes = [1, 2, 4]
    num_patches_list = [1, 16, 64]
    position_embedding_sizes = [32, 128]
    hidden_sizes = [128, 256]
    dtypes = [torch.bfloat16, torch.float16]

    def _test_position_embeddings(
        self, batch_size, num_patches, position_embedding_size, hidden_size, dtype
    ):
        # Create position embedding table [2, position_embedding_size, hidden_size]
        table = torch.randn(2, position_embedding_size, hidden_size, dtype=dtype)

        # Create patch positions [batch, num_patches, 2] with valid indices
        patch_positions = torch.randint(
            0, position_embedding_size, (batch_size, num_patches, 2), dtype=torch.int64
        )

        # Create padding positions [batch, num_patches] — some patches are padding
        padding_positions = torch.zeros(batch_size, num_patches, dtype=torch.bool)
        if num_patches > 1:
            # Mark ~25% as padding
            padding_mask = torch.rand(batch_size, num_patches) < 0.25
            padding_positions = padding_mask
            # Set padding patch positions to -1
            patch_positions[padding_positions] = -1

        # Reference
        ref_out = position_embeddings_ref(patch_positions, padding_positions, table)

        # Kernel
        out = torch.ops.sgl_kernel.position_embeddings_cpu(
            patch_positions, padding_positions, table
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_position_embeddings(self):
        for params in itertools.product(
            self.batch_sizes,
            self.num_patches_list,
            self.position_embedding_sizes,
            self.hidden_sizes,
            self.dtypes,
        ):
            with self.subTest(
                batch_size=params[0],
                num_patches=params[1],
                pos_emb_size=params[2],
                hidden_size=params[3],
                dtype=params[4],
            ):
                self._test_position_embeddings(*params)

    def test_all_padding(self):
        """All patches are padding — output should be all zeros."""
        table = torch.randn(2, 64, 128, dtype=torch.bfloat16)
        patch_positions = torch.full((2, 8, 2), -1, dtype=torch.int64)
        padding_positions = torch.ones(2, 8, dtype=torch.bool)

        out = torch.ops.sgl_kernel.position_embeddings_cpu(
            patch_positions, padding_positions, table
        )

        expected = torch.zeros(2, 8, 128, dtype=torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_no_padding(self):
        """No padding — all patches have valid positions."""
        table = torch.randn(2, 64, 128, dtype=torch.bfloat16)
        patch_positions = torch.randint(0, 64, (2, 8, 2), dtype=torch.int64)
        padding_positions = torch.zeros(2, 8, dtype=torch.bool)

        ref_out = position_embeddings_ref(patch_positions, padding_positions, table)
        out = torch.ops.sgl_kernel.position_embeddings_cpu(
            patch_positions, padding_positions, table
        )

        atol = rtol = precision[torch.bfloat16]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_single_patch(self):
        """Single patch per batch."""
        table = torch.randn(2, 32, 64, dtype=torch.float16)
        patch_positions = torch.tensor([[[5, 10]]], dtype=torch.int64)
        padding_positions = torch.zeros(1, 1, dtype=torch.bool)

        ref_out = position_embeddings_ref(patch_positions, padding_positions, table)
        out = torch.ops.sgl_kernel.position_embeddings_cpu(
            patch_positions, padding_positions, table
        )

        atol = rtol = precision[torch.float16]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
