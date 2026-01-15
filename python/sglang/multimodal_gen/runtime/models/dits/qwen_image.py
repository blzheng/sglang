# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous

from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNorm,
    RMSNorm,
    apply_qk_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    apply_flashinfer_rope_qk_inplace,
)
# from sglang.multimodal_gen.runtime.layers.triton_ops import (
#     fuse_scale_shift_gate_select01_kernel,
#     fuse_scale_shift_kernel,
# )
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(
    attn: "QwenImageCrossAttention", hidden_states, encoder_hidden_states=None
):
    img_query, _ = attn.to_q(hidden_states)
    img_key, _ = attn.to_k(hidden_states)
    img_value, _ = attn.to_v(hidden_states)

    txt_query = txt_key = txt_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        txt_query, _ = attn.add_q_proj(encoder_hidden_states)
        txt_key, _ = attn.add_k_proj(encoder_hidden_states)
        txt_value, _ = attn.add_v_proj(encoder_hidden_states)

    return img_query, img_key, img_value, txt_query, txt_key, txt_value


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError(
                    "When additional_t_cond is True, addition_t_cond must be provided."
                )
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        # self.rope = NDRotaryEmbedding(
        #     rope_dim_list=axes_dim,
        #     rope_theta=theta,
        #     use_real=False,
        #     repeat_interleave_real=False,
        #     dtype=torch.float32 if current_platform.is_mps() else torch.float64,
        # )

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        device = index.device
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            (
                1.0
                / torch.pow(
                    theta,
                    torch.arange(0, dim, 2, device=device).to(torch.float32).div(dim),
                )
            ).to(device=device),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        txt_seq_lens: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`Tuple[int, int, int]` or `List[Tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video.
            txt_seq_lens (`List[int]`):
                A list of integers of length batch_size representing the length of each text prompt.
            device: (`torch.device`):
                The device on which to perform the RoPE computation.
        """
        # When models are initialized under a "meta" device context (e.g. init_empty_weights),
        # tensors created during __init__ become meta tensors. Calling .to(...) on a meta tensor
        # raises "Cannot copy out of meta tensor". Rebuild the frequencies on the target device
        # in that case; otherwise move them if just on a different device.
        if getattr(self.pos_freqs, "device", torch.device("meta")).type == "meta":
            pos_index = torch.arange(4096, device=device)
            neg_index = torch.arange(4096, device=device).flip(0) * -1 - 1
            self.pos_freqs = torch.cat(
                [
                    self.rope_params(pos_index, self.axes_dim[0], self.theta),
                    self.rope_params(pos_index, self.axes_dim[1], self.theta),
                    self.rope_params(pos_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            ).to(device=device)
            self.neg_freqs = torch.cat(
                [
                    self.rope_params(neg_index, self.axes_dim[0], self.theta),
                    self.rope_params(neg_index, self.axes_dim[1], self.theta),
                    self.rope_params(neg_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            ).to(device=device)
        elif self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            # RoPE frequencies are cached via a lru_cache decorator on _compute_video_freqs
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0).to(device=device)
        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = (
            freqs_pos[0][idx : idx + frame]
            .view(frame, 1, 1, -1)
            .expand(frame, height, width, -1)
        )
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(
                frame, height, width, -1
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(
                frame, height, width, -1
            )
        else:
            freqs_height = (
                freqs_pos[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(
            seq_lens, -1
        )
        return freqs.clone().contiguous()


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        device = index.device
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            (
                1.0
                / torch.pow(
                    theta,
                    torch.arange(0, dim, 2, device=device).to(torch.float32).div(dim),
                )
            ).to(device=device),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """

        # When models are initialized under a "meta" device context (e.g. init_empty_weights),
        # tensors created during __init__ become meta tensors. Calling .to(...) on a meta tensor
        # raises "Cannot copy out of meta tensor". Rebuild the frequencies on the target device
        # in that case; otherwise move them if just on a different device.
        if getattr(self.pos_freqs, "device", torch.device("meta")).type == "meta":
            pos_index = torch.arange(4096, device=device)
            neg_index = torch.arange(4096, device=device).flip(0) * -1 - 1
            self.pos_freqs = torch.cat(
                [
                    self.rope_params(pos_index, self.axes_dim[0], self.theta),
                    self.rope_params(pos_index, self.axes_dim[1], self.theta),
                    self.rope_params(pos_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            ).to(device=device)
            self.neg_freqs = torch.cat(
                [
                    self.rope_params(neg_index, self.axes_dim[0], self.theta),
                    self.rope_params(neg_index, self.axes_dim[1], self.theta),
                    self.rope_params(neg_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            ).to(device=device)
        elif self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = (
            freqs_pos[0][idx : idx + frame]
            .view(frame, 1, 1, -1)
            .expand(frame, height, width, -1)
        )
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(
                frame, height, width, -1
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(
                frame, height, width, -1
            )
        else:
            freqs_height = (
                freqs_pos[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(
            seq_lens, -1
        )
        return freqs.clone().contiguous()

    @functools.lru_cache(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = (
            freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        )
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(
                frame, height, width, -1
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(
                frame, height, width, -1
            )
        else:
            freqs_height = (
                freqs_pos[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(
            seq_lens, -1
        )
        return freqs.clone().contiguous()


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        print(x.shape)
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        print(x_rotated.shape)
        print(freqs_cis.shape)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)
# Copied from transformers
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_native(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    q_rotated = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_rotated = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    q_rotated = q_rotated.squeeze(0)
    k_rotated = k_rotated.squeeze(0)

    # embedding is performed in float
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q_rotated * cos) + (rotate_half(q_rotated) * sin)
    k_embed = (k_rotated * cos) + (rotate_half(k_rotated) * sin)

    q_embed = q_embed.to(orig_q_dtype).unsqueeze(0)
    k_embed = k_embed.to(orig_k_dtype).unsqueeze(0)
    return q_embed.flatten(3), k_embed.flatten(3)
    return q_embed.unsqueeze(0), k_embed.unsqueeze(0)
def apply_rotary_embedding_pytorch(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    interleaved: bool = False
) -> torch.Tensor:
    """
    PyTorch版本的rotary embedding实现，替代Triton kernel
    
    Args:
        x: 输入张量，形状为 [..., num_heads, head_size]
        cos: 余弦值张量，形状为 [num_tokens, head_size_half]
        sin: 正弦值张量，形状为 [num_tokens, head_size_half]
        interleaved: 是否使用interleaved模式
    
    Returns:
        应用rotary embedding后的输出张量，形状与x相同
    """
    # 保存原始形状
    original_shape = x.shape
    
    # 处理不同的输入维度
    if x.dim() == 4:
        bsz, num_tokens, num_heads, head_size = x.shape
        x_flat = x.view(bsz * num_tokens * num_heads, head_size)
    elif x.dim() == 3:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1
        x_flat = x.view(num_tokens * num_heads, head_size)
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    # 确保head_size是偶数
    assert head_size % 2 == 0, "head_size must be divisible by 2"
    head_size_half = head_size // 2
    
    # 处理cos和sin的形状
    if interleaved:
        if cos.shape[-1] == head_size:
            # 如果cos/sin包含完整head_size，只取偶数位置
            cos = cos[..., ::2].contiguous()
            sin = sin[..., ::2].contiguous()
    
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # 创建正确的token索引
    # 每个 (batch, token, head) 组合对应一个token索引
    total_rows = bsz * num_tokens * num_heads
    
    # 创建token索引：每个batch中，每个token重复num_heads次
    token_indices = torch.arange(num_tokens, device=x.device)
    token_indices = token_indices.repeat(bsz)  # [bsz * num_tokens]
    token_indices = token_indices.repeat_interleave(num_heads)  # [bsz * num_tokens * num_heads]
    
    # 确保索引数量匹配
    assert token_indices.shape[0] == total_rows, \
        f"token_indices shape {token_indices.shape} doesn't match total_rows {total_rows}"
    
    # 从cos和sin中获取对应的值
    cos_selected = cos[token_indices]  # [total_rows, head_size_half]
    sin_selected = sin[token_indices]  # [total_rows, head_size_half]
    
    output = torch.empty_like(x_flat)
    
    # 分割x为两部分
    if interleaved:
        # interleaved模式：x1 = 偶数索引，x2 = 奇数索引
        x1 = x_flat[:, 0::2]  # [..., head_size_half]
        x2 = x_flat[:, 1::2]  # [..., head_size_half]
    else:
        # 非interleaved模式：前半部分和后半部分
        x1 = x_flat[:, :head_size_half]
        x2 = x_flat[:, head_size_half:]
    
    # 应用rotary embedding公式
    # o1 = x1 * cos - x2 * sin
    # o2 = x1 * sin + x2 * cos
    o1 = x1 * cos_selected - x2 * sin_selected
    o2 = x1 * sin_selected + x2 * cos_selected
    
    # 合并结果
    if interleaved:
        # interleaved模式：交替放置o1和o2
        output[:, 0::2] = o1
        output[:, 1::2] = o2
    else:
        # 非interleaved模式：前半部分放o1，后半部分放o2
        output[:, :head_size_half] = o1
        output[:, head_size_half:] = o2
    
    # 恢复原始形状
    output = output.reshape(*original_shape)
    
    return output

class QwenImageCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,  # query_dim
        num_heads: int,
        head_dim: int,
        window_size=(-1, -1),
        added_kv_proj_dim: int = None,
        out_bias: bool = True,
        qk_norm=True,  # rmsnorm
        eps=1e-6,
        pre_only=False,
        context_pre_only: bool = False,
        parallel_attention=False,
        out_dim: int = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.added_kv_proj_dim = added_kv_proj_dim

        # Use separate Q/K/V projections
        self.inner_dim = out_dim if out_dim is not None else head_dim * num_heads
        self.inner_kv_dim = self.inner_dim
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_k = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_v = ReplicatedLinear(dim, self.inner_dim, bias=True)

        if self.qk_norm:
            self.norm_q = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        if added_kv_proj_dim is not None:
            self.add_q_proj = ReplicatedLinear(
                added_kv_proj_dim, self.inner_dim, bias=True
            )
            self.add_k_proj = ReplicatedLinear(
                added_kv_proj_dim, self.inner_dim, bias=True
            )
            self.add_v_proj = ReplicatedLinear(
                added_kv_proj_dim, self.inner_dim, bias=True
            )

        if context_pre_only is not None and not context_pre_only:
            self.to_add_out = ReplicatedLinear(self.inner_dim, self.dim, bias=out_bias)
        else:
            self.to_add_out = None

        if not pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(
                ReplicatedLinear(self.inner_dim, self.dim, bias=out_bias)
            )
        else:
            self.to_out = None

        self.norm_added_q = RMSNorm(head_dim, eps=eps)
        self.norm_added_k = RMSNorm(head_dim, eps=eps)

        # Scaled dot product attention
        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.AITER,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.SAGE_ATTN,
                AttentionBackendEnum.SAGE_ATTN_3,
            },
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        **cross_attention_kwargs,
    ):
        seq_len_txt = encoder_hidden_states.shape[1]

        img_query, img_key, img_value, txt_query, txt_key, txt_value = (
            _get_qkv_projections(self, hidden_states, encoder_hidden_states)
        )

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (self.num_heads, -1))
        img_key = img_key.unflatten(-1, (self.num_heads, -1))
        img_value = img_value.unflatten(-1, (self.num_heads, -1))

        txt_query = txt_query.unflatten(-1, (self.num_heads, -1))
        txt_key = txt_key.unflatten(-1, (self.num_heads, -1))
        txt_value = txt_value.unflatten(-1, (self.num_heads, -1))

        # Apply QK normalization
        if self.qk_norm:
            img_query, img_key = apply_qk_norm(
                q=img_query,
                k=img_key,
                q_norm=self.norm_q,
                k_norm=self.norm_k,
                head_dim=img_query.shape[-1],
                allow_inplace=True,
            )
            txt_query, txt_key = apply_qk_norm(
                q=txt_query,
                k=txt_key,
                q_norm=self.norm_added_q,
                k_norm=self.norm_added_k,
                head_dim=txt_query.shape[-1],
                allow_inplace=True,
            )

        # Apply RoPE
        if image_rotary_emb is not None:
            if not (
                isinstance(image_rotary_emb[0], torch.Tensor)
                and image_rotary_emb[0].dim() == 2
            ):
                raise RuntimeError("image_rotary_emb must be cos_sin_cache tensors")
            img_cache, txt_cache = image_rotary_emb
            # bsz, seqlen, nheads, d = img_query.shape
            # half_size = img_cache.shape[-1] // 2
            # img_cos = img_cache[:seqlen, :half_size].to(img_query.dtype)
            # img_sin = img_cache[:seqlen, half_size:].to(img_query.dtype)
            # img_cos = img_cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
            # img_sin = img_sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
           
            # img_query = img_query.reshape(bsz * seqlen, nheads, d)
            # img_key = img_key.reshape(bsz * seqlen, nheads, d)

            # bsz, seqlen, nheads, d = txt_query.shape
            # half_size = txt_cache.shape[-1] // 2
            # txt_cos = txt_cache[:seqlen, :half_size].to(txt_query.dtype)
            # txt_sin = txt_cache[:seqlen, half_size:].to(txt_query.dtype)
            # txt_cos = txt_cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
            # txt_sin = txt_sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
           
            # txt_query = txt_query.reshape(bsz * seqlen, nheads, d)
            # txt_key = txt_key.reshape(bsz * seqlen, nheads, d)
            # txt_query = apply_rotary_embedding_pytorch(
            #     txt_query, txt_cos, txt_sin, interleaved=True
            # )
            # img_key = apply_rotary_embedding_pytorch(
            #     img_key, img_cos, img_sin, interleaved=True
            # )
            # txt_query = apply_rotary_embedding_pytorch(
            #     txt_query, txt_cos, txt_sin, interleaved=True
            # )
            # txt_key = apply_rotary_embedding_pytorch(
            #     txt_key, txt_cos, txt_sin, interleaved=True
            # )

            img_query, img_key = apply_flashinfer_rope_qk_inplace(
                img_query, img_key, img_cache, is_neox=False
            )
            txt_query, txt_key = apply_flashinfer_rope_qk_inplace(
                txt_query, txt_key, txt_cache, is_neox=False
            )

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = self.attn(
            joint_query,
            joint_key,
            joint_value,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_len_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_len_txt:, :]  # Image part

        # Apply output projections
        img_attn_output, _ = self.to_out[0](img_attn_output)
        if len(self.to_out) > 1:
            (img_attn_output,) = self.to_out[1](img_attn_output)  # dropout

        txt_attn_output, _ = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.zero_cond_t = zero_cond_t

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = QwenImageCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            added_kv_proj_dim=dim,
            context_pre_only=False,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.img_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def _modulate_triton(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            actual_batch = x.shape[0]
            shift0, shift1 = (
                shift[:actual_batch],
                shift[actual_batch : 2 * actual_batch],
            )
            scale0, scale1 = (
                scale[:actual_batch],
                scale[actual_batch : 2 * actual_batch],
            )
            gate0, gate1 = gate[:actual_batch], gate[actual_batch : 2 * actual_batch]

            if x.is_cuda:
                if not x.is_contiguous():
                    x = x.contiguous()
                if not index.is_contiguous():
                    index = index.contiguous()
                x, gate_result = fuse_scale_shift_gate_select01_kernel(
                    x,
                    scale0=scale0.contiguous(),
                    shift0=shift0.contiguous(),
                    gate0=gate0.contiguous(),
                    scale1=scale1.contiguous(),
                    shift1=shift1.contiguous(),
                    gate1=gate1.contiguous(),
                    index=index,
                )
                return x, gate_result
            else:
                mask = (index == 0).unsqueeze(-1)
                shift_result = torch.where(
                    mask, shift0.unsqueeze(1), shift1.unsqueeze(1)
                )
                scale_result = torch.where(
                    mask, scale0.unsqueeze(1), scale1.unsqueeze(1)
                )
                gate_result = torch.where(mask, gate0.unsqueeze(1), gate1.unsqueeze(1))
                return (
                    fuse_scale_shift_kernel(x, scale_result, shift_result),
                    gate_result,
                )
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)
            return fuse_scale_shift_kernel(x, scale_result, shift_result), gate_result

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb_img_silu: torch.Tensor,
        temb_txt_silu: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod[1](temb_img_silu)  # [B, 6*dim]
        txt_mod_params = self.txt_mod[1](temb_txt_silu)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation

        img_normed = self.img_norm1(hidden_states)

        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)
        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output

        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(
            img_normed2, img_mod2, modulate_index
        )
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


def to_hashable(obj):
    if isinstance(obj, list):
        return tuple(to_hashable(x) for x in obj)
    return obj


class QwenImageTransformer2DModel(CachableDiT, OffloadableDiTMixin):
    """
    The Transformer model introduced in Qwen.

    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    param_names_mapping = QwenImageDitConfig().arch_config.param_names_mapping

    def __init__(
        self,
        config: QwenImageDitConfig,
        hf_config: dict[str, Any],
    ):
        super().__init__(config=config, hf_config=hf_config)
        patch_size = config.arch_config.patch_size
        in_channels = config.arch_config.in_channels
        out_channels = config.arch_config.out_channels
        num_layers = config.arch_config.num_layers
        attention_head_dim = config.arch_config.attention_head_dim
        num_attention_heads = config.arch_config.num_attention_heads
        joint_attention_dim = config.arch_config.joint_attention_dim
        axes_dims_rope = config.arch_config.axes_dims_rope
        self.zero_cond_t = getattr(config.arch_config, "zero_cond_t", False)
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.use_additional_t_cond: bool = getattr(
            config.arch_config, "use_additional_t_cond", False
        )  # For qwen-image-layered now
        self.use_layer3d_rope: bool = getattr(
            config.arch_config, "use_layer3d_rope", False
        )  # For qwen-image-layered now

        if not self.use_layer3d_rope:
            self.rotary_emb = QwenEmbedRope(
                theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
            )
        else:
            self.rotary_emb = QwenEmbedLayer3DRope(
                theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
            )

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            use_additional_t_cond=self.use_additional_t_cond,
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=self.zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.timestep_zero = torch.zeros(
            (1,), dtype=torch.int, device=get_local_torch_device()
        )

        self.layer_names = ["transformer_blocks"]

    @functools.lru_cache(maxsize=50)
    def build_modulate_index(self, img_shapes: tuple[int, int, int], device):
        modulate_index_list = []
        for sample in img_shapes:
            first_size = sample[0][0] * sample[0][1] * sample[0][2]
            total_size = sum(s[0] * s[1] * s[2] for s in sample)
            idx = (torch.arange(total_size, device=device) >= first_size).int()
            modulate_index_list.append(idx)

        modulate_index = torch.stack(modulate_index_list)
        return modulate_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] = None,
        additional_t_cond: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if (
            attention_kwargs is not None
            and attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        hidden_states = self.img_in(hidden_states)

        timestep = (timestep / 1000).to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, self.timestep_zero], dim=0)
            device = timestep.device
            modulate_index = self.build_modulate_index(to_hashable(img_shapes), device)
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

        temb_img_silu = F.silu(temb)
        if self.zero_cond_t:
            temb_txt = temb.chunk(2, dim=0)[0]
            temb_txt_silu = temb_img_silu.chunk(2, dim=0)[0]
        else:
            temb_txt = temb
            temb_txt_silu = temb_img_silu

        image_rotary_emb = freqs_cis
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb_img_silu=temb_img_silu,
                temb_txt_silu=temb_txt_silu,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block // interval_control]
                )
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb_txt)

        output = self.proj_out(hidden_states)
        return output


EntryClass = QwenImageTransformer2DModel