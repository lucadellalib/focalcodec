# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""WavLM (see https://arxiv.org/abs/2110.13900)."""

# Adapted from:
# https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/wavlm/modeling_wavlm.py
# https://github.com/microsoft/unilm/blob/2d5b25f7cecc4ab4ad44f65de5899ef6aa8c53f5/wavlm/WavLM.py
# https://github.com/microsoft/unilm/blob/2d5b25f7cecc4ab4ad44f65de5899ef6aa8c53f5/wavlm/modules.py

import math
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


__all__ = ["WavLM"]


try:
    from torch.nn.attention.flex_attention import flex_attention

    HAS_FLEX_ATTENTION = True

    flex_attention = torch.compile(flex_attention)

    def build_bias_mod(bias: "Tensor") -> "Callable":
        def bias_mod(
            score: "Tensor",
            batch: "Tensor",
            head: "Tensor",
            q_idx: "Tensor",
            k_idx: "Tensor",
        ) -> "Tensor":
            return score + bias[batch, head, q_idx, k_idx]

        return bias_mod

except ImportError:
    HAS_FLEX_ATTENTION = False


class ConvBlock(nn.Module):
    """Convolutional block with dropout, normalization, and activation.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    output_dim:
        Number of output channels.
    kernel_size:
        Size of the convolutional kernel.
    stride:
        Stride for the convolution.
    bias:
        Whether to include a bias term in the convolution.
    dropout:
        Dropout probability.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 512,
        output_dim: "int" = 512,
        kernel_size: "int" = 10,
        stride: "int" = 5,
        bias: "bool" = False,
        dropout: "float" = 0.0,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.dropout_ = dropout
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        input: "Tensor",
        left_context: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        left_context:
            Left context of shape (batch_size, kernel_size - 1, input_dim).
            If None, initialized as zeros.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length // stride, output_dim);
            - updated left context for next chunk.

        """
        input = input.permute(0, 2, 1)
        if self.causal:
            if left_context is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context = left_context.permute(0, 2, 1)
                input = torch.cat([left_context, input], dim=-1)
            left_context = input[..., -self.causal_pad :].permute(0, 2, 1)
        else:
            delta = self.kernel_size - input.shape[-1]
            if delta > 0:
                input = nn.functional.pad(input, [0, delta])
            left_context = None
        output = self.conv(input)
        output = self.dropout(output)
        output = output.permute(0, 2, 1)
        # NOTE: when training, it is recommended to run this layer in fp32
        output = self.norm(output)
        output = self.activation(output)

        return output, left_context


class FeatureExtractor(nn.Module):
    """Feature extractor that applies a series of convolutional layers.

    Parameters
    ----------
    input_dim:
        Number of input channels or features.
    hidden_dims:
        Number of output channels for each convolutional layer.
    kernel_sizes:
        Kernel size for each convolutional layer.
    strides:
        Stride for each convolutional layer.
    bias:
        Whether to include a bias term in the convolutional layers.
    dropout:
        Dropout probability applied after each convolutional block.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1,
        hidden_dims: "Sequence[int]" = (512,) + (512,) * 4 + (512,) * 2,
        kernel_sizes: "Sequence[int]" = (10,) + (3,) * 4 + (2,) * 2,
        strides: "Sequence[int]" = (5,) + (2,) * 4 + (2,) * 2,
        bias: "bool" = False,
        dropout: "float" = 0.0,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.bias = bias
        self.dropout = dropout
        self.causal = causal

        # Modules
        self.layers = nn.ModuleList()
        for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides):
            layer = ConvBlock(
                input_dim, hidden_dim, kernel_size, stride, bias, dropout, causal
            )
            self.layers.append(layer)
            input_dim = hidden_dim

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        left_contexts:
            Left contexts for each layer.
            If provided, each tensor should be of shape (batch_size, kernel_size_i - 1, input_dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length / prod(strides), hidden_dims[-1]);
            - updated left contexts for each layer.

        """
        new_left_contexts: List[Optional[Tensor]] = []
        output = input
        for i, layer in enumerate(self.layers):
            output, new_left_context = layer(
                output,
                None if left_contexts is None else left_contexts[i],
            )
            new_left_contexts.append(new_left_context)
        return output, new_left_contexts


class FeedForward(nn.Module):
    """Feed-forward neural network.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the activation layer.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.in_proj = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(ffn_dim, dim)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        output = self.in_proj(input)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class ConvPositionalEmbedding(nn.Module):
    """Convolutional positional embedding.

    Parameters
    ----------
    dim:
        Number of input/output channels.
    kernel_size:
        Size of the convolutional kernel.
    groups:
        Number of convolutional groups.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        kernel_size: "int" = 128,
        groups: "int" = 16,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.causal = causal
        self.causal_pad = kernel_size - 1
        self.remove = 1 if kernel_size % 2 == 0 else 0

        # Modules
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=0 if causal else kernel_size // 2,
            groups=groups,
        )
        self.activation = nn.GELU()

    def forward(
        self,
        input: "Tensor",
        left_context: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        left_context:
            Left context of shape (batch_size, kernel_size - 1, dim).
            If None, initialized as zeros.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated left context for next chunk.

        """
        input = input.permute(0, 2, 1)
        if self.causal:
            if left_context is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context = left_context.permute(0, 2, 1)
                input = torch.cat([left_context, input], dim=-1)
            left_context = input[..., -self.causal_pad :].permute(0, 2, 1)
        else:
            left_context = None
        output = self.conv(input)
        if not self.causal and self.remove > 0:
            output = output[..., : -self.remove]
        output = self.activation(output)
        output = output.permute(0, 2, 1)

        return output, left_context


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional embeddings.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability for attention weights.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        num_heads: "int" = 16,
        dropout: "float" = 0.0,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.head_dim = dim // num_heads

        # Modules
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        # Parameters
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        input: "Tensor",
        position_bias: "Tensor",
        mask: "Optional[Tensor]" = None,
        curr_pos: "Optional[Tensor]" = None,
        kv_cache: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor]]":
        """Forward pass.

        This method applies relative positional embeddings and multi-head
        attention and handles key-value caching.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        position_bias:
            Precomputed relative positional embeddings for the current input sequence,
            corresponding to positions from `curr_pos` to `curr_pos + seq_length`.
            Shape (num_heads, tgt_seq_length, src_seq_length).
        mask:
            Float mask that is added to the attention scores,
            shape (..., tgt_seq_length, src_seq_length).
        curr_pos:
            Starting position of the current input sequence.
            Default to 0.
        kv_cache:
            Tensor to cache key-value pairs.
            If provided, it should be of shape (batch_size, curr_pos, num_heads, head_dim, 2).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated position `curr_pos + seq_length`;
            - updated key-value cache.

        """
        if curr_pos is None:
            curr_pos = torch.tensor(0, device=input.device)

        B, T, _ = input.shape
        if self.causal:
            next_pos = curr_pos + T
        else:
            next_pos = curr_pos

        if self.causal:
            if kv_cache is None:
                # TODO: avoid hard-coding (might cause issues with ONNX)
                min_cache_size = 512
                kv_cache = torch.zeros(
                    B,
                    max(min_cache_size, T),
                    self.num_heads,
                    self.head_dim,
                    2,
                    device=input.device,
                    dtype=input.dtype,
                )
            elif next_pos > kv_cache.shape[1]:
                # Expand along time dimension
                kv_cache = nn.functional.pad(
                    kv_cache, [0, 0, 0, 0, 0, 0, 0, int(next_pos) - kv_cache.shape[1]]
                )

        qkvs = self.qkv_proj(input).reshape(B, T, -1, self.head_dim)
        qs, ks, vs = qkvs.chunk(3, dim=-2)

        if self.causal and kv_cache is not None:
            kv_cache = kv_cache.type_as(qs)
            kv_cache[:, curr_pos:next_pos, :, :, 0] = ks
            kv_cache[:, curr_pos:next_pos, :, :, 1] = vs

            ks = kv_cache[..., :next_pos, :, :, 0]
            vs = kv_cache[..., :next_pos, :, :, 1]

        # Reshape for scaled_dot_product_attention
        qs = qs.permute(0, 2, 1, 3)  # [B, num_heads, T, head_dim]
        ks = ks.permute(0, 2, 1, 3)  # [B, num_heads, next_pos, head_dim]
        vs = vs.permute(0, 2, 1, 3)  # [B, num_heads, next_pos, head_dim]

        # Compute gated relative position bias
        gated_input = input.reshape(input.shape[:-1] + (self.num_heads, -1))
        gated_input = gated_input.permute(0, 2, 1, 3)

        relative_position_proj = self.gru_rel_pos_linear(gated_input)
        relative_position_proj = relative_position_proj.reshape(
            gated_input.shape[:-1] + (2, 4)
        ).sum(dim=-1)

        gate_a, gate_b = relative_position_proj.sigmoid().chunk(2, dim=-1)
        gate_input = gate_a * (gate_b * self.gru_rel_pos_const.type_as(qs) - 1.0) + 2.0
        gated_position_bias = gate_input * position_bias

        if mask is not None:
            # `mask` must be a float tensor
            gated_position_bias = gated_position_bias + mask

        gated_position_bias = gated_position_bias.type_as(qs)
        output = self._scaled_dot_product_attention(
            qs,
            ks,
            vs,
            gated_position_bias,
        )  # [B, num_heads, T, head_dim]

        # [B, T, num_heads * head_dim]
        output = output.permute(0, 2, 1, 3).reshape(B, T, -1)
        output = self.out_proj(output)  # [B, T, dim]

        return output, next_pos, kv_cache

    def _scaled_dot_product_attention(
        self,
        qs: "Tensor",
        ks: "Tensor",
        vs: "Tensor",
        gated_position_bias: "Tensor",
    ) -> "Tensor":
        return nn.functional.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            attn_mask=gated_position_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )


class MultiHeadFlexAttention(MultiHeadAttention):
    """See documentation of `MultiHeadAttention`."""

    def _scaled_dot_product_attention(
        self,
        qs: "Tensor",
        ks: "Tensor",
        vs: "Tensor",
        gated_position_bias: "Tensor",
    ) -> "Tensor":
        if not torch.is_grad_enabled() and (not self.training or self.dropout == 0.0):
            return self._flex_attention(
                qs,
                ks,
                vs,
                gated_position_bias,
            )
        return nn.functional.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            attn_mask=gated_position_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )

    @torch.jit.ignore
    def _flex_attention(
        self,
        qs: torch.Tensor,
        ks: torch.Tensor,
        vs: torch.Tensor,
        gated_position_bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.attention.flex_attention.flex_attention(
            qs,
            ks,
            vs,
            score_mod=build_bias_mod(gated_position_bias),
        )


class TransformerLayer(nn.Module):
    """Transformer layer comprising self-attention, feed-forward
    and normalization layers.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    num_heads:
        Number of attention heads in the self-attention mechanism.
    dropout:
        Dropout probability applied in the attention and feed-forward layers.
    causal:
        Whether the module should be causal.
    use_flex_attention:
        Whether to use FlexAttention (if available).

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        dropout: "float" = 0.0,
        causal: "bool" = False,
        use_flex_attention: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_ = dropout
        self.causal = causal
        self.use_flex_attention = use_flex_attention
        if use_flex_attention and not HAS_FLEX_ATTENTION:
            warnings.warn(
                f"FlexAttention is not available on this platform and/or PyTorch version"
            )

        # Modules
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = (
            MultiHeadFlexAttention
            if use_flex_attention and HAS_FLEX_ATTENTION
            else MultiHeadAttention
        )(dim, num_heads, dropout, causal)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)

    def forward(
        self,
        input: "Tensor",
        position_bias: "Tensor",
        mask: "Optional[Tensor]" = None,
        curr_pos: "Optional[Tensor]" = None,
        kv_cache: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor]]":
        """See documentation of `MultiHeadAttention.forward`."""
        output = input
        residual = output

        output = self.attention_norm(output)
        output, curr_pos, kv_cache = self.attention(
            output,
            position_bias,
            mask,
            curr_pos,
            kv_cache,
        )
        output = self.dropout(output)
        output = residual + output

        residual = output
        output = self.feed_forward_norm(output)
        output = self.feed_forward(output)
        output = residual + output

        return output, curr_pos, kv_cache


class TransformerEncoder(nn.Module):
    """Transformer encoder with relative positional embeddings and
    convolutional positional embeddings.

    Parameters
    ----------
    num_layers:
        Number of transformer layers in the encoder.
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the feed-forward layer within each transformer layer.
    num_heads:
        Number of attention heads in each transformer layer.
    num_buckets:
        Number of buckets for relative positional embeddings.
    max_distance:
        Maximum distance for relative positional embeddings.
    max_cached_steps:
        Maximum number of time steps for which relative positional
        embeddings are cached to avoid recomputation (improves
        runtime at the cost of increased memory usage).
    dropout:
        Dropout probability applied throughout the model.
    conv_pos:
        Size of the convolutional positional embeddings.
    conv_pos_groups:
        Number of groups in the convolutional positional embeddings.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).
    lookahead_size:
        Maximum number of future tokens each token can attend to
        (used only if causal=True).
    use_flex_attention:
        Whether to use FlexAttention (if available).

    Raises
    ------
    ValueError
        If `max_cached_steps` is greater than 0 but smaller than `window_size`. In this case,
        caching is misconfigured and may result in incorrect or incomplete internal states.

    """

    def __init__(
        self,
        num_layers: "int" = 6,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        num_buckets: "int" = 320,
        max_distance: "int" = 800,
        max_cached_steps: "int" = 2048,
        dropout: "float" = 0.0,
        conv_pos: "int" = 128,
        conv_pos_groups: "int" = 16,
        causal: "bool" = False,
        window_size: "int" = 512,
        lookahead_size: "int" = 3,
        use_flex_attention: "bool" = False,
    ) -> "None":
        if max_cached_steps > 0 and max_cached_steps < window_size:
            raise ValueError(
                f"`max_cached_steps` ({max_cached_steps}) must be either zero "
                f"or at least `window_size` ({window_size})"
            )

        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.max_cached_steps = max_cached_steps
        self.dropout = dropout
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.causal = causal
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        self.use_flex_attention = use_flex_attention
        self.chunk_size = 1 + lookahead_size

        # Needed to compute position bias
        self._num_buckets = num_buckets // 2
        self._num_buckets_minus_one = self._num_buckets - 1
        self._max_exact = self._num_buckets // 2
        self._num_buckets_minus_max_exact = self._num_buckets - self._max_exact
        self._log_max_distance_over_max_exact = math.log(
            self.max_distance / self._max_exact
        )

        # Modules
        self.positional_embedding = ConvPositionalEmbedding(
            dim,
            conv_pos,
            conv_pos_groups,
            causal,
        )
        self.relative_embedding = nn.Embedding(num_buckets, num_heads)
        self.layers = nn.ModuleList(
            TransformerLayer(
                dim,
                ffn_dim,
                num_heads,
                dropout,
                causal,
                use_flex_attention,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

        # Non-persistent buffers
        @torch.no_grad()
        def _update_position_bias(
            this: "nn.Module", *_args: "Any", **_kwargs: "Any"
        ) -> "None":
            if max_cached_steps > 0:
                this.position_bias = this._compute_bias(
                    max_cached_steps,
                    max(max_cached_steps, window_size) if causal else max_cached_steps,
                )

        self.register_load_state_dict_post_hook(_update_position_bias)
        self.register_buffer(
            "position_bias",
            torch.as_tensor([[[float("nan")]]]),
            persistent=False,
        )
        _update_position_bias(self)

    def forward(
        self,
        input: "Tensor",
        curr_pos: "Optional[Tensor]" = None,
        left_context: "Optional[Tensor]" = None,
        kv_caches: "Optional[List[Optional[Tensor]]]" = None,
        length: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor], List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        curr_pos:
            Starting position of the current input sequence.
            Default to 0.
        left_context:
            Left context of shape (batch_size, dim, conv_pos - 1).
        kv_caches:
            Key-value caches for each layer.
            If provided, each tensor should be of shape
            (batch_size, min(curr_pos, window_size - chunk_size), num_heads, head_dim, 2).
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated position `curr_pos + seq_length`;
            - updated left context for next chunk;
            - updated key-value caches for each layer.

        """
        if self.causal:
            return self._forward_causal(input, curr_pos, left_context, kv_caches)
        return self._forward_bidirectional(input, length)

    def _forward_bidirectional(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor], List[Optional[Tensor]]]":
        T = input.shape[1]

        if self.training or T > self.max_cached_steps:
            position_bias = self._compute_bias(T, T)
        else:
            position_bias = self.position_bias[:, :T, :T]

        if length is None:
            key_padding_mask = None
        else:
            B = input.shape[0]
            abs_length = (length * T).ceil().clamp(max=T).long()
            key_padding_mask = (
                torch.arange(T, device=input.device).expand(B, T) < abs_length[:, None]
            )
            inv_key_padding_mask = ~key_padding_mask
            input = input.clone()
            input[inv_key_padding_mask] = 0.0
            key_padding_mask = inv_key_padding_mask.float().masked_fill_(
                inv_key_padding_mask, -float("inf")
            )
            key_padding_mask = key_padding_mask[:, None, None]

        output = input
        positional_embs, _ = self.positional_embedding(output)
        output = output + positional_embs
        output = self.dropout(output)
        position_bias = position_bias.type_as(output)

        new_kv_caches: List[Optional[Tensor]] = []
        for i, layer in enumerate(self.layers):
            output, _, _ = layer(output, position_bias, key_padding_mask)
            new_kv_caches.append(None)

        # NOTE: No output norm
        return output, torch.tensor(0, device=input.device), None, new_kv_caches

    def _forward_causal(
        self,
        input: "Tensor",
        curr_pos: "Optional[Tensor]" = None,
        left_context: "Optional[Tensor]" = None,
        kv_caches: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor], List[Optional[Tensor]]]":
        if curr_pos is None:
            curr_pos = torch.tensor(0, device=input.device)

        T = input.shape[1]
        device = input.device
        next_pos = curr_pos + T
        chunk_size = self.chunk_size

        if self.training or T > self.max_cached_steps:
            position_bias = self._compute_bias(next_pos, next_pos)
        else:
            position_bias = self.position_bias

        # Identify special cases where the rectangular causal mask simplifies to a non-causal mask
        if T <= chunk_size:
            mask = None
        else:
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (T, curr_pos + T), and the only non-masked entries are (i, j) for
            # j in [curr_pos + i - window_size, curr_pos + i + lookahead_size],
            # since row i corresponds to token curr_pos + i
            end_idxes = torch.arange(
                curr_pos + chunk_size,
                next_pos + chunk_size,
                device=device,
            )
            if chunk_size > 1:
                end_idxes = ((end_idxes // chunk_size) * chunk_size).clamp(max=next_pos)
            start_idxes = (end_idxes - self.window_size).clamp(min=0)
            idxes = torch.arange(start_idxes[0], next_pos, device=device)
            mask_ = (idxes[None, :] >= start_idxes[:, None]) & (
                idxes[None, :] < end_idxes[:, None]
            )
            mask = torch.full_like(mask_, fill_value=-float("inf"), dtype=input.dtype)
            mask.masked_fill_(mask_, 0.0)  # (T, min(next_pos, window_size))

        new_kv_caches: List[Optional[Tensor]] = []
        output = input
        positional_embs, left_context = self.positional_embedding(output, left_context)
        output = output + positional_embs
        output = self.dropout(output)
        end_idx = next_pos.clamp(
            max=torch.as_tensor(position_bias.shape[1], device=device)
        )
        start_idx = end_idx - T
        position_bias = position_bias[:, start_idx:end_idx, :next_pos]
        if position_bias.shape[1] > position_bias.shape[2] or (
            mask is not None and mask.shape[1] > position_bias.shape[2]
        ):
            position_bias = self._compute_bias(next_pos, next_pos)[:, start_idx:end_idx]
        position_bias = position_bias.type_as(output)

        curr_pos = curr_pos.clamp(max=self.window_size - chunk_size)
        kv_cache_start_idx = next_pos - self.window_size + chunk_size
        for i, layer in enumerate(self.layers):
            output, _, new_kv_cache = layer(
                output,
                position_bias,
                mask,
                curr_pos,
                None if kv_caches is None else kv_caches[i],  # JIT compilable
            )
            # Prune cache
            if new_kv_cache is not None:
                # new_kv_cache = new_kv_cache[:, kv_cache_start_idx.clamp(min=0):next_pos]
                # Roll cache
                shift = kv_cache_start_idx.clamp(min=0)
                new_kv_cache = torch.cat(
                    [new_kv_cache[:, shift:], new_kv_cache[:, :shift]], dim=1
                )
                new_kv_cache = new_kv_cache[:, : self.window_size]
            new_kv_caches.append(new_kv_cache)
        next_pos = next_pos.clamp(max=self.window_size - chunk_size)

        # NOTE: No output norm
        return output, next_pos, left_context, new_kv_caches

    def _compute_bias(self, query_length: "int", key_length: "int") -> "Tensor":
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(
            self.relative_embedding.weight.device
        )
        values = self.relative_embedding(relative_position_bucket)
        values = values.permute(2, 0, 1)
        return values

    def _relative_positions_bucket(self, relative_positions: "Tensor") -> "Tensor":
        relative_buckets = (relative_positions > 0).to(torch.long) * self._num_buckets
        relative_positions = relative_positions.abs()
        is_small = relative_positions < self._max_exact
        relative_positions_if_large = (
            relative_positions.float() / self._max_exact
        ).log()
        relative_positions_if_large /= self._log_max_distance_over_max_exact
        relative_positions_if_large *= self._num_buckets_minus_max_exact
        relative_positions_if_large += self._max_exact
        relative_positions_if_large = relative_positions_if_large.to(torch.long)
        relative_positions_if_large = relative_positions_if_large.clamp(
            max=self._num_buckets_minus_one
        )
        relative_buckets += torch.where(
            is_small, relative_positions, relative_positions_if_large
        )
        return relative_buckets


class WavLM(nn.Module):
    """WavLM model.

    Parameters
    ----------
    hidden_dims:
        Number of filters for each convolutional layer.
    kernel_sizes:
        Kernel sizes for each convolutional layer.
    strides:
        Strides for each convolutional layer.
    num_layers:
        Number of transformer layers in the encoder.
    dim:
        Dimension of the input and output embeddings in the transformer.
    ffn_dim:
        Dimension of the feed-forward layer within each transformer layer.
    num_heads:
        Number of attention heads in each transformer layer.
    num_buckets:
        Number of buckets for relative positional embeddings.
    max_distance:
        Maximum distance for relative positional embeddings.
    max_cached_steps:
        Maximum number of time steps for which relative positional
        embeddings are cached to avoid recomputation (improves
        runtime at the cost of increased memory usage).
    dropout:
        Dropout probability applied throughout the model.
    conv_pos:
        Size of the convolutional positional embeddings.
    conv_pos_groups:
        Number of groups in the convolutional positional embeddings.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).
    lookahead_size:
        Maximum number of future tokens each token can attend to
        (used only if causal=True).
    use_flex_attention:
        Whether to use FlexAttention (if available).

    """

    def __init__(
        self,
        hidden_dims: "Sequence[int]" = (512,) + (512,) * 4 + (512,) * 2,
        kernel_sizes: "Sequence[int]" = (10,) + (3,) * 4 + (2,) * 2,
        strides: "Sequence[int]" = (5,) + (2,) * 4 + (2,) * 2,
        num_layers: "int" = 6,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        num_buckets: "int" = 320,
        max_distance: "int" = 800,
        max_cached_steps: "int" = 2048,
        dropout: "float" = 0.0,
        conv_pos: "int" = 128,
        conv_pos_groups: "int" = 16,
        causal: "bool" = False,
        window_size: "int" = 512,
        lookahead_size: "int" = 3,
        use_flex_attention: "bool" = False,
    ) -> "None":
        super().__init__()
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.max_cached_steps = max_cached_steps
        self.dropout = dropout
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.causal = causal
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        self.use_flex_attention = use_flex_attention

        # Modules
        self.feature_extractor = FeatureExtractor(
            input_dim=1,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dropout=dropout,
            causal=causal,
        )
        self.norm = nn.LayerNorm(hidden_dims[-1])
        self.feature_proj = nn.Linear(hidden_dims[-1], dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            num_layers,
            dim,
            ffn_dim,
            num_heads,
            num_buckets,
            max_distance,
            max_cached_steps,
            dropout,
            conv_pos,
            conv_pos_groups,
            causal,
            window_size,
            lookahead_size,
            use_flex_attention,
        )
        self.sample_rate = 16000
        self.downsample_factor = torch.Size(strides).numel()
        self.chunk_size = self.downsample_factor * (1 + lookahead_size)

    def forward(
        self,
        input: "Tensor",
        curr_pos: "Optional[Tensor]" = None,
        left_contexts: "Optional[List[Optional[Tensor]]]" = None,
        kv_caches: "Optional[List[Optional[Tensor]]]" = None,
        length: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, List[Optional[Tensor]], List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length).
        curr_pos:
            Starting position of the current input sequence.
            Default to 0.
        left_contexts:
            Left contexts for each feature extractor layer and positional embedding.
            If provided, each tensor should be of shape (batch_size, kernel_size_i - 1, input_dim).
        kv_caches:
            Key-value caches for each encoder layer.
            If provided, each tensor should be of shape (batch_size, min(curr_pos, window_size - 1), num_heads, head_dim, 2).
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated position `curr_pos + seq_length`;
            - updated left contexts for each feature extractor layer and positional embedding;
            - updated key-value caches for each encoder layer.

        """
        # [B, T, 1]
        input = input[..., None]
        encoder_left_context = None if left_contexts is None else left_contexts[-1]
        output, left_contexts = self.feature_extractor(input, left_contexts)
        output = self.norm(output)
        output = self.feature_proj(output)
        output = self.dropout(output)
        output, curr_pos, left_context, kv_caches = self.encoder(
            output,
            curr_pos,
            encoder_left_context,
            kv_caches,
            length,
        )
        left_contexts.append(left_context)
        return output, curr_pos, left_contexts, kv_caches


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 16000
    model = WavLM(causal=True).to(device)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    input = torch.randn(B, T, device=device)
    output, curr_pos, left_contexts, kv_caches = model(input)
    model_jit = torch.jit.script(model)
    output_jit, curr_pos_jit, left_contexts_jit, kv_caches_jit = model_jit(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert curr_pos == curr_pos_jit, curr_pos - curr_pos_jit
    for x, y in zip(left_contexts, left_contexts_jit):
        assert torch.allclose(x, y, atol=1e-6), ((x - y) ** 2).mean().sqrt()
    for x, y in zip(kv_caches, kv_caches_jit):
        assert torch.allclose(x, y, atol=1e-6), ((x - y) ** 2).mean().sqrt()

    print("Model test passed")


def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 16000
    model = WavLM(causal=True).to(device)

    input = torch.randn(B, T, device=device)
    batch_output, batch_curr_pos, batch_left_contexts, batch_kv_caches = model(input)
    single_output = []
    single_curr_pos = []
    single_left_contexts = []
    single_kv_caches = []
    for i in range(B):
        output, curr_pos, left_contexts, kv_caches = model(input[i][None])
        single_output.append(output)
        single_curr_pos.append(curr_pos)
        single_left_contexts.append(left_contexts)
        single_kv_caches.append(kv_caches)
    single_output = torch.cat(single_output)
    assert all(x == single_curr_pos[0] for x in single_curr_pos), single_curr_pos
    single_curr_pos = single_curr_pos[0]
    single_left_contexts = [torch.cat(xs) for xs in zip(*single_left_contexts)]
    single_kv_caches = [torch.cat(xs) for xs in zip(*single_kv_caches)]

    assert torch.allclose(batch_output, single_output, atol=1e-2), (
        ((batch_output - single_output) ** 2).mean().sqrt(),
    )
    assert batch_curr_pos == single_curr_pos, batch_curr_pos - single_curr_pos
    for x, y in zip(batch_left_contexts, single_left_contexts):
        assert torch.allclose(x, y, atol=1e-2), ((x - y) ** 2).mean().sqrt()
    for x, y in zip(batch_kv_caches, single_kv_caches):
        assert torch.allclose(x, y, atol=1e-2), ((x - y) ** 2).mean().sqrt()

    print("Batch invariance test passed")


@torch.no_grad()
def test_causality() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 4 * 512 * 320
    model = WavLM(causal=True).to(device)

    input = torch.randn(B, T, device=device)
    model = torch.jit.script(model)
    output, *_ = model(input)

    incremental_output = []
    state = []
    chunk_size = model.chunk_size
    i = 0
    while i < T:
        output_i, *state = model(input[:, i : i + chunk_size], *state)
        incremental_output.append(output_i)
        i += chunk_size
    incremental_output = torch.cat(incremental_output, dim=1)

    assert torch.allclose(output, incremental_output, atol=1e-2), (
        ((output - incremental_output) ** 2).mean().sqrt(),
    )

    print("Causality test passed")


@torch.no_grad()
def test_onnx() -> "None":
    import io
    import warnings

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 4 * 320
    model = WavLM(causal=True).eval().to(device)

    input = torch.randn(B, T, device=device)
    _, curr_pos, left_contexts, kv_caches = model(input)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        (input, curr_pos, left_contexts, kv_caches),
        f,
        input_names=[
            "input",
            "curr_pos",
            *[f"input_left_contexts.{i}" for i, _ in enumerate(left_contexts)],
            *[f"input_kv_caches.{i}" for i, _ in enumerate(kv_caches)],
        ],
        output_names=[
            "output",
            "next_pos",
            *[f"output_left_contexts.{i}" for i, _ in enumerate(left_contexts)],
            *[f"output_kv_caches.{i}" for i, _ in enumerate(kv_caches)],
        ],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "feature_time"},
            **{
                f"input_left_contexts.{i}": {0: "batch"}
                for i, _ in enumerate(left_contexts)
            },
            **{
                f"output_left_contexts.{i}": {0: "batch"}
                for i, _ in enumerate(left_contexts)
            },
            **{
                f"input_kv_caches.{i}": {0: "batch", 1: "feature_time"}
                for i, _ in enumerate(kv_caches)
            },
            **{
                f"output_kv_caches.{i}": {0: "batch", 1: "feature_time"}
                for i, _ in enumerate(kv_caches)
            },
        },
    )
    onnx_bytes = f.getvalue()

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX")
        return

    # Keep same shape
    input = torch.randn(B, T, device=device)
    _, curr_pos, left_contexts, kv_caches = model(input)

    session = ort.InferenceSession(onnx_bytes)
    inputs_ort = dict(
        zip(
            [x.name for x in session.get_inputs()],
            [input.cpu().numpy(), curr_pos.cpu().numpy()]
            + [x.cpu().numpy() for x in left_contexts]
            + [x.cpu().numpy() for x in kv_caches],
        )
    )
    outputs_ort = session.run([x.name for x in session.get_outputs()], inputs_ort)
    output, curr_pos, left_contexts, kv_caches = model(
        input, curr_pos, left_contexts, kv_caches
    )

    assert torch.allclose(torch.tensor(outputs_ort[0]), output.cpu(), atol=1e-2), (
        ((torch.tensor(outputs_ort[0]) - output.cpu()) ** 2).mean().sqrt(),
    )
    for x, y in zip(outputs_ort[1:], [curr_pos] + left_contexts + kv_caches):
        assert torch.allclose(torch.tensor(x), y.cpu(), atol=1e-2), (
            ((torch.tensor(x) - y.cpu()) ** 2).mean().sqrt()
        )

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_causality()
    test_onnx()
