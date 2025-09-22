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

"""Focal modulation networks (see https://arxiv.org/abs/2203.11926)."""

# Adapted from:
# https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/focalnet/modeling_focalnet.py
# https://github.com/microsoft/FocalNet/blob/v1.0.1/classification/focalnet.py

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Size, Tensor, nn


__all__ = ["FocalDecoder", "FocalEncoder"]


class DynamicTanh(nn.Module):
    """Dynamic tanh activation function.

    See https://arxiv.org/abs/2503.10622.

    Parameters
    ----------
    normalized_shape:
        Input shape for normalization.
    tanhscale_init:
        Initial value for tanh scaling parameter.

    """

    def __init__(
        self,
        normalized_shape: "Union[int, List[int], Size]",
        tanhscale_init: "float" = 0.5,
    ) -> "None":
        super().__init__()
        self.normalized_shape = normalized_shape
        self.tanhscale_init = tanhscale_init

        # Parameters
        self.alpha = nn.Parameter(torch.full((1,), tanhscale_init))

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., *normalized_shape).

        Returns
        -------
            Output tensor of shape (..., *normalized_shape).

        """
        return (self.alpha * input).tanh()

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}("
            f"normalized_shape={self.normalized_shape}, "
            f"tanhscale_init={self.tanhscale_init})"
        )


class FeedForward(nn.Module):
    """Feed-forward neural network module.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the activation and output projection layers.

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
        output = input
        output = self.in_proj(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class ChunkFeedForward(nn.Module):
    """Feed-forward neural network module applied chunk-wise.

    Parameters
    ----------
    chunk_size:
        Number of tokens per chunk.
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the activation and output projection layers.

    """

    def __init__(
        self,
        chunk_size: "int" = 4,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.chunk_size = chunk_size
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.in_proj = nn.Linear(chunk_size * dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(ffn_dim, chunk_size * dim)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., seq_length, dim).

        Returns
        -------
            Output tensor of shape (..., seq_length, dim).

        """
        in_shape = input.shape
        input = input.flatten(start_dim=-2)
        TC = input.shape[-1]
        chunk_size = self.chunk_size * self.dim
        residual = input
        rem_length = TC % chunk_size
        input = nn.functional.pad(
            input,
            [0, chunk_size - rem_length],
        )
        input = input.unflatten(dim=-1, sizes=(-1, chunk_size))
        output = self.in_proj(input)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        output = output.flatten(start_dim=-2)
        output = output[..., :TC]
        output += residual
        output = output.reshape(in_shape)
        return output


class FocalModulation(nn.Module):
    """Focal modulation layer that combines local and global context for processing the input.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    focal_window:
        Size of the initial focal window.
    focal_level:
        Number of focal levels for hierarchical context aggregation.
    focal_factor:
        Scaling factor for focal window sizes across levels.
    dropout:
        Dropout probability applied to the output.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator for stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).
    store_hidden:
        If True, store the hidden states (`self.gates` and `self.modulator`).
        Useful for inspecting the model (e.g. plotting the modulator).

    """

    def __init__(
        self,
        dim: "int" = 1024,
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
        store_hidden: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size
        self.store_hidden = store_hidden

        # Modules
        self.in_proj = nn.Linear(dim, 2 * dim + focal_level + 1)
        self.layers = nn.ModuleList()
        self.activation = nn.GELU()
        self.context_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.causal_pads = []

        for k in range(focal_level):
            kernel_size = focal_factor * k + focal_window
            self.causal_pads.append(kernel_size - 1)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=0 if causal else "same",
                        groups=dim,
                    ),
                    nn.GELU(),
                )
            )

        # Global context
        if causal:
            kernel_size = window_size
            self.causal_pads.append(kernel_size - 1)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        groups=dim,
                    ),
                    nn.GELU(),
                )
            )

        if use_post_norm:
            self.norm = (
                DynamicTanh(dim, tanhscale_init) if self.causal else nn.LayerNorm(dim)
            )
        else:
            # JIT compilable
            self.norm = nn.Identity()

        # JIT compilable
        self.gates = torch.as_tensor(float("nan"))
        self.modulator = torch.as_tensor(float("nan"))

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        left_contexts:
            Left contexts for each layer.
            If provided, each tensor should be of shape (batch_size, kernel_size_i - 1, dim),
            except for the last, which should be of shape (batch_size, window_size - 1, dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated left contexts for each layer.

        """
        input = self.in_proj(input).permute(0, 2, 1)
        query, context, gates = input.split(
            [self.dim, self.dim, self.focal_level + 1], 1
        )

        # Context aggregation
        new_left_contexts: List[Optional[Tensor]] = []
        context_all = 0.0
        for level, layer in enumerate(self.layers):
            causal_pad = self.causal_pads[level]
            if self.causal:
                if left_contexts is None or left_contexts[level] is None:
                    context = nn.functional.pad(
                        context, [causal_pad, 0], mode="replicate"
                    )
                else:
                    left_context = left_contexts[level]
                    # JIT compilable
                    context_ = (
                        [left_context.permute(0, 2, 1), context]
                        if left_context is not None
                        else [context]
                    )
                    context = torch.cat(context_, dim=-1)
                new_left_context = context[..., -causal_pad:].permute(0, 2, 1)
                new_left_contexts.append(new_left_context)
            else:
                new_left_contexts.append(None)
            context = layer(context)
            context_all += context * gates[:, level : level + 1]

        # Global average pooling
        if not self.causal:
            new_left_contexts.append(None)
            context = context.mean(dim=-1)
            context_global = self.activation(context)
            context_global = context_global[..., None] * gates[:, self.focal_level :]
            context_all += context_global

        # Normalize context
        if self.normalize_modulator:
            context_all /= self.focal_level + 1

        # Focal modulation
        modulator = self.context_proj(context_all)
        output = query * modulator
        output = output.permute(0, 2, 1)
        if self.use_post_norm:
            output = self.norm(output)

        output = self.out_proj(output)
        output = self.dropout(output)

        if self.store_hidden:
            self.modulator = modulator
            self.gates = gates

        return output, new_left_contexts


class FocalBlock(nn.Module):
    """Focal block that integrates focal modulation and feed forward layers with
    optional layer scaling.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimensionality of the feed-forward network.
    focal_window:
        Size of the initial focal window in the modulation layer.
    focal_level:
        Number of hierarchical focal levels in the modulation layer.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layer.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layer for
        stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size

        # Modules
        self.modulation_norm = (
            DynamicTanh(dim, tanhscale_init) if self.causal else nn.LayerNorm(dim)
        )
        self.modulation = FocalModulation(
            dim,
            focal_window,
            focal_level,
            focal_factor,
            dropout,
            use_post_norm,
            tanhscale_init,
            normalize_modulator,
            causal,
            window_size,
        )
        self.feed_forward_norm = (
            DynamicTanh(dim, tanhscale_init) if self.causal else nn.LayerNorm(dim)
        )
        self.feed_forward = FeedForward(
            dim,
            ffn_dim,
            dropout,
        )

        if use_layerscale:
            self.modulation_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
            self.feed_forward_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
        else:
            # JIT compilable
            self.modulation_gamma = 1.0
            self.feed_forward_gamma = 1.0

    @property
    def gates(self) -> "Tensor":
        return self.modulation.gates

    @property
    def modulator(self) -> "Tensor":
        return self.modulation.modulator

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        left_contexts:
            Left contexts for each layer.
            If provided, each tensor should be of shape (batch_size, kernel_size_i - 1, dim)
            except for the last, which should be of shape (batch_size, window_size - 1, dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated left contexts for each layer.

        """
        output = input
        if not self.use_post_norm:
            output = self.modulation_norm(output)
        output, left_contexts = self.modulation(output, left_contexts)
        if self.use_post_norm:
            output = self.modulation_norm(output)
        if self.use_layerscale:
            output *= self.modulation_gamma
        output += input

        shortcut = output
        if self.use_post_norm:
            output = self.feed_forward(output)
            output = self.feed_forward_norm(output)
        else:
            output = self.feed_forward_norm(output)
            output = self.feed_forward(output)
        if self.use_layerscale:
            output *= self.feed_forward_gamma
        output += shortcut

        return output, left_contexts


class Snake1d(nn.Module):
    """Snake activation function for 1D inputs, allowing for periodic inductive bias.

    See https://arxiv.org/abs/2006.08195.

    Parameters
    ----------
    dim:
        Dimension of input/output features.

    """

    def __init__(self, dim: "int" = 1024) -> "None":
        super().__init__()
        self.dim = dim

        # Parameters
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        alpha = self.alpha.T
        gate = (alpha * input).sin() ** 2
        output = input + (alpha + 1e-9).reciprocal() * gate
        return output

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(dim={self.dim})"


class DownScale(nn.Module):
    """Downscale 1D input features using convolution
    followed by a Snake activation.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    output_dim:
        Number of output channels.
    kernel_size:
        Size of the convolutional kernel.
    stride:
        Stride of the convolution.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        output_dim: "int" = 1024,
        kernel_size: "int" = 1,
        stride: "int" = 1,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.downscale = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
        )
        self.activation = Snake1d(output_dim)

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
            Left context tensor of shape (batch_size, kernel_size - 1, input_dim).
            If None, initialized as zeros.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length // stride, output_dim);
            - updated left context tensor for next chunk.

        """
        input = input.permute(0, 2, 1)
        if self.causal:
            if left_context is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context = left_context.permute(0, 2, 1)
                input = torch.cat([left_context, input], dim=-1)
                if self.causal_pad == 0:
                    # ONNX compilable
                    input = input[..., 1:]
            # If the module is stateless (causal_pad = 0), return a dummy tensor to comply with the interface
            left_context = input[..., -max(self.causal_pad, 1) :]
            left_context = left_context.permute(0, 2, 1)
        else:
            T = input.shape[-1]
            pad = (T - self.kernel_size) % self.stride
            if pad > 0:
                input = nn.functional.pad(input, [0, pad])
            left_context = None
        output = self.downscale(input)
        output = output.permute(0, 2, 1)
        output = self.activation(output)

        return output, left_context


class UpScale(nn.Module):
    """Upscale 1D input features using Snake activation
    followed by a transposed convolution.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    output_dim:
        Number of output channels.
    kernel_size:
        Size of the transposed convolutional kernel.
    stride:
        Stride of the transposed convolution.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        output_dim: "int" = 1024,
        kernel_size: "int" = 1,
        stride: "int" = 1,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.activation = Snake1d(input_dim)
        self.upscale = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
        )

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
            Left context tensor of shape (batch_size, kernel_size - 1, input_dim).
            If None, initialized as zeros.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length * stride, output_dim);
            - updated left context tensor for next chunk.

        """
        T = input.shape[-2]
        input = input.permute(0, 2, 1)
        if self.causal:
            if left_context is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context = left_context.permute(0, 2, 1)
                input = torch.cat([left_context, input], dim=-1)
                if self.causal_pad == 0:
                    # ONNX compilable
                    input = input[..., 1:]
            # If the module is stateless (causal_pad = 0), return a dummy tensor to comply with the interface
            left_context = input[..., -max(self.causal_pad, 1) :]
            left_context = left_context.permute(0, 2, 1)
        else:
            left_context = None
        input = self.activation(input.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.upscale(input)
        if self.causal:
            output = output[..., -T * self.stride :]
        output = output.permute(0, 2, 1)

        return output, left_context


class FocalDownScale(nn.Module):
    """Focal downscale that combines downscaling and focal modulation.

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    output_dim:
        Dimension of the output features.
    downscale_factor:
        Downscaling factor.
    focal_window:
        Size of the initial focal window in the modulation layer.
    focal_level:
        Number of hierarchical focal levels in the modulation layer.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layer.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for
        stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        output_dim: "int" = 1024,
        downscale_factor: "int" = 1,
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.downscale_factor = downscale_factor
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size

        # Modules
        self.downscale = DownScale(
            input_dim,
            output_dim,
            downscale_factor,
            downscale_factor,
            causal,
        )
        self.focal_block = FocalBlock(
            output_dim,
            output_dim * 4,
            focal_window,
            focal_level,
            focal_factor,
            dropout,
            use_post_norm,
            use_layerscale,
            layerscale_init,
            tanhscale_init,
            normalize_modulator,
            causal,
            window_size,
        )

    @property
    def gates(self) -> "Tensor":
        return self.focal_block.gates

    @property
    def modulator(self) -> "Tensor":
        return self.focal_block.modulator

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
            If provided, each tensor should be of shape (batch_size, kernel_size_i - 1, output_dim),
            except for the first, which should be of shape (batch_size, kernel_size_0 - 1, input_dim),
            and the last, which should be of shape (batch_size, window_size - 1, output_dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length // downscale_factor, output_dim);
            - updated left contexts for each layer.

        """
        output, new_left_context_first = self.downscale(
            input,
            None if left_contexts is None else left_contexts[0],
        )
        output, new_left_contexts_next = self.focal_block(
            output,
            None if left_contexts is None else left_contexts[1:],
        )
        new_left_contexts = [new_left_context_first] + new_left_contexts_next

        return output, new_left_contexts


class FocalUpScale(nn.Module):
    """Focal upscale that combines focal modulation and upscaling.

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    output_dim:
        Dimension of the output features.
    upscale_factor:
        Upscaling factor.
    focal_window:
        Size of the initial focal window in the modulation layer.
    focal_level:
        Number of hierarchical focal levels in the modulation layer.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layer.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for
        stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        output_dim: "int" = 1024,
        upscale_factor: "int" = 1,
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.upscale_factor = upscale_factor
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size

        # Modules
        self.focal_block = FocalBlock(
            input_dim,
            input_dim * 4,
            focal_window,
            focal_level,
            focal_factor,
            dropout,
            use_post_norm,
            use_layerscale,
            layerscale_init,
            tanhscale_init,
            normalize_modulator,
            causal,
            window_size,
        )
        self.upscale = UpScale(
            input_dim, output_dim, upscale_factor, upscale_factor, causal
        )

    @property
    def gates(self) -> "Tensor":
        return self.focal_block.gates

    @property
    def modulator(self) -> "Tensor":
        return self.focal_block.modulator

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
            If provided, each tensor should be of shape (batch_size, kernel_size_i - 1, input_dim),
            except for the second last, which should be of shape (batch_size, window_size - 1, input_dim),
            and the last, which should be of shape (batch_size, kernel_size_k - 1, output_dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length * upscale_factor, output_dim);
            - updated left contexts for each layer.

        """
        output, new_left_contexts_first = self.focal_block(
            input,
            None if left_contexts is None else left_contexts[:-1],
        )
        output, new_left_context_next = self.upscale(
            output,
            None if left_contexts is None else left_contexts[-1],
        )
        new_left_contexts = new_left_contexts_first + [new_left_context_next]

        return output, new_left_contexts


class FocalEncoder(nn.Module):
    """Focal encoder that applies a series of focal downscale layers.

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    output_dim:
        Dimension of the output features.
    hidden_dims:
        Sequence of hidden dimensions in the modulation layers.
    downscale_factors:
        Sequence of downscaling factors for each layer.
    focal_window:
        Size of the initial focal window in the modulation layers.
    focal_level:
        Number of hierarchical focal levels in the modulation layers.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layers.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for
        stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        output_dim: "int" = 12,
        hidden_dims: "Sequence[int]" = (1024, 1024, 1024),
        downscale_factors: "Sequence[int]" = (1, 1, 1),
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.downscale_factors = downscale_factors
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size
        self.downsample_factor = torch.Size(downscale_factors).numel()
        self.chunk_size = self.downsample_factor

        # Modules
        self.layers = nn.ModuleList()
        for hidden_dim, downscale_factor in zip(hidden_dims, downscale_factors):
            layer = FocalDownScale(
                input_dim,
                hidden_dim,
                downscale_factor,
                focal_window,
                focal_level,
                focal_factor,
                dropout,
                use_post_norm,
                use_layerscale,
                layerscale_init,
                tanhscale_init,
                normalize_modulator,
                causal,
                window_size,
            )
            self.layers.append(layer)
            input_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, output_dim)

    @property
    def gates(self) -> "List[Tensor]":
        return [layer.gates for layer in self.layers]

    @property
    def modulators(self) -> "List[Tensor]":
        return [layer.modulator for layer in self.layers]

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[List[Optional[Tensor]]]]]" = None,
    ) -> "Tuple[Tensor, List[List[Optional[Tensor]]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        left_contexts:
            Left contexts for each layer.
            If provided, each tensor in the inner list should be of shape (batch_size, kernel_size_i - 1, output_dim),
            except for the first, which should be of shape (batch_size, kernel_size_0 - 1, input_dim),
            and the last, which should be of shape (batch_size, window_size - 1, output_dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length // prod(downscale_factors), output_dim);
            - updated left contexts for each layer.

        """
        new_left_contexts: List[List[Optional[Tensor]]] = []
        output = input
        for i, layer in enumerate(self.layers):
            output, new_left_contexts_i = layer(
                output,
                None if left_contexts is None else left_contexts[i],
            )
            new_left_contexts.append(new_left_contexts_i)
        output = self.dropout(output)
        output = self.out_proj(output)
        return output, new_left_contexts


class FocalDecoder(nn.Module):
    """Focal decoder that applies a series of focal upscale layers.

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    output_dim:
        Dimension of the output features.
    hidden_dims:
        Sequence of hidden dimensions in the modulation layers.
    upscale_factors:
        Sequence of upscaling factors for each layer.
    focal_window:
        Size of the initial focal window in the modulation layers.
    focal_level:
        Number of hierarchical focal levels in the modulation layers.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layers.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for
        stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).
    last_window_size:
        Maximum number of past tokens each token can attend to
        in the last modulation layer (used only if causal=True).
    lookahead_size:
        Maximum number of future tokens each token can attend to
        (used only if causal=True).

    """

    def __init__(
        self,
        input_dim: "int" = 12,
        output_dim: "int" = 1024,
        hidden_dims: "Sequence[int]" = (1024, 1024, 1024),
        upscale_factors: "Sequence[int]" = (1, 1, 1),
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
        last_window_size: "int" = 512,
        lookahead_size: "int" = 3,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.upscale_factors = upscale_factors
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size
        self.last_window_size = last_window_size
        self.lookahead_size = lookahead_size
        self.upsample_factor = torch.Size(upscale_factors).numel()
        self.chunk_size = 1 + lookahead_size

        # Modules
        hidden_dims = tuple(hidden_dims) + (output_dim,)
        output_dim = hidden_dims[0]
        hidden_dims = hidden_dims[1:]
        self.in_proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for i, (hidden_dim, upscale_factor) in enumerate(
            zip(hidden_dims, upscale_factors)
        ):
            layer = FocalUpScale(
                output_dim,
                hidden_dim,
                upscale_factor,
                focal_window,
                focal_level,
                focal_factor,
                dropout,
                use_post_norm,
                use_layerscale,
                layerscale_init,
                tanhscale_init,
                normalize_modulator,
                causal,
                window_size if i < len(hidden_dims) - 1 else last_window_size,
            )
            self.layers.append(layer)
            output_dim = hidden_dim

        if self.causal and self.lookahead_size > 0:
            self.refiner = ChunkFeedForward(
                self.chunk_size,
                hidden_dims[-1],
                self.chunk_size * hidden_dims[-1],
                dropout,
            )
        else:
            # JIT compilable
            self.refiner = nn.Identity()

    @property
    def gates(self) -> "List[Tensor]":
        return [layer.gates for layer in self.layers]

    @property
    def modulators(self) -> "List[Tensor]":
        return [layer.modulator for layer in self.layers]

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[List[Optional[Tensor]]]]]" = None,
    ) -> "Tuple[Tensor, List[List[Optional[Tensor]]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        left_contexts:
            Left contexts for each layer.
            If provided, each tensor in the inner list should be of shape (batch_size, kernel_size_i - 1, input_dim),
            except for the second last, which should be of shape (batch_size, window_size - 1, input_dim),
            and the last, which should be of shape (batch_size, kernel_size_k - 1, output_dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length * prod(upscale_factors), output_dim);
            - updated left contexts for each layer.

        """
        new_left_contexts: List[List[Optional[Tensor]]] = []
        output = self.in_proj(input)
        output = self.dropout(output)
        for i, layer in enumerate(self.layers):
            output, new_left_contexts_i = layer(
                output,
                None if left_contexts is None else left_contexts[i],
            )
            new_left_contexts.append(new_left_contexts_i)

        if self.causal and self.lookahead_size > 0:
            output = self.refiner(output)

        return output, new_left_contexts


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    encoder = FocalEncoder(causal=True).to(device)
    print(
        f"Encoder size: {sum([x.numel() for x in encoder.state_dict().values()]) / 1e6:.2f}M"
    )
    decoder = FocalDecoder(causal=True).to(device)
    print(
        f"Decoder size: {sum([x.numel() for x in decoder.state_dict().values()]) / 1e6:.2f}M"
    )

    input = torch.randn(B, T, 1024, device=device)
    output, left_contexts = encoder(input)
    output, left_contexts = decoder(output)

    encoder_jit = torch.jit.script(encoder)
    decoder_jit = torch.jit.script(decoder)
    output_jit, left_contexts_jit = encoder_jit(input)
    output_jit, left_contexts_jit = decoder_jit(output_jit)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    output.sum().backward()
    for k, v in encoder.named_parameters():
        assert v.grad is not None, k
    for k, v in decoder.named_parameters():
        assert v.grad is not None, k

    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    for xs, ys in zip(left_contexts, left_contexts_jit):
        for x, y in zip(xs, ys):
            assert torch.allclose(x, y, atol=1e-6), ((x - y) ** 2).mean().sqrt()

    print("Model test passed")


def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 50
    encoder = FocalEncoder(causal=True).to(device)
    decoder = FocalDecoder(causal=True).to(device)

    input = torch.randn(B, T, 1024, device=device)
    batch_output, batch_left_contexts = decoder(encoder(input)[0])
    single_output = []
    single_left_contexts = []
    for i in range(B):
        output, left_contexts = decoder(encoder(input[i][None])[0])
        single_output.append(output)
        single_left_contexts.append(left_contexts)
    single_output = torch.cat(single_output)
    single_left_contexts = [
        [torch.cat(tensors, dim=0) for tensors in zip(*inner_list)]
        for inner_list in zip(*single_left_contexts)
    ]

    assert torch.allclose(batch_output, single_output, atol=1e-2), (
        ((batch_output - single_output) ** 2).mean().sqrt(),
    )
    for xs, ys in zip(batch_left_contexts, single_left_contexts):
        for x, y in zip(xs, ys):
            assert torch.allclose(x, y, atol=1e-2), ((x - y) ** 2).mean().sqrt()

    print("Batch invariance test passed")


@torch.no_grad()
def test_causality() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 10
    encoder = FocalEncoder(causal=True).to(device)
    decoder = FocalDecoder(causal=True).to(device)

    input = torch.randn(B, T, 1024, device=device)
    encoder = torch.jit.script(encoder)
    decoder = torch.jit.script(decoder)
    output, *_ = decoder(encoder(input)[0])

    incremental_output = []
    encoder_state = []
    decoder_state = []
    chunk_size = max(encoder.chunk_size, decoder.chunk_size)
    i = 0
    while i < T:
        output_i, *encoder_state = encoder(input[:, i : i + chunk_size], *encoder_state)
        output_i, *decoder_state = decoder(output_i, *decoder_state)
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
    T = 10
    model = FocalEncoder(causal=True).eval().to(device)

    input = torch.randn(B, T, 1024, device=device)
    _, left_contexts = model(input)
    flat_left_contexts = [x for xs in left_contexts for x in xs]

    f = io.BytesIO()
    torch.onnx.export(
        model,
        (input, left_contexts),
        f,
        input_names=[
            "input",
            *[f"input_left_contexts.{i}" for i, _ in enumerate(flat_left_contexts)],
        ],
        output_names=[
            "output",
            *[f"output_left_contexts.{i}" for i, _ in enumerate(flat_left_contexts)],
        ],
        dynamic_axes={
            "input": {0: "batch", 1: "feature_time"},
            "output": {0: "batch", 1: "latent_time"},
            **{
                f"input_left_contexts.{i}": {0: "batch"}
                for i, _ in enumerate(flat_left_contexts)
            },
            **{
                f"output_left_contexts.{i}": {0: "batch"}
                for i, _ in enumerate(flat_left_contexts)
            },
        },
    )
    onnx_bytes = f.getvalue()

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX")
        return

    input = torch.randn(2 * B, 2 * T, 1024, device=device)
    _, left_contexts = model(input)
    flat_left_contexts = [x for xs in left_contexts for x in xs]

    session = ort.InferenceSession(onnx_bytes)
    inputs_ort = dict(
        zip(
            [x.name for x in session.get_inputs()],
            [input.cpu().numpy()] + [x.cpu().numpy() for x in flat_left_contexts],
        )
    )
    outputs_ort = session.run([x.name for x in session.get_outputs()], inputs_ort)
    output, left_contexts = model(input, left_contexts)
    flat_left_contexts = [x for xs in left_contexts for x in xs]

    assert torch.allclose(torch.tensor(outputs_ort[0]), output.cpu(), atol=1e-2), (
        ((torch.tensor(outputs_ort[0]) - output.cpu()) ** 2).mean().sqrt(),
    )
    for x, y in zip(outputs_ort[1:], flat_left_contexts):
        assert torch.allclose(torch.tensor(x), y.cpu(), atol=1e-2), (
            ((torch.tensor(x) - y.cpu()) ** 2).mean().sqrt()
        )

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_causality()
    test_onnx()
