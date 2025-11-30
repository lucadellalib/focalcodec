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

"""Vocos (see https://arxiv.org/abs/2306.00814)."""

# Adapted from:
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/heads.py
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/models.py
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/modules.py
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/spectral_ops.py

import warnings
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn


__all__ = ["Vocos"]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for 1D input.

    Parameters
    ----------
    dim:
        Number of input channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    layerscale_init:
        Initial value for layer scaling parameter.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        layerscale_init: "Optional[float]" = None,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.layerscale_init = layerscale_init
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=0 if causal else "same", groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(ffn_dim, dim)

        # Parameters
        if layerscale_init is not None:
            self.gamma = nn.Parameter(
                torch.full((dim,), layerscale_init),
            )
        else:
            self.gamma = None

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
        orig_input = input
        if self.causal:
            if left_context is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context = left_context.permute(0, 2, 1)
                input = torch.cat([left_context, input], dim=-1)
            left_context = input[..., -self.causal_pad :].permute(0, 2, 1)
        else:
            left_context = None
        output = self.dwconv(input)
        output = output.permute(0, 2, 1)
        output = self.norm(output)
        output = self.pwconv1(output)
        output = self.activation(output)
        output = self.pwconv2(output)
        if self.gamma is not None:
            output = self.gamma * output
        output = output.permute(0, 2, 1)
        output = orig_input + output
        output = output.permute(0, 2, 1)

        return output, left_context


class VocosBackbone(nn.Module):
    """Vocos backbone.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of hidden channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    layerscale_init:
        Initial value for layer scaling parameter.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        layerscale_init: "Optional[float]" = None,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.layerscale_init = (
            1 / num_layers if layerscale_init is None else layerscale_init
        )
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.embedding = nn.Conv1d(
            input_dim, dim, kernel_size, padding=0 if causal else "same"
        )
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.layers = nn.ModuleList(
            ConvNeXtBlock(dim, ffn_dim, kernel_size, self.layerscale_init, causal)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)

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
            If provided, the first tensor should be of shape (batch_size, kernel_size - 1, input_dim),
            the following tensors should be of shape (batch_size, kernel_size - 1, dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated left contexts for each layer.

        """
        input = input.permute(0, 2, 1)
        left_context_embedding = None if left_contexts is None else left_contexts[0]
        if self.causal:
            if left_context_embedding is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context_embedding = left_context_embedding.permute(0, 2, 1)
                input = torch.cat([left_context_embedding, input], dim=-1)
            new_left_context = input[..., -self.causal_pad :].permute(0, 2, 1)
            new_left_contexts: List[Optional[Tensor]] = [new_left_context]
        else:
            new_left_contexts: List[Optional[Tensor]] = [None]
        output = self.embedding(input)
        output = output.permute(0, 2, 1)
        output = self.input_norm(output)
        for i, layer in enumerate(self.layers):
            output, new_left_context = layer(
                output,
                None if left_contexts is None else left_contexts[i + 1],
            )
            new_left_contexts.append(new_left_context)
        output = self.output_norm(output)
        return output, new_left_contexts


class ISTFT(nn.Module):
    """Custom implementation of inverse STFT with support for non-causal and causal padding.

    Parameters
    ----------
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    win_length:
        Size of window frame and STFT filter.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        win_length: "int" = 1024,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.causal = causal
        self.non_causal_pad = (win_length - hop_length) // 2

        # Buffers (JIT compilable)
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("window_sq", window**2, persistent=False)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, n_fft // 2 + 1).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        # Inverse FFT
        input = input.permute(0, 2, 1)
        ifft = torch.fft.irfft(input, self.n_fft, dim=1, norm="backward")

        if self.causal:
            output = ifft[:, -self.hop_length :].permute(0, 2, 1)
            output = output.flatten(start_dim=1)
            return output

        # Overlap-add
        T = input.shape[-1]
        ifft = ifft * self.window[None, :, None]
        output_size = int((T - 1) * self.hop_length + self.win_length)
        output = nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, self.non_causal_pad : -self.non_causal_pad]

        # Window envelope
        window_sq = self.window_sq.expand(1, T, -1).permute(0, 2, 1)
        window_envelope = nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[0, 0, 0, self.non_causal_pad : -self.non_causal_pad]

        # Normalize
        output /= window_envelope

        return output

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}("
            f"n_fft={self.n_fft}, "
            f"hop_length={self.hop_length}, "
            f"win_length={self.win_length})"
        )


class ISTFTHead(nn.Module):
    """Inverse STFT head.

    Parameters
    ----------
    dim:
        Number of input channels.
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 512,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.causal = causal

        # Modules
        self.proj = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, n_fft, causal)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        output = self.proj(input)
        mag, phase = output.chunk(2, dim=-1)
        mag = mag.exp()
        # Safeguard to prevent excessively large magnitudes
        mag = mag.clamp(max=1e2)
        # Real and imaginary value
        # JIT compilable
        stft = mag * torch.complex(phase.cos(), phase.sin())
        output = self.istft(stft)
        return output


class Vocos(nn.Module):
    """Vocos generator for waveform synthesis.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of input channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    layerscale_init:
        Initial value for layer scaling parameter.
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        layerscale_init: "Optional[float]" = None,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        causal: "bool" = False,
        **kwargs: "Any",
    ) -> "None":
        super().__init__()
        if kwargs.get("input_channels", None) is not None:
            warnings.warn(
                "`input_channels` is deprecated, please use `input_dim` instead",
                DeprecationWarning,
            )
            input_dim = kwargs["input_channels"]
        if kwargs.get("padding", None) is not None:
            warnings.warn(
                "`padding` is deprecated and no longer used. It is now automatically set "
                'to "causal" if the model is causal, "same" otherwise',
                DeprecationWarning,
            )

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.layerscale_init = layerscale_init or 1 / num_layers
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.causal = causal
        self.upsample_factor = hop_length
        self.chunk_size = 1

        # Modules
        self.backbone = VocosBackbone(
            input_dim,
            num_layers,
            dim,
            ffn_dim,
            kernel_size,
            layerscale_init,
            causal,
        )
        self.head = ISTFTHead(dim, n_fft, hop_length, causal)

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
            Left contexts for each backbone layer.
            If provided, the first tensor should be of shape (batch_size, kernel_size - 1, input_dim),
            the following tensors should be of shape (batch_size, kernel_size - 1, dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length * hop_length);
            - updated left contexts for each backbone layer.

        """
        output, left_contexts = self.backbone(input, left_contexts)
        output = self.head(output)
        return output, left_contexts


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = Vocos(causal=True).to(device)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    input = torch.randn(B, T, 1024, device=device)
    output, left_contexts = model(input)
    model_jit = torch.jit.script(model)
    output_jit, left_contexts_jit = model_jit(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    for x, y in zip(left_contexts, left_contexts_jit):
        assert torch.allclose(x, y, atol=1e-6), ((x - y) ** 2).mean().sqrt()

    print("Model test passed")


def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 50
    model = Vocos(causal=True).to(device)

    input = torch.randn(B, T, 1024, device=device)
    batch_output, batch_left_contexts = model(input)
    single_output = []
    single_left_contexts = []
    for i in range(B):
        output, left_contexts = model(input[i][None])
        single_output.append(output)
        single_left_contexts.append(left_contexts)
    single_output = torch.cat(single_output)
    single_left_contexts = [torch.cat(xs) for xs in zip(*single_left_contexts)]

    assert torch.allclose(batch_output, single_output, atol=1e-2), (
        ((batch_output - single_output) ** 2).mean().sqrt(),
    )
    for x, y in zip(batch_left_contexts, single_left_contexts):
        assert torch.allclose(x, y, atol=1e-2), ((x - y) ** 2).mean().sqrt()

    print("Batch invariance test passed")


@torch.no_grad()
def test_causality() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 10
    model = Vocos(causal=True).to(device)

    input = torch.randn(B, T, 1024, device=device)
    model = torch.jit.script(model)
    output, *_ = model(input)

    incremental_output = []
    state = []
    chunk_size = 1
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
    T = 50
    model = Vocos(causal=True).eval().to(device)

    input = torch.randn(B, T, 1024, device=device)
    _, left_contexts = model(input)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        (input, left_contexts),
        f,
        input_names=[
            "input",
            *[f"input_left_contexts.{i}" for i, _ in enumerate(left_contexts)],
        ],
        output_names=[
            "output",
            *[f"output_left_contexts.{i}" for i, _ in enumerate(left_contexts)],
        ],
        dynamic_axes={
            "input": {0: "batch", 1: "feature_time"},
            "output": {0: "batch", 1: "time"},
            **{
                f"input_left_contexts.{i}": {0: "batch"}
                for i, _ in enumerate(left_contexts)
            },
            **{
                f"output_left_contexts.{i}": {0: "batch"}
                for i, _ in enumerate(left_contexts)
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

    session = ort.InferenceSession(onnx_bytes)
    inputs_ort = dict(
        zip(
            [x.name for x in session.get_inputs()],
            [input.cpu().numpy()] + [x.cpu().numpy() for x in left_contexts],
        )
    )
    outputs_ort = session.run([x.name for x in session.get_outputs()], inputs_ort)
    output, left_contexts = model(input, left_contexts)

    assert torch.allclose(torch.tensor(outputs_ort[0]), output.cpu(), atol=1e-2), (
        ((torch.tensor(outputs_ort[0]) - output.cpu()) ** 2).mean().sqrt(),
    )
    for x, y in zip(outputs_ort[1:], left_contexts):
        assert torch.allclose(torch.tensor(x), y.cpu(), atol=1e-2), (
            ((torch.tensor(x) - y.cpu()) ** 2).mean().sqrt()
        )

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_causality()
    test_onnx()
