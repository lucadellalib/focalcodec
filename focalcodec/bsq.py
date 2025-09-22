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

"""Binary spherical quantization (see https://arxiv.org/abs/2406.07548)."""

# Adapted from:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/3e4ce165774d3f5944f12ffb5ccd02848bb38df6/vector_quantize_pytorch/lookup_free_quantization.py

import math
from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = ["BinarySphericalQuantizer"]


class BinarySphericalQuantizer(nn.Module):
    """Binary spherical quantizer that maps inputs to binary codes on the unit hypersphere.

    Parameters
    ----------
    codebook_size:
        Number of binary codes in the codebook.

    """

    def __init__(self, codebook_size: "int" = 4096) -> "None":
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = int(math.log2(codebook_size))

        # Buffers
        self.register_buffer(
            "codebook_value",
            torch.tensor(1 / math.sqrt(self.dim)),
            persistent=False,
        )
        self.register_buffer(
            "mask", 2 ** torch.arange(self.dim - 1, -1, -1), persistent=False
        )
        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self._bits_to_codes(bits) * self.codebook_value
        self.register_buffer("codebook", codebook.float(), persistent=False)

    def forward(self, lats: "Tensor") -> "Tuple[Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            - Output tokens of shape (...);
            - output codes (i.e. quantized latents) of shape (..., dim).

        """
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        return toks, codes

    @torch.jit.export
    def lats_to_codes(self, lats: "Tensor") -> "Tensor":
        """Transform latents into codes (i.e. quantized latents).

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            Output codes of shape (..., dim).

        """
        return torch.where(lats > 0, self.codebook_value, -self.codebook_value)

    @torch.jit.export
    def lats_to_toks(self, lats: "Tensor") -> "Tensor":
        """Transform latents into tokens.

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            Output tokens of shape (...).

        """
        return self.codes_to_toks(lats)

    @torch.jit.export
    def codes_to_toks(self, codes: "Tensor") -> "Tensor":
        """Transform codes (i.e. quantized latents) into tokens.

        Parameters
        ----------
        codes:
            Input codes of shape (..., dim).

        Returns
        -------
            Output tokens of shape (...).

        """
        return ((codes > 0) * self.mask).sum(dim=-1)

    @torch.jit.export
    def toks_to_codes(self, toks: "Tensor") -> "Tensor":
        """Transform tokens into codes (i.e. quantized latents).

        Parameters
        ----------
        toks:
            Input tokens of shape (...).

        Returns
        -------
            Output codes of shape (..., dim).

        """
        # ONNX compilable
        bits = ((toks[..., None] // self.mask) % 2).to(self.codebook.dtype)
        return self._bits_to_codes(bits) * self.codebook_value

    def _bits_to_codes(self, bits: "Tensor") -> "Tensor":
        return bits * 2 - 1

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(codebook_size={self.codebook_size})"


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = BinarySphericalQuantizer().to(device)
    print(model)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    lats = torch.randn(B, T, model.dim, device=device)
    toks, codes = model(lats)
    codes2 = model.lats_to_codes(lats)
    toks2 = model.lats_to_toks(lats)
    toks3 = model.codes_to_toks(codes)
    assert (toks == toks2).all()
    assert (toks == toks3).all()
    assert (codes == codes2).all()
    model_jit = torch.jit.script(model)
    toks_jit, codes_jit = model_jit(lats)
    assert (toks == toks_jit).all()
    assert (codes == codes_jit).all()

    print("Model test passed")


def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 50
    model = BinarySphericalQuantizer().to(device)

    lats = torch.randn(B, T, model.dim, device=device)
    batch_toks, batch_codes = model(lats)

    all_single_toks, all_single_codes = [], []
    for i in range(B):
        single_toks, single_codes = model(lats[i][None])
        all_single_toks.append(single_toks)
        all_single_codes.append(single_codes)
    all_single_toks = torch.cat(all_single_toks)
    all_single_codes = torch.cat(all_single_codes)

    assert (batch_toks == all_single_toks).all()
    assert (batch_codes == all_single_codes).all()

    print("Batch invariance test passed")


@torch.no_grad()
def test_onnx() -> "None":
    import io
    import warnings

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = BinarySphericalQuantizer().eval().to(device)

    lats = torch.randn(B, T, model.dim, device=device)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        (lats,),
        f,
        input_names=["lats"],
        output_names=["toks", "codes"],
        dynamic_axes={
            "lats": {0: "batch", 1: "latent_time"},
            "toks": {0: "batch", 1: "latent_time"},
            "codes": {0: "batch", 1: "latent_time"},
        },
    )
    onnx_bytes = f.getvalue()

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX")
        return

    lats = torch.randn(2 * B, 2 * T, model.dim, device=device)

    session = ort.InferenceSession(onnx_bytes)
    inputs_ort = dict(zip([x.name for x in session.get_inputs()], [lats.cpu().numpy()]))
    outputs_ort = session.run([x.name for x in session.get_outputs()], inputs_ort)
    toks, codes = model(lats)

    assert (toks.cpu() == torch.tensor(outputs_ort[0])).all()
    assert (codes.cpu() == torch.tensor(outputs_ort[1])).all()

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_onnx()
