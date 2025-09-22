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

"""FocalCodec."""

import io
import json
import os
import re
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor, nn


try:
    from .bsq import BinarySphericalQuantizer
    from .focalnet import FocalDecoder, FocalEncoder
    from .version import VERSION
    from .vocos import Vocos
    from .wavenext import WaveNeXt
    from .wavlm import WavLM
except ImportError:
    from bsq import BinarySphericalQuantizer
    from focalnet import FocalDecoder, FocalEncoder
    from version import VERSION
    from vocos import Vocos
    from wavenext import WaveNeXt
    from wavlm import WavLM


__all__ = ["FocalCodec"]


REGISTRY = {
    "BinarySphericalQuantizer": BinarySphericalQuantizer,
    "FocalDecoder": FocalDecoder,
    "FocalEncoder": FocalEncoder,
    "Vocos": Vocos,
    "WaveNeXt": WaveNeXt,
    "WavLM": WavLM,
}

DEFAULT_CONFIGS = [
    "lucadellalib/focalcodec_50hz",
    "lucadellalib/focalcodec_50hz_65k_causal",
    "lucadellalib/focalcodec_50hz_4k_causal",
    "lucadellalib/focalcodec_50hz_2k_causal",
    "lucadellalib/focalcodec_25hz",
    "lucadellalib/focalcodec_12_5hz",
]


class FocalCodec(nn.Module):
    """FocalCodec.

    This class initializes a flexible speech codec system, allowing customizable
    components for encoding, compression, quantization, decompression, and decoding.

    Parameters
    ----------
    encoder_name:
        Encoder registered name (see `REGISTRY`).
    encoder_config:
        Encoder configuration, i.e. keyword arguments for initializing the encoder.
    compressor_name:
        Compressor registered name (see `REGISTRY`).
    compressor_config:
        Compressor configuration, i.e. keyword arguments for initializing the compressor.
    quantizer_name:
        Quantizer registered name (see `REGISTRY`).
    quantizer_config:
        Quantizer configuration, i.e. keyword arguments for initializing the quantizer.
    decompressor_name:
        Decompressor registered name (see `REGISTRY`).
    decompressor_config:
        Decompressor configuration, i.e. keyword arguments for initializing the decompressor.
    decoder_name:
        Decoder registered name (see `REGISTRY`).
    decoder_config:
        Decoder configuration, i.e. keyword arguments for initializing the decoder.

    """

    __version__ = VERSION

    def __init__(
        self,
        encoder_name: "str" = "WavLM",
        encoder_config: "Optional[Dict[str, Any]]" = None,
        compressor_name: "str" = "FocalEncoder",
        compressor_config: "Optional[Dict[str, Any]]" = None,
        quantizer_name: "str" = "BinarySphericalQuantizer",
        quantizer_config: "Optional[Dict[str, Any]]" = None,
        decompressor_name: "str" = "FocalDecoder",
        decompressor_config: "Optional[Dict[str, Any]]" = None,
        decoder_name: "str" = "WaveNeXt",
        decoder_config: "Optional[Dict[str, Any]]" = None,
    ) -> "None":
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_config = encoder_config or {}
        self.compressor_name = compressor_name
        self.compressor_config = compressor_config or {}
        self.quantizer_name = quantizer_name
        self.quantizer_config = quantizer_config or {}
        self.decompressor_name = decompressor_name
        self.decompressor_config = decompressor_config or {}
        self.decoder_name = decoder_name
        self.decoder_config = decoder_config or {}
        self.model_id = None

        # Validate
        for name in [
            encoder_name,
            compressor_name,
            quantizer_name,
            decompressor_name,
            decoder_name,
        ]:
            if name not in REGISTRY:
                raise ValueError(
                    f"Unregistered module: {name}. Available modules: {list(REGISTRY.keys())}"
                )

        # Modules
        self.encoder = REGISTRY[encoder_name](**self.encoder_config)
        self.compressor = REGISTRY[compressor_name](**self.compressor_config)
        self.quantizer = REGISTRY[quantizer_name](**self.quantizer_config)
        self.decompressor = REGISTRY[decompressor_name](**self.decompressor_config)
        self.decoder = REGISTRY[decoder_name](**self.decoder_config)

    @property
    def sample_rate_input(self) -> "int":
        """Return the input sample rate."""
        return self.encoder.sample_rate

    @property
    def sample_rate_output(self) -> "int":
        """Return the output sample rate."""
        return int(
            self.sample_rate_input
            * self.decoder.upsample_factor
            / self.encoder.downsample_factor
        )

    @property
    def sample_rate(self) -> "int":
        """Return the sample rate."""
        if self.sample_rate_input != self.sample_rate_output:
            raise RuntimeError(
                "`sample_rate` is undefined because input and output sample rates "
                f"differ (input={self.sample_rate_input}, output={self.sample_rate_output}). "
                "Please use `sample_rate_input` or `sample_rate_output` explicitly"
            )
        return self.sample_rate_input

    @property
    def causal(self) -> "bool":
        """Whether the model is causal."""
        parts = (
            self.encoder.causal,
            self.compressor.causal,
            self.decompressor.causal,
            self.decoder.causal,
        )
        return all(bool(x) for x in parts)

    @property
    def chunk_size(self) -> "int":
        """Return the chunk size."""
        return self.encoder.chunk_size

    @property
    def latency(self) -> "Optional[float]":
        """Return the theoretical latency in milliseconds."""
        if self.causal:
            return 1000.0 * self.chunk_size / self.sample_rate_input
        return None

    @property
    def codebook(self) -> "Tensor":
        """Return a copy of the quantizer codebook."""
        return self.quantizer.codebook.clone()

    def forward(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        decompressor_state: "Tuple" = (),
        decoder_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> (
        "Union[Tuple[Tensor, Tensor, Tensor], "
        "Tuple[Tensor, Tensor, Tensor, Tuple, Tuple, Tuple, Tuple]]"
    ):
        """Forward pass.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        decompressor_state:
            Decompressor streaming state.
        decoder_state:
            Decoder streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output tokens of shape (batch_size, latent_seq_length);
            - corresponding codes of shape (batch_size, latent_seq_length, latent_dim);
            - reconstructed signal of shape (batch_size, ~seq_length);
            - [updated encoder streaming state];
            - [updated compressor streaming state];
            - [updated decompressor streaming state];
            - [updated decoder streaming state].

        """
        codes, encoder_state, compressor_state = self.sig_to_codes(
            sig,
            encoder_state,
            compressor_state,
            length,
            return_state=True,
        )
        toks = self.codes_to_toks(codes)
        rec_sig, decompressor_state, decoder_state = self.codes_to_sig(
            codes,
            decompressor_state,
            decoder_state,
            matching_set=None,
            topk=-1,
            num_splits=-1,
            output_length=int(
                sig.shape[-1] * (self.sample_rate_output / self.sample_rate_input)
            ),
            return_state=True,
        )
        if return_state:
            return (
                toks,
                codes,
                rec_sig,
                encoder_state,
                compressor_state,
                decompressor_state,
                decoder_state,
            )
        return toks, codes, rec_sig

    # sig -> any
    @torch.jit.export
    def sig_to_feats(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]}":
        """Transform signal into features.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated encoder streaming state].

        """
        feats, *encoder_state = self.encoder(sig, *encoder_state, length=length)
        if return_state:
            return feats, encoder_state
        return feats

    @torch.jit.export
    def sig_to_lats(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple]]":
        """Transform signal into latents.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output latents of shape (batch_size, latent_seq_length, latent_dim);
            - [updated encoder streaming state];
            - [updated compressor streaming state].

        """
        feats, encoder_state = self.sig_to_feats(
            sig, encoder_state, length, return_state=True
        )
        lats, compressor_state = self.feats_to_lats(
            feats, compressor_state, return_state=True
        )
        if return_state:
            return lats, encoder_state, compressor_state
        return lats

    @torch.jit.export
    def sig_to_toks(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple]]":
        """Transform signal into tokens.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output tokens of shape (batch_size, latent_seq_length);
            - [updated encoder streaming state];
            - [updated compressor streaming state].

        """
        feats, encoder_state = self.sig_to_feats(
            sig, encoder_state, length, return_state=True
        )
        toks, compressor_state = self.feats_to_toks(
            feats, compressor_state, return_state=True
        )
        if return_state:
            return toks, encoder_state, compressor_state
        return toks

    @torch.jit.export
    def sig_to_codes(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple]]":
        """Transform signal into codes (i.e. quantized latents).

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output codes of shape (batch_size, latent_seq_length, latent_dim).
            - [updated encoder streaming state];
            - [updated compressor streaming state].

        """
        feats, encoder_state = self.sig_to_feats(
            sig, encoder_state, length, return_state=True
        )
        codes, compressor_state = self.feats_to_codes(
            feats, compressor_state, return_state=True
        )
        if return_state:
            return codes, encoder_state, compressor_state
        return codes

    @torch.jit.export
    def sig_to_qfeats(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        decompressor_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple, Tuple]]":
        """Transform signal into quantized features.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        decompressor_state:
            Decompressor streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output quantized features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated encoder streaming state];
            - [updated compressor streaming state];
            - [updated decompressor streaming state].

        """
        feats, encoder_state = self.sig_to_feats(
            sig, encoder_state, length, return_state=True
        )
        qfeats, compressor_state, decompressor_state = self.feats_to_qfeats(
            feats, compressor_state, decompressor_state, return_state=True
        )
        if return_state:
            return qfeats, encoder_state, compressor_state, decompressor_state
        return qfeats

    @torch.jit.export
    def sig_to_sig(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        decompressor_state: "Tuple" = (),
        decoder_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple, Tuple, Tuple]]":
        """Transform signal into reconstructed signal.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        decompressor_state:
            Decompressor streaming state.
        decoder_state:
            Decoder streaming state.
        length:
            Relative length of each signal in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Reconstructed signal of shape (batch_size, ~seq_length);
            - [updated encoder streaming state];
            - [updated compressor streaming state];
            - [updated decompressor streaming state];
            - [updated decoder streaming state].

        """
        codes, encoder_state, compressor_state = self.sig_to_codes(
            sig,
            encoder_state,
            compressor_state,
            length,
            return_state=True,
        )
        rec_sig, decompressor_state, decoder_state = self.codes_to_sig(
            codes,
            decompressor_state,
            decoder_state,
            matching_set=None,
            topk=-1,
            num_splits=-1,
            output_length=int(
                sig.shape[-1] * (self.sample_rate_output / self.sample_rate_input)
            ),
            return_state=True,
        )
        if return_state:
            return (
                rec_sig,
                encoder_state,
                compressor_state,
                decompressor_state,
                decoder_state,
            )
        return rec_sig

    # feats -> any
    @torch.jit.export
    def feats_to_lats(
        self,
        feats: "Tensor",
        compressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform features into latents.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        compressor_state:
            Compressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output latents of shape (batch_size, latent_seq_length, latent_dim);
            - [updated compressor streaming state].

        """
        lats, *compressor_state = self.compressor(feats, *compressor_state)
        lats = nn.functional.normalize(lats, dim=-1)
        if return_state:
            return lats, compressor_state
        return lats

    @torch.jit.export
    def feats_to_toks(
        self,
        feats: "Tensor",
        compressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform features into tokens.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        compressor_state:
            Compressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output tokens of shape (batch_size, latent_seq_length);
            - [updated compressor streaming state].

        """
        lats, compressor_state = self.feats_to_lats(
            feats, compressor_state, return_state=True
        )
        toks = self.quantizer.lats_to_toks(lats)
        if return_state:
            return toks, compressor_state
        return toks

    @torch.jit.export
    def feats_to_codes(
        self,
        feats: "Tensor",
        compressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform features into codes (i.e. quantized latents).

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        compressor_state:
            Compressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output codes of shape (batch_size, latent_seq_length, latent_dim);
            - [updated compressor streaming state].

        """
        lats, compressor_state = self.feats_to_lats(
            feats, compressor_state, return_state=True
        )
        codes = self.quantizer.lats_to_codes(lats)
        if return_state:
            return codes, compressor_state
        return codes

    @torch.jit.export
    def feats_to_qfeats(
        self,
        feats: "Tensor",
        compressor_state: "Tuple" = (),
        decompressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple]]":
        """Transform features into quantized features.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        compressor_state:
            Compressor streaming state.
        decompressor_state:
            Decompressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output quantized features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated compressor streaming state];
            - [updated decompressor streaming state].

        """
        lats, compressor_state = self.feats_to_lats(
            feats, compressor_state, return_state=True
        )
        codes = self.quantizer.lats_to_codes(lats)
        qfeats, decompressor_state = self.codes_to_qfeats(
            codes, decompressor_state, return_state=True
        )
        if return_state:
            return qfeats, compressor_state, decompressor_state
        return qfeats

    @torch.jit.export
    def feats_to_sig(
        self,
        feats: "Tensor",
        decoder_state: "Tuple" = (),
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform features into signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the input features (see https://arxiv.org/abs/2305.18975). The refined or
        original features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        decoder_state:
            Decoder streaming state.
        matching_set:
            Optional set of candidate features for kNN refinement,
            shape (num_candidates, hidden_dim).
        topk:
            Number of nearest neighbors to consider in the kNN refinement.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency during kNN computation.
        output_length:
            Desired output length of the synthesized signal. If specified, the output
            will be truncated or padded to this length.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output signal, shape (batch_size, output_length) if
              `output_length` is specified, otherwise (batch_size, ~seq_length);
            - [updated decoder streaming state].

        """
        if matching_set is not None:
            feats = self.knn(
                feats,
                matching_set,
                topk,
                num_splits,
            ).mean(dim=-2)
        sig, *decoder_state = self.decoder(feats, *decoder_state)
        if output_length is not None:
            delta = output_length - sig.shape[1]
            if delta < 0:
                sig = sig[:, :output_length]
            elif delta > 0:
                sig = nn.functional.pad(sig, [0, delta], mode="replicate")
        if return_state:
            return sig, decoder_state
        return sig

    # lats -> any
    @torch.jit.export
    def lats_to_toks(self, lats: "Tensor") -> "Tensor":
        """Transform latents into tokens.

        Parameters
        ----------
        lats:
            Input latents of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tokens of shape (batch_size, latent_seq_length).

        """
        toks = self.quantizer.lats_to_toks(lats)
        return toks

    @torch.jit.export
    def lats_to_codes(self, lats: "Tensor") -> "Tensor":
        """Transform latents into codes (i.e. quantized latents).

        Parameters
        ----------
        lats:
            Input latents of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output codes of shape (batch_size, latent_seq_length, latent_dim).

        """
        codes = self.quantizer.lats_to_codes(lats)
        return codes

    @torch.jit.export
    def lats_to_qfeats(
        self,
        lats: "Tensor",
        decompressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform latents into quantized features.

        Parameters
        ----------
        lats:
            Input latents of shape (batch_size, latent_seq_length, latent_dim).
        decompressor_state:
            Decompressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output quantized features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated decompressor streaming state].

        """
        codes = self.quantizer.lats_to_codes(lats)
        qfeats, decompressor_state = self.codes_to_qfeats(
            codes, decompressor_state, return_state=True
        )
        if return_state:
            return qfeats, decompressor_state
        return qfeats

    # toks -> any
    @torch.jit.export
    def toks_to_codes(self, toks: "Tensor") -> "Tensor":
        """Transform tokens into codes (i.e. quantized latents).

        Parameters
        ----------
        toks:
            Input tokens of shape (batch_size, latent_seq_length).

        Returns
        -------
            Output codes of shape (batch_size, latent_seq_length, latent_dim).

        """
        codes = self.quantizer.toks_to_codes(toks)
        return codes

    @torch.jit.export
    def toks_to_qfeats(
        self,
        toks: "Tensor",
        decompressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform tokens into quantized features.

        Parameters
        ----------
        toks:
            Input tokens of shape (batch_size, latent_seq_length).
        decompressor_state:
            Decompressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output quantized features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated decompressor streaming state].

        """
        codes = self.toks_to_codes(toks)
        qfeats, decompressor_state = self.codes_to_qfeats(
            codes, decompressor_state, return_state=True
        )
        if return_state:
            return qfeats, decompressor_state
        return qfeats

    @torch.jit.export
    def toks_to_sig(
        self,
        toks: "Tensor",
        decompressor_state: "Tuple" = (),
        decoder_state: "Tuple" = (),
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple]]":
        """Transform tokens into signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the quantized features (see https://arxiv.org/abs/2305.18975). The refined or
        original quantized features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        toks:
            Input tokens of shape (batch_size, latent_seq_length).
        decompressor_state:
            Decompressor streaming state.
        decoder_state:
            Decoder streaming state.
        matching_set:
            Optional set of candidate features for kNN refinement,
            shape (num_candidates, hidden_dim).
        topk:
            Number of nearest neighbors to consider in the kNN refinement.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency during kNN computation.
        output_length:
            Desired output length of the synthesized signal. If specified, the output
            will be truncated or padded to this length.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output signal, shape (batch_size, output_length) if
              `output_length` is specified, otherwise (batch_size, ~seq_length);
            - [updated decompressor streaming state];
            - [updated decoder streaming state].

        """
        codes = self.toks_to_codes(toks)
        sig, decompressor_state, decoder_state = self.codes_to_sig(
            codes,
            decompressor_state,
            decoder_state,
            matching_set,
            topk,
            num_splits,
            output_length,
            return_state=True,
        )
        if return_state:
            return sig, decompressor_state, decoder_state
        return sig

    # codes -> any
    @torch.jit.export
    def codes_to_toks(self, codes: "Tensor") -> "Tensor":
        """Transform codes (i.e. quantized latents) into tokens.

        Parameters
        ----------
        codes:
            Input codes of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tokens of shape (batch_size, latent_seq_length).

        """
        toks = self.quantizer.codes_to_toks(codes)
        return toks

    @torch.jit.export
    def codes_to_qfeats(
        self,
        codes: "Tensor",
        decompressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple]]":
        """Transform codes (i.e. quantized latents) into quantized features.

        Parameters
        ----------
        codes:
            Input codes of shape (batch_size, latent_seq_length, latent_dim).
        decompressor_state:
            Decompressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output quantized features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated decompressor streaming state].

        """
        qfeats, *decompressor_state = self.decompressor(codes, *decompressor_state)
        if return_state:
            return qfeats, decompressor_state
        return qfeats

    @torch.jit.export
    def codes_to_sig(
        self,
        codes: "Tensor",
        decompressor_state: "Tuple" = (),
        decoder_state: "Tuple" = (),
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple[Tensor, Tuple, Tuple]]":
        """Transform codes (i.e. quantized latents) into signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the quantized features (see https://arxiv.org/abs/2305.18975). The refined or
        original quantized features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        codes:
            Input codes of shape (batch_size, latent_seq_length, latent_dim).
        decompressor_state:
            Decompressor streaming state.
        decoder_state:
            Decoder streaming state.
        matching_set:
            Optional set of candidate features for kNN refinement,
            shape (num_candidates, hidden_dim).
        topk:
            Number of nearest neighbors to consider in the kNN refinement.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency during kNN computation.
        output_length:
            Desired output length of the synthesized signal. If specified, the output
            will be truncated or padded to this length.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output signal, shape (batch_size, output_length) if
              `output_length` is specified, otherwise (batch_size, ~seq_length).
            - [updated decompressor streaming state];
            - [updated decoder streaming state].

        """
        qfeats, decompressor_state = self.codes_to_qfeats(
            codes, decompressor_state, return_state=True
        )
        sig, decoder_state = self.feats_to_sig(
            qfeats,
            decoder_state,
            matching_set,
            topk,
            num_splits,
            output_length,
            return_state=True,
        )
        if return_state:
            return sig, decompressor_state, decoder_state
        return sig

    @torch.jit.export
    def knn(
        self,
        input: "Tensor",
        matching_set: "Tensor",
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        """Perform k-nearest neighbors (kNN) search using cosine distance.

        This method retrieves the `topk` nearest neighbors for each query
        in the `input` tensor from the `matching_set` tensor. Optionally,
        the `matching_set` can be split into smaller subsets to reduce
        memory usage during large-scale computations.

        Parameters
        ----------
        input:
            Query tensor for which nearest neighbors are to be found,
            shape (..., hidden_dim), where `...` represents any
            additional leading dimensions.
        matching_set:
            Set of points to search for neighbors, shape (num_points, hidden_dim).
        topk:
            Number of nearest neighbors to retrieve.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency.

        Returns
        -------
            Tensor containing the nearest neighbors for each query point,
            shape: (..., topk, hidden_dim).

        """
        chunk_size = matching_set.shape[0] // num_splits
        if num_splits > 1:
            matching_subsets = matching_set.split(chunk_size)
        else:
            matching_subsets = [matching_set]
        topk_smallest_dists = []
        topk_smallest_idxes = []
        for i, matching_subset in enumerate(matching_subsets):
            dists = _cosine_distance(input.flatten(end_dim=-2), matching_subset)
            topk_smallest_dists_i, topk_smallest_idxes_i = dists.topk(
                k=min(topk, matching_subset.shape[0]), largest=False, dim=-1
            )
            topk_smallest_dists.append(topk_smallest_dists_i)
            topk_smallest_idxes.append(i * chunk_size + topk_smallest_idxes_i)
        if num_splits > 1:
            dists = torch.cat(topk_smallest_dists, dim=-1)
            idxes = torch.cat(topk_smallest_idxes, dim=-1)
            _, dist_idxes = dists.topk(
                k=min(topk, dists.shape[-1]), largest=False, dim=-1
            )
            output = matching_set[idxes.gather(1, dist_idxes)]
        else:
            output = matching_set[topk_smallest_idxes[0]]
        output = output.reshape(input.shape[:-1] + (-1, input.shape[-1]))
        return output

    def jit(
        self, classes: "Optional[Sequence[Type[nn.Module]]]" = None
    ) -> "FocalCodec":
        """JIT compile selected submodules.

        If `classes` is None, JIT only top-level named children.
        If `classes` is provided, recursively JIT all matching submodules.

        Parameters
        ----------
        classes:
            A list/tuple of module classes to JIT (e.g. [MyBlock, MyAttention]).
            If None, only immediate named children are JIT compiled.

        Returns
        -------
            A new module with some or all children JIT compiled.

        """
        from copy import deepcopy

        scripted = deepcopy(self)

        def _maybe_script(name: "str", module: "nn.Module") -> "nn.Module":
            try:
                return torch.jit.script(module)
            except Exception as e:
                warnings.warn(
                    f"Failed to JIT compile {name} ({type(module).__name__}): {e}"
                )
                return module

        if classes is None:
            # Non-recursive: only top-level children
            for name, module in self.named_children():
                scripted_module = _maybe_script(name, module)
                setattr(scripted, name, scripted_module)
        else:
            # Recursive: only JIT specified types
            def _recursive_jit(module: "nn.Module") -> "nn.Module":
                new_module = module.__class__.__new__(module.__class__)
                new_module.__dict__ = module.__dict__.copy()

                for name, child in module.named_children():
                    if isinstance(child, tuple(classes)):
                        child_scripted = _maybe_script(name, child)
                    else:
                        child_scripted = _recursive_jit(child)
                    setattr(new_module, name, child_scripted)

                return new_module

            # Avoid reference cycles
            try:
                scripted = _recursive_jit(self)
            finally:
                _recursive_jit = None

        return scripted

    def onnx(
        self,
        f: "Union[str, io.BytesIO]",
        batch_size: "Optional[int]" = None,
        chunk_size: "Optional[int]" = None,
        return_state: "bool" = False,
        **kwargs: "Any",
    ) -> "None":
        """Export the model to ONNX format.

        Parameters
        ----------
        f:
            Path to the output ONNX model file. E.g. “model.onnx”.
        batch_size:
            Batch size to use during ONNX inference.
            If None, the batch dimension will be exported as dynamic in the ONNX model.
        chunk_size:
            Chunk size to use during ONNX inference.
            If None, the time dimension will be exported as dynamic in the ONNX model.
        return_state:
            True to return the streaming state(s), False otherwise.
        kwargs:
            Keyword arguments to pass to `torch.onnx.export`
            (see https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export).

        Raises
        ------
        ValueError
            If `return_state` is True but the model is not causal. This combination is invalid
            because non-causal models do not track internal state when exporting to ONNX.

        """

        def _flatten(tree: "Tuple") -> "Tuple":
            if isinstance(tree, (tuple, list)):
                return sum((_flatten(x) for x in tree), ())
            if tree is None:
                return (None,)
            assert isinstance(tree, Tensor)
            return (tree.zero_(),)

        model = self.jit(classes=())  # Equivalent to deep copy
        device = next(model.parameters()).device
        model.eval()

        # TODO: avoid hard-coding (might cause issues with ONNX)
        default_num_tokens = 512
        B = 1 if batch_size is None else batch_size
        T = default_num_tokens * model.chunk_size if chunk_size is None else chunk_size
        sig = torch.randn(B, T, device=device)

        dynamic_axes = {}
        for name in ["sig", "toks", "codes", "rec_sig"]:
            axes = {}
            if batch_size is None:
                axes[0] = "batch"
            if chunk_size is None:
                axes[1] = "time"
            if axes:
                dynamic_axes[name] = axes

        if not return_state:
            torch.onnx.export(
                model,
                (sig,),
                f,
                input_names=["sig"],
                output_names=["toks", "codes", "rec_sig"],
                dynamic_axes=dynamic_axes,
                **kwargs,
            )
            return

        if not model.causal:
            raise ValueError(
                "return_state=True requires a causal model, as non-causal "
                "models do not track internal state during ONNX export"
            )

        with torch.no_grad():
            _, _, _, *state = model(sig, return_state=True)

        # Avoid reference cycles
        try:
            flat_state = _flatten(state)
        finally:
            _flatten = None

        if batch_size is None:
            for i, x in enumerate(flat_state):
                if x.ndim > 1:
                    dynamic_axes[f"input_state.{i}"] = {0: "batch"}
                    dynamic_axes[f"output_state.{i}"] = {0: "batch"}

        length = torch.ones(B, device=device)
        torch.onnx.export(
            model,
            (sig, *state, length, return_state),
            f,
            input_names=[
                "sig",
                *[f"input_state.{i}" for i in range(len(flat_state))],
            ],
            output_names=[
                "toks",
                "codes",
                "rec_sig",
                *[f"output_state.{i}" for i in range(len(flat_state))],
            ],
            dynamic_axes=dynamic_axes,
            **kwargs,
        )

    def info(self) -> "Dict[str, Any]":
        """Return the model information."""
        return {
            "model_id": self.model_id,
            "version": self.__version__,
            "sample_rate_input": self.sample_rate_input,
            "sample_rate_output": self.sample_rate_output,
            "causal": self.causal,
            "chunk_size": self.chunk_size,
            "latency": self.latency,
            "num_total_params": sum([x.numel() for x in self.state_dict().values()]),
        }

    def to_config(
        self,
        config: "str",
        pretrained: "bool" = False,
    ) -> "None":
        """Dump model configuration to a JSON file.

        Parameters
        ----------
        config:
            Path to local JSON file where the configuration should be dumped.
            If the given file path does not end with `.json`, `.json` is automatically appended.
        pretrained:
            Whether to dump the checkpoint along with the configuration.

        """
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        dirpath = os.path.dirname(config_json)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        config = {
            "encoder_name": self.encoder_name,
            "encoder_config": self.encoder_config,
            "compressor_name": self.compressor_name,
            "compressor_config": self.compressor_config,
            "quantizer_name": self.quantizer_name,
            "quantizer_config": self.quantizer_config,
            "decompressor_name": self.decompressor_name,
            "decompressor_config": self.decompressor_config,
            "decoder_name": self.decoder_name,
            "decoder_config": self.decoder_config,
        }

        with open(config_json, "w") as f:
            json.dump(config, f, indent=2)

        if pretrained:
            state_dict = self.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            try:
                from safetensors.torch import save_file as safetensors_save

                checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                safetensors_save(state_dict, checkpoint)
            except Exception:
                # If `safetensors` not available, use `torch`
                checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                torch.save(state_dict, checkpoint)

    def to_pretrained(self, config: "str") -> "None":
        """See documentation of `to_config`."""
        return self.to_config(config, pretrained=True)

    @classmethod
    def from_config(
        cls,
        config: "str",
        pretrained: "bool" = False,
        overrides: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "FocalCodec":
        """Load model from a configuration.

        Parameters
        ----------
        config:
            Configuration source, one of the following:
              - A local JSON file (e.g. "config.json");
              - a Hugging Face repository containing "config.json" (e.g. "username/repo_name");
              - a specific JSON file hosted in a Hugging Face repository (e.g. "username/repo_name/config_xyz.json").
            If the given file path does not end with `.json`, `.json` is automatically appended.
        pretrained:
            Whether to load the corresponding pretrained checkpoint.
              - If True and a JSON file is specified, the method will look for a checkpoint file with the same
                path or URL as the configuration file but with a `.safetensors` or `.pt` extension.
              - If True and a Hugging Face repository is provided, it is assumed that either "model.safetensors"
                or "model.pt" is available.
        overrides:
            Dictionary mapping dot-separated key paths to new values that override entries in the nested configuration.
            For example, {"encoder_config.max_cached_steps": 0}.
        kwargs:
            Additional keyword arguments to pass to `huggingface_hub.hf_hub_download` if
            fetching the configuration from a remote repository.

        Returns
        -------
            A model instance initialized with the given configuration and,
            if specified, pretrained checkpoint.

        Notes
        -----
        When loading from the Hugging Face Hub, the `huggingface-hub` library must be installed.
        You can install it via `pip install huggingface-hub`.

        """

        def _override_config(
            config: "Dict[str, Any]",
            path: "str",
            value: "Any",
        ) -> "None":
            keys = path.split(".")
            tmp = config
            for key in keys[:-1]:
                tmp = tmp.setdefault(key, {})
            tmp[keys[-1]] = value

        model_id = config
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        # Local
        if os.path.exists(config_json):
            with open(config_json) as f:
                config = json.load(f)
            if overrides is not None:
                for path, value in overrides.items():
                    _override_config(config, path, value)
            model = cls(**config)
            if pretrained:
                tgt_keys = list(model.state_dict().keys())
                try:
                    from safetensors.torch import load_file as safetensors_load

                    checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                    state_dict = torch.load(checkpoint, map_location="cpu")
                state_dict = cls._remap_state_dict(state_dict, tgt_keys)
                model.load_state_dict(state_dict)
            model.model_id = model_id
            return model

        # Remote
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("`pip install huggingface-hub` to load this model")

        is_repo = bool(re.fullmatch(r"[\w\-]+/[\w\-.]+", config))

        try:
            repo_id = config if is_repo else os.path.dirname(config_json)
            filename = "config.json" if is_repo else os.path.basename(config_json)
            config_json = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
            with open(config_json) as f:
                config = json.load(f)
            if overrides is not None:
                for path, value in overrides.items():
                    _override_config(config, path, value)
            model = cls(**config)
            if pretrained:
                tgt_keys = list(model.state_dict().keys())
                filename = "model" if is_repo else f"{os.path.splitext(filename)[0]}"
                try:
                    from safetensors.torch import load_file as safetensors_load

                    checkpoint = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.safetensors", **kwargs
                    )
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.pt", **kwargs
                    )
                    state_dict = torch.load(checkpoint, map_location="cpu")
                state_dict = cls._remap_state_dict(state_dict, tgt_keys)
                model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(
                f"Could not load the specified configuration. "
                f"Available default configurations: {DEFAULT_CONFIGS}"
            ) from e
        model.model_id = model_id
        return model

    @classmethod
    def from_pretrained(
        cls,
        config: "str",
        overrides: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "FocalCodec":
        """See documentation of `from_config`."""
        return cls.from_config(config, pretrained=True, overrides=overrides, **kwargs)

    @classmethod
    def _remap_state_dict(
        cls,
        src_state_dict: "Dict[str, Tensor]",
        tgt_keys: "Sequence[str]",
    ) -> "Dict[str, Tensor]":
        # Infer checkpoint version based on its structure
        if any(
            # Separate QKV projections => version 0.0.1
            any(proj in k for proj in (".q_proj", ".k_proj", ".v_proj"))
            for k in src_state_dict
        ):
            # Order-based mapping for compressor and decompressor
            compressor_tgt_keys = sorted(
                [k for k in tgt_keys if k.startswith("compressor.")]
            )
            compressor_src_keys = sorted(
                [k for k in src_state_dict if k.startswith("compressor.")]
            )
            compressor_map = dict(zip(compressor_tgt_keys, compressor_src_keys))

            decompressor_tgt_keys = sorted(
                [k for k in tgt_keys if k.startswith("decompressor.")]
            )
            decompressor_src_keys = sorted(
                [k for k in src_state_dict if k.startswith("decompressor.")]
            )
            decompressor_map = dict(zip(decompressor_tgt_keys, decompressor_src_keys))

            tgt_state_dict = {}
            for name in tgt_keys:
                if name.startswith("encoder.") and "qkv_proj" in name:
                    prefix = name.replace("qkv_proj.weight", "").replace(
                        "qkv_proj.bias", ""
                    )
                    suffix = name.split(".")[-1]  # 'weight' or 'bias'
                    q = src_state_dict[f"{prefix}q_proj.{suffix}"]
                    k = src_state_dict[f"{prefix}k_proj.{suffix}"]
                    v = src_state_dict[f"{prefix}v_proj.{suffix}"]
                    value = torch.cat([q, k, v], dim=0)
                elif name.startswith("encoder."):
                    value = src_state_dict[name]
                elif name.startswith("compressor."):
                    value = src_state_dict[compressor_map[name]]
                elif name.startswith("decompressor."):
                    value = src_state_dict[decompressor_map[name]]
                elif name.startswith("decoder."):
                    value = src_state_dict[name]
                else:
                    raise KeyError(f"Unmapped key: {name}")

                tgt_state_dict[name] = value

            return tgt_state_dict

        return src_state_dict


# Adapted from:
# https://github.com/bshall/knn-vc/blob/848302a262f7299c738af49d74209790ed442a9f/matcher.py#L21
@torch.jit.script
def _cosine_distance(query: "Tensor", target: "Tensor") -> "Tensor":
    # [T, H], [M, K]
    source_norm2 = (query**2).sum(dim=-1)
    target_norm2 = (target**2).sum(dim=-1)
    dotprod = (
        source_norm2[:, None]
        + target_norm2[None]
        - torch.cdist(query[None], target[None])[0] ** 2
    )
    dotprod /= 2
    dists = 1 - dotprod * (source_norm2[:, None] * target_norm2[None]).rsqrt()
    return dists


def test_model(config: "Optional[str]" = None) -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    sig = torch.randn(B, model.sample_rate_input, device=device)
    toks, codes, rec_sig = model(sig)
    model_jit = model.jit()
    toks_jit, codes_jit, rec_sig_jit = model_jit(sig)

    assert (toks == toks_jit).all(), [(toks != toks_jit).sum().item(), toks.numel()]
    assert (codes == codes_jit).all(), [
        (codes != codes_jit).sum().item(),
        codes.numel(),
    ]
    assert torch.allclose(rec_sig, rec_sig_jit, atol=1e-6), (
        ((rec_sig - rec_sig_jit) ** 2).mean().sqrt()
    )

    print("Model test passed")


def test_batch_invariance(config: "Optional[str]" = None) -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)

    sig = torch.randn(B, model.sample_rate_input, device=device)
    batch_toks, batch_codes, batch_rec_sig = model(sig)

    all_single_toks, all_single_codes, all_single_rec_sig = [], [], []
    for i in range(B):
        single_toks, single_codes, single_rec_sig = model(sig[i][None])
        all_single_toks.append(single_toks)
        all_single_codes.append(single_codes)
        all_single_rec_sig.append(single_rec_sig)
    all_single_toks = torch.cat(all_single_toks)
    all_single_codes = torch.cat(all_single_codes)
    all_single_rec_sig = torch.cat(all_single_rec_sig)

    assert (batch_toks != all_single_toks).sum() <= 2, [
        (batch_toks != all_single_toks).sum().item(),
        batch_toks.numel(),
    ]
    assert (batch_codes != all_single_codes).sum() <= 2, [
        (batch_codes != all_single_codes).sum().item(),
        batch_codes.numel(),
    ]
    assert torch.allclose(batch_rec_sig, all_single_rec_sig, atol=1), (
        ((batch_rec_sig - all_single_rec_sig) ** 2).mean().sqrt()
    )

    print("Batch invariance test passed")


@torch.no_grad()
def test_causality(config: "Optional[str]" = None) -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 4 * 512 * 320
    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)

    if not model.causal:
        print("Causality test skipped")
        return

    # NOTE: Randomly initialized quantizer might fail the causality test due to rounding errors
    sig = torch.ones(B, T, device=device)
    model = model.jit()
    toks, codes, rec_sig = model(sig)

    incremental_toks = []
    incremental_codes = []
    incremental_rec_sig = []
    state = []
    chunk_size = model.chunk_size
    i = 0
    while i < T:
        toks_i, codes_i, rec_sig_i, *state = model(
            sig[:, i : i + chunk_size], *state, return_state=True
        )
        incremental_toks.append(toks_i)
        incremental_codes.append(codes_i)
        incremental_rec_sig.append(rec_sig_i)
        i += chunk_size
    incremental_toks = torch.cat(incremental_toks, dim=1)
    incremental_codes = torch.cat(incremental_codes, dim=1)
    incremental_rec_sig = torch.cat(incremental_rec_sig, dim=1)

    assert (toks == incremental_toks).all(), [
        (toks != incremental_toks).sum().item(),
        toks.numel(),
    ]
    assert (codes == incremental_codes).all(), [
        (codes != incremental_codes).sum().item(),
        codes.numel(),
    ]
    assert torch.allclose(rec_sig, incremental_rec_sig, atol=1e-2), (
        ((rec_sig - incremental_rec_sig) ** 2).mean().sqrt()
    )
    print("Causality test passed")


@torch.no_grad()
def test_onnx(config: "Optional[str]" = None) -> "None":
    import numpy as np

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)

    f = io.BytesIO()
    model.onnx(f, return_state=model.causal)
    onnx_bytes = f.getvalue()

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX")
        return

    session = ort.InferenceSession(onnx_bytes)
    inputs_ort = {}
    for input in session.get_inputs():
        name = input.name
        shape_ = input.shape
        type_ = input.type

        # Resolve shape
        shape = []
        for dim in shape_:
            if dim == "batch":
                shape.append(1)
            elif dim == "time":
                shape.append(model.chunk_size)
            elif isinstance(dim, int):
                shape.append(dim)
            else:
                raise ValueError(f"Unrecognized symbolic dimension: {dim}")

        # Resolve dtype
        if type_ == "tensor(float)":
            dtype = np.float32
        elif type_ == "tensor(double)":
            dtype = np.float64
        elif type_ == "tensor(int64)":
            dtype = np.int64
        else:
            raise ValueError(f"Unsupported input type: {type_} for {name}")

        inputs_ort[name] = np.zeros(shape, dtype=dtype)
    inputs_ort["sig"] = np.random.random(inputs_ort["sig"].shape).astype(np.float32)

    def _unflatten(template: "Tuple", flat_iter: "Iterator") -> "Union[Tensor, Tuple]":
        if isinstance(template, (tuple, list)):
            return tuple(_unflatten(sub, flat_iter) for sub in template)
        return torch.tensor(next(flat_iter), device=device)

    # Avoid reference cycles
    try:
        _, _, *template = model(
            torch.tensor(inputs_ort["sig"], device=device), return_state=True
        )
        inputs_torch = _unflatten(template, iter(inputs_ort.values()))
    finally:
        _unflatten = None

    outputs_ort = session.run([x.name for x in session.get_outputs()], inputs_ort)
    toks, codes, rec_sig = model(*inputs_torch)

    assert (toks.cpu() == torch.tensor(outputs_ort[0])).all(), [
        (toks.cpu() != torch.tensor(outputs_ort[0])).sum().item(),
        toks.numel(),
    ]
    assert (codes.cpu() == torch.tensor(outputs_ort[1])).all(), [
        (codes.cpu() != torch.tensor(outputs_ort[1])).sum().item(),
        codes.numel(),
    ]
    assert torch.allclose(rec_sig.cpu(), torch.tensor(outputs_ort[2]), atol=1e-2), (
        ((rec_sig.cpu() - torch.tensor(outputs_ort[2])) ** 2).mean().sqrt()
    )

    print("ONNX test passed")


def test_performance(
    seconds: "float",
    compile: "Optional[str]" = None,
    fp16: "bool" = False,
    config: "Optional[str]" = None,
    mode: "str" = "reconstruct",
) -> "None":
    import torch.utils.benchmark as benchmark

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)

    if compile == "torch.jit.script":
        model = model.jit()
    elif compile == "torch.compile":
        model = torch.compile(model, mode="max-autotune")
    sig = torch.randn(1, int(seconds * model.sample_rate_input), device=device)

    @torch.no_grad()
    def forward(sig: "Tensor") -> "Tensor":
        with torch.autocast(device_type=device.type, enabled=fp16):
            if mode == "encode":
                toks = model.sig_to_toks(sig)
                return toks
            if mode == "reconstruct":
                toks = model.sig_to_toks(sig)
                sig = model.toks_to_sig(toks)
                return sig
            raise NotImplementedError

    # Warmup
    for _ in range(10):
        forward(sig)

    print("=" * 150)
    print(
        f"Input length: {seconds} seconds, Compile: {compile}, fp16: {fp16}, config: {config}, mode: {mode}"
    )
    print("=" * 150)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    forward(sig)
    print(f"Peak memory (MB): {torch.cuda.max_memory_allocated() / 1e6:.2f}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    timer = benchmark.Timer(
        stmt="forward(sig)", globals={"sig": sig, "forward": forward}
    )
    time = timer.timeit(100).mean
    print(f"Latency: {time:.6f}, RTF: {seconds / time:.6f}")
    print("#" * 150)


@torch.no_grad()
def test_offline(config: "Optional[str]" = None) -> "None":
    try:
        import torchaudio
    except ImportError:
        raise ImportError("`pip install torchaudio` to run this script")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = [
        os.path.join("librispeech-dev-clean", "84"),
    ]
    matching_set = {k: [] for k in paths}
    for path in paths:
        for filename in os.listdir(os.path.join(root_dir, "audios", path)):
            filepath = os.path.join(root_dir, "audios", path, filename)
            sig, sample_rate = torchaudio.load(filepath)
            sig = torchaudio.functional.resample(
                sig, sample_rate, model.sample_rate_input
            )
            sig = sig.to(device)
            qfeats = model.sig_to_qfeats(sig)
            matching_set[path].append(qfeats[0])
        matching_set[path] = torch.cat(matching_set[path])

    sig, sample_rate = torchaudio.load(
        os.path.join(root_dir, "audios", "librispeech-dev-clean", "251-118436-0003.wav")
    )
    sig = torchaudio.functional.resample(sig, sample_rate, model.sample_rate_input)
    sig = sig.to(device)

    feats = model.sig_to_feats(sig)
    lats = model.feats_to_lats(feats)
    toks = model.lats_to_toks(lats)
    sig_from_toks = model.toks_to_sig(toks)
    sig_from_toks_vc = model.toks_to_sig(
        toks, matching_set=matching_set["librispeech-dev-clean/84"]
    )

    output_dir = os.path.join(root_dir, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(
        os.path.join(output_dir, "sig.wav"),
        sig.float().cpu(),
        model.sample_rate_input,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_offline.wav"),
        sig_from_toks.float().cpu(),
        model.sample_rate_output,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_vc_offline.wav"),
        sig_from_toks_vc.float().cpu(),
        model.sample_rate_output,
    )


@torch.no_grad()
def test_online(config: "Optional[str]" = None) -> "None":
    try:
        import torchaudio
    except ImportError:
        raise ImportError("`pip install torchaudio` to run this script")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        FocalCodec(
            encoder_config={"causal": True},
            compressor_config={"causal": True},
            decompressor_config={"causal": True},
            decoder_config={"causal": True},
        )
        if config is None
        else FocalCodec.from_pretrained(config)
    )
    model = model.eval().to(device)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = [
        os.path.join("librispeech-dev-clean", "84"),
    ]
    matching_set = {k: [] for k in paths}
    for path in paths:
        for filename in os.listdir(os.path.join(root_dir, "audios", path)):
            filepath = os.path.join(root_dir, "audios", path, filename)
            sig, sample_rate = torchaudio.load(filepath)
            sig = torchaudio.functional.resample(
                sig, sample_rate, model.sample_rate_input
            )
            sig = sig.to(device)
            qfeats = model.sig_to_qfeats(sig)
            matching_set[path].append(qfeats[0])
        matching_set[path] = torch.cat(matching_set[path])

    sig, sample_rate = torchaudio.load(
        os.path.join(root_dir, "audios", "librispeech-dev-clean", "251-118436-0003.wav")
    )
    sig = torchaudio.functional.resample(sig, sample_rate, model.sample_rate_input)
    sig = sig.to(device)

    sig_from_toks = []
    sig_from_toks_vc = []
    feats_state = []
    toks_state = []
    sig_from_feats_state = []
    sig_from_toks_state = []
    sig_from_toks_vc_state = []
    chunk_size = model.chunk_size
    i = 0
    while i < sig.shape[1]:
        feats_i, *feats_state = model.sig_to_feats(
            sig[:, i : i + chunk_size], *feats_state, return_state=True
        )
        toks_i, *toks_state = model.feats_to_toks(
            feats_i, *toks_state, return_state=True
        )
        sig_from_feats_i, *sig_from_feats_state = model.feats_to_sig(
            feats_i, *sig_from_feats_state, return_state=True
        )
        sig_from_toks_i, *sig_from_toks_state = model.toks_to_sig(
            toks_i, *sig_from_toks_state, return_state=True
        )
        sig_from_toks_vc_i, *sig_from_toks_vc_state = model.toks_to_sig(
            toks_i,
            *sig_from_toks_vc_state,
            matching_set=matching_set["librispeech-dev-clean/84"],
            return_state=True,
        )
        sig_from_toks.append(sig_from_toks_i)
        sig_from_toks_vc.append(sig_from_toks_vc_i)
        i += chunk_size
    sig_from_toks = torch.cat(sig_from_toks, dim=1)
    sig_from_toks_vc = torch.cat(sig_from_toks_vc, dim=1)

    output_dir = os.path.join(root_dir, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(
        os.path.join(output_dir, "sig.wav"),
        sig.float().cpu(),
        model.sample_rate_input,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_online.wav"),
        sig_from_toks.float().cpu(),
        model.sample_rate_output,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_vc_online.wav"),
        sig_from_toks_vc.float().cpu(),
        model.sample_rate_output,
    )


if __name__ == "__main__":
    config = "lucadellalib/focalcodec_50hz_4k_causal"
    test_model(config)
    test_batch_invariance(config)
    test_causality(config)
    test_onnx(config)
    test_offline(config)
    test_online(config)
    for seconds in [0.08, 16]:
        test_performance(seconds, config=config, mode="reconstruct")
        test_performance(
            seconds, config=config, compile="torch.jit.script", mode="reconstruct"
        )
        test_performance(seconds, config=config, fp16=True, mode="reconstruct")
        test_performance(
            seconds,
            config=config,
            compile="torch.jit.script",
            fp16=True,
            mode="reconstruct",
        )
