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

"""FocalCodec (see https://arxiv.org/abs/2502.04465) with support for jitting."""

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file as safetensors_load
from torch import Tensor, nn


try:
    from .bsq import BinarySphericalQuantizer
    from .focalnet import FocalDecoder, FocalEncoder
    from .vocos import Vocos
    from .wavlm import WavLM
except ImportError:
    from bsq import BinarySphericalQuantizer
    from focalnet import FocalDecoder, FocalEncoder
    from vocos import Vocos
    from wavlm import WavLM


__all__ = ["FocalCodec"]


REGISTRY = {
    "BinarySphericalQuantizer": BinarySphericalQuantizer,
    "FocalDecoder": FocalDecoder,
    "FocalEncoder": FocalEncoder,
    "Vocos": Vocos,
    "WavLM": WavLM,
}

DEFAULT_CONFIGS = [
    "lucadellalib/focalcodec_50hz",
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

    sample_rate = 16000

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
        decoder_name: "str" = "Vocos",
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

    @classmethod
    def from_config(
        cls,
        config: "str",
        pretrained: "bool" = False,
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
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        # Local
        if os.path.exists(config_json):
            with open(config_json) as f:
                config = json.load(f)
            model = cls(**config)
            if pretrained:
                try:
                    checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                    state_dict = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(state_dict)
            return model

        # Remote
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError("`pip install huggingface-hub` to load this model")

        try:
            list_repo_files(config)
            is_repo = True
        except Exception:
            is_repo = False

        try:
            repo_id = config if is_repo else os.path.dirname(config_json)
            filename = "config.json" if is_repo else os.path.basename(config_json)
            config_json = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
            with open(config_json) as f:
                config = json.load(f)
            model = cls(**config)
            if pretrained:
                filename = "model" if is_repo else f"{os.path.splitext(filename)[0]}"
                try:
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
                model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(
                f"Could not load the specified configuration. "
                f"Available default configurations: {DEFAULT_CONFIGS}"
            ) from e
        return model

    @classmethod
    def from_pretrained(cls, config: "str", **kwargs: "Any") -> "FocalCodec":
        """See documentation of `from_config`."""
        return cls.from_config(config, pretrained=True, **kwargs)

    def forward(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input signal of shape (batch_size, seq_length).
        length:
            Optional tensor representing the relative lengths of each signal in the batch.

        Returns
        -------
            - Output tokens of shape (batch_size, latent_seq_length);
            - corresponding codes of shape (batch_size, latent_seq_length, latent_dim);
            - reconstructed signal of shape (batch_size, ~seq_length).

        """
        toks = self.sig_to_toks(input, length)
        codes = self.toks_to_codes(toks)
        sig = self.codes_to_sig(
            codes,
            matching_set=None,
            topk=-1,
            num_splits=-1,
            output_length=input.shape[-1],
        )
        return toks, codes, sig

    @property
    def codebook(self) -> "Tensor":
        """Return a copy of the quantizer codebook."""
        return self.quantizer.codebook.clone()

    # sig -> any
    @torch.jit.export
    def sig_to_feats(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Transform signal into features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length).
        length:
            Relative lengths of each signal in the batch.

        Returns
        -------
            Output tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        """
        if length is None:
            key_padding_mask = None
        else:
            B, T = input.shape
            abs_length = (length * T).ceil().clamp(max=T).long()
            key_padding_mask = (
                torch.arange(T, device=input.device).expand(B, T) < abs_length[:, None]
            )
        feats = self.encoder(input, key_padding_mask)
        return feats

    @torch.jit.export
    def sig_to_lats(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Transform signal into latents.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length).
        length:
            Relative lengths of each signal in the batch.

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length, latent_dim).

        """
        feats = self.sig_to_feats(input, length)
        lats = self.feats_to_lats(feats)
        return lats

    @torch.jit.export
    def sig_to_toks(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Transform signal into tokens.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length).
        length:
            Relative lengths of each signal in the batch.

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length).

        """
        feats = self.sig_to_feats(input, length)
        toks = self.feats_to_toks(feats)
        return toks

    @torch.jit.export
    def sig_to_codes(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Transform signal into codes (i.e. quantized latents).

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length).
        length:
            Relative lengths of each signal in the batch.

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length, latent_dim).

        """
        feats = self.sig_to_feats(input, length)
        codes = self.feats_to_codes(feats)
        return codes

    @torch.jit.export
    def sig_to_qfeats(
        self,
        input: "Tensor",
        length: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Transform signal into quantized features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length).
        length:
            Relative lengths of each signal in the batch.

        Returns
        -------
            Output tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        """
        feats = self.sig_to_feats(input, length)
        qfeats = self.feats_to_qfeats(feats)
        return qfeats

    # feats -> any
    @torch.jit.export
    def feats_to_lats(
        self,
        input: "Tensor",
    ) -> "Tensor":
        """Transform features into latents.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length, latent_dim).

        """
        lats = self.compressor(input)
        lats = nn.functional.normalize(lats, dim=-1)
        return lats

    @torch.jit.export
    def feats_to_toks(
        self,
        input: "Tensor",
    ) -> "Tensor":
        """Transform features into tokens.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length).

        """
        lats = self.feats_to_lats(input)
        toks = self.quantizer.lats_to_toks(lats)
        return toks

    @torch.jit.export
    def feats_to_codes(
        self,
        input: "Tensor",
    ) -> "Tensor":
        """Transform features into codes (i.e. quantized latents).

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length, latent_dim).

        """
        lats = self.feats_to_lats(input)
        codes = self.quantizer.lats_to_codes(lats)
        return codes

    @torch.jit.export
    def feats_to_qfeats(
        self,
        input: "Tensor",
    ) -> "Tensor":
        """Transform features into quantized features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        Returns
        -------
            Output tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        """
        lats = self.feats_to_lats(input)
        codes = self.quantizer.lats_to_codes(lats)
        qfeats = self.codes_to_qfeats(codes)
        return qfeats

    @torch.jit.export
    def feats_to_sig(
        self,
        input: "Tensor",
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
    ) -> "Tensor":
        """Convert features to signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the input features (see https://arxiv.org/abs/2305.18975). The refined or
        original features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, hidden_seq_length, hidden_dim).
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

        Returns
        -------
            Synthesized signal, shape (batch_size, output_length) if
            `output_length` is specified, otherwise (batch_size, ~seq_length).

        """
        if matching_set is not None:
            input = self.knn(
                input,
                matching_set,
                topk,
                num_splits,
            ).mean(dim=-2)
        sig = self.decoder(input)
        if output_length is not None:
            if sig.shape[1] > output_length:
                sig = sig[:, :output_length]
            elif sig.shape[1] < output_length:
                delta = output_length - sig.shape[1]
                sig = nn.functional.pad(sig, [0, delta], mode="replicate")
        return sig

    # lats -> any
    @torch.jit.export
    def lats_to_toks(self, input: "Tensor") -> "Tensor":
        """Transform latents into tokens.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length).

        """
        toks = self.quantizer.lats_to_toks(input)
        return toks

    @torch.jit.export
    def lats_to_codes(self, input: "Tensor") -> "Tensor":
        """Transform latents into codes (i.e. quantized latents).

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length, latent_dim).

        """
        codes = self.quantizer.lats_to_codes(input)
        return codes

    @torch.jit.export
    def lats_to_qfeats(self, input: "Tensor") -> "Tensor":
        """Transform latents into quantized features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        """
        codes = self.quantizer.lats_to_codes(input)
        qfeats = self.codes_to_qfeats(codes)
        return qfeats

    # toks -> any
    @torch.jit.export
    def toks_to_codes(self, input: "Tensor") -> "Tensor":
        """Transform tokens into codes (i.e. quantized latents).

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length, latent_dim).

        """
        codes = self.quantizer.toks_to_codes(input)
        return codes

    @torch.jit.export
    def toks_to_qfeats(self, input: "Tensor") -> "Tensor":
        """Transform tokens into quantized features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        """
        codes = self.toks_to_codes(input)
        qfeats = self.codes_to_qfeats(codes)
        return qfeats

    @torch.jit.export
    def toks_to_sig(
        self,
        input: "Tensor",
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
    ) -> "Tensor":
        """Convert tokens to signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the quantized features (see https://arxiv.org/abs/2305.18975). The refined or
        original quantized features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length).
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

        Returns
        -------
            Synthesized signal, shape (batch_size, output_length) if
            `output_length` is specified, otherwise (batch_size, ~seq_length).

        """
        codes = self.toks_to_codes(input)
        sig = self.codes_to_sig(codes, matching_set, topk, num_splits, output_length)
        return sig

    # codes -> any
    @torch.jit.export
    def codes_to_toks(self, input: "Tensor") -> "Tensor":
        """Transform codes (i.e. quantized latents) into tokens.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tensor of shape (batch_size, latent_seq_length).

        """
        toks = self.quantizer.codes_to_toks(input)
        return toks

    @torch.jit.export
    def codes_to_qfeats(self, input: "Tensor") -> "Tensor":
        """Transform codes (i.e. quantized latents) into quantized features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length, latent_dim).

        Returns
        -------
            Output tensor of shape (batch_size, hidden_seq_length, hidden_dim).

        """
        qfeats = self.decompressor(input)
        return qfeats

    @torch.jit.export
    def codes_to_sig(
        self,
        input: "Tensor",
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
    ) -> "Tensor":
        """Convert codes (i.e. quantized latents) to signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the quantized features (see https://arxiv.org/abs/2305.18975). The refined or
        original quantized features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, latent_seq_length, latent_dim).
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

        Returns
        -------
            Synthesized signal, shape (batch_size, output_length) if
            `output_length` is specified, otherwise (batch_size, ~seq_length).

        """
        qfeats = self.codes_to_qfeats(input)
        sig = self.feats_to_sig(qfeats, matching_set, topk, num_splits, output_length)
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
            dists = self._cosine_distance(input.flatten(end_dim=-2), matching_subset)
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

    # Adapted from:
    # https://github.com/bshall/knn-vc/blob/848302a262f7299c738af49d74209790ed442a9f/matcher.py#L21
    def _cosine_distance(self, query: "Tensor", target: "Tensor") -> "Tensor":
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


def test_model() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = FocalCodec().to(device)
    print(model)
    print(sum([x.numel() for x in model.state_dict().values()]) / 1e6)

    sig = torch.randn(B, 16000, device=device)
    length = torch.as_tensor([1.0, 0.5, 0.8], device=device)
    toks, codes, rec_sig = model(sig, length)
    model_jit = torch.jit.script(model)
    toks_jit, codes_jit, rec_sig_jit = model_jit(sig, length)
    assert (toks == toks_jit).all(), [(toks != toks_jit).sum().item(), toks.numel()]
    assert (codes == codes_jit).all(), [
        (codes != codes_jit).sum().item(),
        codes.numel(),
    ]
    assert torch.allclose(rec_sig, rec_sig_jit, atol=1e-6), (
        ((rec_sig - rec_sig_jit) ** 2).mean().sqrt()
    )


def test_batch_invariance(
    config: "str" = "lucadellalib/focalcodec_50hz",
) -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = FocalCodec.from_config(config).to(device)
    print(model)

    input = torch.randn(B, 16000, device=device)
    length = torch.as_tensor([1.0, 0.5, 0.8], device=device)
    batch_toks, batch_codes, batch_rec_sig = model(input, length)

    all_single_toks, all_single_codes, all_single_rec_sig = [], [], []
    for i in range(B):
        single_toks, single_codes, single_rec_sig = model(
            input[i][None], length[i][None]
        )
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
    assert torch.allclose(batch_rec_sig, all_single_rec_sig, atol=1e-1), (
        ((batch_rec_sig - all_single_rec_sig) ** 2).mean().sqrt()
    )


def test_performance(
    seconds: "int",
    compile: "Optional[str]" = None,
    fp16: "bool" = False,
    config: "str" = "lucadellalib/focalcodec_50hz",
    mode: "str" = "encode",
) -> "None":
    import logging

    import torch.utils.benchmark as benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocalCodec.from_pretrained(config).to(device)

    if compile == "torch.jit.script":
        model = torch.jit.script(model)
    elif compile == "torch.compile":
        model = torch.compile(model, mode="max-autotune")
    sig = torch.randn(1, seconds * 16000, device=device)

    inference_mode = torch.no_grad
    try:
        inference_mode = torch.inference_mode
    except Exception as e:
        logging.warning(e)

    @inference_mode()
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
    time = timer.timeit(10).mean
    print(f"Latency: {time:.6f}, RTF: {time / seconds:.6f}, iRTF: {seconds / time:.6f}")
    print("#" * 150)


@torch.no_grad()
def test_offline(
    config: "str" = "lucadellalib/focalcodec_50hz",
) -> "None":
    try:
        import torchaudio
    except ImportError:
        raise ImportError("`pip install torchaudio` to run this script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocalCodec.from_pretrained(config).to(device)
    model.eval()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = [
        os.path.join("librispeech-dev-clean", "84"),
    ]
    matching_set = {k: [] for k in paths}
    for path in paths:
        for filename in os.listdir(os.path.join(root_dir, "audios", path)):
            filepath = os.path.join(root_dir, "audios", path, filename)
            sig, sample_rate = torchaudio.load(filepath)
            sig = torchaudio.functional.resample(sig, sample_rate, 16000)
            sig = sig.to(device)
            feats = model.sig_to_feats(sig)
            matching_set[path].append(feats[0])
        matching_set[path] = torch.cat(matching_set[path])

    sig, sample_rate = torchaudio.load(
        os.path.join(
            root_dir, "audios", "librispeech-dev-clean", "251-118436-0003.wav"
        )
    )
    sig = torchaudio.functional.resample(sig, sample_rate, 16000)
    sig = sig.to(device)

    feats = model.sig_to_feats(sig)
    lats = model.feats_to_lats(feats)
    toks = model.lats_to_toks(lats)
    sig_from_feats = model.feats_to_sig(feats)
    sig_from_lats = model.codes_to_sig(lats)
    sig_from_toks = model.toks_to_sig(toks)
    sig_from_toks_vc = model.toks_to_sig(toks, matching_set["librispeech-dev-clean/84"])

    output_dir = os.path.join(root_dir, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(
        os.path.join(output_dir, "sig.wav"),
        sig.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_feats_offline.wav"),
        sig_from_feats.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_lats_offline.wav"),
        sig_from_lats.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_offline.wav"),
        sig_from_toks.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_offline_vc.wav"),
        sig_from_toks_vc.cpu(),
        16000,
    )


@torch.no_grad()
def test_online(
    config: "str" = "lucadellalib/focalcodec_50hz",
) -> "None":

    def overlap_add(
        chunks: "Sequence[Tensor]", chunk_size: "int", overlap_size: "int"
    ) -> "Tensor":
        device = chunks[0].device
        step_size = chunk_size - overlap_size
        batch_size = chunks[0].shape[0]
        num_chunks = len(chunks)
        total_length = step_size * (num_chunks - 1) + chunk_size

        # Stack all processed chunks into a tensor of shape (batch_size, num_chunks, chunk_size)
        chunks_tensor = torch.stack(chunks, dim=1)

        # Create cross-fading window
        fade_in = torch.linspace(0, 1, overlap_size, device=device)
        fade_out = 1 - fade_in
        window = torch.cat(
            [
                fade_out,
                torch.ones(chunk_size - 2 * overlap_size, device=device),
                fade_in,
            ]
        )

        # Reshape the window for broadcasting
        window = window.reshape(1, 1, -1)  # Shape (1, 1, chunk_size)

        # Apply the window to all chunks
        weighted_chunks = (
            chunks_tensor * window
        )  # Shape (batch_size, num_chunks, chunk_size)

        # Initialize output and overlap count
        output = torch.zeros(batch_size, total_length, device=device)
        overlap_count = torch.zeros(batch_size, total_length, device=device)

        # Place weighted chunks in their respective positions
        for i in range(num_chunks):
            start = i * step_size
            end = start + chunk_size
            output[:, start:end] += weighted_chunks[:, i, :]
            overlap_count[:, start:end] += window[0]

        # Normalize by overlap count to handle overlapping regions
        output /= overlap_count
        return output

    try:
        import torchaudio
    except ImportError:
        raise ImportError("`pip install torchaudio` to run this script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocalCodec.from_pretrained(config).to(device)
    model.eval()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [
        os.path.join("librispeech-dev-clean", "84"),
    ]
    matching_set = {k: [] for k in paths}
    for path in paths:
        for filename in os.listdir(os.path.join(root_dir, "audios", path)):
            filepath = os.path.join(root_dir, "audios", path, filename)
            sig, sample_rate = torchaudio.load(filepath)
            sig = torchaudio.functional.resample(sig, sample_rate, 16000)
            sig = sig.to(device)
            feats = model.sig_to_feats(sig)
            matching_set[path].append(feats[0])
        matching_set[path] = torch.cat(matching_set[path])

    sig, sample_rate = torchaudio.load(
        os.path.join(
            root_dir, "audios", "librispeech-dev-clean", "251-118436-0003.wav"
        )
    )
    sig = torchaudio.functional.resample(sig, sample_rate, 16000)
    sig = sig.to(device)

    chunk_size = 8000
    initial_context = 8000
    context_size = 48000
    overlap_size = 1000

    _toks = model.sig_to_toks(torch.randn(1, chunk_size, device=device))
    _rec_sig = model.toks_to_sig(_toks)
    sig_from_toks_chunk_size = _rec_sig.shape[-1]
    _feats = model.sig_to_feats(torch.randn(1, chunk_size, device=device))
    _rec_sig = model.feats_to_sig(_feats)
    sig_from_feats_chunk_size = _rec_sig.shape[-1]

    sig_from_toks_overlap_size = int(
        overlap_size * sig_from_toks_chunk_size / chunk_size
    )
    sig_from_feats_overlap_size = int(
        overlap_size * sig_from_feats_chunk_size / chunk_size
    )

    all_sig_from_feats = []
    all_sig_from_lats = []
    all_sig_from_toks = []
    all_sig_from_toks_vc = []
    frame_idx = 0
    while frame_idx < sig.shape[1]:
        if frame_idx >= initial_context:
            chunk = sig[:, max(0, frame_idx - context_size) : frame_idx]
            feats = model.sig_to_feats(chunk)
            lats = model.feats_to_lats(feats)
            toks = model.lats_to_toks(lats)
            sig_from_feats = model.feats_to_sig(feats)[:, -sig_from_feats_chunk_size:]
            sig_from_lats = model.codes_to_sig(lats)[:, -sig_from_toks_chunk_size:]
            sig_from_toks = model.toks_to_sig(toks)[:, -sig_from_toks_chunk_size:]
            sig_from_toks_vc = model.toks_to_sig(
                toks,
                matching_set["librispeech-dev-clean/84"],
            )[:, -sig_from_toks_chunk_size:]

            all_sig_from_feats.append(sig_from_feats)
            all_sig_from_lats.append(sig_from_lats)
            all_sig_from_toks.append(sig_from_toks)
            all_sig_from_toks_vc.append(sig_from_toks_vc)
        frame_idx += chunk_size - overlap_size

    all_sig_from_feats = overlap_add(
        all_sig_from_feats, sig_from_feats_chunk_size, sig_from_feats_overlap_size
    )
    all_sig_from_lats = overlap_add(
        all_sig_from_lats, sig_from_toks_chunk_size, sig_from_toks_overlap_size
    )
    all_sig_from_toks = overlap_add(
        all_sig_from_toks, sig_from_toks_chunk_size, sig_from_toks_overlap_size
    )
    all_sig_from_toks_vc = overlap_add(
        all_sig_from_toks_vc, sig_from_toks_chunk_size, sig_from_toks_overlap_size
    )

    output_dir = os.path.join(root_dir, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(
        os.path.join(output_dir, "sig.wav"),
        sig.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_feats_online.wav"),
        all_sig_from_feats.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_lats_online.wav"),
        all_sig_from_lats.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_online.wav"),
        all_sig_from_toks.cpu(),
        16000,
    )
    torchaudio.save(
        os.path.join(output_dir, "sig_from_toks_online_vc.wav"),
        all_sig_from_toks_vc.cpu(),
        16000,
    )


if __name__ == "__main__":
    config = "lucadellalib/focalcodec_50hz"
    test_model()
    test_batch_invariance(config)
    test_offline(config=config)
    test_online(config=config)
    for seconds in [1, 2, 4, 8, 16, 32]:
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
