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

"""PyTorch Hub entry point."""

from typing import Any, Dict, Optional

from focalcodec import FocalCodec


# Make sure it is consistent with requirements.txt and README.md
dependencies = [
    "huggingface_hub",
    "safetensors",
    "torch",
]


def focalcodec(
    config: "str" = "lucadellalib/focalcodec_50hz",
    pretrained: "bool" = True,
    overrides: "Optional[Dict[str, Any]]" = None,
    **kwargs: "Any",
) -> "FocalCodec":
    """Load FocalCodec.

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

    """
    codec = FocalCodec.from_config(config, pretrained, overrides, **kwargs)
    return codec


if __name__ == "__main__":
    model = focalcodec()
    print(model)
    print(
        f"Total number of parameters/buffers: "
        f"{sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )
