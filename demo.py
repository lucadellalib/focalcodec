#!/usr/bin/env python3

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

"""FocalCodec demo."""

import argparse
import os
from typing import Optional, Sequence

import torch


try:
    import torchaudio
except ImportError:
    raise ImportError("`pip install torchaudio` to run this script")


def main(
    input_file: "str",
    output_file: "str" = "reconstruction.wav",
    config: "str" = "lucadellalib/focalcodec_50hz",
    reference_files: "Optional[Sequence[str]]" = None,
) -> "None":
    # Load FocalCodec model
    codec = torch.hub.load(
        "lucadellalib/focalcodec:v0.0.1", "focalcodec", config=config, force_reload=True
    )
    codec.eval().requires_grad_(False)

    # Process reference files if provided
    matching_set = None
    if reference_files:
        reference_audio_files = []
        for path in reference_files:
            if os.path.isdir(path):
                # Add all .wav files from the directory
                wav_files = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith(".wav")
                ]
                reference_audio_files.extend(wav_files)
            elif os.path.isfile(path) and path.endswith(".wav"):
                reference_audio_files.append(path)
            else:
                print(f"Skipping invalid path: {path}")

        if reference_audio_files:
            matching_set = []
            for reference_file in reference_audio_files:
                sig, sample_rate = torchaudio.load(reference_file)
                sig = torchaudio.functional.resample(
                    sig, sample_rate, codec.sample_rate
                )
                feats = codec.sig_to_feats(sig)
                matching_set.append(feats[0])
            matching_set = torch.cat(matching_set)
        else:
            print("Warning: No valid reference files found.")

    # Load input audio
    sig, sample_rate = torchaudio.load(input_file)

    # Resample if necessary
    sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate)

    # Encode and decode
    toks = codec.sig_to_toks(sig)
    rec_sig = codec.toks_to_sig(toks, matching_set)

    # Save the reconstructed audio
    torchaudio.save(output_file, rec_sig, codec.sample_rate)
    print(f"Reconstructed audio saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocalCodec demo")

    parser.add_argument(
        "--input_file",
        type=str,
        help="path to the input audio file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="reconstruction.wav",
        help="path to save the reconstructed audio file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lucadellalib/focalcodec_50hz",
        help="FocalCodec configuration",
    )
    parser.add_argument(
        "--reference_files",
        type=str,
        nargs="+",  # Allows specifying multiple files or directories
        default=None,
        help="path(s) to reference audio files or a directory containing reference audio files",
    )

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.config, args.reference_files)
