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

"""FocalCodec demo.

This script performs real-time or offline speech resynthesis and voice conversion using FocalCodec.
Supports both PyTorch and ONNX backends with optional streaming via audio file or microphone input.

Examples:

```bash
# Streaming resynthesis from .wav file
python demo.py \
    audios/librispeech-dev-clean/251-118436-0003.wav \
    --output-file reconstruction.wav \
    --config lucadellalib/focalcodec_50hz_4k_causal \
    --streaming
```

```bash
# Streaming resynthesis from laptop's microphone
python demo.py \
    microphone \
    --output-file reconstruction.wav \
    --config lucadellalib/focalcodec_50hz_4k_causal \
    --streaming
```

```bash
# Streaming voice conversion from .wav file
python demo.py \
    audios/librispeech-dev-clean/251-118436-0003.wav \
    --output-file reconstruction.wav \
    --reference-files audios/librispeech-dev-clean/84 \
    --config lucadellalib/focalcodec_50hz_4k_causal \
    --streaming
```

```bash
# Offline voice conversion from .wav file
python demo.py \
    audios/librispeech-dev-clean/251-118436-0003.wav \
    --output-file reconstruction.wav \
    --reference-files audios/librispeech-dev-clean/84 \
    --config lucadellalib/focalcodec_50hz
```

"""

import argparse
import io
import os
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio


# Suppress warnings
warnings.filterwarnings("ignore")

# Terminal colors
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[94m"
BOLD = "\033[1m"
UNDER = "\033[4m"
RESET = "\033[0m"

# Fetch the latest FocalCodec version from Torch Hub
FORCE_RELOAD = True


@torch.no_grad()
def infer_torch(
    input_file: "str",
    output_file: "str" = "reconstruction.wav",
    config: "str" = "lucadellalib/focalcodec_50hz",
    reference_files: "Optional[Sequence[str]]" = None,
    streaming: "bool" = False,
    jit: "bool" = False,
    device: "str" = "cpu",
    monitor_gpu: "bool" = False,
) -> "None":
    # Load FocalCodec model
    device = torch.device(device)
    codec = torch.hub.load(
        repo_or_dir="lucadellalib/focalcodec",
        model="focalcodec",
        config=config,
        force_reload=FORCE_RELOAD,
    )
    codec.eval().to(device)

    if streaming:
        if not codec.causal:
            raise ValueError("Streaming mode requires all components to be causal")
        print(
            f"{BLUE}{BOLD}"
            f"PyTorch streaming from {UNDER}{'microphone' if input_file == 'microphone' else input_file}{RESET}\n"
            f"{RED}{BOLD}Press Ctrl+C to stop..."
            f"{RESET}"
        )
    elif input_file == "microphone":
        raise ValueError("Set `--streaming` if using microphone input")

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
                print(f"{YELLOW}{BOLD}Skipping invalid path: {path}{RESET}")

        if reference_audio_files:
            # Use offline version to extract target speaker features
            offline_codec = torch.hub.load(
                repo_or_dir="lucadellalib/focalcodec",
                model="focalcodec",
                force_reload=FORCE_RELOAD,
            )
            offline_codec.eval().to(device)
            matching_set = []
            for reference_file in reference_audio_files:
                sig, sr = torchaudio.load(reference_file)
                sig = torchaudio.functional.resample(
                    sig.to(device), sr, offline_codec.sample_rate_input
                )
                feats = offline_codec.sig_to_feats(sig)
                matching_set.append(feats[0])
            matching_set = torch.cat(matching_set)
            del offline_codec
        else:
            raise FileNotFoundError("No valid reference files found")

    # Compile
    if jit:
        codec = codec.jit()

    # Warmup
    state_ = []
    input_ = torch.randn(1, codec.chunk_size, device=device)
    for _ in range(10):
        _, _, _, *state_ = codec(input_, *state_, return_state=True)
    del input_, state_

    # Setup
    sr_in = codec.sample_rate_input
    sr_out = codec.sample_rate_output

    # Offline inference
    if not streaming:
        print(f"{BLUE}{BOLD}PyTorch offline inference{RESET}")
        sig, orig_sr = torchaudio.load(input_file)
        sig = torchaudio.functional.resample(sig, orig_sr, sr_in)
        sig = sig.mean(dim=0, keepdim=True)  # Mono
        sig = sig.to(device)
        toks = codec.sig_to_toks(sig)
        rec_sig = codec.toks_to_sig(toks, matching_set=matching_set)
        rec_sig = rec_sig.cpu()
        torchaudio.save(output_file, rec_sig, sr_out)
        print(f"{BLUE}{BOLD}Reconstructed audio saved to: {UNDER}{output_file}{RESET}")
        return

    # Streaming states
    encoder_state, compressor_state = [], []
    decompressor_state, decoder_state = [], []

    # Inference loop
    chunk_size_in = codec.chunk_size
    chunks_out = []

    try:
        if input_file == "microphone":
            input_stream = sd.InputStream(
                samplerate=sr_in, channels=1, dtype="float32", blocksize=chunk_size_in
            )
        else:

            class AudioFileStream:
                def __init__(self) -> "None":
                    """A wrapper to simulate `sounddevice.InputStream.read()` from an audio file."""
                    sig, orig_sr = torchaudio.load(input_file)
                    sig = torchaudio.functional.resample(sig, orig_sr, sr_in)
                    sig = sig.mean(dim=0, keepdim=True)  # Mono
                    chunks = sig.split(chunk_size_in, dim=-1)
                    self.chunks = iter([x.T.numpy().astype(np.float32) for x in chunks])

                def __enter__(self) -> "AudioFileStream":
                    return self

                def __exit__(self, *_args) -> "None":
                    pass

                def read(self, blocksize: "int") -> "Tuple[np.ndarray, None]":
                    assert (
                        blocksize == chunk_size_in
                    ), f"Block size mismatch: {blocksize} vs {chunk_size_in}"
                    try:
                        chunk = next(self.chunks)
                    except StopIteration:
                        raise StopIteration("End of file stream")
                    return chunk, None

            input_stream = AudioFileStream()

        with input_stream, sd.OutputStream(
            samplerate=sr_out, channels=1, dtype="float32"
        ) as speaker:
            while True:
                try:
                    chunk_in, _ = input_stream.read(chunk_size_in)
                except StopIteration:
                    print(f"{RED}{BOLD}End of file stream{RESET}")
                    break
                chunk_in = torch.tensor(chunk_in.T, device=device)
                if chunk_in.shape[1] < chunk_size_in:
                    chunk_in = torch.nn.functional.pad(
                        chunk_in, (0, chunk_size_in - chunk_in.shape[1])
                    )

                # Inference pipeline
                toks, encoder_state, compressor_state = codec.sig_to_toks(
                    chunk_in,
                    encoder_state,
                    compressor_state,
                    return_state=True,
                )
                rec_sig, decompressor_state, decoder_state = codec.toks_to_sig(
                    toks,
                    decompressor_state,
                    decoder_state,
                    matching_set=matching_set,
                    return_state=True,
                )

                chunk_out = rec_sig[0].cpu()
                chunks_out.append(chunk_out)
                speaker.write(chunk_out.numpy().astype(np.float32))

                # Monitor GPU memory
                if monitor_gpu and device.type == "cuda":
                    max_alloc = torch.cuda.max_memory_allocated() / 1e6
                    max_reserved = torch.cuda.max_memory_reserved() / 1e6
                    print(
                        f"{YELLOW}{BOLD}"
                        f"[GPU Peak] Allocated: {max_alloc:.2f} MB | Reserved: {max_reserved:.2f} MB"
                        f"{RESET}"
                    )

    except KeyboardInterrupt:
        print(f"{RED}{BOLD}\nCtrl+C detected. Stopping streaming...{RESET}")

    finally:
        if chunks_out:
            rec_sig = torch.cat(chunks_out, dim=-1)[None]
            torchaudio.save(output_file, rec_sig, sr_out)
            print(
                f"{BLUE}{BOLD}Reconstructed audio saved to: {UNDER}{output_file}{RESET}"
            )
        else:
            print(f"{RED}{BOLD}No audio to stream{RESET}")


@torch.no_grad()
def infer_onnx(
    input_file: "str",
    output_file: "str" = "reconstruction.wav",
    config: "str" = "lucadellalib/focalcodec_50hz",
    streaming: "bool" = False,
    device: "str" = "cpu",
    monitor_gpu: "bool" = False,
) -> "None":
    print(
        f"{YELLOW}{BOLD}"
        "ONNX support is experimental and might not work as intended"
        f"{RESET}"
    )

    try:
        import onnxruntime as ort
    except ImportError:
        print(f"{RED}{BOLD}`pip install onnxruntime` to use ONNX{RESET}")
        return

    # Load FocalCodec model
    device = torch.device(device)
    codec = torch.hub.load(
        repo_or_dir="lucadellalib/focalcodec",
        model="focalcodec",
        config=config,
        force_reload=FORCE_RELOAD,
    )
    codec.eval().to(device)

    if streaming:
        if not codec.causal:
            raise ValueError("Streaming mode requires all components to be causal")
        print(
            f"{BLUE}{BOLD}"
            f"ONNX streaming from {UNDER}{'microphone' if input_file == 'microphone' else input_file}{RESET}\n"
            f"{RED}{BOLD}Press Ctrl+C to stop..."
            f"{RESET}"
        )
    elif input_file == "microphone":
        raise ValueError("Set `--streaming` if using microphone input")

    # Setup
    sr_in = codec.sample_rate_input
    sr_out = codec.sample_rate_output

    # Export ONNX model
    f = io.BytesIO()
    codec.onnx(
        f,
        batch_size=1,
        chunk_size=codec.chunk_size if streaming else None,
        return_state=True if streaming else False,
    )
    onnx_bytes = f.getvalue()

    # Setup ORT session
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.inter_op_num_threads = 4
    # so.enable_cpu_mem_arena = True
    # so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_bytes,
        sess_options=so,
        provides=[
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # Offline inference
    if not streaming:
        print(f"{BLUE}{BOLD}ONNX offline inference{RESET}")
        sig, orig_sr = torchaudio.load(input_file)
        sig = torchaudio.functional.resample(sig, orig_sr, sr_in)
        sig = sig.mean(dim=0, keepdim=True)  # Mono
        rec_sig = session.run(["rec_sig"], {"sig": sig.numpy().astype(np.float32)})[0]
        rec_sig = torch.tensor(rec_sig)
        torchaudio.save(output_file, rec_sig, sr_out)
        print(f"{BLUE}{BOLD}Reconstructed audio saved to: {UNDER}{output_file}{RESET}")
        return

    # Allocate NumPy buffers for I/O
    input_buffers = {}
    output_buffers = {}
    window_size = codec.encoder.window_size
    for i in session.get_inputs():
        shape = [d if isinstance(d, int) else window_size for d in i.shape]
        input_buffers[i.name] = np.zeros(
            shape,
            dtype={
                "tensor(float)": np.float32,
                "tensor(double)": np.float64,
                "tensor(int64)": np.int64,
            }[i.type],
        )
    for o in session.get_outputs():
        shape = [d if isinstance(d, int) else window_size for d in o.shape]
        output_buffers[o.name] = np.zeros(
            shape,
            dtype={
                "tensor(float)": np.float32,
                "tensor(double)": np.float64,
                "tensor(int64)": np.int64,
            }[o.type],
        )

    # Create IO binding
    io_binding = session.io_binding()
    for name, arr in input_buffers.items():
        ort_val = ort.OrtValue.ortvalue_from_numpy(arr, device.type, 0)
        io_binding.bind_input(
            name=name,
            device_type=ort_val.device_name(),
            device_id=0,
            element_type=arr.dtype,
            shape=ort_val.shape(),
            buffer_ptr=ort_val.data_ptr(),
        )
    for name, arr in output_buffers.items():
        ort_val = ort.OrtValue.ortvalue_from_numpy(arr, device.type, 0)
        io_binding.bind_output(
            name=name,
            device_type=ort_val.device_name(),
            device_id=0,
            element_type=arr.dtype,
            shape=ort_val.shape(),
            buffer_ptr=ort_val.data_ptr(),
        )

    # Streaming states
    state_keys = [k for k in input_buffers.keys() if k.startswith("input_state.")]
    state = [input_buffers[k] for k in state_keys]

    # Inference loop
    chunk_size_in = codec.chunk_size
    chunks_out = []

    try:
        if input_file == "microphone":
            input_stream = sd.InputStream(
                samplerate=sr_in, channels=1, dtype="float32", blocksize=chunk_size_in
            )
        else:

            class AudioFileStream:
                def __init__(self) -> "None":
                    """A wrapper to simulate `sounddevice.InputStream.read()` from an audio file."""
                    sig, orig_sr = torchaudio.load(input_file)
                    sig = torchaudio.functional.resample(sig, orig_sr, sr_in)
                    sig = sig.mean(dim=0, keepdim=True)  # Mono
                    chunks = sig.split(chunk_size_in, dim=-1)
                    self.chunks = iter([x.T.numpy().astype(np.float32) for x in chunks])

                def __enter__(self) -> "AudioFileStream":
                    return self

                def __exit__(self, *_args) -> "None":
                    pass

                def read(self, blocksize: "int") -> "Tuple[np.ndarray, None]":
                    assert (
                        blocksize == chunk_size_in
                    ), f"Block size mismatch: {blocksize} vs {chunk_size_in}"
                    try:
                        chunk = next(self.chunks)
                    except StopIteration:
                        raise StopIteration("End of file stream")
                    return chunk, None

            input_stream = AudioFileStream()

        with input_stream, sd.OutputStream(
            samplerate=sr_out, channels=1, dtype="float32"
        ) as speaker:
            while True:
                try:
                    chunk_in, _ = input_stream.read(chunk_size_in)
                except StopIteration:
                    print(f"{RED}{BOLD}End of file stream{RESET}")
                    break
                chunk_in = torch.tensor(chunk_in.T, device=device)
                if chunk_in.shape[1] < chunk_size_in:
                    chunk_in = torch.nn.functional.pad(
                        chunk_in, (0, chunk_size_in - chunk_in.shape[1])
                    )

                # Inference pipeline
                input_buffers["sig"][...] = chunk_in
                for k, v in zip(state_keys, state):
                    input_buffers[k][...] = v
                session.run_with_iobinding(io_binding)
                outputs = [x for x in output_buffers.values()]
                _, _, rec_sig, *state = outputs
                rec_sig = rec_sig.copy()

                chunk_out = rec_sig[0].astype(np.float32)
                chunks_out.append(torch.tensor(chunk_out))
                speaker.write(chunk_out)

                # Monitor GPU memory
                if monitor_gpu and device.type == "cuda":
                    max_alloc = torch.cuda.max_memory_allocated() / 1e6
                    max_reserved = torch.cuda.max_memory_reserved() / 1e6
                    print(
                        f"{GREEN}{BOLD}[GPU Peak] Allocated: {max_alloc:.2f} MB | Reserved: {max_reserved:.2f} MB{RESET}"
                    )

    except KeyboardInterrupt:
        print(f"{RED}{BOLD}\nCtrl+C detected. Stopping streaming...{RESET}")

    finally:
        if chunks_out:
            rec_sig = torch.cat(chunks_out, dim=-1)[None]
            torchaudio.save(output_file, rec_sig, sr_out)
            print(
                f"{BLUE}{BOLD}Reconstructed audio saved to: {UNDER}{output_file}{RESET}"
            )
        else:
            print(f"{RED}{BOLD}No audio to stream{RESET}")


def main() -> "None":
    parser = argparse.ArgumentParser(description="FocalCodec demo")
    parser.add_argument(
        "input_file",
        metavar="input-file",
        type=str,
        help="path to the input audio file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="reconstruction.wav",
        help="path to the reconstructed audio file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lucadellalib/focalcodec_50hz",
        help="FocalCodec configuration",
    )
    parser.add_argument(
        "--reference-files",
        type=str,
        nargs="+",
        default=None,
        help="path(s) to reference audio files or a directory containing reference audio files (PyTorch only)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="enable real-time streaming mode (requires causal model)",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="use TorchScript JIT compilation for the model (PyTorch only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "onnx"],
        default="torch",
        help="backend to use",
    )
    parser.add_argument(
        "--monitor-gpu",
        action="store_true",
        help="print GPU memory usage during streaming (only if using CUDA)",
    )
    args = parser.parse_args()

    print(
        f"{RED}{BOLD}{UNDER}WARNING{RESET}: "
        f"{RED}{BOLD}the codec might generate unexpectedly loud or unpleasant artifacts. "
        "It is recommended to set your system volume to minimum before playback and raise it gradually."
        f"{RESET}",
    )
    if args.backend == "torch":
        infer_torch(
            args.input_file,
            args.output_file,
            args.config,
            args.reference_files,
            args.streaming,
            args.jit,
            args.device,
            args.monitor_gpu,
        )
    else:
        if args.reference_files:
            raise ValueError(
                "`--reference-files` option is not supported with ONNX inference"
            )
        if args.jit:
            raise ValueError("`--jit` flag is incompatible with ONNX inference")
        infer_onnx(
            args.input_file,
            args.output_file,
            args.config,
            args.streaming,
            args.device,
            args.monitor_gpu,
        )


if __name__ == "__main__":
    main()
