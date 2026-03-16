# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math

import torch
import torch.nn as nn

from .blocks import spectro
from .htdemucs import HTDemucs
from .repo import ModelRepository


class HTDemucsONNXWrapper(nn.Module):
    """
    Wrapper that makes HTDemucs compatible with ONNX export.
    """

    def __init__(self, model: HTDemucs) -> None:
        """
        Initialize the ONNX wrapper.

        :param model: The HTDemucs model to wrap for ONNX export
        """
        super().__init__()
        self.model = model
        self.sources = model.sources
        self.samplerate = model.samplerate
        self.audio_channels = model.audio_channels
        self.nfft = model.nfft
        self.hop_length = model.hop_length

    def forward(
        self, spec_real: torch.Tensor, spec_imag: torch.Tensor, mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        :param spec_real: Real part of spectrogram [B, C, Fq, T]
        :param spec_imag: Imaginary part of spectrogram [B, C, Fq, T]
        :param mix: Raw audio waveform [B, C, samples]
        :return: Tuple of (out_spec_real, out_spec_imag, out_wave) separated spectrograms and waveforms
        """
        B, C, Fq, T = spec_real.shape
        samples = mix.shape[-1]

        # Convert real/imag to CaC format: [ch0_real, ch0_imag, ch1_real, ch1_imag, ...]
        x = torch.stack([spec_real, spec_imag], dim=2).reshape(B, C * 2, Fq, T)

        # Normalize inputs
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt = (mix - meant) / (1e-5 + stdt)

        # Core encoder-transformer-decoder processing
        x, xt = self.model.forward_core(x, xt)

        # Denormalize and reshape frequency branch output
        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # Split CaC back into real/imag
        out_spec_real = x[:, :, 0::2, :, :]
        out_spec_imag = x[:, :, 1::2, :, :]

        # Denormalize and reshape time branch output
        xt = xt.view(B, S, -1, samples)
        xt = xt * stdt[:, None] + meant[:, None]

        return out_spec_real, out_spec_imag, xt


def compute_stft_for_export(
    audio: torch.Tensor, nfft: int, hop_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT for model input, matching HTDemucs preprocessing.

    :param audio: Input audio [B, C, samples]
    :param nfft: FFT size
    :param hop_length: Hop length
    :return: Tuple of (real, imag) spectrograms [B, C, Fq, T]
    """
    # Padding to match HTDemucs._spec
    le = int(math.ceil(audio.shape[-1] / hop_length))
    pad = hop_length // 2 * 3

    # Pad the audio
    padded = torch.nn.functional.pad(
        audio, (pad, pad + le * hop_length - audio.shape[-1]), mode="reflect"
    )

    # Compute STFT
    z = spectro(padded, nfft, hop_length)

    # Trim to expected size
    z = z[..., :-1, :]  # Remove last frequency bin
    z = z[..., 2 : 2 + le]  # Trim time dimension

    # Split into real and imaginary
    real = z.real
    imag = z.imag

    return real, imag


def export_to_onnx(
    model_name: str = "htdemucs",
    output_path: str | None = None,
    opset_version: int = 17,
    segment_seconds: float = 10.0,
    fp16: bool = False,
) -> str:
    """
    Export Demucs model to ONNX format.

    :param model_name: Name of the model to export
    :param output_path: Path to save the ONNX model (defaults to {model_name}.onnx)
    :param opset_version: ONNX opset version
    :param segment_seconds: Audio segment length in seconds
    :param fp16: If True, convert model weights to float16 for smaller size and faster
        WebGPU inference. Inputs/outputs remain float32.
    :return: Path to the exported ONNX model
    :raises ImportError: If the onnx package is not installed
    :raises ValueError: If the model is not an HTDemucs instance
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: uv pip install demucs-next[onnx]"
        )

    if output_path is None:
        suffix = "_fp16" if fp16 else ""
        output_path = f"{model_name}{suffix}.onnx"

    repo = ModelRepository()
    model = repo.get_model(model_name)

    if not isinstance(model, HTDemucs):
        raise ValueError(
            f"Model {model_name} is not a supported model type. "
            f"Expected HTDemucs, got {type(model).__name__}"
        )
    wrapper = HTDemucsONNXWrapper(model)

    model.eval()
    wrapper.eval()

    sample_rate = model.samplerate
    segment_samples = int(segment_seconds * sample_rate)
    nfft = model.nfft
    hop_length = model.hop_length

    batch_size = 1
    audio_channels = model.audio_channels

    dummy_audio = torch.randn(batch_size, audio_channels, segment_samples)
    dummy_spec_real, dummy_spec_imag = compute_stft_for_export(
        dummy_audio, nfft, hop_length
    )

    torch.onnx.export(
        wrapper,
        (dummy_spec_real, dummy_spec_imag, dummy_audio),
        output_path,
        input_names=["spec_real", "spec_imag", "audio"],
        output_names=["out_spec_real", "out_spec_imag", "out_wave"],
        dynamic_axes={
            "spec_real": {0: "batch", 3: "time"},
            "spec_imag": {0: "batch", 3: "time"},
            "audio": {0: "batch", 2: "samples"},
            "out_spec_real": {0: "batch", 4: "time"},
            "out_spec_imag": {0: "batch", 4: "time"},
            "out_wave": {0: "batch", 3: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)

    sources_meta = onnx_model.metadata_props.add()
    sources_meta.key = "sources"
    sources_meta.value = json.dumps(model.sources)

    sample_rate_meta = onnx_model.metadata_props.add()
    sample_rate_meta.key = "sample_rate"
    sample_rate_meta.value = str(model.samplerate)

    channels_meta = onnx_model.metadata_props.add()
    channels_meta.key = "audio_channels"
    channels_meta.value = str(model.audio_channels)

    if fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16

        onnx_model = convert_float_to_float16(
            onnx_model,
            keep_io_types=True,
            op_block_list=[
                "LayerNormalization",
                "InstanceNormalization",
                "ReduceMean",
                "Softmax",
                "Pow",
                "Sqrt",
                "Range",
                "Mod",
            ],
        )

    onnx.save(onnx_model, output_path)

    return output_path
