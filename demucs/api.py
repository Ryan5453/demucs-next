# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from . import __version__
from .apply import Model, ModelEnsemble, apply_model
from .audio import convert_audio, prevent_clip
from .exceptions import (
    LoadAudioError,
    ModelLoadingError,
    ValidationError,
)
from .repo import ModelRepository


class SeparatedSources:
    """
    Container for storing and processing separated audio sources.
    """

    def __init__(
        self,
        sources: dict[str, Tensor],
        sample_rate: int,
        original: Tensor,
    ):
        """
        Initialize a SeparatedSources object.

        :param sources: Mapping of stem names to audio tensors
        :param sample_rate: Sample rate of the audio - comes from the model's sample rate
        :param original: Original unseparated audio
        """
        self.sources = sources
        self.sample_rate = sample_rate
        self.original = original

    def isolate_stem(self, name: str) -> "SeparatedSources":
        """
        Isolate a stem from the separated sources.
        This creates a new SeparatedSources object with the isolated stem and the accompanying complement stem (no_{STEM})

        :param name: Name of the stem to isolate
        :return: New SeparatedSources object with the isolated stem and the accompanying complement stem
        :raises ValidationError: If the requested stem isn't found in the sources
        """
        if name not in self.sources:
            raise ValidationError(
                f"Stem '{name}' not found in sources. Available stems: {list(self.sources.keys())}"
            )

        complement = torch.zeros_like(self.sources[name])
        for source, audio in self.sources.items():
            if source != name:
                complement += audio

        return SeparatedSources(
            sources={name: self.sources[name], f"no_{name}": complement},
            sample_rate=self.sample_rate,
            original=self.original,
        )

    def export_stem(
        self,
        stem_name: str,
        path: Path | str | None = None,
        format: str = "wav",
        clip: str | None = "rescale",
    ) -> Path | bytes:
        """
        Export a stem to either a file path or return as bytes.

        :param stem_name: Name of the stem to export
        :param path: Path to save the stem to. If None, returns raw audio bytes
        :param format: Format to export the stem to, anything supported by FFmpeg
        :param clip: Clipping mode to prevent audio distortion ("rescale", "clamp", "tanh", or None)
        :return: Path to saved file if path provided, otherwise raw audio bytes
        :raises ValidationError: If the stem name is not found
        """
        if stem_name not in self.sources:
            raise ValidationError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        tensor = self.sources[stem_name]

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        tensor = prevent_clip(tensor, mode=clip)

        if path is not None:
            path = Path(path)

            if not path.suffix:
                file_path = path.with_suffix(f".{format}")
            else:
                file_path = path

            file_path.parent.mkdir(exist_ok=True, parents=True)

            encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
            encoder.to_file(file_path)

            return file_path
        else:
            encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
            encoded_tensor = encoder.to_tensor(format=format)
            return bytes(encoded_tensor.untyped_storage())


class Separator:
    """
    Audio source separation using Demucs models.
    """

    def __init__(
        self,
        model: str | Model | ModelEnsemble = "htdemucs",
        device: str = "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
        only_load: str | None = None,
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ):
        """
        Initialize a Separator with the specified model and device.

        :param model: Model to use for separation (name or model instance)
        :param device: Device to use for processing (must be "cpu", "cuda", or "mps")
        :param only_load: If specified, load only the specialized model for this stem
                         (only applicable to bag-of-models like htdemucs_ft)
        :param dtype: Set to torch.float16 for half-precision inference on CUDA.
                     If None, uses default float32. Model weights are cast at init time.
        :param compile: If True, apply ``torch.compile`` for faster inference after an
                       initial warmup. Best for API servers or batch processing; adds
                       significant latency to the first forward pass.
        :raises ValidationError: If device is not valid or only_load stem doesn't exist
        :raises ModelLoadingError: If model fails to load
        """
        # Validate device
        valid_devices = {"cpu", "cuda", "mps"}
        if device not in valid_devices:
            raise ValidationError(
                f"Invalid device '{device}'. Must be one of: {', '.join(sorted(valid_devices))}"
            )

        # Validate dtype — only FP16 on CUDA is supported
        if dtype is not None:
            if dtype != torch.float16:
                raise ValidationError(
                    f"Invalid dtype '{dtype}'. Only torch.float16 is supported."
                )
            if device != "cuda":
                raise ValidationError(
                    "float16 inference is only supported on CUDA. "
                    "CPU and MPS do not benefit from reduced precision."
                )

        self.device = device
        self.dtype = dtype

        # Handle both string model names and model instances
        if isinstance(model, str):
            model_repo = ModelRepository()
            self.model = model_repo.get_model(name=model, only_load=only_load)
        else:
            # model is already a Model instance
            self.model = model

        self.model.eval()
        if self.model is None:
            raise ModelLoadingError("Failed to load model")

        # Validate only_load stem exists in the loaded model
        if only_load and only_load not in self.model.sources:
            raise ValidationError(
                f"Stem '{only_load}' not found in model. "
                f"Available stems: {', '.join(self.model.sources)}"
            )

        self.audio_channels = self.model.audio_channels
        self.sample_rate = self.model.samplerate

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        if self.device in {"cuda", "mps"}:
            self.model.to(self.device)

        # Cast weights to FP16 if requested.
        if self.dtype is not None:
            if isinstance(self.model, ModelEnsemble):
                for m in self.model.models:
                    m.to(dtype=self.dtype)
            else:
                self.model.to(dtype=self.dtype)

        # torch.compile: JIT-compile for fused kernels and reduced launch overhead.
        # reduce-overhead mode enables CUDA graphs for minimal kernel launch cost.
        # Disabled by default because compilation adds ~30s to init time.
        if compile and self.device == "cuda":
            if isinstance(self.model, ModelEnsemble):
                for i, m in enumerate(self.model.models):
                    self.model.models[i] = torch.compile(
                        m, mode="reduce-overhead"
                    )
            else:
                self.model = torch.compile(self.model, mode="reduce-overhead")

            # Warmup pass with synthetic audio to trigger compilation now
            # rather than on the first real call.
            warmup = torch.zeros(
                1, self.audio_channels, self.sample_rate, device=self.device
            )
            if self.dtype is not None:
                warmup = warmup.to(dtype=self.dtype)
            with torch.no_grad():
                self.model(warmup)
            del warmup

    def _to_tensor(self, audio: tuple[Tensor, int] | Path | str | bytes) -> Tensor:
        """
        Convert various input types (tuple of Tensor and sample rate, path, bytes)
        to a 2D float32 tensor on the configured device, matching the model's
        sample rate and channels when possible.
        """
        wav: Tensor
        input_sr: int | None = None

        if isinstance(audio, tuple):
            wav, input_sr = audio
        elif isinstance(audio, (str, Path)):
            try:
                # Use native torchcodec AudioDecoder for better performance
                decoder = AudioDecoder(str(Path(audio)))
                audio_samples = decoder.get_all_samples()
                wav = audio_samples.data
                input_sr = audio_samples.sample_rate
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load file {audio} using torchcodec: {e}. "
                    "Make sure the file format is supported."
                )
        elif isinstance(audio, bytes):
            audio_buffer = BytesIO(audio)
            try:
                # Use native torchcodec AudioDecoder for better performance
                decoder = AudioDecoder(audio_buffer)
                audio_samples = decoder.get_all_samples()
                wav = audio_samples.data
                input_sr = audio_samples.sample_rate
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load audio from bytes using torchcodec: {e}. "
                    "Make sure the audio format is supported."
                )
            finally:
                audio_buffer.close()
        else:
            raise ValueError(
                f"Unsupported audio input type: {type(audio)}. "
                "Expected tuple of (Tensor, sample_rate), file path (str/Path), or bytes."
            )

        # Minimal shape/dtype normalization
        if wav.dim() == 1:
            wav = wav[None]
        if wav.dtype != torch.float32:
            wav = wav.float()

        # Try to match expected sample rate/channels when we know input_sr, or channels mismatch
        if input_sr is not None and input_sr != self.sample_rate:
            wav = convert_audio(wav, input_sr, self.sample_rate, self.audio_channels)
        elif wav.shape[0] != self.audio_channels:
            # Adjust channels without resampling
            wav = convert_audio(
                wav, self.sample_rate, self.sample_rate, self.audio_channels
            )

        return wav

    def separate(
        self,
        audio: tuple[Tensor, int] | Path | str | bytes,
        shifts: int = 1,
        split: bool = True,
        split_size: int | None = None,
        split_overlap: float = 0.25,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        use_only_stem: str | None = None,
        chunk_batch_size: int | None = None,
    ) -> SeparatedSources:
        """
        Separate audio into stems. Accepts tensor with sample rate, file path, or raw bytes.

        :param audio: Audio input - can be:
                     - A tuple of (Tensor, sample_rate) where Tensor is shape [channels, samples]
                     - A file path (str or Path)
                     - Raw audio bytes
        :param shifts: Number of random shifts for equivariant stabilization (1-20)
                      Higher values improve quality but increase processing time
        :param split_overlap: Overlap between split chunks (0.0 to 1.0)
                             Higher values improve quality at chunk boundaries
        :param split: Whether to split the input into chunks for processing
        :param split_size: Length (in seconds) of each chunk (only used if split=True)
        :param progress_callback: Optional callback for progress updates during audio processing
        :param use_only_stem: If specified and model is a ModelEnsemble, only use the model
                             specialized for this stem (performance optimization for Cog)
        :param chunk_batch_size: Number of audio chunks to process simultaneously when splitting.
                               Higher values use more VRAM but improve throughput by saturating GPU compute.
                               Defaults to 4 on CUDA (up to 1.9x faster), 1 on CPU/MPS.
        :return: SeparatedSources object containing the separated stems
        :raises ValidationError: If any parameter value is invalid
        """
        # Validate shifts parameter
        if not isinstance(shifts, int) or shifts < 1 or shifts > 20:
            raise ValidationError(
                f"shifts must be an integer between 1 and 20 (inclusive), got {shifts}"
            )

        # Validate split_overlap parameter
        if (
            not isinstance(split_overlap, (int, float))
            or split_overlap < 0.0
            or split_overlap > 1.0
        ):
            raise ValidationError(
                f"split_overlap must be a float between 0.0 and 1.0 (inclusive), got {split_overlap}"
            )

        # Validate and adjust split_size parameter
        if split_size is not None:
            if not isinstance(split_size, (int, float)) or split_size < 1:
                raise ValidationError(
                    f"split_size must be a number >= 1 if provided, got {split_size}"
                )
            max_allowed = self.model.max_allowed_segment
            if split_size > max_allowed and max_allowed != float("inf"):
                split_size = max_allowed

        # Validate progress_callback parameter
        if progress_callback is not None and not callable(progress_callback):
            raise ValidationError(
                f"progress_callback must be callable if provided, got {type(progress_callback)}"
            )

        # Auto-select chunk_batch_size based on device if not specified
        if chunk_batch_size is None:
            chunk_batch_size = 4 if self.device == "cuda" else 1

        # Validate chunk_batch_size parameter
        if not isinstance(chunk_batch_size, int) or chunk_batch_size < 1:
            raise ValidationError(
                f"chunk_batch_size must be a positive integer, got {chunk_batch_size}"
            )

        # Normalize input to tensor
        wav = self._to_tensor(audio)

        # Separation logic (inlined)
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std()
        wav = (wav - mean) / (1e-5 + std)

        sources_tensor = apply_model(
            self.model,
            wav[None],
            device=self.device,
            shifts=shifts,
            split=split,
            overlap=split_overlap,
            segment=split_size,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
        )[0]

        sources = {}
        for source_idx, source_name in enumerate(self.model.sources):
            sources[source_name] = sources_tensor[source_idx] * std + mean

        return SeparatedSources(sources, self.sample_rate, original=wav)


def select_model(
    isolate_stem: str | None = None,
) -> tuple[str, str | None]:
    """
    Select optimal Demucs model for audio separation.

    This function automatically chooses the best model based on the use case,
    optimizing for quality and performance:

    - For specific stems: Uses specialized fine-tuned models
      - vocals, bass, other -> htdemucs_ft (fine-tuned, better quality)
      - guitar, piano -> htdemucs_6s (6-stem model)
      - drums -> htdemucs (already optimal)

    - Default: htdemucs (balanced quality/performance)

    :param isolate_stem: Specific stem to isolate.
    :return: Tuple of (model_name, only_load_stem)
        - model_name: Name of the recommended model
        - only_load_stem: Stem to load exclusively from ModelEnsemble (for htdemucs_ft),
                         or None to load all stems

    """
    # Stem-specific model selection (quality optimization)
    if isolate_stem:
        if isolate_stem in ["guitar", "piano"]:
            # htdemucs_6s is specialized for 6-stem separation
            return ("htdemucs_6s", None)
        if isolate_stem == "drums":
            # htdemucs already performs best for drums
            return ("htdemucs", None)
        if isolate_stem in ["bass", "other", "vocals"]:
            # htdemucs_ft has fine-tuned models for these stems
            return ("htdemucs_ft", isolate_stem)

    # Default: htdemucs is the best balanced model
    return ("htdemucs", None)


def get_version() -> str:
    """
    Get the version of Demucs you have installed.

    :return: Version string
    """
    return __version__
