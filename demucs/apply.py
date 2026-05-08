# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import (
    Any,
    Callable,
    TypeAlias,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .blocks import center_trim
from .htdemucs import HTDemucs

Model: TypeAlias = HTDemucs


class ModelEnsemble(nn.Module):
    def __init__(
        self,
        models: list[Model],
        weights: list[list[float]] | None = None,
        segment: float | None = None,
    ):
        """
        Represents a model ensemble with specific weights.
        You should call ``apply_model`` rather than calling the forward directly
        for optimal performance.

        :param models: List of Demucs models.
        :param weights: List of per-model weight lists. If ``None``, assumed to
            be all ones, otherwise a list of N lists (N number of models),
            each containing S floats (S number of sources).
        :param segment: Overrides the ``segment`` attribute of each model
            (performed in-place, be careful if you reuse the models passed).
        """
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                if (
                    not isinstance(other, HTDemucs)
                    or segment <= other.max_allowed_segment
                ):
                    other.max_allowed_segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1.0 for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        """
        Return the minimum ``max_allowed_segment`` across all models in the ensemble.

        :return: Maximum allowed segment length in seconds.
        """
        max_allowed_segment = float("inf")
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(
                    max_allowed_segment, float(model.max_allowed_segment)
                )
        return max_allowed_segment

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass is not supported; use ``apply_model`` instead.

        :param x: Input tensor.
        :return: Never returns.
        :raises NotImplementedError: Always raised.
        """
        raise NotImplementedError("Call `apply_model` on this.")


class TensorChunk:
    def __init__(
        self, tensor: Tensor | "TensorChunk", offset: int = 0, length: int | None = None
    ) -> None:
        """
        A lazy view into a tensor along the last dimension.

        :param tensor: Source tensor or another ``TensorChunk`` to wrap.
        :param offset: Start offset along the last dimension.
        :param length: Number of frames to include. If ``None``, extends to the end.
        """
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self) -> list[int]:
        """
        Return the virtual shape with the last dimension reflecting the chunk length.

        :return: Shape as a list of ints.
        """
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length: int) -> Tensor:
        """
        Return the chunk padded (or trimmed) to ``target_length``, centered on the chunk.

        :param target_length: Desired length of the last dimension.
        :return: Padded tensor of the requested length.
        """
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        # Common case: target_length matches the chunk's natural length and the
        # chunk lies fully inside the underlying tensor. Skip the F.pad call,
        # which on MPS still allocates a fresh contiguous tensor even when the
        # padding amount is (0, 0).
        if pad_left == 0 and pad_right == 0:
            out = self.tensor[..., correct_start:correct_end]
        else:
            out = F.pad(
                self.tensor[..., correct_start:correct_end],
                (pad_left, pad_right),
            )
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk: Tensor | TensorChunk) -> TensorChunk:
    """
    Wrap a tensor or pass through an existing ``TensorChunk``.

    :param tensor_or_chunk: A raw tensor or an existing ``TensorChunk``.
    :return: A ``TensorChunk`` instance.
    """
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, Tensor)
        return TensorChunk(tensor_or_chunk)


_SPLIT_WEIGHT_CACHE: dict[
    tuple[int, float, torch.device, torch.dtype], Tensor
] = {}


def _split_weight(
    segment_length: int,
    transition_power: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build (or look up) the triangular cross-fade weight applied to each
    chunk during overlap-add.

    :param segment_length: Length of one segment in samples.
    :param transition_power: Exponent applied to the normalised triangle.
    :param device: Device to allocate the weight tensor on.
    :param dtype: Dtype for the weight tensor.
    :return: 1-D weight tensor of length ``segment_length``.
    """
    key = (segment_length, transition_power, device, dtype)
    cached = _SPLIT_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    half = segment_length // 2
    rising = torch.arange(1, half + 1, device=device, dtype=dtype)
    falling = torch.arange(segment_length - half, 0, -1, device=device, dtype=dtype)
    weight = torch.cat([rising, falling])
    weight = (weight / weight.max()) ** transition_power
    _SPLIT_WEIGHT_CACHE[key] = weight
    return weight


def apply_model(
    model: ModelEnsemble | Model,
    mix: Tensor | TensorChunk,
    device: torch.device | str | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int = 1,
) -> Tensor:
    """
    Apply model to a given mixture, tiling into segments of the model's
    training length (``model.max_allowed_segment * model.samplerate``).

    :param model: Model or ensemble to apply.
    :param mix: Input mixture tensor or chunk.
    :param device: Device for local computation; if ``None``, ``mix.device``.
    :param shifts: If > 0, average over *shifts* random sub-second shifts.
    :param overlap: Overlap ratio between consecutive segments.
    :param transition_power: Exponent on the triangular crossfade weight.
    :param progress_callback: Optional ``callback(event_type, data)``; events:
        ``processing_start``, ``chunk_complete``, ``processing_complete``.
    :param use_only_stem: For ``ModelEnsemble``, only run the sub-model
        specialised for this stem.
    :param chunk_batch_size: Chunks processed in parallel.
    :return: Separated sources tensor.
    :raises ValueError: If ``overlap`` produces a non-positive segment stride.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    kwargs = {
        "shifts": shifts,
        "overlap": overlap,
        "transition_power": transition_power,
        "progress_callback": progress_callback,
        "device": device,
        "use_only_stem": use_only_stem,
        "chunk_batch_size": chunk_batch_size,
    }
    out: float | Tensor
    res: float | Tensor
    if isinstance(model, ModelEnsemble):
        # Special treatment for model ensemble.
        # We explicitely apply multiple times `apply_model` so that the random shifts
        # are different for each model.

        # Optimization: If use_only_stem is specified, only run the specialized model
        if use_only_stem:
            # Find which model specializes in this stem
            try:
                stem_index = model.sources.index(use_only_stem)
            except ValueError:
                # Stem doesn't exist, fall through to run all models
                pass
            else:
                # Find the model that specializes in this stem
                model_index = None
                for i, weights in enumerate(model.weights):
                    if (
                        len(weights) > stem_index
                        and abs(weights[stem_index] - 1.0) < 1e-6
                        and all(
                            abs(w) < 1e-6
                            for j, w in enumerate(weights)
                            if j != stem_index
                        )
                    ):
                        model_index = i
                        break

                if model_index is not None:
                    sub_kwargs = dict(kwargs)
                    sub_kwargs.pop("use_only_stem")
                    return apply_model(model.models[model_index], mix, **sub_kwargs)

        # Run all models in the ensemble — all sub-models are already on device
        estimates: float | Tensor = 0.0
        totals = [0.0] * len(model.sources)
        for sub_model, model_weights in zip(model.models, model.weights):
            sub_kwargs = dict(kwargs)
            sub_kwargs.pop("use_only_stem")
            out = apply_model(sub_model, mix, **sub_kwargs)

            for k, inst_weight in enumerate(model_weights):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight
            estimates += out
            del out

        assert isinstance(estimates, Tensor)
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]
        return estimates

    # Move/eval the model only when needed. ``Module.to`` and ``Module.eval``
    # iterate every submodule even when the result is a no-op, so we cheaply
    # check first and skip the recursion when the model is already where we
    # want it. ``Separator.__init__`` always sets eval mode and places the
    # model on its device, so the typical call hits the fast path.
    first_param = next(model.parameters(), None)
    if first_param is not None and first_param.device != device:
        model.to(device)
    if model.training:
        model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    if shifts:
        kwargs["shifts"] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.0
        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            res = apply_model(model, shifted, **kwargs)
            shifted_out = res
            out += shifted_out[..., max_shift - offset :]
        out /= shifts
        assert isinstance(out, Tensor)
        return out

    # Allocate accumulators on the inference *device* rather than mix.device.
    # Previously each chunk's output was transferred from device->host (e.g.
    # MPS/CUDA -> CPU) before being added to the accumulator, forcing a
    # synchronization for every chunk. Accumulating on-device lets the
    # backend keep work in flight and we transfer once at the end.
    out_device = device
    out = torch.zeros(
        batch,
        len(model.sources),
        channels,
        length,
        device=out_device,
    )
    sum_weight = torch.zeros(length, device=out_device)
    segment = model.max_allowed_segment
    assert segment > 0.0
    segment_length: int = int(model.samplerate * segment)
    stride = int((1 - overlap) * segment_length)
    if stride < 1:
        raise ValueError(
            f"split overlap {overlap} produces an invalid stride for segment length {segment_length}"
        )
    offsets = range(0, length, stride)
    weight = _split_weight(segment_length, transition_power, out_device, torch.float32)

    if isinstance(model, HTDemucs):
        chunk_valid_length: int = segment_length
    elif hasattr(model, "valid_length"):
        chunk_valid_length = model.valid_length(segment_length)  # type: ignore
    else:
        chunk_valid_length = segment_length

    # Move mix to the inference device once instead of per chunk in the loop.
    mix_dev: Tensor | TensorChunk
    if isinstance(mix, TensorChunk):
        inner = mix.tensor
        if inner.device != device:
            inner = inner.to(device)
        mix_dev = TensorChunk(inner, mix.offset, mix.length)
    else:
        mix_dev = mix if mix.device == device else mix.to(device)

    # Collect all chunks
    all_chunks = [
        (offset, TensorChunk(mix_dev, offset, segment_length)) for offset in offsets
    ]
    total_chunks = len(all_chunks)

    if progress_callback:
        progress_callback("processing_start", {"total_chunks": total_chunks})

    completed_chunks = 0
    for batch_start in range(0, total_chunks, chunk_batch_size):
        batch_items = all_chunks[batch_start : batch_start + chunk_batch_size]

        # Chunks already live on `device`; just pad and stack.
        padded = torch.cat(
            [chunk.padded(chunk_valid_length) for _, chunk in batch_items],
            dim=0,
        )  # [N, channels, chunk_valid_length]

        with torch.inference_mode():
            batch_out = model(padded)  # [N, sources, channels, ...]

        for i, (offset, chunk) in enumerate(batch_items):
            chunk_out = center_trim(batch_out[i : i + 1], chunk.length)
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            out[..., offset : offset + segment_length] += w * chunk_out
            sum_weight[offset : offset + segment_length] += w

            completed_chunks += 1
            if progress_callback:
                progress_callback(
                    "chunk_complete",
                    {
                        "completed_chunks": completed_chunks,
                        "total_chunks": total_chunks,
                    },
                )

    if progress_callback:
        progress_callback("processing_complete", {"total_chunks": total_chunks})
    # No `assert sum_weight.min() > 0`: `.min()` forces a host sync, and the
    # triangular weight + validated stride guarantee full coverage.
    out /= sum_weight
    if out.device != mix.device:
        out = out.to(mix.device)
    assert isinstance(out, Tensor)
    return out
