# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Demucs requires PyTorch 2.9+ but Cog does not support that yet
# Cog ends up serving us an incorrect version, so we need to force
# priority for PyTorch's embedded version
import os

os.environ.pop("LD_LIBRARY_PATH", None)

from io import BytesIO

from cog import BaseModel, BasePredictor, Input, Path

from demucs import ModelRepository, Separator, select_model


class Output(BaseModel):
    class Config:
        extra = "allow"


class Predictor(BasePredictor):
    separators: dict[str, "Separator"] = {}

    def setup(self) -> None:
        repo = ModelRepository()

        for model_name in repo.list_models().keys():
            separator = Separator(
                model=model_name,
                load_all=True,
            )
            self.separators[model_name] = separator

    def predict(
        self,
        audio: Path = Input(description="The audio file to separate"),
        model: str = Input(
            description="Model to use for separation",
            default="auto",
            choices=[
                "auto",
                "hdemucs_mmi",
                "htdemucs",
                "htdemucs_ft",
                "htdemucs_6s",
            ],
        ),
        format: str = Input(
            description="Output audio format, anything supported by FFmpeg",
            default="wav",
        ),
        isolate_stem: str = Input(
            description="Only creates a {stem} and no_{stem} stem/file",
            default="none",
            choices=[
                "none",
                "drums",
                "bass",
                "other",
                "vocals",
                "guitar",
                "piano",
            ],
        ),
        shifts: int = Input(
            description="Number of random shifts for equivariant stabilization, more increases quality but increases processing time linearly",
            default=1,
            ge=1,
            le=20,
        ),
        split: bool = Input(
            description="Split audio into chunks to save memory",
            default=True,
        ),
        split_size: int | None = Input(
            description="Size of each chunk in seconds, smaller values use less GPU memory but process slower",
            default=None,
            ge=1,
        ),
        split_overlap: float = Input(
            description="Overlap between split chunks, higher values improve quality at chunk boundaries",
            default=0.25,
            ge=0.0,
            le=1.0,
        ),
        clip_mode: str = Input(
            description="Method to prevent audio clipping in output, or None for no clipping prevention",
            default="rescale",
            choices=["none", "rescale", "clamp", "tanh"],
        ),
    ) -> Output:
        if model == "auto":
            model, _ = select_model(
                audio=audio,
                isolate_stem=None if isolate_stem == "none" else isolate_stem,
            )
        separator = self.separators[model]

        if isolate_stem != "none":
            separated = separator.separate(
                audio=audio,
                shifts=shifts,
                split=split,
                split_size=split_size,
                split_overlap=split_overlap,
                use_only_stem=isolate_stem,
            )
            separated = separated.isolate_stem(isolate_stem)
        else:
            separated = separator.separate(
                audio=audio,
                shifts=shifts,
                split=split,
                split_size=split_size,
                split_overlap=split_overlap,
            )

        output_data = {}
        for stem in separated.sources:
            audio_bytes = separated.export_stem(
                stem, format=format, clip=None if clip_mode == "none" else clip_mode
            )
            buf = BytesIO(audio_bytes)
            buf.name = f"{stem}.{format}"  # Set filename for mime type detection
            output_data[stem] = buf

        return Output(**output_data)
