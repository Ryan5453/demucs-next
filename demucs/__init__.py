# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

# Suppress PyTorch internal resize warnings from STFT/iSTFT
# These are benign and originate from PyTorch issue #134323
warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized since it had shape",
    category=UserWarning,
)

__version__ = "1.0.0.dev0"

from .api import (
    SeparatedSources,
    Separator,
    get_version,
    select_model,
)
from .repo import ModelRepository

__all__ = [
    "__version__",
    "Separator",
    "SeparatedSources",
    "ModelRepository",
    "get_version",
    "select_model",
]
