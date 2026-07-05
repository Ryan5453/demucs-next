# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class DemucsError(Exception):
    """
    Base exception class for all Demucs-specific errors.
    """

    pass


class LoadAudioError(DemucsError):
    """
    Exception raised when audio loading fails.
    """

    pass


class ModelLoadingError(DemucsError):
    """
    Exception raised when model loading fails.
    """

    pass


class ValidationError(DemucsError, ValueError):
    """
    Exception raised when a parameter value is invalid.

    Also a ``ValueError`` so callers catching the builtin keep working.
    """

    pass
