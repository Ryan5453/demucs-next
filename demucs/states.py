# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import inspect
import warnings
from collections.abc import Callable
from pathlib import Path

import torch

# Known deprecated parameters that are present in older model checkpoints
# but are no longer used in the current model classes. These are silently ignored.
_DEPRECATED_PARAMS = frozenset(
    {
        # Legacy Wiener filtering parameters
        "wiener_iters",
        "end_iters",
        "wiener_residual",
        # Removed sparse attention parameters (xformers APIs deprecated in 0.0.34)
        "t_sparse_self_attn",
        "t_sparse_cross_attn",
        "t_mask_type",
        "t_mask_random_seed",
        "t_sparse_attn_window",
        "t_global_window",
        "t_sparsity",
        "t_auto_sparsity",
    }
)


def load_model(path_or_package: dict | str | Path, strict: bool = False) -> torch.nn.Module:
    """
    Load a model from a serialized dict or a file path.

    :param path_or_package: A dict (already loaded) or path to a serialized model file
    :param strict: If True, do not drop unknown parameters
    :return: The loaded model with state restored
    :raises ValueError: If path_or_package is not a dict, str, or Path
    """
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, "cpu", weights_only=False)
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                if key not in _DEPRECATED_PARAMS:
                    warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def set_state(model: torch.nn.Module, state: dict) -> None:
    """
    Set the state dict on a model.

    :param model: The model to load state into
    :param state: The state dict to load
    """
    model.load_state_dict(state)


def capture_init(init: Callable) -> Callable:
    """
    Decorator that captures the args and kwargs passed to __init__.

    :param init: The __init__ method to wrap
    :return: Wrapped __init__ that stores args/kwargs on the instance
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
