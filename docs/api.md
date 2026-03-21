# Demucs API

The Demucs Python API is primarily comprised of two classes: `Separator` and `SeparatedSources`.

## Separator

The `Separator` class is a high level representation of a Demucs audio source separation model. When you want to separate an audio file into its constituent stems, you will first need to create an instance of the `Separator` class which will load the model into memory for use.

```python
separator = Separator(
    model: str | Model | ModelEnsemble = "htdemucs",
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    only_load: str | None = None,
    dtype: torch.dtype | None = None,
    compile: bool = False,
)
```

A `Separator` takes the following parameters:

- `model` - The model to use for separation. While just passing in a string is the easiest, you can use `ModelRepository` to load models manually and then pass them in.
- `device` - The device/backend to use for loading and running the model. Demucs can usually auto-detect the best backend to use based on the availability of the hardware using the heuristic above.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to bag-of-models like htdemucs_ft).
- `dtype` - Optional, set to `torch.float16` for half-precision inference on CUDA. If `None`, uses default float32.
- `compile` - Optional, if `True`, compiles only the `HTDemucs.forward_core()` neural network body on CUDA. This avoids the complex STFT/iSTFT path and improves steady-state throughput for long-lived jobs. Use `Separator.warmup(batch_sizes=[...])` to precompile the batch sizes you expect to serve. Default is `False`.

### Attributes

After construction, the following attributes are available on a `Separator` instance:

- `device` - The device being used for processing (`str`).
- `model` - The loaded model instance (`Model | ModelEnsemble`).
- `audio_channels` - Number of audio channels the model expects (`int`).
- `sample_rate` - Sample rate the model operates at (`int`).

If you enable compilation, you can optionally pre-warm the compiled path for the batch sizes you expect to run:

```python
separator.warmup(batch_sizes=[4, 1])
```

This is mainly useful for long-lived CUDA services such as Cog or Replicate, where paying the compile cost during setup is preferable to paying it on the first live request.

Once you have a `Separator` instance, you can use the `separate` method to separate an audio file into its constituent stems.

```python
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
```

When separating audio, you have the ability to specify the following parameters:

- `audio` - The audio to separate. Can be a tuple of (Tensor, sample_rate), a file path, or raw audio bytes.
- `shifts` - The number of random shifts for equivariant stabilization. In simple terms, this is a technique to make the model more robust to small changes in the audio, such as small shifts in time or pitch. More shifts mean generally higher quality separation but also longer processing time.
- `split` - Whether to split the audio into chunks.
- `split_size` - The size of each chunk in seconds.
- `split_overlap` - The overlap between split chunks. Must be in the range `[0.0, 1.0)`.
- `progress_callback` - A callback function to receive progress updates. View the [Progress Callbacks](#progress-callbacks) section for more information.
- `use_only_stem` - If specified, perform the separation using only the specialized model for this stem. In most cases you should use `only_load` when creating the `Separator` instance instead of this.
- `chunk_batch_size` - Number of split chunks to process in parallel. Defaults to `4` on CUDA and `1` on CPU/MPS.

## SeparatedSources

After running `Separator.separate`, you will be returned a `SeparatedSources` instance. This instance contains the separated audio sources, the sample rate of the audio, and the original audio.

### Attributes

- `sources` - Dictionary mapping stem names (e.g. `"vocals"`, `"drums"`) to their audio tensors (`dict[str, Tensor]`). You can iterate the keys to get available stem names.
- `sample_rate` - Sample rate of the separated audio (`int`), inherited from the model.
- `original` - The original unseparated audio tensor (`Tensor`).

If you're happy with the pure audio stems, you have the ability to export them to an audio container (rather than the Tensors that are stored in the `SeparatedSources` instance).

```python
def export_stem(
    self,
    stem_name: str,
    path: Path | str | None = None,
    format: str = "wav",
    clip: str | None = "rescale",
) -> Path | bytes:
```

When exporting a stem, you have the ability to specify the following parameters:

- `stem_name` - The name of the stem to export.
- `path` - The path to save the stem to. If not provided, the stem will be returned as raw audio bytes.
- `format` - The format to export the stem to. Anything supported by FFmpeg.
- `clip` - The clipping mode to use to prevent audio distortion.

However, Demucs provides an option to be able to isolate a single stem from the `SeparatedSources` instance. This returns a new `SeparatedSources` instance with the chosen stem and an accompanying complement stem (no_{STEM}) that is the sum of all other stems.

```python
def isolate_stem(self, name: str) -> "SeparatedSources":
```

## Auto Model Selection

As Demucs provides many models to perform audio source separation, it is often difficult to know which model to use for a given task. Demucs provides a function to attempt to select the best model for a given task.

```python
def select_model(
    isolate_stem: str | None = None,
) -> tuple[str, str | None]:
```

If you are attempting to isolate a single stem, pass in the name of the stem to the `isolate_stem` parameter.

This will return a tuple of the model name and the stem to exclusively load from the model. When creating a `Separator` instance, you pass these in as the `model` and `only_load` parameters respectively.

## ModelRepository

Demucs provides a `ModelRepository` class to more deeply control the model loading process. This is used internally by the `Separator` class but can be used directly to load models manually to then pass to Separator itself.

`ModelRepository` is initialized with no parameters. (i.e. `repo = ModelRepository()`)

### get_cache_info

```python
def get_cache_info(self) -> dict[str, dict]:
```

This will return a dictionary of information about the cached models.

```python
{
    "model_name": {
        "layers": {       # A dictionary mapping layer checksums to their cache information
            "checksum": {
                "path": Path,      # Path to the cached layer file
                "size_bytes": int, # Size of the layer in bytes
            }
        },
        "size_bytes": int, # Total size of the model in bytes
    },
    ...
}
```

### get_model

```python
def get_model(self, name: str, only_load: str | None = None, progress_callback: Callable[[str, dict[str, Any]], None] | None = None) -> Model | ModelEnsemble:
```

When using the `get_model` method, the following parameters are available:

- `name` - The name of the model to load.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to bag-of-models like htdemucs_ft).
- `progress_callback` - Optional, a callback function to receive progress updates. View the [Progress Callbacks](#progress-callbacks) section for more information.

This will return either a `Model` or `ModelEnsemble` instance corresponding to the given model name.

### list_models

```python
def list_models(self) -> dict[str, dict]:
```

This will return a dictionary of all available models.

```python
{
    "model_name": {
        "description": str, # Description of the model
        "layers": list,     # List of layer configurations
        # ... other metadata fields
    }
}
```

### remove_model

```python
def remove_model(self, name: str) -> bool:
```

Pass in the name of the model you would like to remove and it will remove the weights from the filesystem.

### get_cache_dir

```python
def get_cache_dir(self) -> Path:
```

This will return the directory where the models are cached. This path is fully resolved.

## Progress Callbacks

Demucs provides a callback-based system for monitoring progress during long-running operations like model downloads and audio processing. This system is designed to be UI-agnostic, allowing you to implement a progress display into your own CLI or other application.

All Demucs progress callbacks are designed to use the same API. You should implement a method that matches the following signature:

```python
def progress_callback(event: str, data: dict[str, Any]) -> Any:
    pass
```

### Model Downloading

When using `ModelRepository.get_model` (or creating a `Separator` which calls it internally), the callback receives the following events:

- `download_start`: Fired when the download process begins.
  - `model_name`: Name of the model being downloaded.
  - `total_layers`: Total number of layers to download.
- `layer_start`: Fired when a specific layer starts downloading.
  - `model_name`: Name of the model.
  - `layer_index`: Index of the current layer (1-based).
  - `total_layers`: Total number of layers.
  - `layer_size_bytes`: Size of the layer in bytes.
- `layer_progress`: Fired periodically during download and loading.
  - `model_name`: Name of the model.
  - `layer_index`: Index of the current layer.
  - `total_layers`: Total number of layers.
  - `progress_percent`: Percentage complete (0-100).
  - `downloaded_bytes`: Bytes downloaded so far.
  - `total_bytes`: Total bytes to download.
  - `phase`: Optional. Can be "loading" or "verifying" during those stages.
- `layer_complete`: Fired when a layer is successfully loaded and cached.
  - `model_name`: Name of the model.
  - `layer_index`: Index of the current layer.
  - `total_layers`: Total number of layers.
  - `cached`: Optional. True if the layer was found in cache.
- `download_complete`: Fired when all layers are downloaded and loaded.
  - `model_name`: Name of the model.
  - `total_layers`: Total number of layers.

### Audio Separation

When using `Separator.separate`, the callback receives the following events (only if `split=True`):

- `processing_start`: Fired before processing chunks.
  - `total_chunks`: Total number of chunks to process.
- `chunk_complete`: Fired after each chunk is processed.
  - `completed_chunks`: Number of chunks completed so far.
  - `total_chunks`: Total number of chunks.
- `processing_complete`: Fired after all chunks are processed.
  - `total_chunks`: Total number of chunks.

## Version

You can get the version of the `demucs` package you have installed:

```python
def get_version() -> str:
```

Returns the version string (e.g. `"5.0.0"`).
