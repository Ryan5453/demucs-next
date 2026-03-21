# demucs-next

> [!WARNING]
> `demucs-next` is still in alpha not recommended for production use.

Demucs is a state-of-the-art music source separation model capable of separating drums, bass, and vocals from the rest of the accompaniment.
This is a fork of the [author](https://github.com/adefossez)'s [fork](https://github.com/adefossez/demucs) of the [original Demucs repository](https://github.com/facebookresearch/demucs). It has been updated to use modern versions of Python, PyTorch, and TorchCodec.

## Installation

### Prerequisites

#### FFmpeg

- FFmpeg v4+ available in your `PATH`
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optionally, a working C/C++ compiler such as `g++` if you plan to use `--compile`

### Temporary Installation using UV

With UV, you can use the `uvx` command to run Demucs without installing it permanently on your system. This sets up a temporary virtual enviornment for the duration of the command. 

```bash
uvx demucs-next separate audio_file.mp3
```

**Note**: Demucs does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or PyTorch's default CUDA version (currently 12.8) on Linux when using uvx. Demucs will fall back to CPU if one of the above conditions are not met.

### Install using UV

Create a virtual environment backed by a `uv`-managed Python:

```bash
uv python install 3.12
uv venv --managed-python --python 3.12
source .venv/bin/activate
```

Using a `uv`-managed Python is recommended because it will include the Python headers needed by PyTorch / Triton.

Then install Demucs into that environment:

```bash
uv pip install demucs-next --torch-backend=auto
```

The `--torch-backend=auto` flag automatically detects your GPU and installs the appropriate version of PyTorch compatible with your system.

### Install without UV

Install Demucs using the following command:

```bash
pip install demucs-next
```

**Note**: Demucs does not specify a specific PyTorch wheel. If you want to use a GPU, view the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and append the correct index URL.

## Usage

After installing Demucs, you can use it like the following:

```bash
# View separation options
demucs separate --help

# Separate one audio file
demucs separate audio_file.mp3

# Separate multiple audio files
demucs separate audio_file_1.mp3 audio_file_2.mp3

# Separate all audio files in a directory
demucs separate /path/to/music/folder
```

## Cog Usage

Demucs provides a [Cog](https://github.com/replicate/cog), which allows you to easily deploy a Demucs model as a REST API. You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

## API Usage

Demucs provides a Python API for separating audio files. Please refer to the [API docs](docs/api.md) for more information.

## Changelog

The [changelog](docs/changelog.md) contains information about the changes between versions of demucs-next. 