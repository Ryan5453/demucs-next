# ONNX Export

`demucs-next` includes the ability to export Demucs models to the ONNX format for deployment in browsers, mobile, or other runtimes. 
This is how [demucs.app](https://demucs.app) runs Demucs in-browser.

## Export

You can use the CLI to export a Demucs model to the ONNX format:

```bash
demucs onnx --model htdemucs --output htdemucs.onnx
```

## Model Interface

**Inputs:**
- `spec_real`: Real part of STFT `[B, 2, 2048, T]`
- `spec_imag`: Imaginary part of STFT `[B, 2, 2048, T]`
- `audio`: Raw waveform `[B, 2, samples]`

**Outputs:**
- `out_spec_real`: Separated spectrograms (real) `[B, S, 2, 2048, T]`
- `out_spec_imag`: Separated spectrograms (imag) `[B, S, 2, 2048, T]`
- `out_wave`: Time-domain branch output `[B, S, 2, samples]`

Where `S` = number of sources (4 for htdemucs, 6 for htdemucs_6s).

## Inference Steps

The ONNX model contains only the core neural network - STFT and iSTFT are not included. You'll need to implement these yourself or use an existing FFT library.

### 1. Preprocessing (STFT)

```python
# Constants
NFFT = 4096
HOP = 1024
SEGMENT = 441000  # 10 seconds at 44.1kHz

# Pad audio to segment length
audio = pad(audio, SEGMENT)

# Demucs padding
le = ceil(samples / HOP)
pad_amount = HOP // 2 * 3  # 384
audio_padded = reflect_pad(audio, (pad_amount, pad_amount + le * HOP - samples))

# STFT
z = stft(audio_padded, n_fft=4096, hop_length=1024, window=hann, normalized=True, center=True)

# Trim
z = z[..., :-1, :]      # Remove last freq bin: 2049 -> 2048
z = z[..., 2:2+le]      # Trim time: remove 2 frames each side

spec_real, spec_imag = z.real, z.imag
```

### 2. Run Inference

```python
out_real, out_imag, out_wave = session.run(
    ["out_spec_real", "out_spec_imag", "out_wave"],
    {"spec_real": spec_real, "spec_imag": spec_imag, "audio": audio}
)
```

### 3. Postprocessing (iSTFT + Combine)

```python
for each source:
    # Pad spectrogram back
    z = out_real[s] + 1j * out_imag[s]
    z = pad(z, freq=(0, 1), time=(2, 2))  # Reverse the trimming
    
    # iSTFT
    freq_audio = istft(z, hop_length=1024, window=hann, normalized=True, length=target_len)
    
    # Trim Demucs padding
    freq_audio = freq_audio[..., pad_amount:pad_amount+samples]
    
    # Combine branches
    output[s] = freq_audio + out_wave[s]
```

## Embedded Metadata

The ONNX model includes metadata you can read at runtime:

```python
import onnx
import json

model = onnx.load("htdemucs.onnx")
metadata = {prop.key: prop.value for prop in model.metadata_props}

sources = json.loads(metadata["sources"])         # ["drums", "bass", "other", "vocals"]
sample_rate = int(metadata["sample_rate"])        # 44100
audio_channels = int(metadata["audio_channels"])  # 2
```